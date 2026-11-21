"""Analytical path balancing for a placed and routed Zircon RV app.

usage: python pb_plan.py <bin dir> --stream N [--out-json path_balancing.json]
           [--out-place manual.place] [--max-len 16] [--report]

Model (cycle counts measured in RTL waveforms of the conv1d kernel; the REG,
ROM and PE numbers agree with aha-agate's auto_balance model):
- every REG on a route is one cycle (a 2-deep FIFO with registered ready);
- a PE adds 2 cycles from its input port to its output port;
- a ROM MEM is rigid and adds 4 cycles;
- a streaming MEM (fifo / dual_read lake config) is elastic: each read port
  can run behind its write side, so each read port is a free source;
- a FIFO Pond feeding a PE input is elastic on that input (it never stalls its
  writer); it adds 1 cycle, and how long a word waits in it is its occupancy,
  which must stay below the 1-D FIFO guard (28 words here);
- a repeat Pond (weights, biases) is an always-ready source;
- a path-balance Pond of length b inserted on a PE input or output adds b
  cycles (lake get_path_balancing_pond: b interconnect FIFOs);
- an explicit register with k initial tokens shifts its stream by k tokens,
  i.e. reduces the path's effective latency by k;
- the lanes of one E64 multi-bank GLB tile leave together (one input group
  node) and are written together (one output group node).

A rigid fork-join (two rigid forks that meet, including two different sources
meeting at several PEs) runs at full rate only if every cycle of rigid edges is
balanced: there is a firing time t(v) per node with t(v) = t(u) + L(e) for
every rigid edge e = (u, v). The solver chooses the times of the free nodes
(sources, MEM read ports) and the path-balance Ponds (one per PE tile, none in
a tile that already holds a Pond, length in a set that divides the stream) to
minimize the remaining slack on rigid edges. Remaining slack on a rigid edge
is a rate loss; slack on a FIFO Pond edge is that Pond's occupancy.

The model assumes every edge carries one token per cycle. An edge at rate r
tolerates a long branch up to (2 / r - 1) times the short one, so slack on
low-rate edges (e.g. the half-rate quantizer side of the fused MX kernel) is
overstated; check the IO handshakes of a waveform before inserting Ponds.

Inserting the planned Ponds (second run): copy the --out-json file to the
app's path_balancing_configs/path_balancing.json and --out-place to the app's
manual.place, then run with POND_PATH_BALANCING=1 and
MANUAL_PATH_BALANCE_PLACER=1. No committed kernel needs this today.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pulp

PE_LAT, ROM_LAT, POND_LAT, MEM_LAT = 2, 4, 1, 14
POND_OCC_LIMIT = 28

ap = argparse.ArgumentParser()
ap.add_argument("bin")
ap.add_argument("--stream", type=int, required=True, help="tokens per lane through every balanced PE")
ap.add_argument("--max-len", type=int, default=16)
ap.add_argument("--out-json", default="")
ap.add_argument("--out-place", default="")
ap.add_argument("--report", action="store_true")
ap.add_argument("--no-balance", action="store_true", help="only analyze (no balance ponds)")
args = ap.parse_args()
b = Path(args.bin)

# ---------------------------------------------------------------- parsing
id2name = {}
for line in (b / "design.id_to_name").read_text().splitlines():
    if ": " in line:
        k, v = line.split(": ", 1)
        id2name[k.strip()] = v.strip()
name2id = {v: k for k, v in id2name.items()}
place = {}
for line in (b / "design.place").read_text().splitlines()[2:]:
    f = line.split()
    if len(f) >= 4 and f[-1].startswith("#"):
        place[f[-1][1:]] = (int(f[-3]), int(f[-2]))
nets = {}
for line in (b / "design.packed").read_text().split("ID to Names:")[0].splitlines():
    m = re.match(r"^(e\d+): (.*)$", line.strip())
    if m:
        pins = re.findall(r"\((\w+), (\w+)\)", m.group(2))
        nets[m.group(1)] = (pins[0], pins[1:])
routes = defaultdict(list)
cur = None
for line in (b / "design.route").read_text().splitlines():
    line = line.strip()
    if line.startswith("Net ID:"):
        cur = line.split()[2]
        routes[cur] = []
    elif line.startswith("Segment:"):
        routes[cur].append([])
    elif line and cur is not None:
        routes[cur][-1].append(line)

top_json = json.loads((b / "design_top.json").read_text())
mods = top_json["namespaces"]["global"]["modules"]
top = max((v for v in mods.values() if "instances" in v), key=lambda v: len(v["instances"]))
insts = top["instances"]
conns = top.get("connections", [])

glb_cfg = json.loads((b / "glb_bank_config.json").read_text())
multibank = {}
for direction in ("inputs", "outputs"):
    for entry in glb_cfg.get(direction, []):
        for _, e in entry.items():
            for x in e.get("x_coord", []):
                multibank[x] = e.get("E64_packed", [1])[0] != 0


def node_key(node):
    m = re.match(r"REG (T\d+_\w+) \((\d+), (\d+), (\d+), \d+\)", node)
    if m:
        return ("REG", m.group(1), int(m.group(3)), int(m.group(4)))
    m = re.match(r"PORT (\w+) \((\d+), (\d+), \d+\)", node)
    if m:
        return ("PORT", m.group(1), int(m.group(2)), int(m.group(3)))
    return None


def pin_key(inst, port):
    x, y = place[inst]
    if inst.startswith("r"):
        return ("REG", id2name[inst].rsplit("@", 1)[1], x, y)
    return ("PORT", port, x, y)


def route_dists(segs):
    succ = defaultdict(list)
    has_pred = set()
    for seg in segs:
        for a, c in zip(seg, seg[1:]):
            succ[a].append(c)
            has_pred.add(c)
    roots = [n for seg in segs for n in seg[:1] if n not in has_pred]
    dist = {}
    stack = [(r, 0) for r in roots]
    while stack:
        n, d = stack.pop()
        if n in dist:
            continue
        dist[n] = d
        for c in succ[n]:
            stack.append((c, d + (1 if c.startswith("REG") else 0)))
    by_key = {}
    for n, d in dist.items():
        k = node_key(n)
        if k:
            by_key[k] = min(d, by_key.get(k, 1 << 30))
    return by_key


route_keys = {net: route_dists(segs) for net, segs in routes.items()}


def net_latencies(net):
    """REG count from the net source to each sink pin (register chains are
    stored under the first net of the chain)."""
    src, sinks = nets[net]
    if net not in routes and src[0].startswith("r"):
        sk = pin_key(*src)
        for keys in route_keys.values():
            if sk in keys:
                return {s: (keys[pin_key(*s)] - keys[sk]) if pin_key(*s) in keys else None for s in sinks}
        return {s: None for s in sinks}
    keys = route_keys.get(net, {})
    return {s: keys.get(pin_key(*s)) for s in sinks}


# ----------------------------------------------------------- classification
def short(i):
    return id2name.get(i, i).split("@")[0]


reg_init = {}
for i, n in id2name.items():
    if i[0] == "r":
        meta = insts.get(n.split("@")[0], {}).get("metadata", {})
        reg_init[i] = int(meta.get("extra_data", 0) or 0)

mem_kind = {}
for i, n in id2name.items():
    if i[0] == "m":
        meta = insts.get(n, {}).get("metadata", {})
        mem_kind[i] = "rom" if meta.get("is_rom") else "stream"

# Ponds packed into PE tiles: reader PE name -> pond config type
pond_type_of_pe = {}
for a, c in conns:
    for src, dst in ((a, c), (c, a)):
        si, sp = src.split(".", 1)
        di = dst.split(".", 1)[0]
        if "_path_balance_pond" in si:
            # named after its host PE id; on a PE output it drives other tiles
            host = id2name.get(si.split("_path_balance_pond")[0])
            if host:
                pond_type_of_pe[host] = "path_balance"
            continue
        if sp.startswith("data_out_pond") and insts.get(si, {}).get("genref") == "cgralib.Pond" \
                and insts.get(di, {}).get("modref") == "global.PE":
            cfg = insts[si].get("metadata", {}).get("lake_rv_config", {})
            if isinstance(cfg, str):
                cfg = json.loads(cfg)
            typ = cfg.get("type", "?")
            if "_path_balance_pond" in si:
                typ = "path_balance"
            pond_type_of_pe.setdefault(di, typ)
pb_json = b / "path_balancing.json"
pb_len = json.loads(pb_json.read_text())["balance_lengths"] if pb_json.exists() else {}
pe_has_pond = set()
for (src, sport), sinks in nets.values():
    for s in sinks:
        if s[1].startswith("PondTop_input"):
            pe_has_pond.add(s[0])


def io_group(inst):
    x = place[inst][0]
    if multibank.get(x, True):
        return f"GLB{x // 2}"
    return f"IO{x}"


# balance Pond direction and PE port, from path_balancing.json
pb_out, pb_port = {}, {}
if pb_json.exists():
    for pe, (to_pond, port) in json.loads(pb_json.read_text())["pe_to_pond"].items():
        pb_out[pe] = bool(to_pond)
        if not to_pond:
            pb_port[pe] = f"PE_input_width_17_num_{port[-1]}"

# ------------------------------------------------------------------- edges
# Graph nodes: PE ids, ROM ids, MEM read ports "m3:o0", MEM write side "m3:i",
# IO groups "in:GLB6" / "out:GLB10" / "in:IO10".
edges = []  # dict(src, dst, port, lat, kind) kind in rigid / pond / pondrep / free


def node_of_source(inst, port):
    if inst[0] == "I":
        return f"in:{io_group(inst)}"
    if inst[0] == "m":
        if mem_kind[inst] == "rom":
            return inst
        return f"{inst}:o{port[-1]}"
    return inst


def node_of_sink(inst, port):
    if inst[0] == "I":
        return f"out:{io_group(inst)}"
    if inst[0] == "m":
        return inst if mem_kind[inst] == "rom" else f"{inst}:i"
    return inst


reg_out = defaultdict(list)  # reg -> net ids it drives
for net, ((src, sport), sinks) in nets.items():
    if src[0] == "r":
        reg_out[src].append(net)


def walk(src_node, net, lat0, init0):
    lat = net_latencies(net)
    for s in nets[net][1]:
        inst, port = s
        if lat[s] is None:
            continue
        l = lat0 + lat[s]
        if inst[0] == "r":
            for n2 in reg_out[inst]:
                walk(src_node, n2, l, init0 + reg_init.get(inst, 0))
            continue
        if node_of_sink(inst, port) == src_node:
            continue  # in-tile Pond -> own PE
        dst = node_of_sink(inst, port)
        kind = "rigid"
        if port.startswith("PondTop_input"):
            pname = id2name[inst]
            typ = pond_type_of_pe.get(pname, "?")
            kind = {"get_fifo_pond": "pond", "path_balance": "rigid"}.get(typ, "pondrep")
            if typ == "path_balance":
                # a balance Pond in front of this PE input: b cycles
                l += pb_len[inst]
                port = pb_port.get(inst, "PE_input_width_17_num_0")
            else:
                port = "PE_input_width_17_num_0"
        edges.append(dict(src=src_node, dst=dst, port=port, lat=l - init0, kind=kind, inst=inst))


for net, ((src, sport), sinks) in nets.items():
    if src[0] == "r":
        continue
    if src[0] == "p" and sport.startswith("PondTop_output"):
        if all(s[0] == src for s in sinks):
            continue  # in-tile Pond -> PE: modeled on the Pond's input edge
        # a balance Pond on this PE's output drives the interconnect: b cycles
        walk(src, net, pb_len.get(src, 0), 0)
        continue
    if src[0] == "p" and any(s[0] == src and s[1].startswith("PondTop_input") for s in sinks) \
            and pond_type_of_pe.get(id2name[src]) == "path_balance" and pb_out.get(src):
        continue  # PE -> its own output balance Pond (the Pond's output net carries on)
    walk(node_of_source(src, sport), net, 0, 0)

nodes = sorted({e["src"] for e in edges} | {e["dst"] for e in edges})


def node_delay(n):
    if n.startswith(("in:", "out:")) or ":o" in n or n.endswith(":i"):
        return 0
    if n[0] == "M" and "_path_balance_pond" in id2name.get(n, ""):
        # an unpacked balance Pond placed in its host PE tile: b cycles
        return pb_len[id2name[n].split("_path_balance_pond")[0]]
    if n[0] == "m":
        return ROM_LAT
    if n[0] == "r":
        return 0
    return PE_LAT


# ---------------------------------------------------------------- solver
# t[n]: fire time of n (inputs consumed); output leaves at t[n] + delay(n).
prob = pulp.LpProblem("path_balance", pulp.LpMinimize)
t = {n: pulp.LpVariable(f"t_{re.sub(r'[^A-Za-z0-9]', '_', n)}", lowBound=0) for n in nodes}
# MEM streaming: read ports run after the write side, otherwise free.
mem_ports = defaultdict(list)
for n in nodes:
    if ":o" in n:
        mem_ports[n.split(":")[0]].append(n)
for m, ports in mem_ports.items():
    if f"{m}:i" in t:
        for p in ports:
            prob += t[p] >= t[f"{m}:i"] + MEM_LAT

# A length-1 balance Pond is a 1-D schedule over the whole stream, and Pond
# extent counters are 11-bit signed (at most 1024); longer streams need b >= 2
# (lake get_path_balancing_pond, as in archipelago auto_path_balancing).
min_len = 1 if args.stream <= 1024 else 2
lengths = [v for v in range(min_len, args.max_len + 1) if args.stream % v == 0]
site_vars = {}  # (pe, site) -> {length: binary}; site "in<k>" or "out"
usable = lambda pe: pe[0] == "p" and pe not in pe_has_pond and not args.no_balance


def site(pe, s):
    if (pe, s) not in site_vars:
        site_vars[(pe, s)] = {v: pulp.LpVariable(f"z_{pe}_{s}_{v}", cat="Binary") for v in lengths}
    return pulp.lpSum(v * z for v, z in site_vars[(pe, s)].items())


resid = []
pond_occ = []
for k, e in enumerate(edges):
    u, v = e["src"], e["dst"]
    arrive = t[u] + node_delay(u) + e["lat"]
    if usable(u):
        arrive = arrive + site(u, "out")
    if e["kind"] == "pondrep":
        continue
    if e["kind"] == "pond":
        occ = pulp.LpVariable(f"occ_{k}", lowBound=0, upBound=POND_OCC_LIMIT)
        prob += t[v] == arrive + POND_LAT + occ
        pond_occ.append((e, occ))
        continue
    if v.endswith(":i"):
        continue  # a streaming MEM's write side never makes its writer wait
    if usable(v) and e["port"].startswith("PE_input"):
        arrive = arrive + site(v, f"in{e['port'][-1]}")
    r = pulp.LpVariable(f"r_{k}", lowBound=0)
    prob += t[v] == arrive + r
    resid.append((e, r))
# one balance Pond per PE tile
by_pe = defaultdict(list)
for (pe, s), zs in site_vars.items():
    by_pe[pe] += list(zs.values())
for pe, zs in by_pe.items():
    prob += pulp.lpSum(zs) <= 1
# A Pond input fed by this PE's own output would not see the fork's stalls
# (no fan-out handshake inside a tile): never put a balance Pond on the output
# of a PE that also writes an in-tile Pond. (Our FIFO Ponds sit in the reader's
# tile, so this only guards against future layouts.)
prob += 1000 * pulp.lpSum(r for _, r in resid) + pulp.lpSum(
    v * z for zs in site_vars.values() for v, z in zs.items()) + 0.001 * pulp.lpSum(t.values())
status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))
print(f"[INFO] {len(nodes)} nodes, {len(edges)} edges, {len(site_vars)} balance sites, "
      f"solver {pulp.LpStatus[status]}")

# ---------------------------------------------------------------- report
val = lambda x: int(round(pulp.value(x) or 0))
chosen = {}
for (pe, s), zs in site_vars.items():
    for v, z in zs.items():
        if val(z):
            chosen[pe] = (s, v)
tot_resid = sum(val(r) for _, r in resid)
print(f"[INFO] balance Ponds: {len(chosen)} (total {sum(v for _, v in chosen.values())} cycles); "
      f"remaining rigid slack: {tot_resid} cycles on {sum(1 for _, r in resid if val(r))} edges")
bad = sorted(((val(r), e) for e, r in resid if val(r)), key=lambda x: -x[0])
for r, e in bad[:40]:
    print(f"  slack {r:3d}  {short(e['src']) if e['src'][0] in 'pm' else e['src']} -> "
          f"{short(e['dst']) if e['dst'][0] in 'pm' else e['dst']} ({e['port'][-1] if 'input' in e['port'] else e['port']})")
occs = sorted(((val(o), e) for e, o in pond_occ), key=lambda x: -x[0])
print(f"[INFO] FIFO Pond occupancy: max {occs[0][0] if occs else 0} words (limit {POND_OCC_LIMIT})")
if args.report:
    for o, e in occs[:12]:
        print(f"  occ {o:3d}  {short(e['src']) if e['src'][0] in 'pm' else e['src']} -> Pond of {short(e['dst'])}")
for pe, (s, v) in sorted(chosen.items(), key=lambda kv: short(kv[0])):
    print(f"  pond {v:2d} at {s:4s} of {short(pe)} {place[pe]}")

if args.out_json:
    meta = {"balance_lengths": {}, "total_stream_lengths": {}, "name_to_id": {}, "pe_to_pond": {}}
    for pe, (s, v) in chosen.items():
        meta["balance_lengths"][pe] = v
        meta["total_stream_lengths"][pe] = args.stream
        meta["name_to_id"][id2name[pe]] = pe
        meta["pe_to_pond"][pe] = [True, ""] if s == "out" else [False, f"data{s[2:]}"]
    Path(args.out_json).write_text(json.dumps(meta, indent=4) + "\n")
    print(f"[INFO] wrote {args.out_json}")
if args.out_place:
    lines = [f"{i} {x} {y}" for i, (x, y) in sorted(place.items()) if i[0] in "pm"]
    Path(args.out_place).write_text("\n".join(lines) + "\n")
    print(f"[INFO] wrote {args.out_place} ({len(lines)} tiles)")
