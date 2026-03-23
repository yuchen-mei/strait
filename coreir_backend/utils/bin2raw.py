import argparse
import numpy as np
import os
import sys

def convert_bin_to_raw(input_path, output_path, target_dtype):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)

    try:
        tensor_fp32 = np.fromfile(input_path, dtype=np.float32)

        if target_dtype in ['int8', 'uint8']:
            if target_dtype == 'int8':
                data = tensor_fp32.astype(np.int8).view(np.uint8).astype(np.uint16)
            else:
                data = tensor_fp32.astype(np.uint8).astype(np.uint16)

        elif target_dtype in ['int16', 'uint16']:
            if target_dtype == 'int16':
                data = tensor_fp32.astype(np.int16).view(np.uint16)
            else:
                data = tensor_fp32.astype(np.uint16)

        elif target_dtype == 'float16':
            data = tensor_fp32.astype(np.float16).view(np.uint16)

        elif target_dtype == 'bfloat16':
            data = (tensor_fp32.view(np.uint32) >> 16).astype(np.uint16)

        else:
            raise ValueError(f"Unsupported dtype: {target_dtype}")

        data_swapped = data.byteswap()
        data_swapped.tofile(output_path)

        print(f"[bin2raw] Converted {len(tensor_fp32)} fp32 values to {target_dtype} and saved to {output_path}")

    except Exception as e:
        raise RuntimeError(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an fp32 binary tensor to a byte-swapped .raw file.")
    parser.add_argument("input", help="Path to the input .bin file")
    parser.add_argument("output", help="Path to the output .raw file")
    parser.add_argument("--dtype", default="int16",
                        choices=["int8", "uint8", "int16", "uint16", "float16", "bfloat16"],
                        help="Target data type representation (default: int16)")

    args = parser.parse_args()

    convert_bin_to_raw(args.input, args.output, args.dtype)