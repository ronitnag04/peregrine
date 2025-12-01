import os
import glob
import numpy as np


RESOURCE_FILES = [
    "thr_rob.npy",
    "thr_load_queue.npy",
    "thr_store_queue.npy",
    "thr_alu_issue.npy",
    "thr_fp_issue.npy",
    "thr_ls_issue.npy",
    "thr_load_ls_pipes_lower.npy",
    "thr_load_ls_pipes_upper.npy",
    "thr_icache_fills.npy",
    "thr_fetch_buffers.npy",
]

# Resources with two parameters (p0, p1); others have one parameter (p0 only)
DOUBLE_PARAM_RESOURCES = {
    "load_ls_pipes_lower",
    "load_ls_pipes_upper",
}


def load_resource_file(path, res_name):
    """
    Load a single resource .npy file.

    Layout (current C++ writer):
      single-param resources:
        [p0, thr_0, ..., thr_{N-1}]
      double-param resources:
        [p0, p1, thr_0, ..., thr_{N-1}]

    Returns:
        params: (num_combos, 1 or 2) array of parameter values (p0[, p1])
        thr:    (num_combos, num_windows) array of throughputs
    """
    arr = np.load(path)  # shape: (num_combos, param_cols + num_windows)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D array, got shape {arr.shape}")

    num_combos, total_cols = arr.shape

    double_params = res_name in DOUBLE_PARAM_RESOURCES
    param_cols = 2 if double_params else 1
    num_windows = total_cols - param_cols
    if num_windows <= 0:
        raise ValueError(
            f"{path}: invalid layout, total_cols={total_cols}, "
            f"param_cols={param_cols}"
        )

    params = arr[:, :param_cols]
    thr = arr[:, param_cols:]
    return params, thr


def load_all(output_dir="output"):
    """
    Load all known throughput npy files from 'output_dir'.

    Returns:
        dict: {resource_name: (params, thr)}
              where:
                params: (num_combos, 1 or 2)
                thr:    (num_combos, num_windows)
    """
    results = {}
    for fname in RESOURCE_FILES:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        res_name = os.path.splitext(fname)[0].replace("thr_", "")
        params, thr = load_resource_file(path, res_name)
        results[res_name] = (params, thr)
    return results


def summarize(output_dir="output"):
    """
    Print a small human-readable summary for each resource.
    """
    results = load_all(output_dir)
    if not results:
        print(f"No throughput files found in {output_dir}")
        return

    for res_name, (params, thr) in results.items():
        num_combos, num_windows = thr.shape
        param_count = params.shape[1]

        print("=" * 60)
        print(f"Resource: {res_name}")
        print(f"  Num param combos: {num_combos}")
        print(f"  Num windows     : {num_windows}")
        print(f"  Param count     : {param_count}")

        for i in range(num_combos):
            row_params = params[i]

            if param_count == 1:
                p_str = f"p0={row_params[0]:.0f}"
            else:
                p_str = f"p0={row_params[0]:.0f}, p1={row_params[1]:.0f}"

            vals = thr[i]
            avg = float(np.mean(vals))
            mn = float(np.min(vals))
            mx = float(np.max(vals))
            print(f"    combo {i}: {p_str}")
            print(f"      avg={avg:.4f}, min={mn:.4f}, max={mx:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and summarize throughput .npy files from ./output"
    )
    parser.add_argument(
        "-d", "--dir", default="output", help="Output directory (default: output)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available .npy files and exit"
    )
    args = parser.parse_args()

    if args.list:
        files = glob.glob(os.path.join(args.dir, "thr_*.npy"))
        if not files:
            print(f"No files matching thr_*.npy in {args.dir}")
        else:
            print("Found files:")
            for f in sorted(files):
                print("  ", f)
    else:
        summarize(args.dir)
