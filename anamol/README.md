Here is a comprehensive README for your team, documenting the structure, build process, and usage of the analytical modeling tools.

-----

# Analytical Models

This repository contains a C++20-based analytical modeling framework for estimating processor throughput, along with Python scripts for post-processing and analysis.

## 1\. Project Structure

```text
analytical/
├── Makefile              # Build automation (compile, test, benchmark)
├── include/              # C++ header files (models.h, parser.h, instr.h)
├── src/                  # C++ source files (main logic, models, parsing)
├── tests/                # Unit tests for parameters and models
├── traces/               # Input trace files (e.g., trace.csv)
├── output/               # Output directory for generated .npy throughput files
├── python/               # Post-processing scripts
│   ├── anamol_post.py         # Interactive DataFrame loading/analysis
│   ├── example_read_thr.py    # CLI tool to summarize output .npy files
│   └── example_build_cdfs.py  # Generates CDF feature vectors from outputs
```

## 2\. Prerequisites

  * **Compiler**: A C++20 compliant compiler (Makefile defaults to `g++-15`).
  * **Libraries**: OpenMP (`libomp`) for parallel execution.
  * **Python**: Python 3.x with `numpy` and `pandas` installed for post-processing.

## 3\. Building and Testing

The project uses a `Makefile` for all build operations.

### Build Targets

  * **`make` / `make all`**: Compiles the main executable and all tests.
  * **`make anamol`**: Compiles the main executable (`build/main`) and copies it to the root directory as `anamol`.
  * **`make clean`**: Removes the `build/` directory.

### Running Tests & Benchmarks

  * **`make test_params.run`**: Runs unit tests for configuration parameters.
  * **`make benchmark`**: Compares performance between the serial and OpenMP (parallel) versions of the model.

## 4\. Running the Model

The core tool reads a trace file, calculates throughputs using a sliding window, and exports the results to the `output/` directory.

### Basic Usage

You can run the executable directly after building:

```bash
./anamol -t traces/trace.csv -w 400
```

### Command Line Arguments

| Flag | Long Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| `-t` | `--tracefile` | Path to the input CSV trace file. | `trace.csv` |
| `-w` | `--window` | Sliding window size (instructions per window). | `400` |

### Makefile Shortcuts

You can also run the model using Make convenience targets:

```bash
# Runs ./anamol with default parameters
make anamol.run

# Override defaults using Make variables
make anamol.run TRACE_CSV=traces/my_trace.csv WINDOW_SIZE=1000
```

## 5\. Output Data

The model exports throughput data into the `output/` directory as **NumPy (.npy)** files. Each file corresponds to a specific hardware resource (e.g., `thr_rob.npy`, `thr_alu_issue.npy`).

**File Format:**

  * Files are 2D arrays: `[rows, columns]`
  * **Columns**: Parameters (1 or 2 columns) + Windows (remaining columns).
  * **Rows**: Different parameter combinations (e.g., varying ROB sizes).

## 6\. Python Post-Processing

Scripts in the `python/` directory are provided to analyze the raw `.npy` outputs.

### Summarizing Results

Use `example_read_thr.py` to get a quick textual summary of the generated throughputs (min, max, average) for every parameter combination.

```bash
python python/example_read_thr.py --dir output
```

### Building CDFs

Use `example_build_cdfs.py` to convert raw throughput traces into Cumulative Distribution Function (CDF) feature vectors.

```bash
python python/example_build_cdfs.py --input-dir output --output-dir output_cdf
```

This will:

1.  Read all `.npy` files from `output/`.
2.  Compute raw and weighted percentiles.
3.  Save the resulting features to `output_cdf/`.

### Interactive Analysis

For Jupyter Notebooks or interactive shells, use `anamol_post.py`. It provides helper functions to load all resources directly into Pandas DataFrames:

```python
from analytical.python.anamol_post import build_cdfs_and_dataframes_per_resource

# Loads all data into a dictionary of DataFrames
dfs = build_cdfs_and_dataframes_per_resource(input_dir="output")
print(dfs['rob'].head())
```