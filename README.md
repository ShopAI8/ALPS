# Efficient Filtered Vector Search with Diverse Labels: A Trie-Free and Adaptive Routing Approach

This repository provides **ALPS**, an enhanced implementation for **Filtered Approximate Nearest Neighbor Search (Filtered ANNS)**. ALPS improves **UNG (Unified Navigating Graph)** with an accelerated variant named **TFNG (Trie-Free Navigating Graph)**, and further integrates the core ideas of **TFNG**, **FAVOR**, and **Pre-Filtering**. It introduces a new **AI-based intelligent routing** mechanism that adaptively schedules algorithms using a machine learning model. In addition, the improved version, **ALPS+**, addresses cache thrashing caused by frequent switching among algorithms.

---

## 1. Preparation

### 1.1 Data Preparation

Download the example **Genome** dataset from [Hugging Face](https://huggingface.co/datasets/Paper4Review/SmartRoute_data) into the `data/Genome` directory in advance.

Please note that `data` is the default directory for storing datasets. Keep the downloaded Genome files under `data/Genome` using the original directory structure from Hugging Face.

The Genome dataset contains the following files:

1. `*_random_300/`: These directories contain the query files used for testing.
2. `Genome_base_labels.txt`: This file contains the labels corresponding to the base vectors.
3. `Genome_base.bin`: This binary file contains the base vectors.
4. `Genome_base.fvecs`: This file stores the base vectors in `fvecs` format.

### 1.2 Environment Setup

We conduct experiments on a Linux server with two Intel Xeon Platinum 8360Y processors, 144 threads, and 1 TB RAM. All algorithms are implemented in C++ and python. The project is built with CMake 4.3.2 and Boost 1.85.0.

The experimental environment is as follows:

- CPU: 2 × Intel Xeon Platinum 8360Y processors
- Threads: 144
- Memory: 1 TB RAM
- Compiler: GCC 11.4.0
- CMake: 4.3.2
- Boost: 1.85.0
- Python: 3.10.20

The Docker image installs the Python and C++ dependencies and precompiles
CRoaring, UNG/TFNG, and FAVOR. FAVOR and pre-filtering are internal execution
paths selected by ALPS; the experiment entry point exposes only ALPS, ALPS+,
and TFNG. Dataset files and experiment outputs are kept outside the image.


### 1.2.1 Build the image

Run this from the directory containing this README and the Dockerfile:

```bash
cd your_path/ALPS
docker build --build-arg BUILD_JOBS=8 -t alps:cpu .
```

The image supports ALPS, ALPS+, and TFNG. ACORN, NaviX, Curator, and the two
Milvus/Knowhere baselines are intentionally excluded from this build.
`BUILD_JOBS` can be reduced when the host has limited memory.

### 1.2.2 Start the container

Replace the two host paths below with absolute paths:

```bash
mkdir -p /absolute/path/FilterVectorResults
docker run --rm -it \
  --name alps-dev \
  --shm-size=16g \
  -v /absolute/path/FilterVectorData:/data \
  -v /absolute/path/FilterVectorResults:/results \
  alps:cpu
```

Do not mount another directory over `/workspace/ALPS`, because doing so
would hide the source and binaries built into the image. No `conda activate` is
needed; the container's Python virtual environment is already on `PATH`.

### 1.2.3 Verify the environment inside the container

```bash
python --version
cmake --version
python -c "import numpy, pandas, sklearn, xgboost, onnx; print('Python dependencies OK')"
test -x /opt/alps-build/ung/apps/search_UNG_index
test -x /opt/alps-build/favor/app/build_index
echo "C++ binaries OK"
```



### 2 Repository Structure

The main structure of the repository is as follows:

```text
ALPS/
├── UNG/                    # UNG / TFNG implementation and data-processing scripts
├── FAVOR/                  # Internal high-selectivity path used by ALPS
├── Genome_model/           # Routering model for Genome
├── data/                   # Dataset directory
├── experiment_json/        # Example experiment configurations
├── build_hybrid.sh         # Unified build script
├── generate_gt.sh          # Ground-truth generation script
├── search.sh               # Search and result aggregation script
├── exp.sh                  # Main experiment entry script
├── final_exp.sh            # Supplementary experiment entry script
├── generate_queries.sh     # Query generation script
├── generate_queries_config.json
└── environment.yml         # Recommended Conda environment
```

## 3. Experiments

Experiment configurations are stored in the `experiment_json/` directory.

### 3.1 Running Experiments

You can modify the experimental parameters and dataset paths in the corresponding configuration files before running the experiments.

Run the following command to start an experiment:

```bash
./exp.sh experiment_json/experiments-Genome.json
```

### 3.2 Experiment Workflow

1. **Configuration Parsing**: `exp.sh` reads the `experiments` array from the JSON configuration file.
2. **Index Construction**: `build_hybrid.sh` is invoked to build indexes. The script supports the `parallel` mode, which builds the UNG and FAVOR indexes simultaneously.
3. **Ground-Truth Generation**: `generate_gt.sh` is invoked to compute the true nearest neighbors, which are used for recall evaluation.
4. **Search Execution**: `search.sh` is invoked to run the search process. It loads the pretrained ONNX router model for online scheduling.

---



### 3.3 Details of Experiment Parameters

The following parameters in the configuration files or scripts determine the behavior of the algorithms:

| Parameter | Description | Values / Notes |
| :--- | :--- | :--- |
| `ROUTING_MODE` | Determines the routing logic. | `0`: **TFNG**; `1`: **ALPS**; `5`: **ALPS+**. |
| `BASELINE_ALG` | Selects TFNG when `ROUTING_MODE=0`. | `15`: **TFNG**. |
| `BUILD_MODE` | Specifies the index construction mode. | `serial`, `parallel`, `all`, `ung_only`, `favor_only`, `skip`, or `compile`. |
| `Lsearch` | Search parameter for TFNG. | Similar to `efSearch` in HNSW; controls the search depth. |
| `efs_start/step` | FAVOR-HNSW search parameters used internally by ALPS. | Controls the internal graph-search breadth; ALPS class 0 defaults to this ef-aligned path. |

---

### 3.4 Model Training and Deployment

The decision model for intelligent routing is trained using Python scripts:

- **Training**: Use `selector/smart_route_train.py`.
- **Deployment**: Export the trained model to `.onnx` format and place it in the `SelectModels` directory, where it can be loaded by the C++ `MethodSelector`.
