# Efficient Filtered Vector Search with Diverse Labels: A Trie-Free and Adaptive Routing Approach

This repository provides **ALPS**, an enhanced implementation for **Filtered Approximate Nearest Neighbor Search (Filtered ANNS)**. ALPS improves **UNG (Unified Navigating Graph)** with an accelerated variant named **TFNG (Trie-Free Navigating Graph)**, and further integrates the core ideas of **TFNG**, **FAVOR**, and **Pre-Filtering**. It introduces a new **AI-based intelligent routing** mechanism that adaptively schedules algorithms using a machine learning model. In addition, the improved version, **ALPS+**, addresses cache thrashing caused by frequent switching among algorithms.

---

## 1. Preparation

### 1.1 Environment Setup

We conduct experiments on a Linux server with two Intel Xeon Gold 6342 processors, 144 threads, and 1 TB RAM. All algorithms are implemented in C++ and python. The project is built with CMake 4.3.2 and Boost 1.85.0.

The experimental environment is as follows:

- CPU: 2 × Intel Xeon Gold 6342 processors
- Threads: 144
- Memory: 1 TB RAM
- Compiler: GCC 11.4.0
- CMake: 4.3.2
- Boost: 1.85.0
- Python: 3.10.20

Configure the Python environment for the experiments using Conda.

Run the following command to create the required environment:

```bash
conda env create -f environment.yml
```

### 1.2 Data Preparation

Download the required datasets from [Hugging Face](https://huggingface.co/datasets/Paper4Review/SmartRoute_data) into the `data` folder in advance.

Please note that `data` is the default directory for storing datasets. The dataset collection includes eight datasets: Amazon, BookReviews, Genome, Laion, Music, Reviews, Tiktok, and VariousImg. Please store each dataset under the `data` directory using the same directory structure.

Each dataset contains the following files:

1. `*_random_300/`: This folder contains the query files used for testing. The wildcard `*` represents the dataset name.
2. `*_base_labels.txt`: This file contains the ground-truth labels corresponding to the base vectors.
3. `*_base.bin`: This binary file contains the base vectors of the dataset.
4. `*_base.fvecs`: This file stores the base vectors in `fvecs` format.

### 1.3 Repository Structure

The main structure of the repository is as follows:

```text
FilterVectorCode/
├── ACORN/                  # Implementation related to ACORN
├── NaviX/                  # Implementation related to NaviX
├── UNG/                    # UNG / TFNG implementation and data-processing scripts
├── knowhere/               # Dependency for the Milvus baseline
├── FAVOR/                  # Dependency for the FAVOR baseline
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

## 2. Experiments

Experiment configurations are stored in the `experiment_json/` directory.

### 2.1 Running Experiments

You can modify the experimental parameters and dataset paths in the corresponding configuration files before running the experiments.

Run the following command to start an experiment:

```bash
./exp.sh experiment_json/experiments-Genome-200-random-300-mix-len.json
```

### 2.2 Experiment Workflow

1. **Configuration Parsing**: `exp.sh` reads the `experiments` array from the JSON configuration file.
2. **Index Construction**: `build_hybrid.sh` is invoked to build indexes. The script supports the `parallel` mode, which builds the UNG and FAVOR indexes simultaneously.
3. **Ground-Truth Generation**: `generate_gt.sh` is invoked to compute the true nearest neighbors, which are used for recall evaluation.
4. **Search Execution**: `search.sh` is invoked to run the search process. It loads the pretrained ONNX router model for online scheduling.

---

## 3. Docker Environment

The Docker image installs the Python and C++ dependencies and precompiles
CRoaring, NaviX, UNG, ACORN, and FAVOR. Dataset files and experiment outputs
are deliberately kept outside the image.

### 3.1 Check the host data layout

The host dataset root must contain one directory per dataset. For example:

```text
/absolute/path/FilterVectorData/
└── Genome/
    ├── Genome_base.bin
    ├── Genome_base.fvecs
    ├── Genome_base_labels.txt
    ├── Genome_base_labels_info.log
    ├── tree_roots.txt
    └── query_select_200_A_B_C-weighted-sub-base-123456789_random_300/
        ├── Genome_query.fvecs
        ├── Genome_query_labels.txt
        └── Genome_query_source_groups.txt
```

The exact query directory name must match `query_dir_name` in the selected
experiment JSON. If `*_base.bin` or `*_query.bin` is absent, the scripts create
it from the corresponding `fvecs` file, so mount the data directory read-write.

### 3.2 Build the image

Run this from the directory containing this README and the Dockerfile:

```bash
cd your_path/ALPS
docker build --build-arg BUILD_JOBS=8 -t alps:cpu .
```

The default image supports ALPS/UNG/TFNG, ACORN, FAVOR, NaviX, pre-filtering,
and Curator. Building can take several minutes and needs substantial RAM.
`BUILD_JOBS` can be reduced when the host has limited memory.

The two Milvus baselines additionally require the large Knowhere/Conan build:

```bash
docker build \
  --build-arg BUILD_JOBS=8 \
  --build-arg ENABLE_KNOWHERE=1 \
  -t alps:cpu-knowhere .
```

### 3.3 Start the container

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

### 3.4 Verify the environment inside the container

```bash
python --version
cmake --version
python -c "import numpy, pandas, sklearn, xgboost, onnx; print('Python dependencies OK')"
test -x /opt/alps-build/ung/apps/search_UNG_index
test -x /opt/alps-build/acorn/demos/test_acorn
test -x /opt/alps-build/favor/app/build_index
echo "C++ binaries OK"
```

### 3.5 Run one algorithm first

The environment variables provided by the image automatically replace the
absolute `data_dir` and `output_dir` stored in the JSON with `/data/<dataset>`
and `/results`. Start with pre-filtering to validate the full index → ground
truth → search pipeline without requiring a selector model:

```bash
ALPS_ALGORITHMS=pre-filter \
bash exp.sh \
  experiment_json/202604-200-random-300-mix-len/experiments-Genome-200-random-300-mix-len.json
```

Run several selected algorithms with a comma-separated list:

```bash
ALPS_ALGORITHMS='UNG-nTfalse,ACORN-gamma,FAVOR,pre-filter' \
bash exp.sh \
  experiment_json/202604-200-random-300-mix-len/experiments-Genome-200-random-300-mix-len.json
```

Unset `ALPS_ALGORITHMS` to run every algorithm in the JSON. Only do that with
the `alps:cpu-knowhere` image if the JSON includes `Milvus-IVF` or
`Milvus-HNSW`.

The first run builds dataset indexes and ground truth and can be very slow for
the full datasets. Later runs reuse existing files under `/results`. Search
automatically runs without `perf` hardware counters when the container lacks
permission; set `ALPS_ENABLE_PERF=0` to disable probing explicitly.

### 3.6 Train the selector model

The training script now reads its result root from `ALPS_RESULTS_DIR`, which is
set to `/results` in the image:

```bash
python selector/smart_route_train.py --configs FAVOR --full-model-only
```

This step expects the EDA/training CSV layout used by the script under
`/results/EDA_Plots_try`. After exporting a model, set `selector_model_path` in
the experiment JSON (or mount/copy it into the corresponding dataset result
directory) before running `ALPS` or `ALPS+`.

---

## 4. Core Parameters

The following parameters in the configuration files or scripts determine the behavior of the algorithms:

| Parameter | Description | Values / Notes |
| :--- | :--- | :--- |
| `ROUTING_MODE` | Determines the routing logic. | `0`: Baseline mode; `1`: **ALPS**; `5`: **ALPS+**. |
| `BASELINE_ALG` | Specifies the algorithm when `ROUTING_MODE=0`. | `0`: UNG; `2`: ACORN-gamma; `4`: NaviX; `5`: Pre-Filtering; `6`: ACORN-1; `8`: UNG+; `9`: Milvus-IVF; `10`: Milvus-HNSW; `11`: FAVOR; `12`: FAVOR-HNSW; `13`: Curator; `14`: UNG++; `15`: TFNG. |
| `BUILD_MODE` | Specifies the index construction mode. | `parallel`: Build all indexes in parallel; `acorn_only`: Build only the ACORN index. |
| `Lsearch` | Search parameter for UNG. | Similar to `efSearch` in HNSW; controls the search depth. |
| `efs_start/step` | Search parameters for ACORN/NaviX. | Used to dynamically adjust the filtering strength during search. |

---

## 5. Model Training and Deployment

The decision model for intelligent routing is trained using Python scripts:

- **Training**: Use `selector/smart_route_train.py`.
- **Deployment**: Export the trained model to `.onnx` format and place it in the `SelectModels` directory, where it can be loaded by the C++ `MethodSelector`.
