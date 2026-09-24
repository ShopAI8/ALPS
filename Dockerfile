# syntax=docker/dockerfile:1
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=8
ARG PREBUILD=1
ARG ENABLE_KNOWHERE=0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        g++-11 \
        gcc-11 \
        git \
        jq \
        libaio-dev \
        libboost-filesystem-dev \
        libboost-program-options-dev \
        libcurl4-openssl-dev \
        libdouble-conversion-dev \
        libevent-dev \
        libgflags-dev \
        libgtest-dev \
        liblapack-dev \
        libopenblas-dev \
        libomp-dev \
        liburing-dev \
        ninja-build \
        nlohmann-json3-dev \
        numactl \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/alps-venv
ENV PATH="/opt/alps-venv/bin:${PATH}"

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir "cmake>=3.24,<5" -r /tmp/requirements-docker.txt

WORKDIR /workspace/ALPS20260922
COPY . .
RUN chmod +x build_hybrid.sh exp.sh generate_gt.sh search.sh generate_queries.sh

# UNG expects CRoaring at this exact in-tree location.
RUN cmake -S UNG/codes/third_party/CRoaring \
          -B UNG/codes/third_party/CRoaring/build \
          -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_ROARING_TESTS=OFF \
          -DROARING_USE_CPM=OFF \
    && cmake --build UNG/codes/third_party/CRoaring/build --parallel "${BUILD_JOBS}"

# Knowhere powers the two Milvus baselines and is intentionally opt-in because
# its Conan dependency build is large. Core ALPS/UNG/ACORN/FAVOR/NaviX does not
# require it.
RUN if [[ "${ENABLE_KNOWHERE}" == "1" ]]; then \
        python -m pip install --no-cache-dir conan==2.25.1; \
        conan profile detect --force; \
        conan remote add default-conan-local2 \
          https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2 --force; \
        make -C knowhere -j"${BUILD_JOBS}"; \
    fi

# Compile once into a shared build directory. exp.sh reuses these artifacts for
# every dataset instead of recompiling them under build_para_<dataset>.
RUN mkdir -p /opt/alps-build /tmp/alps-empty-data /tmp/alps-build-output \
    && if [[ "${PREBUILD}" == "1" ]]; then \
         if [[ "${ENABLE_KNOWHERE}" == "1" ]]; then \
           export KNOWHERE_INCLUDE_DIR=/workspace/ALPS20260922/knowhere/include; \
           export KNOWHERE_LIBRARY=/workspace/ALPS20260922/knowhere/build/Release/libknowhere.so; \
         else \
           export KNOWHERE_INCLUDE_DIR=/nonexistent/knowhere/include; \
           export KNOWHERE_LIBRARY=/nonexistent/libknowhere.so; \
         fi; \
         BUILD_JOBS="${BUILD_JOBS}" \
         NAVIX_BUILD_DIR=/opt/alps-build/navix \
         UNG_BUILD_DIR=/opt/alps-build/ung \
         ACORN_BUILD_DIR=/opt/alps-build/acorn \
         FAVOR_BUILD_DIR=/opt/alps-build/favor \
         bash build_hybrid.sh \
           --build_mode compile \
           --query_dir_name unused \
           --dataset DockerSmoke \
           --data_dir /tmp/alps-empty-data \
           --exp_output_dir /tmp/alps-build-output \
           --max_degree 32 \
           --Lbuild 100 \
           --alpha 1.2 \
           --num_cross_edges 6 \
           --num_entry_points 16 \
           --acorn_n 1 \
           --acorn_m 32 \
           --acorn_m_beta 64 \
           --acorn_gamma 80; \
       fi \
    && rm -rf /tmp/alps-empty-data /tmp/alps-build-output

ENV ALPS_BUILD_ROOT=/opt/alps-build \
    ALPS_DATA_ROOT=/data \
    ALPS_OUTPUT_ROOT=/results \
    ALPS_RESULTS_DIR=/results \
    ALPS_ENABLE_PERF=auto \
    LD_LIBRARY_PATH=/workspace/ALPS20260922/UNG/codes/third_party/onnxruntime-linux-x64-1.16.3/lib:/workspace/ALPS20260922/knowhere/build/Release

VOLUME ["/data", "/results"]
CMD ["bash"]
