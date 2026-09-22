cmake_minimum_required(VERSION 3.15)
project(ANNS)

# cmake setting
set(CMAKE_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_KNOWHERE_MILVUS_BASELINE "Enable Milvus baseline via Knowhere engine" ON)
set(ENABLE_KNOWHERE_MILVUS_BASELINE ON CACHE BOOL "Enable Milvus baseline via Knowhere engine")
option(UNG_ENABLE_RABITQ "Enable RabitQ side-index acceleration" OFF)
set(KNOWHERE_INCLUDE_DIR "" CACHE PATH "Knowhere include directory (should contain knowhere/index/index_factory.h)")
set(KNOWHERE_LIBRARY "" CACHE FILEPATH "Path to libknowhere.so or libknowhere.a")

# Add the ACORN/Faiss project as a subdirectory of the UNG project
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable Faiss GPU support")
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Faiss Python bindings")
set(BUILD_TESTING OFF CACHE BOOL "Disable Faiss tests")
add_subdirectory(../../ACORN ${CMAKE_BINARY_DIR}/acorn_build)

# Boost
find_package(Boost REQUIRED COMPONENTS program_options filesystem)

if(NOT Boost_FOUND)
    message(FATAL_ERROR "no Boost support")
endif()

message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

# build options
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")
option(UNG_FORCE_GLIBCXX_ABI0 "Force -D_GLIBCXX_USE_CXX11_ABI=0 for legacy prebuilt dependencies" OFF)
if(UNG_FORCE_GLIBCXX_ABI0)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
    message(WARNING "UNG_FORCE_GLIBCXX_ABI0=ON: building with old libstdc++ ABI (0).")
else()
    message(STATUS "UNG_FORCE_GLIBCXX_ABI0=OFF: using default libstdc++ ABI (usually 1).")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -DNDEBUG -Ofast -march=native -mtune=native")

# include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third_party/CRoaring/include)
include_directories(${PROJECT_SOURCE_DIR}/../../FAVOR/include)
if(UNG_ENABLE_RABITQ)
    include_directories(${PROJECT_SOURCE_DIR}/../../../RaBitQ/RaBitQ-Library/include)
endif()

if(ENABLE_KNOWHERE_MILVUS_BASELINE)
    if(NOT EXISTS "${KNOWHERE_INCLUDE_DIR}/knowhere/index/index_factory.h")
        message(FATAL_ERROR "Knowhere is mandatory but KNOWHERE_INCLUDE_DIR is invalid: ${KNOWHERE_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${KNOWHERE_LIBRARY}")
        message(FATAL_ERROR "Knowhere is mandatory but KNOWHERE_LIBRARY not found: ${KNOWHERE_LIBRARY}")
    endif()
else()
    message(STATUS "Knowhere (Milvus) baseline disabled. OK for non-Milvus algorithms.")
endif()
find_library(URING_LIB NAMES uring)
if(URING_LIB)
    message(STATUS "liburing found at: ${URING_LIB}")
    add_link_options(-Wl,--no-as-needed)
else()
    message(WARNING "liburing not found. Knowhere/Folly link may fail on io_uring symbols.")
endif()
if(ENABLE_KNOWHERE_MILVUS_BASELINE)
    add_compile_definitions(ENABLE_KNOWHERE_MILVUS_BASELINE=1)
    add_compile_definitions(GLOG_USE_GLOG_EXPORT=1)
    include_directories(${KNOWHERE_INCLUDE_DIR})
    get_filename_component(KNOWHERE_LIB_DIR "${KNOWHERE_LIBRARY}" DIRECTORY)
    set(KNOWHERE_MILVUS_COMMON_INCLUDE "${KNOWHERE_LIB_DIR}/milvus-common-src/include")
    if(EXISTS "${KNOWHERE_MILVUS_COMMON_INCLUDE}/filemanager/FileManager.h")
        include_directories(${KNOWHERE_MILVUS_COMMON_INCLUDE})
        message(STATUS "Knowhere extra include enabled: ${KNOWHERE_MILVUS_COMMON_INCLUDE}")
    endif()
    file(GLOB KNOWHERE_CONAN_NLOHMANN_CANDIDATES "$ENV{HOME}/.conan2/p/b/nlohm*/p/include")
    if(KNOWHERE_CONAN_NLOHMANN_CANDIDATES)
        list(GET KNOWHERE_CONAN_NLOHMANN_CANDIDATES 0 KNOWHERE_CONAN_NLOHMANN_INCLUDE)
        include_directories(BEFORE ${KNOWHERE_CONAN_NLOHMANN_INCLUDE})
        message(STATUS "Knowhere nlohmann include enabled: ${KNOWHERE_CONAN_NLOHMANN_INCLUDE}")
    endif()
    file(GLOB KNOWHERE_CONAN_INCLUDE_DIRS
        "$ENV{HOME}/.conan2/p/*/p/include"
        "$ENV{HOME}/.conan2/p/b/*/p/include")
    if(KNOWHERE_CONAN_INCLUDE_DIRS)
        list(FILTER KNOWHERE_CONAN_INCLUDE_DIRS EXCLUDE REGEX ".*/boost[^/]*/p/include$")
        list(REMOVE_DUPLICATES KNOWHERE_CONAN_INCLUDE_DIRS)
        include_directories(BEFORE ${KNOWHERE_CONAN_INCLUDE_DIRS})
        list(LENGTH KNOWHERE_CONAN_INCLUDE_DIRS KNOWHERE_CONAN_INCLUDE_COUNT)
        message(STATUS "Knowhere conan include roots enabled: ${KNOWHERE_CONAN_INCLUDE_COUNT}")
    endif()
    message(STATUS "Knowhere baseline enabled. include=${KNOWHERE_INCLUDE_DIR}, lib=${KNOWHERE_LIBRARY}")
endif()

# Find pre-built CRoaring library
set(ROARING_LIB_DIR "${PROJECT_SOURCE_DIR}/third_party/CRoaring/build/src")
find_library(ROARING_LIB
    NAMES roaring libroaring
    PATHS ${ROARING_LIB_DIR}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)
if(NOT ROARING_LIB)
    message(FATAL_ERROR "CRoaring library not found. Searched in ${ROARING_LIB_DIR} and subdirectories. Please check the build or path.")
else()
    message(STATUS "CRoaring library found at: ${ROARING_LIB}")
endif()

# ONNX
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.16.3" CACHE PATH "Path to ONNX Runtime directory")
if(EXISTS ${ONNXRUNTIME_DIR})
    include_directories(${ONNXRUNTIME_DIR}/include)
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
    )
    message(STATUS "Found ONNX Runtime at: ${ONNXRUNTIME_DIR}")
else()
    message(FATAL_ERROR "ONNX Runtime not found at specified path: ${ONNXRUNTIME_DIR}. Please check the path in CMakeLists.txt")
endif()

# ==============================================================================
# Curator (multi-tenant hierarchical index) — isolated build, no global includes
# All Curator headers are PRIVATE to the curator_faiss target
# ==============================================================================
add_subdirectory(curator)

# ==============================================================================
# Include NaviX, the standalone Faiss 1.10.0 variant
# ==============================================================================
set(NAVIX_ROOT "${PROJECT_SOURCE_DIR}/../../NaviX")
include_directories(${NAVIX_ROOT})

# Allow external build scripts to provide the actual NAVIX_BUILD_DIR
if(NOT NAVIX_BUILD_DIR)
    set(NAVIX_BUILD_DIR "${NAVIX_ROOT}/build")
endif()

# Locate libfaiss_avx2.a or libfaiss.a
find_library(NAVIX_LIB
    NAMES faiss_avx2 faiss
    PATHS "${NAVIX_BUILD_DIR}/faiss_navix" "${NAVIX_BUILD_DIR}/faiss"
    NO_DEFAULT_PATH
)

if(NOT NAVIX_LIB)
    message(FATAL_ERROR "NaviX library not found in ${NAVIX_BUILD_DIR}! Please build NaviX first.")
else()
    message(STATUS "NaviX library found at: ${NAVIX_LIB}")
endif()
# ==============================================================================
# ==============================================================================

# add subdirectories
add_subdirectory(vamana)
add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(test)
add_subdirectory(apps)
