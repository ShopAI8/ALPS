#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// NaviX FAISS headers
#include <faiss_navix/impl/IDSelector.h>  // BEFORE Index.h to avoid forward-decl conflict
#include <faiss_navix/Clustering.h>
#include <faiss_navix/Index.h>
#include <faiss_navix/IndexFlat.h>
#include <faiss_navix/IndexIVF.h>
#include <faiss_navix/MetricType.h>
#include <faiss_navix/impl/FaissAssert.h>
#include <faiss_navix/impl/platform_macros.h>
#include <faiss_navix/invlists/DirectMap.h>
#include <faiss_navix/invlists/InvertedLists.h>
#include <faiss_navix/utils/Heap.h>
#include <faiss_navix/utils/distances.h>
#include <faiss_navix/utils/utils.h>
#include <faiss_navix/utils/prefetch.h>

// ScopeDeleter utilities (not in NaviX)
template <class T> struct ScopeDeleter {
    const T* ptr;
    explicit ScopeDeleter(const T* ptr = nullptr) : ptr(ptr) {}
    ~ScopeDeleter() { delete ptr; }
    void release() { ptr = nullptr; }
    void set(const T* ptr_in) { ptr = ptr_in; }
};
template <class T> struct ScopeDeleter1 {
    const T* ptr;
    explicit ScopeDeleter1(const T* ptr = nullptr) : ptr(ptr) {}
    ~ScopeDeleter1() { delete ptr; }
    void release() { ptr = nullptr; }
    void set(const T* ptr_in) { ptr = ptr_in; }
};

// Curator-specific types
using tid_t   = int32_t;
using vid_t   = uint64_t;
using label_t = uint32_t;
using Buffer    = std::vector<vid_t>;
using AccessMap = std::unordered_map<int64_t, std::unordered_set<tid_t>>;

// Import all faiss_navix into faiss (only affects Curator compilation units)
namespace faiss {
    using namespace faiss_navix;
    using ext_vid_t = label_t;
    using int_vid_t = vid_t;
    using ext_lid_t = tid_t;
    using int_lid_t = tid_t;
}

