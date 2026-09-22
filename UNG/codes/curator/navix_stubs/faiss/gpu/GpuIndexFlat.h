/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss_navix/gpu/GpuIndex.h>
#include <memory>

namespace faiss_navix {

struct IndexFlat;
struct IndexFlatL2;
struct IndexFlatIP;

} // namespace faiss_navix

namespace faiss_navix {
namespace gpu {

class FlatIndex;

struct GpuIndexFlatConfig : public GpuIndexConfig {
    /// Whether or not data is stored as float16
    bool ALIGNED(8) useFloat16 = false;

    /// Deprecated: no longer used
    /// Previously used to indicate whether internal storage of vectors is
    /// transposed
    bool storeTransposed = false;
};

/// Wrapper around the GPU implementation that looks like
/// faiss_navix::IndexFlat; copies over centroid data from a given
/// faiss_navix::IndexFlat
class GpuIndexFlat : public GpuIndex {
   public:
    /// Construct from a pre-existing faiss_navix::IndexFlat instance, copying
    /// data over to the given GPU
    GpuIndexFlat(
            GpuResourcesProvider* provider,
            const faiss_navix::IndexFlat* index,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    GpuIndexFlat(
            std::shared_ptr<GpuResources> resources,
            const faiss_navix::IndexFlat* index,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    /// Construct an empty instance that can be added to
    GpuIndexFlat(
            GpuResourcesProvider* provider,
            int dims,
            faiss_navix::MetricType metric,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    GpuIndexFlat(
            std::shared_ptr<GpuResources> resources,
            int dims,
            faiss_navix::MetricType metric,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    ~GpuIndexFlat() override;

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const faiss_navix::IndexFlat* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss_navix::IndexFlat* index) const;

    /// Returns the number of vectors we contain
    size_t getNumVecs() const;

    /// Clears all vectors from this index
    void reset() override;

    /// This index is not trained, so this does nothing
    void train(idx_t n, const float* x) override;

    /// Overrides to avoid excessive copies
    void add(idx_t, const float* x) override;

    /// Reconstruction methods; prefer the batch reconstruct as it will
    /// be more efficient
    void reconstruct(idx_t key, float* out) const override;

    /// Batch reconstruction method
    void reconstruct_n(idx_t i0, idx_t num, float* out) const override;

    /// Batch reconstruction method
    void reconstruct_batch(idx_t n, const idx_t* keys, float* out)
            const override;

    /// Compute residual
    void compute_residual(const float* x, float* residual, idx_t key)
            const override;

    /// Compute residual (batch mode)
    void compute_residual_n(
            idx_t n,
            const float* xs,
            float* residuals,
            const idx_t* keys) const override;

    /// For internal access
    inline FlatIndex* getGpuData() {
        return data_.get();
    }

   protected:
    void resetIndex_(int dims);

    /// Flat index does not require IDs as there is no storage available for
    /// them
    bool addImplRequiresIDs_() const override;

    /// Called from GpuIndex for add
    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;

    /// Called from GpuIndex for search
    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override;

   protected:
    /// Our configuration options
    const GpuIndexFlatConfig flatConfig_;

    /// Holds our GPU data containing the list of vectors
    std::unique_ptr<FlatIndex> data_;
};

/// Wrapper around the GPU implementation that looks like
/// faiss_navix::IndexFlatL2; copies over centroid data from a given
/// faiss_navix::IndexFlat
class GpuIndexFlatL2 : public GpuIndexFlat {
   public:
    /// Construct from a pre-existing faiss_navix::IndexFlatL2 instance, copying
    /// data over to the given GPU
    GpuIndexFlatL2(
            GpuResourcesProvider* provider,
            faiss_navix::IndexFlatL2* index,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    GpuIndexFlatL2(
            std::shared_ptr<GpuResources> resources,
            faiss_navix::IndexFlatL2* index,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    /// Construct an empty instance that can be added to
    GpuIndexFlatL2(
            GpuResourcesProvider* provider,
            int dims,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    GpuIndexFlatL2(
            std::shared_ptr<GpuResources> resources,
            int dims,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(faiss_navix::IndexFlat* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss_navix::IndexFlat* index);
};

/// Wrapper around the GPU implementation that looks like
/// faiss_navix::IndexFlatIP; copies over centroid data from a given
/// faiss_navix::IndexFlat
class GpuIndexFlatIP : public GpuIndexFlat {
   public:
    /// Construct from a pre-existing faiss_navix::IndexFlatIP instance, copying
    /// data over to the given GPU
    GpuIndexFlatIP(
            GpuResourcesProvider* provider,
            faiss_navix::IndexFlatIP* index,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    GpuIndexFlatIP(
            std::shared_ptr<GpuResources> resources,
            faiss_navix::IndexFlatIP* index,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    /// Construct an empty instance that can be added to
    GpuIndexFlatIP(
            GpuResourcesProvider* provider,
            int dims,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    GpuIndexFlatIP(
            std::shared_ptr<GpuResources> resources,
            int dims,
            GpuIndexFlatConfig config = GpuIndexFlatConfig());

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(faiss_navix::IndexFlat* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss_navix::IndexFlat* index);
};

} // namespace gpu
} // namespace faiss_navix
