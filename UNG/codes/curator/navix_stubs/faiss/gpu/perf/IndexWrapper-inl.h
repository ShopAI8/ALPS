/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss_navix/impl/FaissAssert.h>

namespace faiss_navix {
namespace gpu {

template <typename GpuIndex>
IndexWrapper<GpuIndex>::IndexWrapper(
        int numGpus,
        std::function<std::unique_ptr<GpuIndex>(GpuResourcesProvider*, int)>
                init) {
    FAISS_NAVIX_ASSERT(numGpus <= faiss_navix::gpu::getNumDevices());
    for (int i = 0; i < numGpus; ++i) {
        auto res = std::unique_ptr<faiss_navix::gpu::StandardGpuResources>(
                new StandardGpuResources);

        subIndex.emplace_back(init(res.get(), i));
        resources.emplace_back(std::move(res));
    }

    if (numGpus > 1) {
        // create proxy
        replicaIndex =
                std::unique_ptr<faiss_navix::IndexReplicas>(new faiss_navix::IndexReplicas);

        for (auto& index : subIndex) {
            replicaIndex->addIndex(index.get());
        }
    }
}

template <typename GpuIndex>
faiss_navix::Index* IndexWrapper<GpuIndex>::getIndex() {
    if ((bool)replicaIndex) {
        return replicaIndex.get();
    } else {
        FAISS_NAVIX_ASSERT(!subIndex.empty());
        return subIndex.front().get();
    }
}

template <typename GpuIndex>
void IndexWrapper<GpuIndex>::runOnIndices(std::function<void(GpuIndex*)> f) {
    if ((bool)replicaIndex) {
        replicaIndex->runOnIndex([f](int, faiss_navix::Index* index) {
            f(dynamic_cast<GpuIndex*>(index));
        });
    } else {
        FAISS_NAVIX_ASSERT(!subIndex.empty());
        f(subIndex.front().get());
    }
}

template <typename GpuIndex>
void IndexWrapper<GpuIndex>::setNumProbes(size_t nprobe) {
    runOnIndices([nprobe](GpuIndex* index) { index->nprobe = nprobe; });
}

} // namespace gpu
} // namespace faiss_navix
