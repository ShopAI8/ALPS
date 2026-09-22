/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss_navix/Index.h>
#include <faiss_navix/MetricType.h>
#include <faiss_navix/gpu/GpuIndicesOptions.h>
#include <faiss_navix/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss_navix/gpu/utils/DeviceVector.cuh>
#include <faiss_navix/gpu/utils/Tensor.cuh>

namespace faiss_navix {
namespace gpu {

class GpuResources;

void runIVFFlatScan(
        Tensor<float, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        idx_t maxListLength,
        int k,
        faiss_navix::MetricType metric,
        bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res);

} // namespace gpu
} // namespace faiss_navix
