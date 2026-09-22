/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss_navix/AutoTune.h>
#include <faiss_navix/Index.h>

namespace faiss_navix {
namespace gpu {

/// parameter space and setters for GPU indexes
struct GpuParameterSpace : faiss_navix::ParameterSpace {
    /// initialize with reasonable parameters for the index
    void initialize(const faiss_navix::Index* index) override;

    /// set a combination of parameters on an index
    void set_index_parameter(
            faiss_navix::Index* index,
            const std::string& name,
            double val) const override;
};

} // namespace gpu
} // namespace faiss_navix
