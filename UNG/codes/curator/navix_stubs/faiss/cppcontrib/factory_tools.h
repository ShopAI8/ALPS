/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <string>

namespace faiss_navix {

struct Index;
struct IndexBinary;

std::string reverse_index_factory(const faiss_navix::Index* index);
std::string reverse_index_factory(const faiss_navix::IndexBinary* index);

} // namespace faiss_navix
