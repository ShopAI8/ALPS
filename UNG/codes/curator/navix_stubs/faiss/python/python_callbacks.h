/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss_navix/IVFlib.h>
#include <faiss_navix/impl/IDSelector.h>
#include <faiss_navix/impl/io.h>
#include <faiss_navix/invlists/InvertedLists.h>
#include "Python.h"

//  all callbacks have to acquire the GIL on input

/***********************************************************
 * Callbacks for IO reader and writer
 ***********************************************************/

struct PyCallbackIOWriter : faiss_navix::IOWriter {
    PyObject* callback;
    size_t bs; // maximum write size

    /** Callback: Python function that takes a bytes object and
     *  returns the number of bytes successfully written.
     */
    explicit PyCallbackIOWriter(PyObject* callback, size_t bs = 1024 * 1024);

    size_t operator()(const void* ptrv, size_t size, size_t nitems) override;

    ~PyCallbackIOWriter() override;
};

struct PyCallbackIOReader : faiss_navix::IOReader {
    PyObject* callback;
    size_t bs; // maximum buffer size

    /** Callback: Python function that takes a size and returns a
     * bytes object with the resulting read */
    explicit PyCallbackIOReader(PyObject* callback, size_t bs = 1024 * 1024);

    size_t operator()(void* ptrv, size_t size, size_t nitems) override;

    ~PyCallbackIOReader() override;
};

/***********************************************************
 * Callbacks for IDSelector
 ***********************************************************/

struct PyCallbackIDSelector : faiss_navix::IDSelector {
    PyObject* callback;

    explicit PyCallbackIDSelector(PyObject* callback);

    bool is_member(faiss_navix::idx_t id) const override;

    ~PyCallbackIDSelector() override;
};

/***********************************************************
 * Callbacks for IVF index sharding
 ***********************************************************/

struct PyCallbackShardingFunction : faiss_navix::ivflib::ShardingFunction {
    PyObject* callback;

    explicit PyCallbackShardingFunction(PyObject* callback);

    int64_t operator()(int64_t i, int64_t shard_count) override;

    ~PyCallbackShardingFunction() override;

    PyCallbackShardingFunction(const PyCallbackShardingFunction&) = delete;
    PyCallbackShardingFunction(PyCallbackShardingFunction&&) noexcept = default;
    PyCallbackShardingFunction& operator=(const PyCallbackShardingFunction&) =
            default;
    PyCallbackShardingFunction& operator=(PyCallbackShardingFunction&&) =
            default;
};
