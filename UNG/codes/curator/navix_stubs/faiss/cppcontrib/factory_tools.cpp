/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss_navix/cppcontrib/factory_tools.h>

#include <map>

#include <faiss_navix/IndexBinaryFlat.h>
#include <faiss_navix/IndexBinaryHNSW.h>
#include <faiss_navix/IndexBinaryIVF.h>
#include <faiss_navix/IndexHNSW.h>
#include <faiss_navix/IndexIDMap.h>
#include <faiss_navix/IndexIVFFlat.h>
#include <faiss_navix/IndexIVFPQFastScan.h>
#include <faiss_navix/IndexLSH.h>
#include <faiss_navix/IndexNSG.h>
#include <faiss_navix/IndexPQFastScan.h>
#include <faiss_navix/IndexPreTransform.h>
#include <faiss_navix/IndexRefine.h>

namespace faiss_navix {

namespace {

const std::map<faiss_navix::ScalarQuantizer::QuantizerType, std::string> sq_types = {
        {faiss_navix::ScalarQuantizer::QT_8bit, "SQ8"},
        {faiss_navix::ScalarQuantizer::QT_4bit, "SQ4"},
        {faiss_navix::ScalarQuantizer::QT_6bit, "SQ6"},
        {faiss_navix::ScalarQuantizer::QT_fp16, "SQfp16"},
        {faiss_navix::ScalarQuantizer::QT_bf16, "SQbf16"},
        {faiss_navix::ScalarQuantizer::QT_8bit_direct_signed, "SQ8_direct_signed"},
        {faiss_navix::ScalarQuantizer::QT_8bit_direct, "SQ8_direct"},
};

int get_hnsw_M(const faiss_navix::IndexHNSW* index) {
    if (index->hnsw.cum_nneighbor_per_level.size() > 1) {
        return index->hnsw.cum_nneighbor_per_level[1] / 2;
    }
    // Avoid runtime error, just return 0.
    return 0;
}

int get_hnsw_M(const faiss_navix::IndexBinaryHNSW* index) {
    if (index->hnsw.cum_nneighbor_per_level.size() > 1) {
        return index->hnsw.cum_nneighbor_per_level[1] / 2;
    }
    // Avoid runtime error, just return 0.
    return 0;
}

} // namespace

// Reference for reverse_index_factory:
// https://github.com/facebookresearch/faiss/blob/838612c9d7f2f619811434ec9209c020f44107cb/contrib/factory_tools.py#L81
std::string reverse_index_factory(const faiss_navix::Index* index) {
    std::string prefix;
    if (dynamic_cast<const faiss_navix::IndexFlat*>(index)) {
        return "Flat";
    } else if (
            const faiss_navix::IndexIVF* ivf_index =
                    dynamic_cast<const faiss_navix::IndexIVF*>(index)) {
        const faiss_navix::Index* quantizer = ivf_index->quantizer;

        if (dynamic_cast<const faiss_navix::IndexFlat*>(quantizer)) {
            prefix = "IVF" + std::to_string(ivf_index->nlist);
        } else if (
                const faiss_navix::MultiIndexQuantizer* miq =
                        dynamic_cast<const faiss_navix::MultiIndexQuantizer*>(
                                quantizer)) {
            prefix = "IMI" + std::to_string(miq->pq.M) + "x" +
                    std::to_string(miq->pq.nbits);
        } else if (
                const faiss_navix::IndexHNSW* hnsw_index =
                        dynamic_cast<const faiss_navix::IndexHNSW*>(quantizer)) {
            prefix = "IVF" + std::to_string(ivf_index->nlist) + "_HNSW" +
                    std::to_string(get_hnsw_M(hnsw_index));
        } else {
            prefix = "IVF" + std::to_string(ivf_index->nlist) + "(" +
                    reverse_index_factory(quantizer) + ")";
        }

        if (dynamic_cast<const faiss_navix::IndexIVFFlat*>(ivf_index)) {
            return prefix + ",Flat";
        } else if (
                auto sq_index =
                        dynamic_cast<const faiss_navix::IndexIVFScalarQuantizer*>(
                                ivf_index)) {
            return prefix + "," + sq_types.at(sq_index->sq.qtype);
        } else if (
                const faiss_navix::IndexIVFPQ* ivfpq_index =
                        dynamic_cast<const faiss_navix::IndexIVFPQ*>(ivf_index)) {
            return prefix + ",PQ" + std::to_string(ivfpq_index->pq.M) + "x" +
                    std::to_string(ivfpq_index->pq.nbits);
        } else if (
                const faiss_navix::IndexIVFPQFastScan* ivfpqfs_index =
                        dynamic_cast<const faiss_navix::IndexIVFPQFastScan*>(
                                ivf_index)) {
            return prefix + ",PQ" + std::to_string(ivfpqfs_index->pq.M) + "x" +
                    std::to_string(ivfpqfs_index->pq.nbits) + "fs";
        }
    } else if (
            const faiss_navix::IndexPreTransform* pretransform_index =
                    dynamic_cast<const faiss_navix::IndexPreTransform*>(index)) {
        if (pretransform_index->chain.size() != 1) {
            // Avoid runtime error, just return empty string for logging.
            return "";
        }
        const faiss_navix::VectorTransform* vt = pretransform_index->chain.at(0);
        if (const faiss_navix::OPQMatrix* opq_matrix =
                    dynamic_cast<const faiss_navix::OPQMatrix*>(vt)) {
            prefix = "OPQ" + std::to_string(opq_matrix->M) + "_" +
                    std::to_string(opq_matrix->d_out);
        } else if (
                const faiss_navix::ITQTransform* itq_transform =
                        dynamic_cast<const faiss_navix::ITQTransform*>(vt)) {
            prefix = "ITQ" + std::to_string(itq_transform->itq.d_out);
        } else if (
                const faiss_navix::PCAMatrix* pca_matrix =
                        dynamic_cast<const faiss_navix::PCAMatrix*>(vt)) {
            assert(pca_matrix->eigen_power == 0);
            prefix = "PCA" +
                    std::string(pca_matrix->random_rotation ? "R" : "") +
                    std::to_string(pca_matrix->d_out);
        } else {
            // Avoid runtime error, just return empty string for logging.
            return "";
        }
        return prefix + "," + reverse_index_factory(pretransform_index->index);
    } else if (
            const faiss_navix::IndexHNSW* hnsw_index =
                    dynamic_cast<const faiss_navix::IndexHNSW*>(index)) {
        return "HNSW" + std::to_string(get_hnsw_M(hnsw_index));
    } else if (
            const faiss_navix::IndexNSG* nsg_index =
                    dynamic_cast<const faiss_navix::IndexNSG*>(index)) {
        return "NSG" + std::to_string(nsg_index->nsg.R) + "," +
                reverse_index_factory(nsg_index->storage);
    } else if (
            const faiss_navix::IndexRefine* refine_index =
                    dynamic_cast<const faiss_navix::IndexRefine*>(index)) {
        return reverse_index_factory(refine_index->base_index) + ",Refine(" +
                reverse_index_factory(refine_index->refine_index) + ")";
    } else if (
            const faiss_navix::IndexPQFastScan* pqfs_index =
                    dynamic_cast<const faiss_navix::IndexPQFastScan*>(index)) {
        return std::string("PQ") + std::to_string(pqfs_index->pq.M) + "x" +
                std::to_string(pqfs_index->pq.nbits) + "fs";
    } else if (
            const faiss_navix::IndexPQ* pq_index =
                    dynamic_cast<const faiss_navix::IndexPQ*>(index)) {
        return std::string("PQ") + std::to_string(pq_index->pq.M) + "x" +
                std::to_string(pq_index->pq.nbits);
    } else if (
            const faiss_navix::IndexLSH* lsh_index =
                    dynamic_cast<const faiss_navix::IndexLSH*>(index)) {
        std::string result = "LSH";
        if (lsh_index->rotate_data) {
            result += "r";
        }
        if (lsh_index->train_thresholds) {
            result += "t";
        }
        return result;
    } else if (
            const faiss_navix::IndexScalarQuantizer* sq_index =
                    dynamic_cast<const faiss_navix::IndexScalarQuantizer*>(index)) {
        return sq_types.at(sq_index->sq.qtype);
    } else if (
            const faiss_navix::IndexIDMap* idmap =
                    dynamic_cast<const faiss_navix::IndexIDMap*>(index)) {
        return std::string("IDMap,") + reverse_index_factory(idmap->index);
    }
    // Avoid runtime error, just return empty string for logging.
    return "";
}

std::string reverse_index_factory(const faiss_navix::IndexBinary* index) {
    std::string prefix;
    if (dynamic_cast<const faiss_navix::IndexBinaryFlat*>(index)) {
        return "BFlat";
    } else if (
            const faiss_navix::IndexBinaryIVF* ivf_index =
                    dynamic_cast<const faiss_navix::IndexBinaryIVF*>(index)) {
        const faiss_navix::IndexBinary* quantizer = ivf_index->quantizer;

        if (dynamic_cast<const faiss_navix::IndexBinaryFlat*>(quantizer)) {
            return "BIVF" + std::to_string(ivf_index->nlist);
        } else if (
                const faiss_navix::IndexBinaryHNSW* hnsw_index =
                        dynamic_cast<const faiss_navix::IndexBinaryHNSW*>(
                                quantizer)) {
            return "BIVF" + std::to_string(ivf_index->nlist) + "_HNSW" +
                    std::to_string(get_hnsw_M(hnsw_index));
        }
        // Add further cases for BinaryIVF here.
    } else if (
            const faiss_navix::IndexBinaryHNSW* hnsw_index =
                    dynamic_cast<const faiss_navix::IndexBinaryHNSW*>(index)) {
        return "BHNSW" + std::to_string(get_hnsw_M(hnsw_index));
    }
    // Avoid runtime error, just return empty string for logging.
    return "";
}

} // namespace faiss_navix
