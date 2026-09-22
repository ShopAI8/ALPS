#pragma once

#include <bitset>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include <roaring/roaring.hh>
#include "storage.h"
#include "config.h"

// Forward declare — no Curator/FAISS headers leak into SODA code
namespace faiss {
    class MultiTenantIndexIVFHierarchical;
}

namespace ANNS {

// Opaque context: Curator internals are only visible in curator_wrapper.cpp.
// Curator now owns its own inverted index — no dependency on UNG framework.
struct CuratorContext {
    std::unique_ptr<faiss::MultiTenantIndexIVFHierarchical> index;

    // Parameters
    int nlist = 32;
    int nprobe = 1200;
    int max_leaf_size = 256;
    int search_ef = 128;
    int beam_size = 1;
    bool ready = false;

    // Self-contained inverted index (built from base storage, not UNG)
    // label -> roaring bitmap of vector IDs
    std::unordered_map<LabelType, roaring::Roaring> inverted;
    IdxType num_points = 0;

    CuratorContext();
    ~CuratorContext();
};

// Build Curator index + inverted index from base storage.
// Self-contained — does NOT need UNG index or inverted indices.
void curator_build_index(
    CuratorContext& ctx,
    const std::shared_ptr<IStorage>& base_storage);

// Build only the FAISS tree (k-means + add vectors + grant access).
// Assumes ctx.inverted is already populated.
void curator_build_faiss_tree(
    CuratorContext& ctx,
    const std::shared_ptr<IStorage>& base_storage);

// Save Curator index to disk (inverted index + params + FAISS tree)
void curator_save_index(const CuratorContext& ctx, const std::string& path);

// Load Curator index from disk (loads inverted index + FAISS tree if available)
void curator_load_index(
    CuratorContext& ctx,
    const std::shared_ptr<IStorage>& base_storage,
    const std::string& path);

// Timing breakdown
struct CuratorSearchStats {
    double bitmap_time_ms = 0.0;
    double search_time_ms = 0.0;
    size_t exact_cand_size = 0;
    float global_p_pass = 0.0f;
};

// Run one Curator containment query.
// If precomputed_mask != nullptr, the qualified set is taken from the
// framework's bitset filter map (same bipartite-graph/bitset path as the
// pre-filter baseline) and ctx.inverted is NOT consulted; only the
// mask->vector extraction cost is charged to bitmap_time_ms. Otherwise
// falls back to CuratorContext's self-contained inverted index.
void curator_search_query_detailed(
    const CuratorContext& ctx,
    int query_id,
    const char* query_vec,
    const std::vector<LabelType>& query_labels,
    IdxType K,
    std::pair<IdxType, float>* results_out,
    CuratorSearchStats& stats_out,
    const std::bitset<16000000>* precomputed_mask = nullptr);

// Update the FAISS index's internal search_ef (used for Lsearch sweep).
// Must be called after build_curator, before search.
void curator_update_search_ef(CuratorContext& ctx, int search_ef);

}  // namespace ANNS
