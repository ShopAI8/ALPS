#include "curator_wrapper.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>

#include <roaring/roaring.hh>

// Curator FAISS includes — only visible in this .cpp, NOT leaking to SODA
#include "faiss/MultiTenantIndexIVFHierarchical.h"

namespace ANNS {

CuratorContext::CuratorContext() = default;
CuratorContext::~CuratorContext() = default;

void curator_build_faiss_tree(
    CuratorContext& ctx,
    const std::shared_ptr<IStorage>& base_storage)
{
    const size_t dim = base_storage->get_dim();
    const IdxType n = ctx.num_points;

    auto t_build0 = std::chrono::high_resolution_clock::now();

    ctx.index = std::make_unique<faiss::MultiTenantIndexIVFHierarchical>(
        dim, ctx.nlist, faiss::METRIC_L2,
        1000, 0.001f,        // bf_capacity, bf_false_pos (curator-v2 defaults)
        128,                 // max_sl_size (original default)
        20,                  // clus_niter
        ctx.max_leaf_size,
        ctx.nprobe,
        1.6f, 0.4f,          // prune_thres, variance_boost
        ctx.search_ef,
        ctx.beam_size,
        false);

    // Gather all vectors
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> all_vecs((size_t)n * dim);
    for (IdxType i = 0; i < n; ++i) {
        const char* vec = base_storage->get_vector(i);
        std::memcpy(all_vecs.data() + (size_t)i * dim, vec, dim * sizeof(float));
    }
    double t_gather = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // Train
    std::cout << "[Curator] Training (nlist=" << ctx.nlist << ")..." << std::endl;
    auto t_train0 = std::chrono::high_resolution_clock::now();
    ctx.index->train(n, all_vecs.data(), 0);
    double t_train = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_train0).count();
    std::cout << "[Curator] Training done in " << t_train << "s" << std::endl;

    // Add vectors
    std::cout << "[Curator] Adding " << n << " vectors..." << std::endl;
    auto t_add0 = std::chrono::high_resolution_clock::now();
    std::vector<faiss_navix::idx_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    ctx.index->add_vector_with_ids(n, all_vecs.data(), ids.data());
    double t_add = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_add0).count();

    // Grant access — remap SODA labels to compact int32 range
    std::cout << "[Curator] Granting access (" << ctx.inverted.size()
              << " labels)..." << std::endl;
    auto t_grant0 = std::chrono::high_resolution_clock::now();
    std::unordered_map<LabelType, int32_t> label_remap;
    int32_t next_id = 0;
    size_t total_grants = 0;

    for (IdxType vid = 0; vid < n; ++vid) {
        const auto& lbls = base_storage->get_label_set(vid);
        for (auto lbl : lbls) {
            auto it = label_remap.find(lbl);
            int32_t tid;
            if (it != label_remap.end()) {
                tid = it->second;
            } else {
                tid = next_id++;
                label_remap[lbl] = tid;
            }
            ctx.index->grant_access(vid, static_cast<faiss::ext_lid_t>(tid));
            ++total_grants;
        }
        if ((vid + 1) % 100000 == 0) {
            std::cout << "  " << (vid + 1) << "/" << n
                      << " vectors, " << total_grants << " grants, "
                      << next_id << " unique labels" << std::endl;
        }
    }
    double t_grant = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_grant0).count();

    double t_build = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_build0).count();
    std::cout << "[Curator] FAISS tree built in " << t_build << "s"
              << " (gather=" << t_gather << "s"
              << ", train=" << t_train << "s, add=" << t_add << "s"
              << ", grant=" << t_grant << "s)" << std::endl;
}

void curator_build_index(
    CuratorContext& ctx,
    const std::shared_ptr<IStorage>& base_storage)
{
    if (ctx.ready) return;

    const size_t dim = base_storage->get_dim();
    const IdxType n = ctx.num_points = base_storage->get_num_points();

    std::cout << "[Curator] Building: n=" << n << " dim=" << dim
              << " nlist=" << ctx.nlist << " max_leaf_size=" << ctx.max_leaf_size
              << " nprobe=" << ctx.nprobe << " search_ef=" << ctx.search_ef
              << " beam_size=" << ctx.beam_size
              << std::endl;

    // ---- Phase 1: Build self-contained inverted index ----
    auto t_inv0 = std::chrono::high_resolution_clock::now();
    std::cout << "[Curator] Building inverted index..." << std::endl;
    ctx.inverted.clear();
    for (IdxType vid = 0; vid < n; ++vid) {
        const auto& lbls = base_storage->get_label_set(vid);
        for (auto lbl : lbls) {
            ctx.inverted[lbl].add(vid);
        }
    }
    double t_inv = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_inv0).count();
    std::cout << "[Curator] Inverted index: " << ctx.inverted.size()
              << " unique labels, time=" << t_inv << "s" << std::endl;

    // ---- Phase 2: Build FAISS tree ----
    curator_build_faiss_tree(ctx, base_storage);

    ctx.ready = true;
}

void curator_search_query_detailed(
    const CuratorContext& ctx,
    int query_id,
    const char* query_vec,
    const std::vector<LabelType>& query_labels,
    IdxType K,
    std::pair<IdxType, float>* results_out,
    CuratorSearchStats& stats_out,
    const std::bitset<16000000>* precomputed_mask)
{
    (void)query_id;

    // Debug: print first 3 queries
    static int debug_cnt = 0;
    bool debug = (debug_cnt++ < 3);

    // ---- Phase 1: Qualified vectors ----
    auto mask_t0 = std::chrono::high_resolution_clock::now();
    std::vector<faiss::ext_vid_t> qualified;

    if (precomputed_mask != nullptr) {
        // Reuse the framework's bitset filter map (same bipartite-graph path
        // as the pre-filter baseline). Only the extraction to a vector — the
        // format Curator's bitmap search consumes — is charged here.
        const uint64_t* bit_words =
            reinterpret_cast<const uint64_t*>(precomputed_mask);
        const size_t num_words =
            ctx.num_points / 64 + (ctx.num_points % 64 != 0 ? 1 : 0);
        qualified.reserve(ctx.num_points / 64);
        for (size_t w = 0; w < num_words; ++w) {
            uint64_t word = bit_words[w];
            while (word != 0) {
                const int bit_idx = __builtin_ctzll(word);
                const IdxType vec_id = w * 64 + bit_idx;
                if (vec_id >= ctx.num_points) break;
                qualified.push_back(static_cast<faiss::ext_vid_t>(vec_id));
                word &= word - 1;
            }
        }
        stats_out.exact_cand_size = qualified.size();
    } else if (query_labels.empty()) {
        qualified.resize(ctx.num_points);
        std::iota(qualified.begin(), qualified.end(), 0);
        stats_out.exact_cand_size = ctx.num_points;
        if (debug) std::cout << "[Curator DEBUG] qid=" << query_id
                             << " labels=EMPTY qualified=" << qualified.size() << std::endl;
    } else {
        // Sort labels by inverted list size (smallest first for efficiency)
        auto sorted_lbls = query_labels;
        std::sort(sorted_lbls.begin(), sorted_lbls.end(),
            [&](LabelType a, LabelType b) {
                auto ca = ctx.inverted.count(a) ? ctx.inverted.at(a).cardinality() : 0ULL;
                auto cb = ctx.inverted.count(b) ? ctx.inverted.at(b).cardinality() : 0ULL;
                return ca < cb;
            });

        const roaring::Roaring* result_rb = nullptr;
        roaring::Roaring tmp;

        if (ctx.inverted.count(sorted_lbls[0])) {
            if (sorted_lbls.size() == 1) {
                result_rb = &ctx.inverted.at(sorted_lbls[0]);
            } else {
                tmp = ctx.inverted.at(sorted_lbls[0]);
                for (size_t i = 1; i < sorted_lbls.size() && !tmp.isEmpty(); ++i) {
                    if (ctx.inverted.count(sorted_lbls[i])) {
                        tmp &= ctx.inverted.at(sorted_lbls[i]);
                    } else {
                        tmp = roaring::Roaring();
                        break;
                    }
                }
                result_rb = &tmp;
            }
        }

        if (result_rb && !result_rb->isEmpty()) {
            qualified.reserve(result_rb->cardinality());
            for (auto it = result_rb->begin(); it != result_rb->end(); ++it) {
                qualified.push_back(static_cast<faiss::ext_vid_t>(*it));
            }
            stats_out.exact_cand_size = qualified.size();
        } else {
            stats_out.exact_cand_size = 0;
        }
    }

    if (debug) std::cout << "[Curator DEBUG] qid=" << query_id
                         << " n_labels=" << query_labels.size()
                         << " qualified=" << qualified.size()
                         << " inv_size=" << ctx.inverted.size()
                         << " n_points=" << ctx.num_points << std::endl;

    stats_out.bitmap_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - mask_t0).count();
    stats_out.global_p_pass = ctx.num_points > 0
        ? static_cast<float>(stats_out.exact_cand_size) / ctx.num_points
        : 0.0f;

    // ---- Phase 2: Curator search ----
    auto search_t0 = std::chrono::high_resolution_clock::now();

    // Matches original Curator: only an EMPTY qualified set skips the search
    // (search_with_bitmap_filter returns early on size 0). For
    // 0 < |qualified| < K, search runs and returns min(K, |qualified|)
    // results with the remaining slots set to -1.
    if (qualified.empty()) {
        for (IdxType k = 0; k < K; ++k) {
            results_out[k].first = static_cast<IdxType>(-1);
            results_out[k].second = std::numeric_limits<float>::max();
        }
        if (debug) std::cout << "[Curator DEBUG] qid=" << query_id
                             << " SKIP search (qualified empty)" << std::endl;
    } else {
        std::vector<float> dists(K);
        std::vector<faiss_navix::idx_t> labels(K);
        ctx.index->search_with_bitmap_filter(
            1, reinterpret_cast<const float*>(query_vec), K,
            qualified.data(), qualified.size(),
            dists.data(), labels.data());
        for (IdxType k = 0; k < K; ++k) {
            results_out[k].first = static_cast<IdxType>(labels[k]);
            results_out[k].second = dists[k];
        }
        if (debug) {
            std::cout << "[Curator DEBUG] qid=" << query_id << " top-3 results: ";
            for (IdxType k = 0; k < std::min(K, (IdxType)3); ++k)
                std::cout << "(" << results_out[k].first << "," << results_out[k].second << ") ";
            std::cout << std::endl;
        }
    }

    stats_out.search_time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - search_t0).count();
}

// ==============================================================================
// FAISS tree serialization (BFS format)
// ==============================================================================
static void curator_save_faiss_tree(
    const CuratorContext& ctx, const std::string& path)
{
    if (!ctx.index) return;
    const auto& idx = *ctx.index;

    // Auto-create parent directory
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[Curator] ERROR: cannot write FAISS tree to " << path << std::endl;
        return;
    }

    const size_t dim = idx.d;
    const size_t n = ctx.num_points;

    // Header
    const uint32_t magic = 0x46414953;  // "FAIS"
    const uint32_t version = 3;  // v3: CURATOR_MAX_LEAF_SIZE_LOG2 10->16 (leaf-overflow fix changes vid encoding)
    ofs.write(reinterpret_cast<const char*>(&magic), 4);
    ofs.write(reinterpret_cast<const char*>(&version), 4);

    uint64_t u64 = dim;  ofs.write(reinterpret_cast<const char*>(&u64), 8);
    u64 = n;             ofs.write(reinterpret_cast<const char*>(&u64), 8);

    // --- Vectors ---
    const float* xb = idx.storage->get_xb();
    ofs.write(reinterpret_cast<const char*>(xb), n * dim * sizeof(float));

    // --- id_allocator (ext_vid_t → int_vid_t) ---
    u64 = idx.id_allocator.label_to_id.size();
    ofs.write(reinterpret_cast<const char*>(&u64), 8);
    for (const auto& [label, vid] : idx.id_allocator.label_to_id) {
        uint32_t ext = label;
        ofs.write(reinterpret_cast<const char*>(&ext), 4);
        uint64_t internal = vid;
        ofs.write(reinterpret_cast<const char*>(&internal), 8);
    }

    // --- tid_allocator ---
    u64 = idx.tid_allocator.id_to_label.size();
    ofs.write(reinterpret_cast<const char*>(&u64), 8);
    for (size_t i = 0; i < idx.tid_allocator.id_to_label.size(); ++i) {
        int32_t ext = idx.tid_allocator.id_to_label[i];
        if (ext == faiss::TenantIdAllocator::INVALID_ID) continue;
        int32_t internal = static_cast<int32_t>(i);
        ofs.write(reinterpret_cast<const char*>(&ext), 4);
        ofs.write(reinterpret_cast<const char*>(&internal), 4);
    }

    // --- vid_to_storage_idx ---
    u64 = idx.vid_to_storage_idx.size();
    ofs.write(reinterpret_cast<const char*>(&u64), 8);
    for (const auto& [vid, sidx] : idx.vid_to_storage_idx) {
        uint64_t v = vid;
        ofs.write(reinterpret_cast<const char*>(&v), 8);
        u64 = sidx;
        ofs.write(reinterpret_cast<const char*>(&u64), 8);
    }

    // --- Tree structure (BFS) ---
    std::vector<const faiss::TreeNode*> bfs;
    bfs.push_back(idx.tree_root);
    for (size_t i = 0; i < bfs.size(); ++i) {
        for (auto* child : bfs[i]->children)
            bfs.push_back(child);
    }
    u64 = bfs.size();
    ofs.write(reinterpret_cast<const char*>(&u64), 8);

    for (const auto* node : bfs) {
        uint64_t lvl = node->level;
        ofs.write(reinterpret_cast<const char*>(&lvl), 8);
        uint64_t sid = node->sibling_id;
        ofs.write(reinterpret_cast<const char*>(&sid), 8);
        uint64_t nid = node->node_id;
        ofs.write(reinterpret_cast<const char*>(&nid), 8);
        uint8_t leaf = node->children.empty() ? 1 : 0;
        ofs.write(reinterpret_cast<const char*>(&leaf), 1);

        // centroid
        ofs.write(reinterpret_cast<const char*>(node->centroid),
                  dim * sizeof(float));

        if (leaf) {
            // vector_indices
            u64 = node->vector_indices.size();
            ofs.write(reinterpret_cast<const char*>(&u64), 8);
            ofs.write(reinterpret_cast<const char*>(node->vector_indices.data.data()),
                      node->vector_indices.size() * sizeof(faiss::int_vid_t));
        } else {
            uint64_t nc = node->children.size();
            ofs.write(reinterpret_cast<const char*>(&nc), 8);
        }

        // shortlists
        u64 = node->shortlists.size();
        ofs.write(reinterpret_cast<const char*>(&u64), 8);
        for (const auto& [tid, sl] : node->shortlists) {
            int32_t t = tid;
            ofs.write(reinterpret_cast<const char*>(&t), 4);
            u64 = sl.size();
            ofs.write(reinterpret_cast<const char*>(&u64), 8);
            ofs.write(reinterpret_cast<const char*>(sl.data.data()),
                      sl.size() * sizeof(faiss::int_vid_t));
        }
    }

    size_t size_mb = static_cast<size_t>(ofs.tellp()) / 1024 / 1024;
    ofs.close();
    std::cout << "[Curator] FAISS tree saved to " << path
              << " (" << bfs.size() << " nodes, " << size_mb << " MB)" << std::endl;
}

static bool curator_load_faiss_tree(
    CuratorContext& ctx,
    const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    uint32_t magic, version;
    ifs.read(reinterpret_cast<char*>(&magic), 4);
    ifs.read(reinterpret_cast<char*>(&version), 4);
    if (magic != 0x46414953 || version != 3) {
        std::cout << "[Curator] FAISS tree file is stale (version "
                  << version << " != 3), rebuilding" << std::endl;
        return false;
    }

    uint64_t dim_u64, n_u64;
    ifs.read(reinterpret_cast<char*>(&dim_u64), 8);
    ifs.read(reinterpret_cast<char*>(&n_u64), 8);
    const size_t dim = static_cast<size_t>(dim_u64);
    const size_t n = static_cast<size_t>(n_u64);

    std::cout << "[Curator] Loading FAISS tree: n=" << n << " dim=" << dim
              << " nlist=" << ctx.nlist << std::endl;

    // Create index shell
    ctx.index = std::make_unique<faiss::MultiTenantIndexIVFHierarchical>(
        dim, ctx.nlist, faiss::METRIC_L2,
        1000, 0.001f, 128, 20, ctx.max_leaf_size,
        ctx.nprobe, 1.6f, 0.4f,
        ctx.search_ef, ctx.beam_size, false);

    auto& idx = *ctx.index;

    // --- Load vectors ---
    std::vector<float> vecs(n * dim);
    ifs.read(reinterpret_cast<char*>(vecs.data()), n * dim * sizeof(float));
    if (idx.own_fields) {
        idx.storage->add(n, vecs.data());
    }

    // --- Load id_allocator (label_to_id) ---
    uint64_t num_mappings;
    ifs.read(reinterpret_cast<char*>(&num_mappings), 8);
    for (uint64_t i = 0; i < num_mappings; ++i) {
        uint32_t label; uint64_t vid;
        ifs.read(reinterpret_cast<char*>(&label), 4);
        ifs.read(reinterpret_cast<char*>(&vid), 8);
        idx.id_allocator.add_mapping(label, vid);
    }

    // --- Load tid_allocator ---
    // Reconstruct the mappings directly. Do NOT use allocate_id() for padding:
    // it registers a dummy label and throws "label already exists" on the
    // second call, which broke loading whenever there were >= 2 tenants.
    uint64_t num_tids;
    ifs.read(reinterpret_cast<char*>(&num_tids), 8);
    for (uint64_t i = 0; i < num_tids; ++i) {
        int32_t ext, internal;
        ifs.read(reinterpret_cast<char*>(&ext), 4);
        ifs.read(reinterpret_cast<char*>(&internal), 4);
        if (static_cast<size_t>(internal) >= idx.tid_allocator.id_to_label.size()) {
            idx.tid_allocator.id_to_label.resize(
                internal + 1, faiss::TenantIdAllocator::INVALID_ID);
        }
        idx.tid_allocator.label_to_id[ext] = internal;
        idx.tid_allocator.id_to_label[internal] = ext;
    }

    // --- Load vid_to_storage_idx ---
    uint64_t num_vs;
    ifs.read(reinterpret_cast<char*>(&num_vs), 8);
    for (uint64_t i = 0; i < num_vs; ++i) {
        uint64_t vid, sidx;
        ifs.read(reinterpret_cast<char*>(&vid), 8);
        ifs.read(reinterpret_cast<char*>(&sidx), 8);
        idx.vid_to_storage_idx[vid] = static_cast<size_t>(sidx);
    }

    // --- Load tree (BFS) ---
    uint64_t num_nodes;
    ifs.read(reinterpret_cast<char*>(&num_nodes), 8);

    struct FlatNode {
        uint64_t level, sibling_id, node_id;
        uint8_t is_leaf;
        uint64_t num_children = 0;
        std::vector<float> centroid;
        std::vector<faiss::int_vid_t> vector_indices;
        std::unordered_map<faiss::int_lid_t, std::vector<faiss::int_vid_t>> shortlists;
    };
    std::vector<FlatNode> flat_nodes(num_nodes);

    for (uint64_t i = 0; i < num_nodes; ++i) {
        auto& fn = flat_nodes[i];
        ifs.read(reinterpret_cast<char*>(&fn.level), 8);
        ifs.read(reinterpret_cast<char*>(&fn.sibling_id), 8);
        ifs.read(reinterpret_cast<char*>(&fn.node_id), 8);
        ifs.read(reinterpret_cast<char*>(&fn.is_leaf), 1);

        fn.centroid.resize(dim);
        ifs.read(reinterpret_cast<char*>(fn.centroid.data()), dim * sizeof(float));

        if (fn.is_leaf) {
            uint64_t nv;
            ifs.read(reinterpret_cast<char*>(&nv), 8);
            fn.vector_indices.resize(nv);
            ifs.read(reinterpret_cast<char*>(fn.vector_indices.data()),
                     nv * sizeof(faiss::int_vid_t));
        } else {
            ifs.read(reinterpret_cast<char*>(&fn.num_children), 8);
        }

        uint64_t ns;
        ifs.read(reinterpret_cast<char*>(&ns), 8);
        for (uint64_t j = 0; j < ns; ++j) {
            int32_t tid;
            ifs.read(reinterpret_cast<char*>(&tid), 4);
            uint64_t sz;
            ifs.read(reinterpret_cast<char*>(&sz), 8);
            auto& sl = fn.shortlists[tid];
            sl.resize(sz);
            ifs.read(reinterpret_cast<char*>(sl.data()), sz * sizeof(faiss::int_vid_t));
        }
    }

    // --- Reconstruct tree from flat nodes ---
    // Delete old tree_root (created by constructor)
    if (idx.tree_root) delete idx.tree_root;

    // Build TreeNode* from FlatNode, link parent/children
    std::vector<faiss::TreeNode*> nodes(num_nodes, nullptr);
    for (uint64_t i = 0; i < num_nodes; ++i) {
        auto& fn = flat_nodes[i];

        // TreeNode's constructor copies the centroid internally.
        auto* node = new faiss::TreeNode(
            fn.level, fn.sibling_id, nullptr, fn.centroid.data(), dim, 1000, 0.001f);
        node->node_id = fn.node_id;

        // Restore vector_indices (for leaf)
        if (fn.is_leaf) {
            for (auto vid : fn.vector_indices)
                node->vector_indices.insert(vid);
        }

        // Restore shortlists
        for (auto& [tid, sl] : fn.shortlists) {
            node->shortlists[tid] = faiss::ShortList(sl);
            node->bf.insert(tid);
        }

        nodes[i] = node;
    }

    // Link parent-children.
    // The save format is BFS (root first; each node's children pushed in
    // order), so node i's children are simply the next `num_children` unused
    // nodes in file order. This also preserves positional order: children[k]
    // has sibling_id == k, which build_temp_index_for_filter relies on when
    // indexing children[child_idx] directly.
    uint64_t next_child = 1;
    bool link_ok = true;
    for (uint64_t i = 0; i < num_nodes && link_ok; ++i) {
        if (flat_nodes[i].is_leaf) continue;
        const uint64_t nc = flat_nodes[i].num_children;
        if (next_child + nc > num_nodes) { link_ok = false; break; }
        for (uint64_t k = 0; k < nc; ++k) {
            faiss::TreeNode* child = nodes[next_child + k];
            child->parent = nodes[i];
            nodes[i]->children.push_back(child);
        }
        next_child += nc;
    }
    if (!link_ok || next_child != num_nodes) {
        std::cerr << "[Curator] ERROR: inconsistent tree structure in " << path
                  << " (linked " << next_child << " of " << num_nodes
                  << " nodes). Rebuilding." << std::endl;
        // Break parent-child ownership before deleting to avoid double free,
        // then delete every node individually.
        for (auto* node : nodes) node->children.clear();
        for (auto* node : nodes) delete node;
        idx.tree_root = nullptr;
        return false;
    }

    // Ancestor bloom filters must cover all descendant tenants (grant_access
    // maintains this online; restore it here). BFS order guarantees children
    // come after parents, so a reverse sweep is a post-order update.
    for (uint64_t i = num_nodes; i-- > 0;) {
        nodes[i]->bf = nodes[i]->recompute_bloom_filter();
    }

    idx.tree_root = nodes[0];
    idx.ntotal = n;

    std::cout << "[Curator] FAISS tree loaded from " << path
              << " (" << num_nodes << " nodes)" << std::endl;
    return true;
}

void curator_save_index(const CuratorContext& ctx, const std::string& path) {
    // Auto-create parent directory
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[Curator] ERROR: cannot write to " << path << std::endl;
        return;
    }

    // Header: magic + version + params
    const uint32_t magic = 0x43555238;  // "CUR8"
    const uint32_t version = 2;  // v2: aligned Curator params, fixed tid_allocator load
    ofs.write(reinterpret_cast<const char*>(&magic), 4);
    ofs.write(reinterpret_cast<const char*>(&version), 4);
    ofs.write(reinterpret_cast<const char*>(&ctx.num_points), 4);
    ofs.write(reinterpret_cast<const char*>(&ctx.nlist), 4);
    ofs.write(reinterpret_cast<const char*>(&ctx.nprobe), 4);
    ofs.write(reinterpret_cast<const char*>(&ctx.max_leaf_size), 4);
    ofs.write(reinterpret_cast<const char*>(&ctx.search_ef), 4);
    ofs.write(reinterpret_cast<const char*>(&ctx.beam_size), 4);

    // Inverted index: number of labels, then for each: label + roaring bitmap
    uint32_t n_labels = ctx.inverted.size();
    ofs.write(reinterpret_cast<const char*>(&n_labels), 4);
    for (const auto& [label, rb] : ctx.inverted) {
        ofs.write(reinterpret_cast<const char*>(&label), 4);
        uint32_t rb_size = rb.getSizeInBytes();
        ofs.write(reinterpret_cast<const char*>(&rb_size), 4);
        std::vector<char> buf(rb_size);
        rb.write(buf.data());
        ofs.write(buf.data(), rb_size);
    }

    size_t inv_size_mb = static_cast<size_t>(ofs.tellp()) / 1024 / 1024;
    ofs.close();
    std::cout << "[Curator] Inverted index saved to " << path
              << " (" << n_labels << " labels, "
              << inv_size_mb << " MB)" << std::endl;

    // Also save FAISS tree alongside
    std::string faiss_path = std::filesystem::path(path)
        .replace_extension("").string() + "_faiss.bin";
    curator_save_faiss_tree(ctx, faiss_path);
}

void curator_load_index(
    CuratorContext& ctx,
    const std::shared_ptr<IStorage>& base_storage,
    const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cout << "[Curator] No saved index at " << path << ", will build" << std::endl;
        curator_build_index(ctx, base_storage);
        curator_save_index(ctx, path);
        return;
    }

    // Header
    uint32_t magic, version;
    ifs.read(reinterpret_cast<char*>(&magic), 4);
    ifs.read(reinterpret_cast<char*>(&version), 4);
    if (magic != 0x43555238 || version != 2) {
        std::cerr << "[Curator] Stale or bad index file at " << path
                  << " (need version 2), rebuilding" << std::endl;
        curator_build_index(ctx, base_storage);
        curator_save_index(ctx, path);
        return;
    }

    // Params
    ifs.read(reinterpret_cast<char*>(&ctx.num_points), 4);
    ifs.read(reinterpret_cast<char*>(&ctx.nlist), 4);
    ifs.read(reinterpret_cast<char*>(&ctx.nprobe), 4);
    ifs.read(reinterpret_cast<char*>(&ctx.max_leaf_size), 4);
    ifs.read(reinterpret_cast<char*>(&ctx.search_ef), 4);
    ifs.read(reinterpret_cast<char*>(&ctx.beam_size), 4);

    // Inverted index
    uint32_t n_labels;
    ifs.read(reinterpret_cast<char*>(&n_labels), 4);
    ctx.inverted.clear();
    for (uint32_t i = 0; i < n_labels; ++i) {
        LabelType label;
        ifs.read(reinterpret_cast<char*>(&label), 4);
        uint32_t rb_size;
        ifs.read(reinterpret_cast<char*>(&rb_size), 4);
        std::vector<char> buf(rb_size);
        ifs.read(buf.data(), rb_size);
        ctx.inverted[label] = roaring::Roaring::read(buf.data());
    }

    std::cout << "[Curator] Loaded inverted index from " << path
              << " (" << n_labels << " labels)" << std::endl;

    // Try loading FAISS tree; rebuild if not available
    std::string faiss_path = std::filesystem::path(path)
        .replace_extension("").string() + "_faiss.bin";
    if (curator_load_faiss_tree(ctx, faiss_path)) {
        ctx.ready = true;
    } else {
        std::cout << "[Curator] No saved FAISS tree, rebuilding..." << std::endl;
        curator_build_faiss_tree(ctx, base_storage);
        curator_save_faiss_tree(ctx, faiss_path);
        ctx.ready = true;
    }
}

void curator_update_search_ef(CuratorContext& ctx, int search_ef) {
    ctx.search_ef = search_ef;
    if (ctx.index) {
        ctx.index->search_ef = static_cast<size_t>(search_ef);
    }
}

}  // namespace ANNS
