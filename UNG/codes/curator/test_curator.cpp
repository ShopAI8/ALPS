// Minimal Curator test: loads SODA binary data, builds Curator, runs queries.
// No SODA library dependency — just Curator + roaring + standard C++.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <roaring/roaring.hh>
#include "faiss/MultiTenantIndexIVFHierarchical.h"

static std::vector<float> load_bin(const std::string& path, uint32_t& n, uint32_t& dim) {
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&dim), 4);
    std::vector<float> v(n * dim);
    f.read(reinterpret_cast<char*>(v.data()), n * dim * 4);
    return v;
}

static std::vector<std::vector<uint32_t>> load_labels(const std::string& path) {
    std::vector<std::vector<uint32_t>> r;
    std::ifstream f(path);
    for (std::string line; std::getline(f, line);) {
        std::vector<uint32_t> lbls;
        if (!line.empty())
            for (size_t p = 0; p < line.size();) {
                auto np = line.find(',', p);
                if (np == std::string::npos) np = line.size();
                lbls.push_back(std::stoul(line.substr(p, np - p)));
                p = np + 1;
            }
        r.push_back(lbls);
    }
    return r;
}

static std::vector<float> load_fvecs(const std::string& path, uint32_t& n, uint32_t& dim) {
    std::ifstream f(path, std::ios::binary);
    std::vector<float> v;
    n = 0;
    while (f.peek() != EOF) {
        int32_t d;
        f.read(reinterpret_cast<char*>(&d), 4);
        if (!f) break;
        dim = d;
        v.resize(v.size() + d);
        f.read(reinterpret_cast<char*>(v.data() + n * d), d * 4);
        ++n;
    }
    return v;
}

static void load_gt(const std::string& path,
                    uint32_t nq, uint32_t K,
                    std::vector<std::pair<uint32_t, float>>& gt) {
    gt.resize(nq * K);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(gt.data()), nq * K * 8);
}

int main(int argc, char** argv) {
    std::string ds = argc > 1 ? argv[1] : "Reviews";
    std::string data_dir = "/noraiddata/lijiakang/FilterVector/FilterVectorData/" + ds;
    std::string result_dir = "/noraiddata/lijiakang/FilterVector/FilterVectorResults/" + ds;
    std::string qdir = "query_select_200_A_B_C-sub-base-123456789_random_300";

    uint32_t K = 10;
    std::cout << "=== Curator Test: " << ds << " ===" << std::endl;

    // Load base
    uint32_t nb, dim;
    auto X = load_bin(data_dir + "/" + ds + "_base.bin", nb, dim);
    auto labels = load_labels(data_dir + "/" + ds + "_base_labels.txt");
    std::cout << "Base: " << nb << " x " << dim << ", " << labels.size() << " label rows" << std::endl;

    // Load queries
    uint32_t nq, qdim;
    auto Q = load_fvecs(data_dir + "/" + qdir + "/" + ds + "_query.fvecs", nq, qdim);
    auto qlabels = load_labels(data_dir + "/" + qdir + "/" + ds + "_query_labels.txt");
    std::cout << "Queries: " << nq << " x " << qdim << std::endl;

    // Load GT
    std::string gt_file = result_dir + "/GroundTruth/GT_" + qdir + "_K" + std::to_string(K) +
                          "/" + ds + "_gt_labels_containment.bin";
    std::vector<std::pair<uint32_t, float>> gt;
    load_gt(gt_file, nq, K, gt);
    std::cout << "GT: " << nq << " x " << K << std::endl;

    // Build inverted index for fast containment
    std::unordered_map<uint32_t, roaring::Roaring> inverted;
    for (uint32_t i = 0; i < nb; ++i)
        for (auto lbl : labels[i]) inverted[lbl].add(i);
    std::cout << "Inverted: " << inverted.size() << " unique labels" << std::endl;

    // Build Curator
    int nlist = 32, nprobe = 32, max_leaf = 128, search_ef = 320;
    std::cout << "\n[Build] nlist=" << nlist << " nprobe=" << nprobe << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    faiss::MultiTenantIndexIVFHierarchical curator(
        dim, nlist, faiss::METRIC_L2,
        1000, 0.001f, 128, 20, max_leaf,
        nprobe, 1.6f, 0.2f, search_ef, 1, false);

    curator.train(nb, X.data(), 0);
    std::vector<int64_t> ids(nb);
    std::iota(ids.begin(), ids.end(), 0);
    curator.add_vector_with_ids(nb, X.data(), ids.data());

    // Grant access
    std::unordered_map<uint32_t, int32_t> remap;
    int32_t next = 0;
    for (uint32_t i = 0; i < nb; ++i) {
        for (auto lbl : labels[i]) {
            auto it = remap.find(lbl);
            int32_t tid = (it != remap.end()) ? it->second : (remap[lbl] = next++, next - 1);
            curator.grant_access(i, static_cast<int32_t>(tid));
        }
        if ((i + 1) % 100000 == 0)
            std::cout << "  access " << (i + 1) << "/" << nb << " (" << next << " labels)" << std::endl;
    }
    double build_s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "Build: " << build_s << "s, " << next << " unique labels" << std::endl;

    // Query
    uint32_t test_n = std::min<uint32_t>(1000, nq);
    std::cout << "\n[Query] " << test_n << " queries..." << std::endl;
    double total_ms = 0, total_recall = 0;

    for (uint32_t i = 0; i < test_n; ++i) {
        // Containment filtering
        std::vector<uint32_t> qual;
        if (qlabels[i].empty()) {
            qual.resize(nb); std::iota(qual.begin(), qual.end(), 0);
        } else {
            auto sorted = qlabels[i];
            std::sort(sorted.begin(), sorted.end(), [&](auto a, auto b) {
                auto ca = inverted.count(a) ? inverted.at(a).cardinality() : 0ull;
                auto cb = inverted.count(b) ? inverted.at(b).cardinality() : 0ull;
                return ca < cb;
            });
            if (inverted.count(sorted[0])) {
                roaring::Roaring res = inverted.at(sorted[0]);
                for (size_t j = 1; j < sorted.size() && !res.isEmpty(); ++j)
                    if (inverted.count(sorted[j])) res &= inverted.at(sorted[j]);
                    else { res = {}; break; }
                qual.reserve(res.cardinality());
                for (auto it = res.begin(); it != res.end(); ++it) qual.push_back(*it);
            }
        }
        if (qual.size() < K) continue;

        auto tq = std::chrono::high_resolution_clock::now();
        std::vector<float> dists(K);
        std::vector<int64_t> lids(K);
        curator.search_with_bitmap_filter(1, Q.data() + i * dim, K,
                                          qual.data(), qual.size(),
                                          dists.data(), lids.data());
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - tq).count();
        total_ms += ms;

        std::unordered_set<int64_t> gt_set;
        for (uint32_t j = 0; j < K; ++j)
            if (gt[i * K + j].first != (uint32_t)-1)
                gt_set.insert(gt[i * K + j].first);
        int hits = 0;
        for (uint32_t j = 0; j < K; ++j) hits += gt_set.count(lids[j]);
        total_recall += gt_set.empty() ? 1.0 : (double)hits / gt_set.size();

        if ((i + 1) % 200 == 0)
            std::cout << "  " << (i + 1) << "/" << test_n
                      << " avg_t=" << total_ms/(i+1) << "ms"
                      << " avg_R@" << K << "=" << total_recall/(i+1) << std::endl;
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Recall@" << K << ": " << total_recall / test_n << std::endl;
    std::cout << "Avg time:  " << total_ms / test_n << " ms" << std::endl;
    std::cout << "Build:     " << build_s << " s" << std::endl;
    return 0;
}
