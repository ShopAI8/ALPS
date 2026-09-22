#include <omp.h>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <vector>
#include <queue>
#include <stack>
#include <bitset>
#include <cstring>
#include <cctype>
#include <cmath>

#include <random>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <filesystem>
#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <iomanip>
#include <atomic> 
#include <mutex>
#include <ThreadPool.h>
#include <shared_mutex>

#include "utils.h"
#include "vamana/vamana.h"
#include "include/uni_nav_graph.h"
#include <roaring/roaring.h>
#include <roaring/roaring.hh>
#ifdef ENABLE_KNOWHERE_MILVUS_BASELINE
#include <knowhere/binaryset.h>
#include <knowhere/bitsetview.h>
#include <knowhere/comp/knowhere_config.h>
#include <knowhere/comp/index_param.h>
#include <knowhere/dataset.h>
#include <knowhere/index/index_factory.h>
#include <knowhere/version.h>
#endif

namespace fs = boost::filesystem;
using BitsetType = boost::dynamic_bitset<>;

std::shared_timed_mutex lock_m;

// 文件格式常量
const std::string FVEC_EXT = ".fvecs";
const std::string TXT_EXT = ".txt";
const size_t FVEC_HEADER_SIZE = sizeof(uint32_t);
struct QueryTask
{
   ANNS::IdxType vec_id;         // 向量ID
   std::vector<uint32_t> labels; // 查询标签集
};

struct TreeInfo
{
   ANNS::IdxType root_group_id;
   ANNS::LabelType root_label_id; // 存储概念上的根标签ID
   size_t coverage_count;
   // 用于按覆盖率降序排序
   bool operator<(const TreeInfo &other) const
   {
      return coverage_count > other.coverage_count;
   }
};

namespace ANNS
{
   namespace
   {
#ifdef ENABLE_KNOWHERE_MILVUS_BASELINE
      struct MilvusKnowhereCacheMeta {
         std::string index_kind;
         int64_t rows = 0;
         int64_t dim = 0;
         int nlist = 0;
         int nprobe = 0;
         double build_time_ms = -1.0;
         double train_time_ms = -1.0;
         double add_time_ms = -1.0;
         double index_sizeof_mb = 0.0;
         double serialized_index_size_mb = 0.0;
      };

      bool
      save_milvus_knowhere_meta(const std::string& meta_path, const MilvusKnowhereCacheMeta& meta) {
         std::ofstream out(meta_path, std::ios::trunc);
         if (!out.is_open()) {
            return false;
         }
         out << std::fixed << std::setprecision(6);
         out << "index_kind=" << meta.index_kind << "\n";
         out << "rows=" << meta.rows << "\n";
         out << "dim=" << meta.dim << "\n";
         out << "nlist=" << meta.nlist << "\n";
         out << "nprobe=" << meta.nprobe << "\n";
         if (meta.build_time_ms >= 0.0) {
            out << "build_time_ms=" << meta.build_time_ms << "\n";
         }
         if (meta.train_time_ms >= 0.0) {
            out << "train_time_ms=" << meta.train_time_ms << "\n";
         }
         if (meta.add_time_ms >= 0.0) {
            out << "add_time_ms=" << meta.add_time_ms << "\n";
         }
         if (meta.index_sizeof_mb > 0.0) {
            out << "index_sizeof_mb=" << meta.index_sizeof_mb << "\n";
         }
         if (meta.serialized_index_size_mb > 0.0) {
            out << "serialized_index_size_mb=" << meta.serialized_index_size_mb << "\n";
         }
         return out.good();
      }

      bool
      load_milvus_knowhere_meta(const std::string& meta_path, MilvusKnowhereCacheMeta& meta) {
         std::ifstream in(meta_path);
         if (!in.is_open()) {
            return false;
         }
         std::string line;
         while (std::getline(in, line)) {
            const auto pos = line.find('=');
            if (pos == std::string::npos) {
               continue;
            }
            const std::string key = line.substr(0, pos);
            const std::string value = line.substr(pos + 1);
            if (key == "index_kind") {
               meta.index_kind = value;
            } else if (key == "rows") {
               meta.rows = std::stoll(value);
            } else if (key == "dim") {
               meta.dim = std::stoll(value);
            } else if (key == "nlist") {
               meta.nlist = std::stoi(value);
            } else if (key == "nprobe") {
               meta.nprobe = std::stoi(value);
            } else if (key == "build_time_ms") {
               meta.build_time_ms = std::stod(value);
            } else if (key == "train_time_ms") {
               meta.train_time_ms = std::stod(value);
            } else if (key == "add_time_ms") {
               meta.add_time_ms = std::stod(value);
            } else if (key == "index_sizeof_mb") {
               meta.index_sizeof_mb = std::stod(value);
            } else if (key == "serialized_index_size_mb") {
               meta.serialized_index_size_mb = std::stod(value);
            }
         }
         return !meta.index_kind.empty() && meta.rows > 0 && meta.dim > 0;
      }

      bool
      save_milvus_knowhere_index_file(const knowhere::Index<knowhere::IndexNode>& index,
                                      const std::string& file_path,
                                      uint64_t* serialized_size_bytes = nullptr) {
         knowhere::BinarySet binset;
         if (index.Serialize(binset) != knowhere::Status::success || binset.binary_map_.empty()) {
            return false;
         }
         const auto& bin = binset.binary_map_.begin()->second;
         if (bin == nullptr || bin->data == nullptr || bin->size <= 0) {
            return false;
         }
         std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
         if (!out.is_open()) {
            return false;
         }
         out.write(reinterpret_cast<const char*>(bin->data.get()), static_cast<std::streamsize>(bin->size));
         if (serialized_size_bytes != nullptr) {
            *serialized_size_bytes = static_cast<uint64_t>(bin->size);
         }
         return out.good();
      }
#endif

      std::string trim_copy(std::string value) {
         const auto begin = value.find_first_not_of(" \t\r\n");
         if (begin == std::string::npos) {
            return "";
         }
         const auto end = value.find_last_not_of(" \t\r\n");
         return value.substr(begin, end - begin + 1);
      }

      int map_router_algo_label_to_baseline_id(const std::string& raw_label) {
         const std::string label = trim_copy(raw_label);
         if (label == "ACORN-gamma") return 2;
         if (label == "ACORN-gamma-improved") return 3;
         if (label == "ACORN-1") return 6;
         // SODA's "NaviX" family is expected to route into the
         // NaviX-ACORN baseline, which runs on the ACORN index.
         if (label == "NaviX" || label == "NaviX-ACORN") return 4;
         if (label == "FAVOR" || label == "FAVOR-HNSW") return 12;
         if (label == "UNG+") return 8;
         // The wide-table selector uses the label UNG++ for the sorted-LNG
         // implementation (baseline id 15).
         if (label == "UNG++" || label == "UNG++-sorted-lng") return 15;
         if (label == "pre-filter") return 5;
         if (label == "Curator") return 13;
         return -1;
      }

      std::optional<int> load_router_target_algo_id_from_class_labels(const fs::path& labels_path) {
         std::ifstream in(labels_path.string());
         if (!in.is_open()) {
            return std::nullopt;
         }

         std::string line;
         while (std::getline(in, line)) {
            line = trim_copy(line);
            if (line.empty()) {
               continue;
            }

            const auto comma_pos = line.find(',');
            if (comma_pos == std::string::npos) {
               continue;
            }

            const std::string class_id_str = trim_copy(line.substr(0, comma_pos));
            const std::string label = trim_copy(line.substr(comma_pos + 1));

            if (class_id_str != "0") {
               continue;
            }

            const int mapped_id = map_router_algo_label_to_baseline_id(label);
            if (mapped_id >= 0) {
               return mapped_id;
            }

            std::cerr << "[SODA] Unsupported class_labels entry for class 0: "
                      << label << std::endl;
            return std::nullopt;
         }

         return std::nullopt;
      }

      std::optional<int> load_router_target_algo_id_from_env() {
         const char* env_value = std::getenv("ROUTER_ZERO_ALGORITHM");
         if (env_value == nullptr) {
            return std::nullopt;
         }

         const std::string label = trim_copy(env_value);
         if (label.empty()) {
            return std::nullopt;
         }

         const int mapped_id = map_router_algo_label_to_baseline_id(label);
         if (mapped_id >= 0) {
            return mapped_id;
         }

         std::cerr << "[SODA] Unsupported ROUTER_ZERO_ALGORITHM override: "
                   << label << std::endl;
         return std::nullopt;
      }

      std::optional<fs::path> find_latest_router_dir(const fs::path& root_dir) {
         if (!fs::exists(root_dir) || !fs::is_directory(root_dir)) {
            return std::nullopt;
         }

         bool found = false;
         std::time_t newest_time = std::numeric_limits<std::time_t>::min();
         fs::path newest_dir;

         for (fs::recursive_directory_iterator it(root_dir), end; it != end; ++it) {
            if (!fs::is_regular_file(it->path()) || it->path().filename() != "router.onnx") {
               continue;
            }

            const auto parent_dir = it->path().parent_path();
            const std::time_t write_time = fs::last_write_time(it->path());
            if (!found || write_time > newest_time) {
               newest_time = write_time;
               newest_dir = parent_dir;
               found = true;
            }
         }

         if (!found) {
            return std::nullopt;
         }

         return newest_dir;
      }

      std::optional<fs::path> resolve_smart_route_model_dir(const std::string& selector_model_prefix) {
         const fs::path base_dir(selector_model_prefix);
         const std::vector<fs::path> direct_candidates = {
            base_dir,
            base_dir / "smart_route",
         };

         for (const auto& candidate_dir : direct_candidates) {
            if (fs::exists(candidate_dir / "router.onnx")) {
               return candidate_dir;
            }
         }

         const auto dynamic_dir = find_latest_router_dir(base_dir / "smart_route_dynamic");
         if (dynamic_dir.has_value()) {
            return dynamic_dir;
         }

         return std::nullopt;
      }

      struct FavorCacheMeta {
         int64_t rows = 0;
         int64_t dim = 0;
         double build_time_ms = -1.0;
         double index_sizeof_mb = 0.0;
         double serialized_index_size_mb = 0.0;
      };

      bool
      save_favor_meta(const std::string& meta_path, const FavorCacheMeta& meta) {
         std::ofstream out(meta_path, std::ios::trunc);
         if (!out.is_open()) {
            return false;
         }
         out << std::fixed << std::setprecision(6);
         out << "rows=" << meta.rows << "\n";
         out << "dim=" << meta.dim << "\n";
         if (meta.build_time_ms >= 0.0) {
            out << "build_time_ms=" << meta.build_time_ms << "\n";
         }
         if (meta.index_sizeof_mb > 0.0) {
            out << "index_sizeof_mb=" << meta.index_sizeof_mb << "\n";
         }
         if (meta.serialized_index_size_mb > 0.0) {
            out << "serialized_index_size_mb=" << meta.serialized_index_size_mb << "\n";
         }
         return out.good();
      }

      bool
      load_favor_meta(const std::string& meta_path, FavorCacheMeta& meta) {
         std::ifstream in(meta_path);
         if (!in.is_open()) {
            return false;
         }
         std::string line;
         while (std::getline(in, line)) {
            const auto pos = line.find('=');
            if (pos == std::string::npos) {
               continue;
            }
            const std::string key = line.substr(0, pos);
            const std::string value = line.substr(pos + 1);
            if (key == "rows") {
               meta.rows = std::stoll(value);
            } else if (key == "dim") {
               meta.dim = std::stoll(value);
            } else if (key == "build_time_ms") {
               meta.build_time_ms = std::stod(value);
            } else if (key == "index_sizeof_mb") {
               meta.index_sizeof_mb = std::stod(value);
            } else if (key == "serialized_index_size_mb") {
               meta.serialized_index_size_mb = std::stod(value);
            }
         }
         return meta.rows > 0 && meta.dim > 0;
      }

      template <typename dist_t>
      uint64_t
      estimate_favor_logical_size_bytes(const favor::FAVOR<dist_t>& index) {
         const uint64_t cur_count = static_cast<uint64_t>(index.cur_element_count.load());
         uint64_t total_bytes = cur_count * static_cast<uint64_t>(index.size_data_per_element_);

         for (uint64_t i = 0; i < cur_count; ++i) {
            const int level = index.element_levels_[i];
            if (level > 0) {
               total_bytes += static_cast<uint64_t>(index.size_links_per_element_) *
                              static_cast<uint64_t>(level);
            }
         }

         // Match the "logical memory / sizeof-like" accounting style used by other baselines:
         // count the major allocated bookkeeping arrays, but not allocator/container slack.
         total_bytes += static_cast<uint64_t>(index.max_elements_) * sizeof(void*);
         total_bytes += static_cast<uint64_t>(index.max_elements_) * sizeof(int);
         return total_bytes;
      }

      int
      compute_aligned_efs(
         ANNS::IdxType Lsearch,
         int lsearch_start,
         int lsearch_step,
         int efs_start,
         int efs_step_slow,
         int efs_step_fast,
         int lsearch_threshold) {
         int current_efs;
         if (Lsearch <= lsearch_threshold) {
            current_efs = efs_start + ((Lsearch - lsearch_start) / lsearch_step) * efs_step_slow;
         } else {
            int num_steps_in_slow_zone = (lsearch_threshold - lsearch_start) / lsearch_step;
            int efs_at_threshold = efs_start + num_steps_in_slow_zone * efs_step_slow;
            int num_steps_in_fast_zone = (Lsearch - lsearch_threshold) / lsearch_step;
            current_efs = efs_at_threshold + num_steps_in_fast_zone * efs_step_fast;
         }
         return std::max(current_efs, efs_start);
      }

      void
      upsert_build_time_metric(const std::string& csv_path, const std::string& metric_name, const std::string& metric_value) {
         std::vector<std::pair<std::string, std::string>> rows;
         rows.emplace_back("Index Name", "Build Time (ms)");

         bool found = false;
         std::ifstream in(csv_path);
         if (in.is_open()) {
            std::string line;
            bool first_line = true;
            while (std::getline(in, line)) {
               if (line.empty()) {
                  continue;
               }
               const auto pos = line.find(',');
               if (pos == std::string::npos) {
                  continue;
               }
               std::string key = line.substr(0, pos);
               std::string value = line.substr(pos + 1);
               if (first_line && key == "Index Name") {
                  first_line = false;
                  continue;
               }
               first_line = false;
               if (key == metric_name) {
                  rows.emplace_back(metric_name, metric_value);
                  found = true;
               } else {
                  rows.emplace_back(std::move(key), std::move(value));
               }
            }
         }
         if (!found) {
            rows.emplace_back(metric_name, metric_value);
         }

         std::ofstream out(csv_path, std::ios::trunc);
         if (!out.is_open()) {
            return;
         }
         for (const auto& row : rows) {
            out << row.first << "," << row.second << "\n";
         }
      }

      template <class Function>
      inline void FavorParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
         if (numThreads == 0) {
            numThreads = std::thread::hardware_concurrency();
         }

         if (numThreads <= 1) {
            for (size_t id = start; id < end; ++id) {
               fn(id, 0);
            }
            return;
         }

         std::vector<std::thread> threads;
         std::atomic<size_t> current(start);
         std::exception_ptr lastException = nullptr;
         std::mutex lastExceptMutex;

         for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.emplace_back([&, threadId] {
               while (true) {
                  size_t id = current.fetch_add(1);
                  if (id >= end) {
                     break;
                  }
                  try {
                     fn(id, threadId);
                  } catch (...) {
                     std::lock_guard<std::mutex> lock(lastExceptMutex);
                     if (!lastException) {
                        lastException = std::current_exception();
                     }
                     current = end;
                     break;
                  }
               }
            });
         }

         for (auto& thread : threads) {
            thread.join();
         }
         if (lastException) {
            std::rethrow_exception(lastException);
         }
      }

      class FavorContainmentFilter final {
         public:
            FavorContainmentFilter(ANNS::IStorage* base_storage,
                                   const std::vector<ANNS::LabelType>& query_labels)
               : base_storage_(base_storage), query_labels_(query_labels) {}

            bool check(size_t internal_id) const {
               const auto& base_labels = base_storage_->get_label_set(static_cast<ANNS::IdxType>(internal_id));
               if (query_labels_.empty()) {
                  return true;
               }
               if (base_labels.size() < query_labels_.size()) {
                  return false;
               }
               return std::includes(base_labels.begin(), base_labels.end(),
                                    query_labels_.begin(), query_labels_.end());
            }

            std::string getCacheKey() const {
               std::string key;
               key.reserve(query_labels_.size() * 8);
               for (auto label : query_labels_) {
                  key += std::to_string(label);
                  key.push_back(',');
               }
               return key;
            }

         private:
            ANNS::IStorage* base_storage_;
            const std::vector<ANNS::LabelType>& query_labels_;
      };

      class ACORNRabitQDistanceBackend final : public faiss::ACORNDistanceBackend
      {
         public:
            struct Snapshot
            {
               double ctx_prepare_time_ms = 0.0;
               double ctx_rotate_time_ms = 0.0;
               double ctx_q2c_time_ms = 0.0;
               double ctx_wrapper_time_ms = 0.0;
               double bin_time_ms = 0.0;
               double full_time_ms = 0.0;
               size_t bin_calls = 0;
               size_t full_calls = 0;
               size_t prepare_calls = 0;
            };

            explicit ACORNRabitQDistanceBackend(const ANNS::rabitq::RabitQSideIndex *side_index)
                : side_index_(side_index), shared_metrics_(std::make_shared<SharedMetrics>())
            {
            }

            ~ACORNRabitQDistanceBackend() override
            {
               flush_local_to_shared();
            }

            std::unique_ptr<faiss::ACORNDistanceBackend> clone() const override
            {
               return std::unique_ptr<faiss::ACORNDistanceBackend>(
                   new ACORNRabitQDistanceBackend(side_index_, shared_metrics_));
            }

            bool prepare_query(const float *x) override
            {
               if (side_index_ == nullptr || !side_index_->enabled() || x == nullptr)
               {
                  return false;
               }
               query_ctx_ = ANNS::rabitq::RabitQSideIndex::QueryContext{};
               top_full_worst_ = std::priority_queue<float>();
               auto begin = std::chrono::high_resolution_clock::now();
               ANNS::rabitq::RabitQSideIndex::InitTiming init_timing;
               const bool ok = side_index_->init_query(reinterpret_cast<const char *>(x), query_ctx_, &init_timing);
               const uint64_t elapsed_ns = static_cast<uint64_t>(
                   std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::high_resolution_clock::now() - begin)
                       .count());
               if (ok)
               {
                  local_metrics_.ctx_prepare_time_ns += elapsed_ns;
                  local_metrics_.ctx_rotate_time_ns += ms_to_ns(init_timing.rotate_ms);
                  local_metrics_.ctx_q2c_time_ns += ms_to_ns(init_timing.q_to_centroids_ms);
                  local_metrics_.ctx_wrapper_time_ns += ms_to_ns(init_timing.wrapper_ms);
                  local_metrics_.prepare_calls += 1;
               }
               return ok;
            }

            float distance(faiss::idx_t i) override
            {
               if (side_index_ == nullptr || i < 0)
               {
                  return std::numeric_limits<float>::max();
               }
               // Two-stage accounting for ACORN RabitQ path:
               // 1) estimate_bin for coarse timing/call stats
               // 2) estimate_full for final returned distance (keeps correctness aligned with exact scoring semantics)
               float low_dist = 0.0F;
               local_metrics_.bin_calls += 1;
               distance_calls_ += 1;
               float bin_dist = 0.0F;
               if (should_sample(distance_calls_))
               {
                  auto bin_begin = std::chrono::high_resolution_clock::now();
                  bin_dist = side_index_->estimate_bin(static_cast<ANNS::IdxType>(i), query_ctx_, &low_dist);
                  const uint64_t bin_elapsed_ns = static_cast<uint64_t>(
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::high_resolution_clock::now() - bin_begin)
                          .count());
                  local_metrics_.bin_time_ns += (bin_elapsed_ns * kSamplePeriod);
               }
               else
               {
                  bin_dist = side_index_->estimate_bin(static_cast<ANNS::IdxType>(i), query_ctx_, &low_dist);
               }

               // Adaptive two-stage refinement:
               // - warmup: compute some full distances to bootstrap a cutoff
               // - steady-state: only refine when low_bound looks competitive
               bool should_refine = false;
               if (top_full_worst_.size() < full_warmup_)
               {
                  should_refine = true;
               }
               else if (!top_full_worst_.empty())
               {
                  const float cutoff = top_full_worst_.top();
                  should_refine = (low_dist < cutoff);
               }

               if (!should_refine)
               {
                  return bin_dist;
               }

               local_metrics_.full_calls += 1;
               float full_dist = 0.0F;
               if (should_sample(local_metrics_.full_calls))
               {
                  auto full_begin = std::chrono::high_resolution_clock::now();
                  full_dist = side_index_->estimate_full(static_cast<ANNS::IdxType>(i), query_ctx_, nullptr);
                  const uint64_t full_elapsed_ns = static_cast<uint64_t>(
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::high_resolution_clock::now() - full_begin)
                          .count());
                  local_metrics_.full_time_ns += (full_elapsed_ns * kSamplePeriod);
               }
               else
               {
                  full_dist = side_index_->estimate_full(static_cast<ANNS::IdxType>(i), query_ctx_, nullptr);
               }

               if (top_full_worst_.size() < full_keep_)
               {
                  top_full_worst_.push(full_dist);
               }
               else if (full_dist < top_full_worst_.top())
               {
                  top_full_worst_.pop();
                  top_full_worst_.push(full_dist);
               }
               return full_dist;
            }

            Snapshot snapshot() const
            {
               flush_local_to_shared();
               Snapshot out;
               out.ctx_prepare_time_ms = ns_to_ms(shared_metrics_->ctx_prepare_time_ns.load(std::memory_order_relaxed));
               out.ctx_rotate_time_ms = ns_to_ms(shared_metrics_->ctx_rotate_time_ns.load(std::memory_order_relaxed));
               out.ctx_q2c_time_ms = ns_to_ms(shared_metrics_->ctx_q2c_time_ns.load(std::memory_order_relaxed));
               out.ctx_wrapper_time_ms = ns_to_ms(shared_metrics_->ctx_wrapper_time_ns.load(std::memory_order_relaxed));
               out.bin_time_ms = ns_to_ms(shared_metrics_->bin_time_ns.load(std::memory_order_relaxed));
               out.full_time_ms = ns_to_ms(shared_metrics_->full_time_ns.load(std::memory_order_relaxed));
               out.bin_calls = static_cast<size_t>(shared_metrics_->bin_calls.load(std::memory_order_relaxed));
               out.full_calls = static_cast<size_t>(shared_metrics_->full_calls.load(std::memory_order_relaxed));
               out.prepare_calls = static_cast<size_t>(shared_metrics_->prepare_calls.load(std::memory_order_relaxed));
               return out;
            }

         private:
            struct LocalMetrics
            {
               uint64_t ctx_prepare_time_ns = 0;
               uint64_t ctx_rotate_time_ns = 0;
               uint64_t ctx_q2c_time_ns = 0;
               uint64_t ctx_wrapper_time_ns = 0;
               uint64_t bin_time_ns = 0;
               uint64_t full_time_ns = 0;
               uint64_t bin_calls = 0;
               uint64_t full_calls = 0;
               uint64_t prepare_calls = 0;
            };

            struct SharedMetrics
            {
               std::atomic<uint64_t> ctx_prepare_time_ns{0};
               std::atomic<uint64_t> ctx_rotate_time_ns{0};
               std::atomic<uint64_t> ctx_q2c_time_ns{0};
               std::atomic<uint64_t> ctx_wrapper_time_ns{0};
               std::atomic<uint64_t> bin_time_ns{0};
               std::atomic<uint64_t> full_time_ns{0};
               std::atomic<uint64_t> bin_calls{0};
               std::atomic<uint64_t> full_calls{0};
               std::atomic<uint64_t> prepare_calls{0};
            };

            ACORNRabitQDistanceBackend(
                const ANNS::rabitq::RabitQSideIndex *side_index,
                std::shared_ptr<SharedMetrics> shared_metrics)
                : side_index_(side_index), shared_metrics_(std::move(shared_metrics))
            {
            }

            static uint64_t ms_to_ns(double ms)
            {
               return static_cast<uint64_t>(std::max(0.0, ms) * 1000000.0);
            }

            static double ns_to_ms(uint64_t ns)
            {
               return static_cast<double>(ns) / 1000000.0;
            }

            static bool should_sample(uint64_t counter)
            {
               return (counter % kSamplePeriod) == 0;
            }

            void flush_local_to_shared() const
            {
               if (local_flushed_)
               {
                  return;
               }
               local_flushed_ = true;
               if (local_metrics_.ctx_prepare_time_ns)
                  shared_metrics_->ctx_prepare_time_ns.fetch_add(local_metrics_.ctx_prepare_time_ns, std::memory_order_relaxed);
               if (local_metrics_.ctx_rotate_time_ns)
                  shared_metrics_->ctx_rotate_time_ns.fetch_add(local_metrics_.ctx_rotate_time_ns, std::memory_order_relaxed);
               if (local_metrics_.ctx_q2c_time_ns)
                  shared_metrics_->ctx_q2c_time_ns.fetch_add(local_metrics_.ctx_q2c_time_ns, std::memory_order_relaxed);
               if (local_metrics_.ctx_wrapper_time_ns)
                  shared_metrics_->ctx_wrapper_time_ns.fetch_add(local_metrics_.ctx_wrapper_time_ns, std::memory_order_relaxed);
               if (local_metrics_.bin_time_ns)
                  shared_metrics_->bin_time_ns.fetch_add(local_metrics_.bin_time_ns, std::memory_order_relaxed);
               if (local_metrics_.full_time_ns)
                  shared_metrics_->full_time_ns.fetch_add(local_metrics_.full_time_ns, std::memory_order_relaxed);
               if (local_metrics_.bin_calls)
                  shared_metrics_->bin_calls.fetch_add(local_metrics_.bin_calls, std::memory_order_relaxed);
               if (local_metrics_.full_calls)
                  shared_metrics_->full_calls.fetch_add(local_metrics_.full_calls, std::memory_order_relaxed);
               if (local_metrics_.prepare_calls)
                  shared_metrics_->prepare_calls.fetch_add(local_metrics_.prepare_calls, std::memory_order_relaxed);
            }

            const ANNS::rabitq::RabitQSideIndex *side_index_ = nullptr;
            ANNS::rabitq::RabitQSideIndex::QueryContext query_ctx_{};
            std::shared_ptr<SharedMetrics> shared_metrics_;
            mutable LocalMetrics local_metrics_{};
            mutable bool local_flushed_ = false;
            uint64_t distance_calls_ = 0;
            std::priority_queue<float> top_full_worst_{};
            size_t full_warmup_ = 128;
            size_t full_keep_ = 256;
            static constexpr uint64_t kSamplePeriod = 32;
      };
   } // namespace


   void UniNavGraph::configure_rabitq_build(bool enable, size_t total_bits)
   {
      if (!enable)
      {
         _build_rabitq_side_index = false;
         _rabitq_total_bits = total_bits;
         return;
      }

      if (total_bits < 1 || total_bits > 9)
      {
         std::cerr << "[RabitQ] Invalid total_bits=" << total_bits
                   << ", expected [1, 9]. Disable RabitQ side-index build." << std::endl;
         _build_rabitq_side_index = false;
         _rabitq_total_bits = 4;
         return;
      }

      _build_rabitq_side_index = true;
      _rabitq_total_bits = total_bits;
   }

   void UniNavGraph::set_ung_distance_mode(const std::string &mode)
   {
      std::string lowered = mode;
      std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
         return static_cast<char>(std::tolower(c));
      });

      if (lowered == "rabitq")
      {
         _ung_distance_mode = UngDistanceMode::RabitQ;
         if (!_rabitq_side_index.enabled() &&
             !_acorn_rabitq_side_index.enabled() &&
             !_acorn_1_rabitq_side_index.enabled())
         {
            std::cerr << "[RabitQ] Distance mode is set to rabitq, but side index is not available. "
                      << "UNG will fallback to exact distance at query time." << std::endl;
         }
      }
      else
      {
         if (lowered != "exact")
         {
            std::cerr << "[RabitQ] Unknown ung_distance_mode='" << mode
                      << "', fallback to exact." << std::endl;
         }
         _ung_distance_mode = UngDistanceMode::Exact;
      }
   }

   void UniNavGraph::prepare_rabitq_query_contexts(std::shared_ptr<IStorage> &query_storage,
                                                   const std::string &query_bin_file)
   {
      if (_ung_distance_mode != UngDistanceMode::RabitQ || !_rabitq_side_index.enabled())
      {
         return;
      }
      if (!query_storage)
      {
         return;
      }

      const auto num_queries = query_storage->get_num_points();
      const bool need_refresh =
          (_rabitq_cached_query_storage != query_storage.get()) ||
          (_rabitq_query_ctx_cache.size() != num_queries);

      if (!need_refresh)
      {
         return;
      }

      std::string cache_file_path;
      if (!query_bin_file.empty())
      {
         fs::path query_path(query_bin_file);
         cache_file_path = (query_path.parent_path() / "query_rabitq_ctx_cache.bin").string();
      }

      if (!cache_file_path.empty())
      {
         std::vector<std::unique_ptr<ANNS::rabitq::RabitQSideIndex::QueryContext>> loaded_cache;
         if (_rabitq_side_index.load_query_context_cache(cache_file_path, loaded_cache) &&
             loaded_cache.size() == static_cast<size_t>(num_queries))
         {
            _rabitq_query_ctx_cache = std::move(loaded_cache);
            _rabitq_query_ctx_prepare_ms.assign(num_queries, 0.0);
            _rabitq_query_ctx_rotate_ms.assign(num_queries, 0.0);
            _rabitq_query_ctx_q2c_ms.assign(num_queries, 0.0);
            _rabitq_query_ctx_wrapper_ms.assign(num_queries, 0.0);
            _rabitq_cached_query_storage = query_storage.get();
            std::cout << "[RabitQ] Query contexts loaded from cache: " << cache_file_path << std::endl;
            return;
         }
      }

      std::cout << "[RabitQ] Preparing query contexts for reuse ..." << std::endl;
      auto prep_all_start = std::chrono::high_resolution_clock::now();
      _rabitq_query_ctx_cache.clear();
      _rabitq_query_ctx_cache.resize(num_queries);
      _rabitq_query_ctx_prepare_ms.assign(num_queries, 0.0);
      _rabitq_query_ctx_rotate_ms.assign(num_queries, 0.0);
      _rabitq_query_ctx_q2c_ms.assign(num_queries, 0.0);
      _rabitq_query_ctx_wrapper_ms.assign(num_queries, 0.0);

      std::atomic<bool> all_ok{true};
      const IdxType invalid_query_id = std::numeric_limits<IdxType>::max();
      std::atomic<IdxType> first_failed_query_id{invalid_query_id};
      std::atomic<IdxType> prepared_count{0};
      const IdxType progress_step = 1000;
      const int omp_threads = std::max<int>(1, std::min<int>(
          static_cast<int>(num_queries), omp_get_max_threads()));
      std::cout << "[RabitQ] Query context prepare OpenMP threads: "
                << omp_threads << std::endl;

#pragma omp parallel for schedule(dynamic, 64) num_threads(omp_threads)
      for (IdxType query_id = 0; query_id < num_queries; ++query_id)
      {
         if (!all_ok.load(std::memory_order_relaxed))
         {
            continue;
         }

         auto per_start = std::chrono::high_resolution_clock::now();
         auto ctx = std::make_unique<ANNS::rabitq::RabitQSideIndex::QueryContext>();
         ANNS::rabitq::RabitQSideIndex::InitTiming init_timing;
         if (!_rabitq_side_index.init_query(query_storage->get_vector(query_id), *ctx, &init_timing))
         {
            all_ok.store(false, std::memory_order_relaxed);
            IdxType expected = invalid_query_id;
            first_failed_query_id.compare_exchange_strong(expected, query_id, std::memory_order_relaxed);
            continue;
         }
         _rabitq_query_ctx_prepare_ms[query_id] = std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - per_start
         ).count();
         _rabitq_query_ctx_rotate_ms[query_id] = init_timing.rotate_ms;
         _rabitq_query_ctx_q2c_ms[query_id] = init_timing.q_to_centroids_ms;
         _rabitq_query_ctx_wrapper_ms[query_id] = init_timing.wrapper_ms;
         _rabitq_query_ctx_cache[query_id] = std::move(ctx);

         const IdxType done = prepared_count.fetch_add(1, std::memory_order_relaxed) + 1;
         if (done == num_queries || done % progress_step == 0)
         {
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - prep_all_start
            ).count();
            const double progress = (num_queries > 0)
                                        ? (100.0 * static_cast<double>(done) /
                                           static_cast<double>(num_queries))
                                        : 100.0;
#pragma omp critical(rabitq_progress_log)
            {
               std::cout << "[RabitQ] Query context prepare progress: "
                         << done << "/" << num_queries
                         << " (" << std::fixed << std::setprecision(1)
                         << progress << "%), elapsed="
                         << std::setprecision(1) << elapsed_ms << " ms"
                         << std::defaultfloat << std::endl;
            }
         }
      }

      if (!all_ok.load(std::memory_order_relaxed))
      {
         const IdxType failed_id = first_failed_query_id.load(std::memory_order_relaxed);
         std::cerr << "[RabitQ] Failed to prepare query context for query "
                   << failed_id << ", fallback to per-query initialization." << std::endl;
         _rabitq_query_ctx_cache.clear();
         _rabitq_query_ctx_prepare_ms.clear();
         _rabitq_query_ctx_rotate_ms.clear();
         _rabitq_query_ctx_q2c_ms.clear();
         _rabitq_query_ctx_wrapper_ms.clear();
         _rabitq_cached_query_storage = nullptr;
         return;
      }

      if (all_ok.load(std::memory_order_relaxed))
      {
         _rabitq_cached_query_storage = query_storage.get();
         const double total_ms = std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - prep_all_start
         ).count();
         std::cout << "[RabitQ] Query contexts prepared in " << total_ms
                   << " ms and will be reused across Lsearch runs." << std::endl;
         if (!cache_file_path.empty())
         {
            if (_rabitq_side_index.save_query_context_cache(cache_file_path, _rabitq_query_ctx_cache))
            {
               std::cout << "[RabitQ] Query contexts saved to cache: " << cache_file_path << std::endl;
            }
            else
            {
               std::cerr << "[RabitQ] Failed to save query context cache: " << cache_file_path << std::endl;
            }
         }
      }
   }

   void UniNavGraph::build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                           std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                           IdxType max_degree, IdxType Lbuild, float alpha, std::string dataset,
                           ANNS::AcornInUng new_cross_edge)
   {
      auto all_start_time = std::chrono::high_resolution_clock::now();
      _base_storage = base_storage;
      _num_points = base_storage->get_num_points();
      _distance_handler = distance_handler;
      std::cout << "- Scenario: " << scenario << std::endl;

      _dataset = dataset;

      // index parameters
      _index_name = index_name;
      _num_cross_edges = num_cross_edges;
      _max_degree = max_degree;
      _Lbuild = Lbuild;
      _alpha = alpha;
      _num_threads = num_threads;
      _scenario = scenario;

      std::cout << "Dividing groups and building the trie tree index ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      build_trie_and_divide_groups();
      _graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      _global_graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      std::cout << "begin prepare_group_storages_graphs" << std::endl;
      prepare_group_storages_graphs();
      _label_processing_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
      std::cout << "- Finished in " << _label_processing_time << " ms" << std::endl;

      // build graph index for each group
      build_graph_for_all_groups();
      // build_global_vamana_graph();
      build_vector_and_attr_graph(); // fxy_add

      // for label equality scenario, there is no need for label navigating graph and cross-group edges
      if (_scenario == "equality")
      {
         add_offset_for_uni_nav_graph();
      }
      else
      {

         // build the label navigating graph
         build_label_nav_graph();
         get_descendants_info(); // fxy_add
         {
            std::cout << "\n--- [分析] 正在统计 LNG 边数 ---" << std::endl;
            uint64_t total_lng_edges = 0;
            for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)// Walk all groups and sum their out-degree (child/superset count).
            {
               total_lng_edges += _label_nav_graph->out_neighbors[group_id].size();
            }
            std::cout << "  - 组 (Nodes) 总数: " << _num_groups << std::endl;
            std::cout << "  - LNG 边 (Edges) 总数: " << total_lng_edges << std::endl;
            if (_num_groups > 0) {
                 std::cout << "  - 平均出度 (边/组): " << static_cast<double>(total_lng_edges) / _num_groups << std::endl;
            }
            std::cout << "--- [分析] LNG 边数统计完毕 ---\n" << std::endl;
         }

         // calculate the coverage ratio
         cal_f_coverage_ratio(); // fxy_add

         // initialize_lng_descendants_coverage_bitsets();
         auto roaring_start_time = std::chrono::high_resolution_clock::now();
         initialize_roaring_bitsets();
         _build_roaring_bitsets_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - roaring_start_time).count();

         // precompute the node depths in the label navigating graph,used in hard sandwitch
         // _precompute_lng_node_depths();

         std::cout << "new_cross_edge.ung_and_acorn: " << new_cross_edge.ung_and_acorn << std::endl;
         if (!new_cross_edge.ung_and_acorn)
            build_cross_group_edges();
         else
         {
            _num_cross_edges = _num_cross_edges / 2;
            finalize_intra_group_graphs(); // Extract the local-to-global ID conversion step from `build_cross_group_edges`.

            add_new_distance_oriented_edges(
                dataset,
                num_threads,
                new_cross_edge);
         }
      }

      if (_build_rabitq_side_index)
      {
         std::cout << "Building RabitQ side index for UNG search..." << std::endl;
         auto rabitq_start = std::chrono::high_resolution_clock::now();

         // Build RabitQ clusters by first metadata label (metadata[i][0]),
         // instead of UNG internal group_id, to align with ACORN RabitQ grouping.
         std::vector<IdxType> rabitq_point_to_group(_num_points, 1);
         std::unordered_map<int64_t, IdxType> first_label_to_group;
         IdxType next_group_id = 1;
         for (IdxType point_id = 0; point_id < _num_points; ++point_id)
         {
            const auto &label_set = _base_storage->get_label_set(point_id);
            const int64_t first_label = label_set.empty() ? -1LL : static_cast<int64_t>(label_set[0]);
            auto it = first_label_to_group.find(first_label);
            if (it == first_label_to_group.end())
            {
               first_label_to_group[first_label] = next_group_id;
               rabitq_point_to_group[point_id] = next_group_id;
               ++next_group_id;
            }
            else
            {
               rabitq_point_to_group[point_id] = it->second;
            }
         }
         const IdxType rabitq_num_groups = next_group_id - 1;
         std::cout << "[RabitQ] Grouping by metadata[i][0], num_groups=" << rabitq_num_groups << std::endl;

         const bool ok = _rabitq_side_index.build(
             _base_storage,
             rabitq_point_to_group,
             rabitq_num_groups,
             _rabitq_total_bits
         );
         if (!ok)
         {
            _rabitq_build_time_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - rabitq_start)
                                       .count();
            std::cerr << "[RabitQ] Failed to build side index. Falling back to exact distance mode." << std::endl;
         }
         else
         {
            _rabitq_build_time_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - rabitq_start)
                                       .count();
            std::cout << "[RabitQ] Side index built in "
                      << _rabitq_build_time_ms
                      << " ms" << std::endl;
         }
      }
      else
      {
         _rabitq_build_time_ms = 0.0;
      }

      // index time
      _index_time = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - all_start_time)
                        .count();
   }

   void UniNavGraph::build_trie_and_divide_groups()
   {
      // create groups for base label sets
      IdxType new_group_id = 1;
      for (auto vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         auto group_id = _trie_index.insert(label_set, new_group_id);

         // deal with new label setinver
         if (group_id + 1 > _group_id_to_vec_ids.size())
         {
            _group_id_to_vec_ids.resize(group_id + 1);
            _group_id_to_label_set.resize(group_id + 1);
            _group_id_to_label_set[group_id] = label_set;
         }
         _group_id_to_vec_ids[group_id].emplace_back(vec_id);
      }

      // logs
      _num_groups = new_group_id - 1;
      std::cout << "- Number of groups: " << _num_groups << std::endl;
   }

   void UniNavGraph::get_min_super_sets(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                        bool avoid_self, bool need_containment)
   {
      min_super_set_ids.clear();

      // obtain the candidates
      std::vector<std::shared_ptr<TrieNode>> candidates;
      _trie_index.get_super_set_entrances(query_label_set, candidates, avoid_self, need_containment); // Search candidate supersets.

      // ================= [DEBUG] =================
      // Warn when the candidate set is so large that it can easily stall the query.
      if (candidates.size() > 1000000) 
      {
         #pragma omp critical
         {
            std::cout << "[WARNING] Performance Hazard! Group Label Size: " << query_label_set.size()
                        << " | Candidates Found: " << candidates.size() 
                        << " (This will be slow!)" << std::endl;
            if (!query_label_set.empty()) {
                  std::cout << "   -> Labels: " << query_label_set[0] << " ... " << std::endl;
            }
         }
      }
      // ===================================================

      // special cases
      if (candidates.empty())
         return;
      if (candidates.size() == 1)
      {
         min_super_set_ids.emplace_back(candidates[0]->group_id);
         return;
      }

      // obtain the minimum size
      // Sort all candidates by label-set size in ascending order.
      std::sort(candidates.begin(), candidates.end(),
                [](const std::shared_ptr<TrieNode> &a, const std::shared_ptr<TrieNode> &b)
                {
                   return a->label_set_size < b->label_set_size;
                });
      auto min_size = _group_id_to_label_set[candidates[0]->group_id].size();

      // get the minimum super sets
      for (auto candidate : candidates)
      {
         const auto &cur_group_id = candidate->group_id;
         const auto &cur_label_set = _group_id_to_label_set[cur_group_id];
         bool is_min = true;

         // check whether contains existing minimum super sets (label ids are in ascending order)
         if (cur_label_set.size() > min_size)
         {
            for (auto min_group_id : min_super_set_ids)
            {
               const auto &min_label_set = _group_id_to_label_set[min_group_id];
               if (std::includes(cur_label_set.begin(), cur_label_set.end(), min_label_set.begin(), min_label_set.end()))
               {
                  is_min = false;
                  break;
               }

            }
         }

         // add to the minimum super sets
         if (is_min)
            min_super_set_ids.emplace_back(cur_group_id);
      }
   }

   void UniNavGraph::get_min_super_sets_debug(const std::vector<LabelType> &query_label_set,
                                              std::vector<IdxType> &min_super_set_ids,
                                              bool avoid_self, bool need_containment,
                                              std::atomic<int> &print_counter, bool is_new_trie_method, bool is_rec_more_start, QueryStats &stats,
                                              bool skip_group_id_check)
   {
// #if ENABLE_ENTRY_DEBUG_OUTPUT
      // --- Timer variables ---
      double time_trie_lookup = 0.0;
      double time_sorting = 0.0;
      double time_filtering_loop = 0.0;
      auto function_start_time = std::chrono::high_resolution_clock::now();
// #endif

      min_super_set_ids.clear();

      // --- 1. Measure trie candidate lookup time ---
// #if ENABLE_ENTRY_DEBUG_OUTPUT
      auto start_trie = std::chrono::high_resolution_clock::now();
// #endif

      // // 对应伪代码 L1-L8: 增加快速路径，首先尝试精确匹配
      // auto exact_match_node = _trie_index.find_exact_match(query_label_set);
      // if (exact_match_node != nullptr && exact_match_node->group_id > 0)
      // {
      //    min_super_set_ids.emplace_back(exact_match_node->group_id);
      //    return;
      // }

      std::vector<std::shared_ptr<TrieNode>> candidates;

      if (is_new_trie_method)
      {
         // --- Run the new recursive method ---
         TrieSearchMetricsRecursive trie_metrics_m2 = {};
         if (is_rec_more_start)
            _trie_index.get_super_set_entrances_new_more_sp_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m2);
         else
            _trie_index.get_super_set_entrances_new_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m2);
         stats.recursive_calls = trie_metrics_m2.recursive_calls;
         stats.pruning_events = trie_metrics_m2.pruning_events;
         stats.trie_nodes_traversed = trie_metrics_m2.recursive_calls + trie_metrics_m2.nodes_processed_in_bfs;
         if (stats.recursive_calls > 0)
            stats.pruning_efficiency = static_cast<float>(stats.pruning_events) / stats.recursive_calls;
         else
            stats.pruning_efficiency = 0.0f;
      }
      else
      {
         // --- Run the legacy shortcut method ---
         TrieMethod1Metrics trie_metrics_m1 = {};
         _trie_index.get_super_set_entrances_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m1, skip_group_id_check);
         stats.successful_checks = trie_metrics_m1.successful_checks;
         stats.trie_nodes_traversed = trie_metrics_m1.upward_traversals + trie_metrics_m1.bfs_nodes_processed;
         stats.redundant_upward_steps = trie_metrics_m1.redundant_upward_steps;
         if (stats.candidate_set_size > 0)
            stats.shortcut_hit_ratio = static_cast<float>(stats.successful_checks) / stats.candidate_set_size;
         else
            stats.shortcut_hit_ratio = 0.0f;
      }

// #if ENABLE_ENTRY_DEBUG_OUTPUT
      auto end_trie = std::chrono::high_resolution_clock::now();
      time_trie_lookup = std::chrono::duration<double, std::milli>(end_trie - start_trie).count();
      stats.els_trie_time = time_trie_lookup;
      stats.els_trie_nodes = static_cast<uint64_t>(stats.trie_nodes_traversed);
// #endif

      // Handle trivial cases early.
      stats.els_C_trie = candidates.size();
      if (candidates.empty())
         return;
      if (candidates.size() == 1)
      {
         min_super_set_ids.emplace_back(candidates[0]->group_id);
         stats.els_M_min = min_super_set_ids.size();
         return;
      }
      bool skip_filter = this->skip_els_filter;  // If true, skip filtering and return every candidate group id directly.
      //bool skip_filter = false; //TODO
      if (skip_filter)
      {
         // Fast path: skip sorting and filtering, and return candidate group ids as-is.
         for (const auto& candidate : candidates)
         {
            min_super_set_ids.emplace_back(candidate->group_id);
         }
      }
      
      
      // --- 2. Measure sorting time ---
// #if ENABLE_ENTRY_DEBUG_OUTPUT
      auto start_sort = std::chrono::high_resolution_clock::now();
// #endif
      if (!skip_filter)
      {
         std::sort(candidates.begin(), candidates.end(),
                [](const std::shared_ptr<TrieNode> &a, const std::shared_ptr<TrieNode> &b)
                {
                   return a->label_set_size < b->label_set_size;
                });
      }
      
// #if ENABLE_ENTRY_DEBUG_OUTPUT
      auto end_sort = std::chrono::high_resolution_clock::now();
      time_sorting = std::chrono::duration<double, std::milli>(end_sort - start_sort).count();
// #endif

      // --- 3. Measure filtering-loop time ---
// #if ENABLE_ENTRY_DEBUG_OUTPUT
      auto start_filter = std::chrono::high_resolution_clock::now();
// #endif
      if (!skip_filter)
      {
         auto min_size = _group_id_to_label_set[candidates[0]->group_id].size();
         std::vector<IdxType> filtered_ids;
         for (auto candidate : candidates)
         {
            const auto &cur_group_id = candidate->group_id;
            const auto &cur_label_set = _group_id_to_label_set[cur_group_id];
            bool is_min = true;

            if (cur_label_set.size() > min_size)
            {
               for (auto min_group_id : filtered_ids)
               {
                  const auto &min_label_set = _group_id_to_label_set[min_group_id];
                  ++stats.els_containment_calls;
                  auto cur_it = cur_label_set.begin();
                  auto min_it = min_label_set.begin();
                  while (cur_it != cur_label_set.end() && min_it != min_label_set.end()) {
                     ++stats.els_label_steps;
                     if (*min_it < *cur_it) break;
                     if (!(*cur_it < *min_it)) ++min_it;
                     ++cur_it;
                  }
                  if (min_it == min_label_set.end())
                  {
                     is_min = false;
                     break;
                  }
               }
            }

            if (is_min)
            {
               // min_super_set_ids.emplace_back(cur_group_id);
               filtered_ids.emplace_back(cur_group_id);
            }
         }
         // Replace the original result set with the filtered ids.
         min_super_set_ids = std::move(filtered_ids);
      }
      
// #if ENABLE_ENTRY_DEBUG_OUTPUT
      auto end_filter = std::chrono::high_resolution_clock::now();
      time_filtering_loop = std::chrono::duration<double, std::milli>(end_filter - start_filter).count();

      auto function_total_time = std::chrono::duration<double, std::milli>(end_filter - function_start_time).count();

      // Record the timing breakdown in `stats`.
      stats.els_trie_time = time_trie_lookup;
      stats.els_sort_time = time_sorting;
      stats.els_filter_time = time_filtering_loop;
      stats.els_total_time = function_total_time;
      stats.els_M_min = min_super_set_ids.size();
// #endif
   }

   // fxy_add: bitmap-driven exact ELS. F_pass = ∩ _group_attr_roaring_inv[label];
   // minimal elements = F_pass \ ∪ Desc(g). No trie traversal, no O(Y^2) shrink.
   void UniNavGraph::get_min_super_sets_bitmap(const std::vector<LabelType> &query_label_set,
                                               std::vector<IdxType> &min_super_set_ids) const
   {
      min_super_set_ids.clear();

      // Step 1: F_pass = intersection of group-roaring bitmaps over query labels.
      roaring::Roaring f_pass;
      bool first = true;
      for (auto label : query_label_set)
      {
         auto it = _attr_to_id.find(label);
         if (it == _attr_to_id.end())          // label absent -> no qualified group
            return;
         if (first) { f_pass = _group_attr_roaring_inv[it->second]; first = false; }
         else        f_pass &= _group_attr_roaring_inv[it->second];
         if (f_pass.isEmpty())                 // early out
            return;
      }

      // Step 2: covered = union of proper-supersets of every qualified group.
      roaring::Roaring covered;
      for (uint32_t g : f_pass)
         covered |= _lng_descendants_rb[g];    // _lng_descendants_rb[g] = proper supersets of g

      // Step 3: minimal elements = F_pass \ covered.
      f_pass -= covered;
      for (uint32_t g : f_pass)
         min_super_set_ids.push_back(g);
   }

   // UNG++ ELS: obtain all qualified groups by bitmap, then scan them in
   // increasing label-set cardinality.  For each unmarked candidate, traverse
   // the LNG downward and mark its supersets.  A shared visited array ensures
   // each LNG node/edge is expanded at most once per query.
   void UniNavGraph::get_min_super_sets_sorted_lng(
       const std::vector<LabelType> &query_label_set,
       std::vector<IdxType> &min_super_set_ids,
       QueryStats &stats) const
   {
      min_super_set_ids.clear();
      const auto total_start = std::chrono::high_resolution_clock::now();

      auto bitmap_start = std::chrono::high_resolution_clock::now();
      roaring::Roaring f_pass;
      bool first = true;
      for (auto label : query_label_set) {
         auto it = _attr_to_id.find(label);
         if (it == _attr_to_id.end()) { stats.els_A_all = 0; return; }
         if (first) { f_pass = _group_attr_roaring_inv[it->second]; first = false; }
         else f_pass &= _group_attr_roaring_inv[it->second];
         if (f_pass.isEmpty()) { stats.els_A_all = 0; return; }
      }
      stats.els_A_all = f_pass.cardinality();
      stats.els_bitmap_time = std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - bitmap_start).count();

      std::vector<IdxType> candidates;
      candidates.reserve(stats.els_A_all);
      for (uint32_t g : f_pass) candidates.push_back(static_cast<IdxType>(g));

      auto sort_start = std::chrono::high_resolution_clock::now();
      std::sort(candidates.begin(), candidates.end(), [&](IdxType a, IdxType b) {
         const auto sa = _group_id_to_label_set[a].size();
         const auto sb = _group_id_to_label_set[b].size();
         return sa != sb ? sa < sb : a < b;
      });
      stats.elspp_sort_time = std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - sort_start).count();

      std::vector<uint8_t> marked(_num_groups + 1, 0);
      std::vector<uint8_t> visited(_num_groups + 1, 0);
      std::vector<uint8_t> is_candidate(_num_groups + 1, 0);
      for (IdxType g : candidates) if (g <= _num_groups) is_candidate[g] = 1;

      auto scan_start = std::chrono::high_resolution_clock::now();
      for (IdxType g : candidates) {
         ++stats.els_scan_count;
         if (marked[g]) { ++stats.els_skip_count; continue; }
         min_super_set_ids.push_back(g);
         ++stats.els_selected_count;

         auto lng_start = std::chrono::high_resolution_clock::now();
         std::queue<IdxType> q;
         if (g <= _num_groups && !visited[g]) { visited[g] = 1; q.push(g); }
         while (!q.empty()) {
            IdxType cur = q.front(); q.pop();
            ++stats.els_lng_nodes;
            if (cur < is_candidate.size() && is_candidate[cur]) {
               ++stats.els_mark_tests;
               if (!marked[cur]) { marked[cur] = 1; ++stats.els_new_marks; }
               else ++stats.els_duplicate_marks;
            }
            if (!_label_nav_graph || cur >= _label_nav_graph->out_neighbors.size()) continue;
            for (IdxType child : _label_nav_graph->out_neighbors[cur]) {
               ++stats.els_lng_edges;
               if (child <= _num_groups && !visited[child]) {
                  visited[child] = 1;
                  q.push(child);
               }
            }
         }
         stats.elspp_lng_time += std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - lng_start).count();
      }
      stats.elspp_scan_time = std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - scan_start).count();
      stats.els_M_min = min_super_set_ids.size();
      stats.elspp_total_time = std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - total_start).count();
   }

   // Store candidate groups with scores so sorting stays simple.
   struct ScoredCandidate
   {
      double score;
      ANNS::IdxType group_id;

      // Reverse the comparison so `std::sort` yields descending scores.
      bool operator<(const ScoredCandidate &other) const
      {
         return score > other.score; // Larger scores rank earlier.
      }
   };

   std::vector<ANNS::IdxType> UniNavGraph::select_entry_groups(
       const std::vector<ANNS::IdxType> &minimum_entry_sets,
       ANNS::SelectionMode mode,
       size_t top_k,
       double beta,
       ANNS::IdxType true_query_group_id) const
   {
      bool verbose = false;
      assert(_label_nav_graph != nullptr && "Error: _label_nav_graph should not be null.");

      if (top_k == 0 && true_query_group_id == 0)
      {
         return minimum_entry_sets;
      }
      if (minimum_entry_sets.empty() && true_query_group_id == 0)
      {
         return {};
      }

      // --- Step 1: run BFS and collect reachable candidate nodes ---
      std::queue<ANNS::IdxType> q;
      std::vector<int> lng_distance(_num_groups + 1, -1);

      // === Core optimization 1: keep a compact list of BFS-reached nodes with positive distance ===
      std::vector<ANNS::IdxType> reachable_nodes;
      reachable_nodes.reserve(_num_groups / 10); // Pre-allocate to reduce reallocations.

      for (const auto &group_id : minimum_entry_sets)
      {
         if (group_id > 0 && group_id <= _num_groups && lng_distance[group_id] == -1)
         {
            q.push(group_id);
            lng_distance[group_id] = 0;
         }
      }

      while (!q.empty())
      {
         ANNS::IdxType current_group = q.front();
         q.pop();

         if (current_group >= _label_nav_graph->out_neighbors.size())
            continue;

         for (const auto &child_group : _label_nav_graph->out_neighbors[current_group])
         {
            if (child_group > 0 && child_group <= _num_groups && lng_distance[child_group] == -1)
            {
               lng_distance[child_group] = lng_distance[current_group] + 1;
               q.push(child_group);
               // === Core optimization 2: append newly discovered candidates to the compact list ===
               reachable_nodes.push_back(child_group);
            }
         }
      }

      // --- Step 2: score only the reachable nodes ---
      std::vector<ANNS::ScoredCandidate> candidates;
      candidates.reserve(reachable_nodes.size());

      // === Core optimization 3: iterate over the much smaller `reachable_nodes` list ===
      for (const auto group_id : reachable_nodes)
      {
         const double group_size = static_cast<double>(_group_id_to_vec_ids[group_id].size());
         double score = 0.0;
         if (mode == ANNS::SelectionMode::SizeOnly)
         {
            score = group_size;
         }
         else
         { // SizeAndDistance with new formula
            const int dist = lng_distance[group_id];
            // New scoring rule: prioritize depth first, then break ties by group size.
            score = static_cast<double>(dist) * static_cast<double>(_num_points) + group_size;
         }
         candidates.push_back({score, group_id});
      }

      // --- Step 3: partially sort and select ---
      size_t num_to_sort = std::min(candidates.size(), top_k + 5); // Keep a small safety buffer.
      if (num_to_sort > 0)
      {
         std::partial_sort(candidates.begin(),
                           candidates.begin() + num_to_sort,
                           candidates.end());
      }

      std::vector<ANNS::IdxType> final_entry_groups = minimum_entry_sets;
      std::unordered_set<ANNS::IdxType> existing_groups(minimum_entry_sets.begin(), minimum_entry_sets.end());

      bool oracle_injected = false;
      bool oracle_existed = false;

      if (true_query_group_id > 0 && true_query_group_id <= _num_groups)
      {
         if (existing_groups.find(true_query_group_id) == existing_groups.end())
         {
            final_entry_groups.push_back(true_query_group_id);
            existing_groups.insert(true_query_group_id);
            oracle_injected = true;
         }
         else
         {
            oracle_existed = true;
         }
      }

      for (size_t i = 0; i < num_to_sort; ++i)
      {
         const auto &candidate = candidates[i];
         if (final_entry_groups.size() >= minimum_entry_sets.size() + top_k)
         {
            break;
         }
         if (existing_groups.find(candidate.group_id) == existing_groups.end())
         {
            final_entry_groups.push_back(candidate.group_id);
            existing_groups.insert(candidate.group_id);
         }
      }

      // --- Condensed debug logging ---
      if (verbose)
      {
         if (oracle_injected)
         {
            std::cout << "[Info] Oracle group " << true_query_group_id << " was successfully injected." << std::endl;
         }
         else if (oracle_existed && true_query_group_id > 0)
         {
            std::cout << "[Info] Oracle group " << true_query_group_id << " was already in the initial set." << std::endl;
         }

         size_t added_count = final_entry_groups.size() > minimum_entry_sets.size() ? final_entry_groups.size() - minimum_entry_sets.size() : 0;
         if (oracle_injected)
            added_count--;

         std::cout << "[Info] Final selected entry groups count: " << final_entry_groups.size()
                   << " (Initial: " << minimum_entry_sets.size()
                   << ", Heuristically Added: " << added_count << ")" << std::endl;
      }

      return final_entry_groups;
   }

   void UniNavGraph::prepare_group_storages_graphs()
   {
      _new_vec_id_to_group_id.resize(_num_points);

      // reorder the vectors
      _group_id_to_range.resize(_num_groups + 1);
      _new_to_old_vec_ids.resize(_num_points);
      IdxType new_vec_id = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         _group_id_to_range[group_id].first = new_vec_id;
         for (auto old_vec_id : _group_id_to_vec_ids[group_id])
         {
            _new_to_old_vec_ids[new_vec_id] = old_vec_id;
            _new_vec_id_to_group_id[new_vec_id] = group_id;
            ++new_vec_id;
         }
         _group_id_to_range[group_id].second = new_vec_id;
      }

      // reorder the underlying storage
      _base_storage->reorder_data(_new_to_old_vec_ids);

      // init storage and graph for each group
      _group_storages.resize(_num_groups + 1);
      _group_graphs.resize(_num_groups + 1);
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         auto start = _group_id_to_range[group_id].first;
         auto end = _group_id_to_range[group_id].second;
         _group_storages[group_id] = create_storage(_base_storage, start, end);
         _group_graphs[group_id] = std::make_shared<Graph>(_graph, start, end);
      }
   }

   void UniNavGraph::build_graph_for_all_groups()
   {
      std::cout << "Building graph for each group ..." << std::endl;
      omp_set_num_threads(_num_threads);
      auto start_time = std::chrono::high_resolution_clock::now();

      // build vamana index
      if (_index_name == "Vamana")
      {
         _vamana_instances.resize(_num_groups + 1);
         _group_entry_points.resize(_num_groups + 1);

#pragma omp parallel for schedule(dynamic, 1)
         for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         {
            // if (group_id % 100 == 0)
            //    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;

            // if there are less than _max_degree points in the group, just build a complete graph
            const auto &range = _group_id_to_range[group_id];
            if (range.second - range.first <= _max_degree)
            {
               build_complete_graph(_group_graphs[group_id], range.second - range.first);
               _vamana_instances[group_id] = std::make_shared<Vamana>(_group_storages[group_id], _distance_handler,
                                                                      _group_graphs[group_id], 0);

               // build the vamana graph
            }
            else
            {
               _vamana_instances[group_id] = std::make_shared<Vamana>(false);
               _vamana_instances[group_id]->build(_group_storages[group_id], _distance_handler,
                                                  _group_graphs[group_id], _max_degree, _Lbuild, _alpha, 1);
            }

            // set entry point
            _group_entry_points[group_id] = _vamana_instances[group_id]->get_entry_point() + range.first;
         }

         // if none of the above
      }
      else
      {
         std::cerr << "Error: invalid index name " << _index_name << std::endl;
         exit(-1);
      }
      _build_graph_time = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count();
      std::cout << "\r- Finished in " << _build_graph_time << " ms" << std::endl;
   }

   // Build the global Vamana graph.
   void UniNavGraph::build_global_vamana_graph()
   {
      std::cout << "Building global Vamana graph..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _global_vamana = std::make_shared<Vamana>(false);

      _global_vamana->build(_base_storage, _distance_handler, _global_graph, _max_degree, _Lbuild, _alpha, 1);

      auto build_time = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      _global_vamana_entry_point = _global_vamana->get_entry_point();

      std::cout << "- Global Vamana graph built in " << build_time << " ms" << std::endl;
   }
   //===================================== Begin preprocessing: build the vector-attribute bipartite graph =========================================
   // Build the vector-attribute bipartite graph.
   void UniNavGraph::build_vector_and_attr_graph()
   {
      std::cout << "Building vector-attribute bipartite graph..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // Reset the attribute-to-id and reverse-id mappings.
      _attr_to_id.clear();
      _id_to_attr.clear();
      _vector_attr_graph.clear();

      // First pass: collect all unique attributes and assign ids.
      AtrType attr_id = 0;
      for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         for (const auto &label : label_set)
         {
            if (_attr_to_id.find(label) == _attr_to_id.end())
            {
               _attr_to_id[label] = attr_id;
               _id_to_attr[attr_id] = label;
               attr_id++;
            }
         }
      }

      // Initialize the graph storage for vector nodes plus attribute nodes.
      _vector_attr_graph.resize(_num_points + static_cast<size_t>(attr_id));

      // Second pass: populate the graph edges.
      for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         for (const auto &label : label_set)
         {
            AtrType a_id = _attr_to_id[label];

            // Add bidirectional edges.
            // Vector node ids are in [0, _num_points - 1].
            // Attribute node ids are in [_num_points, _num_points + attr_id - 1].
            _vector_attr_graph[vec_id].push_back(_num_points + static_cast<IdxType>(a_id));
            _vector_attr_graph[_num_points + static_cast<IdxType>(a_id)].push_back(vec_id);
         }
      }

      // Update summary statistics.
      _num_attributes = attr_id;
      _build_vector_attr_graph_time = std::chrono::duration<double, std::milli>(
                                          std::chrono::high_resolution_clock::now() - start_time)
                                          .count();
      std::cout << "- Finish in " << _build_vector_attr_graph_time << " ms" << std::endl;
      std::cout << "- Total edges: " << count_graph_edges() << std::endl;

      // Optional: persist the graph for debugging.
      // save_bipartite_graph_info();
   }

   // Count edges in the vector-attribute bipartite graph.
   size_t UniNavGraph::count_graph_edges() const
   {
      size_t total_edges = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         total_edges += neighbors.size();
      }
      return total_edges / 2; // Each undirected edge is stored twice.
   }

   // Save bipartite-graph debug information as text.
   void UniNavGraph::save_bipartite_graph_info() const
   {
      std::ofstream outfile("bipartite_graph_info.txt");
      if (!outfile.is_open())
      {
         std::cerr << "Warning: Could not open file to save bipartite graph info" << std::endl;
         return;
      }

      outfile << "Bipartite Graph Information\n";
      outfile << "==========================\n";
      outfile << "Total vectors: " << _num_points << "\n";
      outfile << "Total attributes: " << _num_attributes << "\n";
      outfile << "Total edges: " << count_graph_edges() << "\n\n";

      // Dump the attribute-id mapping.
      outfile << "Attribute to ID Mapping:\n";
      for (const auto &pair : _attr_to_id)
      {
         outfile << pair.first << " -> " << static_cast<IdxType>(pair.second) << "\n";
      }
      outfile << "\n";

      // Dump a small sample of graph connectivity.
      outfile << "Sample Graph Connections (first 10 vectors and attributes):\n";
      outfile << "Vector connections:\n";
      for (IdxType i = 0; i < std::min(_num_points, static_cast<IdxType>(10)); ++i)
      {
         outfile << "Vector " << i << " connects to attributes: ";
         for (auto a_node : _vector_attr_graph[i])
         {
            AtrType attr_id = static_cast<AtrType>(a_node - _num_points);
            outfile << _id_to_attr.at(attr_id) << " ";
         }
         outfile << "\n";
      }

      outfile << "\nAttribute connections:\n";
      for (AtrType i = 0; i < std::min(_num_attributes, static_cast<AtrType>(5)); ++i)
      {
         outfile << "Attribute " << _id_to_attr.at(i) << " connects to vectors: ";
         for (auto v_node : _vector_attr_graph[_num_points + static_cast<IdxType>(i)])
         {
            outfile << v_node << " ";
         }
         outfile << "\n";
      }

      outfile.close();
      std::cout << "- Bipartite graph info saved to bipartite_graph_info.txt" << std::endl;
   }

   uint32_t UniNavGraph::compute_checksum() const
   {
      // 简单的校验和计算示例
      uint32_t sum = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         for (IdxType node : neighbors)
         {
            sum ^= (node << (sum % 32));
         }
      }
      return sum;
   }

   // Save the bipartite graph to disk.
   void UniNavGraph::save_bipartite_graph(const std::string &filename)
   {
      std::ofstream out(filename, std::ios::binary);
      if (!out)
      {
         throw std::runtime_error("Cannot open file for writing: " + filename);
      }

      // 1. Write the file header and version marker.
      const char header[8] = {'B', 'I', 'P', 'G', 'R', 'P', 'H', '1'};
      out.write(header, 8);

      // 2. Write basic metadata.
      out.write(reinterpret_cast<const char *>(&_num_points), sizeof(IdxType));
      out.write(reinterpret_cast<const char *>(&_num_attributes), sizeof(AtrType));

      // 3. Write the attribute mapping table.
      // 3.1 Write the number of entries first.
      uint64_t map_size = _attr_to_id.size();
      out.write(reinterpret_cast<const char *>(&map_size), sizeof(uint64_t));

      // 3.2 Write each mapping entry. `LabelType` is `uint16_t`, so we can store it directly.
      for (const auto &[label, id] : _attr_to_id)
      {
         out.write(reinterpret_cast<const char *>(&label), sizeof(LabelType));
         out.write(reinterpret_cast<const char *>(&id), sizeof(AtrType));
      }

      // 4. Write adjacency-list data.
      // 4.1 Write the total node count first.
      uint64_t total_nodes = _vector_attr_graph.size();
      out.write(reinterpret_cast<const char *>(&total_nodes), sizeof(uint64_t));

      // 4.2 Write each node's neighbor list.
      for (const auto &neighbors : _vector_attr_graph)
      {
         // Write the neighbor count first.
         uint32_t neighbor_count = neighbors.size();
         out.write(reinterpret_cast<const char *>(&neighbor_count), sizeof(uint32_t));

         // Then write the neighbor ids.
         if (!neighbors.empty())
         {
            out.write(reinterpret_cast<const char *>(neighbors.data()),
                      neighbors.size() * sizeof(IdxType));
         }
      }

      // 5. Write the trailing checksum.
      uint32_t checksum = compute_checksum();
      out.write(reinterpret_cast<const char *>(&checksum), sizeof(uint32_t));

      std::cout << "Successfully saved bipartite graph to " << filename
                << " (" << out.tellp() << " bytes)" << std::endl;
   }

   // Load the bipartite graph from disk.
   void UniNavGraph::load_bipartite_graph(const std::string &filename)
   {
      std::cout << "Loading bipartite graph from " << filename << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      std::ifstream in(filename, std::ios::binary);
      if (!in)
      {
         throw std::runtime_error("Cannot open file for reading: " + filename);
      }

      // 1. Validate the file header.
      char header[8];
      in.read(header, 8);
      if (std::string(header, 8) != "BIPGRPH1")
      {
         throw std::runtime_error("Invalid file format");
      }

      // 2. Read the basic metadata.
      in.read(reinterpret_cast<char *>(&_num_points), sizeof(IdxType));
      in.read(reinterpret_cast<char *>(&_num_attributes), sizeof(AtrType));

      // 3. Read the attribute mapping table.
      _attr_to_id.clear();
      _id_to_attr.clear();

      // 3.1 Read the number of entries.
      uint64_t map_size;
      in.read(reinterpret_cast<char *>(&map_size), sizeof(uint64_t));

      // 3.2 Read each mapping entry.
      for (uint64_t i = 0; i < map_size; ++i)
      {
         LabelType label;
         AtrType id;

         in.read(reinterpret_cast<char *>(&label), sizeof(LabelType));
         in.read(reinterpret_cast<char *>(&id), sizeof(AtrType));

         _attr_to_id[label] = id;
         _id_to_attr[id] = label;
      }

      // 4. Read adjacency-list data.
      _vector_attr_graph.clear();

      // 4.1 Read the total node count.
      uint64_t total_nodes;
      in.read(reinterpret_cast<char *>(&total_nodes), sizeof(uint64_t));
      _vector_attr_graph.resize(total_nodes);

      // 4.2 Read each node's neighbor list.
      for (uint64_t i = 0; i < total_nodes; ++i)
      {
         uint32_t neighbor_count;
         in.read(reinterpret_cast<char *>(&neighbor_count), sizeof(uint32_t));

         _vector_attr_graph[i].resize(neighbor_count);
         if (neighbor_count > 0)
         {
            in.read(reinterpret_cast<char *>(_vector_attr_graph[i].data()),
                    neighbor_count * sizeof(IdxType));
         }
      }

      // 5. Verify the checksum.
      uint32_t stored_checksum;
      in.read(reinterpret_cast<char *>(&stored_checksum), sizeof(uint32_t));

      uint32_t computed_checksum = compute_checksum();
      if (stored_checksum != computed_checksum)
      {
         throw std::runtime_error("Checksum verification failed");
      }

      std::cout << "- Loaded bipartite graph with " << _num_points << " vectors and "
                << _num_attributes << " attributes in "
                << std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - start_time)
                       .count()
                << " ms" << std::endl;
   }

   // Compare two vector-attribute bipartite graphs.
   bool UniNavGraph::compare_graphs(const ANNS::UniNavGraph &g1, const ANNS::UniNavGraph &g2)
   {
      // 1. Validate basic graph properties.
      if (g1._num_points != g2._num_points)
      {
         std::cerr << "Mismatch in _num_points: "
                   << g1._num_points << " vs " << g2._num_points << std::endl;
         return false;
      }

      if (g1._num_attributes != g2._num_attributes)
      {
         std::cerr << "Mismatch in _num_attributes: "
                   << g1._num_attributes << " vs " << g2._num_attributes << std::endl;
         return false;
      }

      // 2. Validate the attribute mappings.
      if (g1._attr_to_id.size() != g2._attr_to_id.size())
      {
         std::cerr << "Mismatch in _attr_to_id size" << std::endl;
         return false;
      }

      for (const auto &[label, id] : g1._attr_to_id)
      {
         auto it = g2._attr_to_id.find(label);
         if (it == g2._attr_to_id.end())
         {
            std::cerr << "Label " << label << " missing in g2" << std::endl;
            return false;
         }
         if (it->second != id)
         {
            std::cerr << "Mismatch ID for label " << label
                      << ": " << id << " vs " << it->second << std::endl;
            return false;
         }
      }

      // 3. Validate the reverse attribute mappings.
      for (const auto &[id, label] : g1._id_to_attr)
      {
         auto it = g2._id_to_attr.find(id);
         if (it == g2._id_to_attr.end())
         {
            std::cerr << "ID " << id << " missing in g2" << std::endl;
            return false;
         }
         if (it->second != label)
         {
            std::cerr << "Mismatch label for ID " << id
                      << ": " << label << " vs " << it->second << std::endl;
            return false;
         }
      }

      // 4. Validate the adjacency lists.
      if (g1._vector_attr_graph.size() != g2._vector_attr_graph.size())
      {
         std::cerr << "Mismatch in graph size" << std::endl;
         return false;
      }

      for (size_t i = 0; i < g1._vector_attr_graph.size(); ++i)
      {
         const auto &neighbors1 = g1._vector_attr_graph[i];
         const auto &neighbors2 = g2._vector_attr_graph[i];

         if (neighbors1.size() != neighbors2.size())
         {
            std::cerr << "Mismatch in neighbors count for node " << i << std::endl;
            return false;
         }

         for (size_t j = 0; j < neighbors1.size(); ++j)
         {
            if (neighbors1[j] != neighbors2[j])
            {
               std::cerr << "Mismatch in neighbor " << j << " for node " << i
                         << ": " << neighbors1[j] << " vs " << neighbors2[j] << std::endl;
               return false;
            }
         }
      }

      std::cout << "Graphs are identical!" << std::endl;

      return true;
   }
   //===================================== End preprocessing: build the vector-attribute bipartite graph =========================================

   //===================================== Begin query-time bitmap computation =========================================

   // Build the query bitmap.
   std::pair<std::bitset<10000001>, double> UniNavGraph::compute_attribute_bitmap(const std::vector<LabelType> &query_attributes) const
   {
      // 1. Start from an all-true bitmap, meaning every point is initially valid.
      std::bitset<10000001> bitmap;
      bitmap.set(); // Set every bit to 1.
      double per_query_bitmap_time = 0.0;

      // 2. Process each query attribute.
      for (LabelType attr_label : query_attributes)
      {
         // 2.1 Look up the attribute id.
         auto it = _attr_to_id.find(attr_label);
         if (it == _attr_to_id.end())
         {
            // Missing attribute means no point can satisfy the full conjunction.
            return {std::bitset<10000001>(), 0.0};
         }

         // 2.2 Convert it to the attribute-node id.
         AtrType attr_id = it->second;
         IdxType attr_node_id = _num_points + static_cast<IdxType>(attr_id);

         // 2.3 Build a temporary bitmap for the current attribute.
         std::bitset<10000001> temp_bitmap;
         auto start_time = std::chrono::high_resolution_clock::now();
         for (IdxType vec_id : _vector_attr_graph[attr_node_id])
         {
            temp_bitmap.set(vec_id); // Mark matching vectors with bit value 1.
         }

         // 2.4 Intersect it with the running bitmap.
         bitmap &= temp_bitmap; // Use bitwise AND for conjunction.

         per_query_bitmap_time += std::chrono::duration<double, std::milli>(
                                      std::chrono::high_resolution_clock::now() - start_time)
                                      .count();
      }

      return {bitmap, per_query_bitmap_time};
   }

   // Build a bitmap from group ids.
   roaring::Roaring UniNavGraph::compute_bitmap_from_groups(const std::vector<IdxType> &group_ids) const
   {
      roaring::Roaring final_bitmap;
      for (IdxType group_id : group_ids)
      {
         // Ignore invalid group ids.
         if (group_id > 0 && group_id <= _num_groups)
         {
            // Union in the coverage set for each group.
            final_bitmap |= _covered_sets_rb[group_id];
         }
      }
      return final_bitmap;
   }

   // Batch-compute bitmaps produced by UNG filtering.
   std::vector<roaring::Roaring> UniNavGraph::batch_compute_ung_bitmaps(
       const ANNS::UniNavGraph &index,
       const std::shared_ptr<ANNS::IStorage> &query_storage,
       uint32_t num_threads,
       bool is_new_trie_method,
       bool is_rec_more_start)
   {
      auto num_queries = query_storage->get_num_points();
      std::vector<roaring::Roaring> all_bitmaps(num_queries);

// omp_set_num_threads(num_threads);
#pragma omp parallel
      {
         // Keep one shared atomic counter instead of recreating it inside the loop.
         static std::atomic<int> trie_debug_print_counter{0};

#pragma omp for
         for (int id = 0; id < num_queries; ++id)
         {
            // 1. Read the current query labels.
            const auto &query_labels = query_storage->get_label_set(id);

            // 2. Compute the minimum superset groups.
            std::vector<ANNS::IdxType> entry_group_ids;

            // Create a temporary stats object for `get_min_super_sets_debug`.
            ANNS::QueryStats dummy_stats;

            const_cast<ANNS::UniNavGraph &>(index).get_min_super_sets_debug(
                query_labels,
                entry_group_ids,
                false, true, // Default behavior: `avoid_self=false`, `need_containment=true`.
                trie_debug_print_counter,
                is_new_trie_method,
                is_rec_more_start,
                dummy_stats,false);

            // 3. Convert the group ids into a bitmap efficiently.
            all_bitmaps[id] = index.compute_bitmap_from_groups(entry_group_ids);
         }
      }
      return all_bitmaps;
   }

   // fxy_add
   const std::bitset<16000000>& UniNavGraph::get_exact_cand_size_and_mask(
      const std::vector<LabelType>& query_labels,
      size_t& cand_size,
      bool use_optimized) const
   {
      // Reuse heap-backed buffers to avoid blowing up the thread-local stack.
      static thread_local std::unique_ptr<std::bitset<16000000>> final_bitmap_ptr;
      static thread_local std::unique_ptr<std::bitset<16000000>> temp_bitmap_ptr;
      if (!final_bitmap_ptr) final_bitmap_ptr = std::make_unique<std::bitset<16000000>>();
      if (!temp_bitmap_ptr) temp_bitmap_ptr = std::make_unique<std::bitset<16000000>>();
      auto& final_bitmap = *final_bitmap_ptr;
      auto& temp_bitmap = *temp_bitmap_ptr;
      
      // Special case: an empty query degenerates to the full set.
      if (query_labels.empty()) {
         final_bitmap.set(); 
         cand_size = _num_points;
         return final_bitmap;
      }

      std::vector<const std::vector<IdxType>*> valid_vec_lists;
      valid_vec_lists.reserve(query_labels.size()); 

      for (LabelType attr_label : query_labels) {
         auto it = _attr_to_id.find(attr_label);
         if (it == _attr_to_id.end()) {
               final_bitmap.reset(); 
               cand_size = 0;
               return final_bitmap;
         }
         AtrType attr_id = it->second;
         IdxType attr_node_id = _num_points + static_cast<IdxType>(attr_id);
         valid_vec_lists.push_back(&_vector_attr_graph[attr_node_id]);
      }

      if (use_optimized) {
         // ==========================================
         // Use the optimized path for SODA-family modes or when forced on.
         // ==========================================
      std::sort(valid_vec_lists.begin(), valid_vec_lists.end(), 
         [](const std::vector<IdxType>* a, const std::vector<IdxType>* b) {
               return a->size() < b->size();
         });

      final_bitmap.reset(); 
      const auto& first_list = *valid_vec_lists[0];
         for (IdxType vec_id : first_list) final_bitmap.set(vec_id);

      for (size_t i = 1; i < valid_vec_lists.size(); ++i) {
         if (final_bitmap.none()) {
               cand_size = 0;
               return final_bitmap;
         }
         const auto& vec_list = *valid_vec_lists[i];
         for (IdxType vec_id : vec_list) temp_bitmap.set(vec_id);
         final_bitmap &= temp_bitmap;
         for (IdxType vec_id : vec_list) temp_bitmap.reset(vec_id);
      }
      } else {
         // ==========================================
         // Naive baseline path for measuring the raw algorithm behavior.
         // ==========================================
         final_bitmap.set(); // Expensive: mark the entire universe as valid.
         for (size_t i = 0; i < valid_vec_lists.size(); ++i) {
               const auto& vec_list = *valid_vec_lists[i];
               temp_bitmap.reset(); // Expensive: clear the entire bitmap.
               for (IdxType vec_id : vec_list) temp_bitmap.set(vec_id);
               final_bitmap &= temp_bitmap;
         }
      }
      
      cand_size = final_bitmap.count();
      return final_bitmap;
   }

   // fxy_add
   void UniNavGraph::search_baseline_exact(
    const char* query,
    const std::bitset<16000000>& final_bitmap,
    IdxType K,
    std::pair<IdxType, float>* results,
    size_t& num_distance_calcs,
    bool use_optimized_bitset)
   {
      num_distance_calcs = 0;
      for (IdxType i = 0; i < K; ++i) {
         results[i] = {-1, std::numeric_limits<float>::max()};
      }

      std::priority_queue<std::pair<float, IdxType>> top_k_heap; 
      auto dim = _base_storage->get_dim();

      if (use_optimized_bitset) {
         // =======================================================
         // Optimized FastSmartRoute path: word-level skipping plus hardware bit ops.
         // =======================================================
         const uint64_t* bit_words = reinterpret_cast<const uint64_t*>(&final_bitmap);
         size_t num_words = _num_points / 64 + (_num_points % 64 != 0 ? 1 : 0);

         for (size_t w = 0; w < num_words; ++w) {
               uint64_t word = bit_words[w];
               if (word == 0) continue; // Skip 64-vector blocks that are entirely zero.

               while (word != 0) {
                  // Use a hardware primitive to locate the next set bit quickly.
                  int bit_idx = __builtin_ctzll(word);
                  IdxType vec_id = w * 64 + bit_idx;
                  
                  if (vec_id >= _num_points) break;

                  float dist = _distance_handler->compute(query, _base_storage->get_vector(vec_id), dim);
                  num_distance_calcs++;

                  if (top_k_heap.size() < K) {
                     top_k_heap.push({dist, vec_id});
                  } else if (dist < top_k_heap.top().first) {
                     top_k_heap.pop();
                     top_k_heap.push({dist, vec_id});
                  }
                  
                  word &= (word - 1); // Clear the lowest set bit that was just processed.
               }
         }
      } else {
         // =======================================================
         // Original baseline path for standalone pre-filter runs: check one bit at a time.
         // =======================================================
         for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id) {
               if (final_bitmap.test(vec_id)) { // Keep the original form with bounds-safe access.
                  float dist = _distance_handler->compute(query, _base_storage->get_vector(vec_id), dim);
                  num_distance_calcs++;

                  if (top_k_heap.size() < K) {
                     top_k_heap.push({dist, vec_id});
                  } else if (dist < top_k_heap.top().first) {
                     top_k_heap.pop();
                     top_k_heap.push({dist, vec_id});
                  }
               }
         }
      }

      // 将堆中结果倒序写入
      size_t valid_k = top_k_heap.size();
      for (int i = valid_k - 1; i >= 0; --i) {
         results[i].first = top_k_heap.top().second; 
         results[i].second = top_k_heap.top().first;
         top_k_heap.pop();
      }
   }

   // fxy_add
   void UniNavGraph::build_vector_inverted_indices() {
    std::cout << "\n--- Building Vector-level Roaring Inverted Indices for GlobalPpass ---" << std::endl;
    
    _vec_attr_roaring_inv.clear();
    _vec_attr_roaring_inv.resize(_num_attributes);

    for (AtrType attr_id = 0; attr_id < _num_attributes; ++attr_id) {
        IdxType attr_node_id = _num_points + static_cast<IdxType>(attr_id);
        for (IdxType vec_id : _vector_attr_graph[attr_node_id]) {
            _vec_attr_roaring_inv[attr_id].add(vec_id);
        }
    }
    
    std::cout << "- Vector-level Roaring Inverted Indices built directly from bipartite graph." << std::endl;
}

   // fxy_add
   void UniNavGraph::search_baseline_exact_roaring(
    const char* query,
    const roaring::Roaring& valid_bitmap,
    IdxType K,
    std::pair<IdxType, float>* results,
    size_t& num_distance_calcs)
   {
      num_distance_calcs = 0;
      for (IdxType i = 0; i < K; ++i) {
         // results[i] = {-1, std::numeric_limits<float>::max()};
         results[i].first = static_cast<IdxType>(-1);
        results[i].second = std::numeric_limits<float>::max();
      }

      std::priority_queue<std::pair<float, IdxType>> top_k_heap; 
      auto dim = _base_storage->get_dim();

      // CRoaring 直接支持对有效位进行极速迭代
      for (uint32_t vec_id : valid_bitmap) {
          float dist = _distance_handler->compute(query, _base_storage->get_vector(vec_id), dim);
          num_distance_calcs++;

          if (top_k_heap.size() < K) {
             top_k_heap.push({dist, vec_id});
          } else if (dist < top_k_heap.top().first) {
             top_k_heap.pop();
             top_k_heap.push({dist, vec_id});
          }
      }

      size_t valid_k = top_k_heap.size();
      for (int i = valid_k - 1; i >= 0; --i) {
         results[i].first = top_k_heap.top().second; 
         results[i].second = top_k_heap.top().first;
         top_k_heap.pop();
      }
   }

   bool UniNavGraph::run_milvus_knowhere_baseline(
      IdxType query_id,
      const char* query,
      const std::vector<LabelType>& query_labels,
      IdxType Lsearch,
      IdxType K,
      int milvus_baseline_alg,
      SearchQueue& cur_result,
      QueryStats& stats,
      float& num_cmps_out) {
#ifdef ENABLE_KNOWHERE_MILVUS_BASELINE
      auto search_time_start_ms = std::chrono::high_resolution_clock::now();
      const bool use_hnsw = (milvus_baseline_alg == 10);
      if ((!use_hnsw && !_milvus_knowhere_ready) || (use_hnsw && !_milvus_knowhere_hnsw_ready)) {
         std::cerr << "[Milvus-IVF-Knowhere] index is not ready for query " << query_id
                   << " (baseline_alg=" << milvus_baseline_alg << ")" << std::endl;
         return false;
      }

      stats.query_length = query_labels.size();
      stats.acorn_filter_type = 3;

      roaring::Roaring valid_candidates;
      bool allow_all = query_labels.empty();
      bool valid_query = true;
      auto filter_build_start_ms = std::chrono::high_resolution_clock::now();
      if (!allow_all) {
         std::vector<const roaring::Roaring*> valid_rb;
         valid_rb.reserve(query_labels.size());
         for (auto label : query_labels) {
            auto it = _attr_to_id.find(label);
            if (it == _attr_to_id.end() || static_cast<size_t>(it->second) >= _vec_attr_roaring_inv.size()) {
               valid_query = false;
               break;
            }
            valid_rb.push_back(&_vec_attr_roaring_inv[it->second]);
         }
         if (valid_query && !valid_rb.empty()) {
            std::sort(valid_rb.begin(), valid_rb.end(),
                      [](const roaring::Roaring* a, const roaring::Roaring* b) {
                         return a->cardinality() < b->cardinality();
                      });
            valid_candidates = *valid_rb[0];
            for (size_t i = 1; i < valid_rb.size(); ++i) {
               if (valid_candidates.isEmpty()) break;
               valid_candidates &= *valid_rb[i];
            }
         }
      }
      stats.exact_cand_size = allow_all ? _num_points : (valid_query ? valid_candidates.cardinality() : 0);
      stats.candidate_set_size = stats.exact_cand_size;
      stats.global_p_pass = (_num_points > 0) ? static_cast<float>(stats.candidate_set_size) / static_cast<float>(_num_points) : 0.0f;

      // Milvus-style pipeline:
      // 1) evaluate scalar predicate to obtain "valid rows";
      // 2) convert to Knowhere bitset where bit=1 means "filtered out".
      // We currently have no delete/time-travel bitset in this standalone baseline,
      // so final filtered bitset is simply inverse(valid_rows).
      std::vector<uint8_t> valid_bits((_num_points + 7) / 8, 0x00);
      if (allow_all) {
         std::fill(valid_bits.begin(), valid_bits.end(), 0xFF);
      } else if (valid_query) {
         for (uint32_t vec_id : valid_candidates) {
            valid_bits[vec_id >> 3] |= static_cast<uint8_t>(1u << (vec_id & 7));
         }
      }
      std::vector<uint8_t> filter_bits(valid_bits.size(), 0x00);
      for (size_t i = 0; i < valid_bits.size(); ++i) {
         filter_bits[i] = static_cast<uint8_t>(~valid_bits[i]);
      }
      // Clear tail bits outside [_num_points) to avoid accidental filtering noise.
      if ((_num_points & 7U) != 0U) {
         const uint8_t tail_mask = static_cast<uint8_t>((1u << (_num_points & 7U)) - 1u);
         filter_bits.back() &= tail_mask;
      }
      stats.bitmap_time_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - filter_build_start_ms)
                                  .count();

      const int64_t dim = static_cast<int64_t>(_base_storage->get_dim());
      const float* query_ptr = reinterpret_cast<const float*>(query);
      auto query_ds = knowhere::GenDataSet(1, dim, query_ptr);

      // Keep thread policy consistent: outer framework controls query-level parallelism.
      static std::once_flag kh_search_pool_once;
      std::call_once(kh_search_pool_once, []() {
         knowhere::KnowhereConfig::SetSearchThreadPoolSize(1);
      });

      const int max_nprobe = std::max<int>(1, std::min<int>(_milvus_knowhere_nlist, 256));
      const int runtime_nprobe = std::max<int>(
          1, std::min<int>(max_nprobe, static_cast<int>(std::ceil(static_cast<double>(std::max<IdxType>(1, Lsearch)) / 16.0))));
      // For Milvus-HNSW baseline, use Lsearch directly as ef to align with
      // the experiment framework semantics ("candidate queue length").
      const int runtime_ef = std::max<int>(1, static_cast<int>(Lsearch));

      knowhere::Json search_json;
      search_json[knowhere::meta::DIM] = dim;
      search_json[knowhere::meta::TOPK] = static_cast<int64_t>(K);
      search_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
      if (!use_hnsw && _milvus_knowhere_is_ivf) {
         search_json[knowhere::indexparam::NPROBE] = runtime_nprobe;
      }
      if (use_hnsw) {
         search_json[knowhere::indexparam::EF] = runtime_ef;
      }

      knowhere::BitsetView bitset(filter_bits.data(), _num_points);
      auto core_search_start_ms = std::chrono::high_resolution_clock::now();
      auto& active_index = use_hnsw ? _milvus_knowhere_hnsw_index : _milvus_knowhere_index;
      auto kh_results = active_index.Search(query_ds, search_json, bitset);
      if (!kh_results.has_value()) {
         std::cerr << "[Milvus-IVF-Knowhere] search failed for query " << query_id << std::endl;
         return false;
      }

      const auto* ids = kh_results.value()->GetIds();
      const auto* dists = kh_results.value()->GetDistance();
      cur_result.clear();
      if (ids != nullptr && dists != nullptr) {
         for (IdxType i = 0; i < K; ++i) {
            if (ids[i] >= 0) {
               cur_result.insert(static_cast<IdxType>(ids[i]), dists[i]);
            }
         }
      }

      stats.core_search_time_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - core_search_start_ms)
                                       .count();
      stats.search_time_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - search_time_start_ms)
                                  .count();
      stats.num_distance_calcs = 0;
      stats.num_nodes_visited = 0;
      stats.acorn_efs_used = use_hnsw ? runtime_ef : (_milvus_knowhere_is_ivf ? runtime_nprobe : 0);
      num_cmps_out = 0.0f;
      return true;
#else
      (void)query_id;
      (void)query;
      (void)query_labels;
      (void)Lsearch;
      (void)K;
      (void)milvus_baseline_alg;
      (void)cur_result;
      (void)stats;
      (void)num_cmps_out;
      std::cerr << "[Milvus-IVF-Knowhere] baseline requested but binary was built without ENABLE_KNOWHERE_MILVUS_BASELINE." << std::endl;
      return false;
#endif
   }

   bool UniNavGraph::run_favor_baseline(
      IdxType query_id,
      const char* query,
      const std::vector<LabelType>& query_labels,
      IdxType Lsearch,
      IdxType K,
      SearchQueue& cur_result,
      QueryStats& stats,
      float& num_cmps_out) {
      (void)query_id;
      if (!_favor_ready || _favor_index == nullptr || _base_storage == nullptr) {
         std::cerr << "[FAVOR] index is not ready." << std::endl;
         return false;
      }

      _favor_index->setEf(std::max<size_t>(static_cast<size_t>(K), static_cast<size_t>(std::max<IdxType>(1, Lsearch))));

      auto search_time_start_ms = std::chrono::high_resolution_clock::now();
      stats.query_length = query_labels.size();
      stats.acorn_filter_type = 4;

      FavorContainmentFilter filter(_base_storage.get(), query_labels);

      roaring::Roaring valid_candidates;
      bool allow_all = query_labels.empty();
      bool valid_query = true;
      auto filter_build_start_ms = std::chrono::high_resolution_clock::now();
      if (!allow_all) {
         std::vector<const roaring::Roaring*> valid_rb;
         valid_rb.reserve(query_labels.size());
         for (auto label : query_labels) {
            auto it = _attr_to_id.find(label);
            if (it == _attr_to_id.end() || static_cast<size_t>(it->second) >= _vec_attr_roaring_inv.size()) {
               valid_query = false;
               break;
            }
            valid_rb.push_back(&_vec_attr_roaring_inv[it->second]);
         }
         if (valid_query && !valid_rb.empty()) {
            std::sort(valid_rb.begin(), valid_rb.end(),
                      [](const roaring::Roaring* a, const roaring::Roaring* b) {
                         return a->cardinality() < b->cardinality();
                      });
            valid_candidates = *valid_rb[0];
            for (size_t i = 1; i < valid_rb.size(); ++i) {
               if (valid_candidates.isEmpty()) break;
               valid_candidates &= *valid_rb[i];
            }
         }
      }
      stats.exact_cand_size = allow_all ? _num_points : (valid_query ? valid_candidates.cardinality() : 0);
      stats.candidate_set_size = stats.exact_cand_size;
      stats.global_p_pass = (_num_points > 0) ? static_cast<float>(stats.candidate_set_size) / static_cast<float>(_num_points) : 0.0f;
      stats.bitmap_time_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - filter_build_start_ms)
                                  .count();

      const float* query_ptr = reinterpret_cast<const float*>(query);
      auto core_search_start_ms = std::chrono::high_resolution_clock::now();
      auto favor_results = _favor_index->searchKnnGeneric(query_ptr, static_cast<size_t>(K), filter);
      stats.core_search_time_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - core_search_start_ms)
                                       .count();

      cur_result.clear();
      const hnswlib::labeltype invalid_favor_label =
          std::numeric_limits<hnswlib::labeltype>::max();
      while (!favor_results.empty()) {
         auto item = favor_results.top();
         favor_results.pop();
         // FAVOR pads fewer-than-K results with labeltype(-1). Since labeltype is
         // size_t, filter it explicitly before converting to our IdxType space.
         if (item.second != invalid_favor_label &&
             item.second < static_cast<hnswlib::labeltype>(_new_to_old_vec_ids.size())) {
            cur_result.insert(static_cast<IdxType>(item.second), item.first);
         }
      }

      stats.search_time_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - search_time_start_ms)
                                  .count();
      stats.num_distance_calcs = 0;
      stats.num_nodes_visited = 0;
      stats.acorn_efs_used = std::max<IdxType>(1, Lsearch);
      num_cmps_out = 0.0f;
      return true;
   }

   bool UniNavGraph::run_favor_hnsw_baseline(
      IdxType query_id,
      const char* query,
      const std::vector<LabelType>& query_labels,
      IdxType Lsearch,
      int favor_efs,
      IdxType K,
      SearchQueue& cur_result,
      QueryStats& stats,
      float& num_cmps_out) {
      (void)query_id;
      if (!_favor_ready || _favor_index == nullptr || _base_storage == nullptr) {
         std::cerr << "[FAVOR-HNSW] index is not ready." << std::endl;
         return false;
      }

      const int effective_ef = std::max<int>(favor_efs, static_cast<int>(K));
      _favor_index->setEf(static_cast<size_t>(effective_ef));

      auto search_time_start_ms = std::chrono::high_resolution_clock::now();
      stats.query_length = query_labels.size();
      stats.acorn_filter_type = 4;

      FavorContainmentFilter filter(_base_storage.get(), query_labels);

      roaring::Roaring valid_candidates;
      bool allow_all = query_labels.empty();
      bool valid_query = true;
      auto filter_build_start_ms = std::chrono::high_resolution_clock::now();
      if (!allow_all) {
         std::vector<const roaring::Roaring*> valid_rb;
         valid_rb.reserve(query_labels.size());
         for (auto label : query_labels) {
            auto it = _attr_to_id.find(label);
            if (it == _attr_to_id.end() || static_cast<size_t>(it->second) >= _vec_attr_roaring_inv.size()) {
               valid_query = false;
               break;
            }
            valid_rb.push_back(&_vec_attr_roaring_inv[it->second]);
         }
         if (valid_query && !valid_rb.empty()) {
            std::sort(valid_rb.begin(), valid_rb.end(),
                      [](const roaring::Roaring* a, const roaring::Roaring* b) {
                         return a->cardinality() < b->cardinality();
                      });
            valid_candidates = *valid_rb[0];
            for (size_t i = 1; i < valid_rb.size(); ++i) {
               if (valid_candidates.isEmpty()) break;
               valid_candidates &= *valid_rb[i];
            }
         }
      }
      stats.exact_cand_size = allow_all ? _num_points : (valid_query ? valid_candidates.cardinality() : 0);
      stats.candidate_set_size = stats.exact_cand_size;
      stats.global_p_pass = (_num_points > 0) ? static_cast<float>(stats.candidate_set_size) / static_cast<float>(_num_points) : 0.0f;
      stats.bitmap_time_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - filter_build_start_ms)
                                  .count();

      const float* query_ptr = reinterpret_cast<const float*>(query);
      auto core_search_start_ms = std::chrono::high_resolution_clock::now();
      auto favor_results = _favor_index->searchKnnGenericGraphOnly(
          query_ptr,
          static_cast<size_t>(K),
          stats.global_p_pass,
          filter);
      stats.core_search_time_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - core_search_start_ms)
                                       .count();

      cur_result.clear();
      const hnswlib::labeltype invalid_favor_label =
          std::numeric_limits<hnswlib::labeltype>::max();
      while (!favor_results.empty()) {
         auto item = favor_results.top();
         favor_results.pop();
         if (item.second != invalid_favor_label &&
             item.second < static_cast<hnswlib::labeltype>(_new_to_old_vec_ids.size())) {
            cur_result.insert(static_cast<IdxType>(item.second), item.first);
         }
      }

      stats.search_time_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - search_time_start_ms)
                                  .count();
      stats.num_distance_calcs = 0;
      stats.num_nodes_visited = 0;
      stats.acorn_efs_used = effective_ef;
      num_cmps_out = 0.0f;
      return true;
   }

   void UniNavGraph::search_baseline_rabitq(
      const std::bitset<16000000>& final_bitmap,
      IdxType K,
      std::pair<IdxType, float>* results,
      size_t& num_distance_calcs,
      ANNS::rabitq::RabitQSideIndex::QueryContext& query_ctx,
      QueryStats& stats,
      bool use_optimized_bitset,
      const ANNS::rabitq::RabitQSideIndex* side_index)
   {
      const ANNS::rabitq::RabitQSideIndex* active_side_index =
          (side_index != nullptr) ? side_index : &_rabitq_side_index;

      num_distance_calcs = 0;
      for (IdxType i = 0; i < K; ++i) {
         results[i].first = static_cast<IdxType>(-1);
         results[i].second = std::numeric_limits<float>::max();
      }

      std::priority_queue<std::pair<float, IdxType>> top_k_heap;

      auto process_candidate = [&](IdxType vec_id) {
         float low_dist = 0.0F;
         auto bin_start = std::chrono::high_resolution_clock::now();
         float dist = active_side_index->estimate_bin(vec_id, query_ctx, &low_dist);
         stats.rabitq_bin_time_ms += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - bin_start
         ).count();
         stats.rabitq_bin_calls += 1;
         num_distance_calcs += 1;

         bool should_refine = false;
         if (top_k_heap.size() >= K) {
            should_refine = low_dist < top_k_heap.top().first;
         }
         if (should_refine) {
            float refined_low = 0.0F;
            auto full_start = std::chrono::high_resolution_clock::now();
            dist = active_side_index->estimate_full(vec_id, query_ctx, &refined_low);
            stats.rabitq_full_time_ms += std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - full_start
            ).count();
            stats.rabitq_full_calls += 1;
            num_distance_calcs += 1;
            (void)refined_low;
         }

         if (top_k_heap.size() < K) {
            top_k_heap.push({dist, vec_id});
         } else if (dist < top_k_heap.top().first) {
            top_k_heap.pop();
            top_k_heap.push({dist, vec_id});
         }
      };

      if (use_optimized_bitset) {
         const uint64_t* bit_words = reinterpret_cast<const uint64_t*>(&final_bitmap);
         size_t num_words = _num_points / 64 + (_num_points % 64 != 0 ? 1 : 0);
         for (size_t w = 0; w < num_words; ++w) {
            uint64_t word = bit_words[w];
            if (word == 0) continue;
            while (word != 0) {
               int bit_idx = __builtin_ctzll(word);
               IdxType vec_id = static_cast<IdxType>(w * 64 + bit_idx);
               if (vec_id >= _num_points) break;
               process_candidate(vec_id);
               word &= (word - 1);
            }
         }
      } else {
         for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id) {
            if (final_bitmap.test(vec_id)) {
               process_candidate(vec_id);
            }
         }
      }

      size_t valid_k = top_k_heap.size();
      for (int i = static_cast<int>(valid_k) - 1; i >= 0; --i) {
         results[i].first = top_k_heap.top().second;
         results[i].second = top_k_heap.top().first;
         top_k_heap.pop();
      }
   }

   void UniNavGraph::search_baseline_rabitq_roaring(
      const roaring::Roaring& valid_bitmap,
      IdxType K,
      std::pair<IdxType, float>* results,
      size_t& num_distance_calcs,
      ANNS::rabitq::RabitQSideIndex::QueryContext& query_ctx,
      QueryStats& stats,
      const ANNS::rabitq::RabitQSideIndex* side_index)
   {
      const ANNS::rabitq::RabitQSideIndex* active_side_index =
          (side_index != nullptr) ? side_index : &_rabitq_side_index;

      num_distance_calcs = 0;
      for (IdxType i = 0; i < K; ++i) {
         results[i].first = static_cast<IdxType>(-1);
         results[i].second = std::numeric_limits<float>::max();
      }

      std::priority_queue<std::pair<float, IdxType>> top_k_heap;

      for (uint32_t vec_id : valid_bitmap) {
         float low_dist = 0.0F;
         auto bin_start = std::chrono::high_resolution_clock::now();
         float dist = active_side_index->estimate_bin(static_cast<IdxType>(vec_id), query_ctx, &low_dist);
         stats.rabitq_bin_time_ms += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - bin_start
         ).count();
         stats.rabitq_bin_calls += 1;
         num_distance_calcs += 1;

         bool should_refine = false;
         if (top_k_heap.size() >= K) {
            should_refine = low_dist < top_k_heap.top().first;
         }
         if (should_refine) {
            float refined_low = 0.0F;
            auto full_start = std::chrono::high_resolution_clock::now();
            dist = active_side_index->estimate_full(static_cast<IdxType>(vec_id), query_ctx, &refined_low);
            stats.rabitq_full_time_ms += std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - full_start
            ).count();
            stats.rabitq_full_calls += 1;
            num_distance_calcs += 1;
            (void)refined_low;
         }

         if (top_k_heap.size() < K) {
            top_k_heap.push({dist, static_cast<IdxType>(vec_id)});
         } else if (dist < top_k_heap.top().first) {
            top_k_heap.pop();
            top_k_heap.push({dist, static_cast<IdxType>(vec_id)});
         }
      }

      size_t valid_k = top_k_heap.size();
      for (int i = static_cast<int>(valid_k) - 1; i >= 0; --i) {
         results[i].first = top_k_heap.top().second;
         results[i].second = top_k_heap.top().first;
         top_k_heap.pop();
      }
   }

   //====================================end 查询过程：计算bitmap=========================================
   void UniNavGraph::build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points)
   {
      for (auto i = 0; i < num_points; ++i)
         for (auto j = 0; j < num_points; ++j)
            if (i != j)
               graph->neighbors[i].emplace_back(j);
   }

   //=====================================begin LNG中每个f覆盖率计算=========================================
   // fxy_add: 递归打印孩子节点及其覆盖率
   void print_children_recursive(const std::shared_ptr<ANNS::LabelNavGraph> graph, IdxType group_id, std::ofstream &outfile, int indent_level)
   {
      // 根据缩进级别生成前缀空格
      std::string indent(indent_level * 4, ' '); // 每一层缩进4个空格

      // 打印当前节点
      outfile << "\n"
              << indent << "[" << group_id << "] (" << graph->coverage_ratio[group_id] << ")";

      // 如果有子节点，递归打印子节点
      if (!graph->out_neighbors[group_id].empty())
      {
         for (size_t i = 0; i < graph->out_neighbors[group_id].size(); ++i)
         {
            auto child_group_id = graph->out_neighbors[group_id][i];
            print_children_recursive(graph, child_group_id, outfile, indent_level + 1); // 增加缩进级别
         }
      }
   }

   // fxy_add: 输出覆盖率及孩子节点,调用print_children_recursive
   void output_coverage_ratio(const std::shared_ptr<ANNS::LabelNavGraph> _label_nav_graph, IdxType _num_groups, std::ofstream &outfile)
   {
      outfile << "Coverage Ratio for each group\n";
      outfile << "=====================================\n";
      outfile << "Format: [GroupID] -> CoverageRatio \n";
      outfile << "-> [child_GroupID (CoverageRatio)]\n\n";

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // 打印当前组的覆盖率
         outfile << "[" << group_id << "] -> " << _label_nav_graph->coverage_ratio[group_id];

         // 打印孩子节点
         if (!_label_nav_graph->out_neighbors[group_id].empty())
         {
            outfile << " ->";
            for (size_t i = 0; i < _label_nav_graph->out_neighbors[group_id].size(); ++i)
            {
               auto child_group_id = _label_nav_graph->out_neighbors[group_id][i];
               print_children_recursive(_label_nav_graph, child_group_id, outfile, 1); // 第一层子节点缩进为1
            }
         }
         outfile << "\n";
      }
   }

   // fxy_add：计算每个label set的向量覆盖比率
   void UniNavGraph::cal_f_coverage_ratio()
   {
      std::cout << "Calculating coverage ratio..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // Step 0: 初始化covered_sets
      _label_nav_graph->coverage_ratio.clear();
      _label_nav_graph->covered_sets.clear();
      _label_nav_graph->covered_sets.resize(_num_groups + 1);

      // Step 1: 初始化每个 group 的覆盖集合
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &vec_ids = _group_id_to_vec_ids[group_id];
         if (vec_ids.empty())
            continue;

         _label_nav_graph->covered_sets[group_id].insert(vec_ids.begin(), vec_ids.end());
      }

      // Step 2: 找出所有叶子节点（出度为 0）
      std::queue<IdxType> q;
      std::vector<int> out_degree(_num_groups + 1, 0); // 复制一份出度用于拓扑传播

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         out_degree[group_id] = _label_nav_graph->out_neighbors[group_id].size();
         if (out_degree[group_id] == 0)
            q.push(group_id);
      }
      std::cout << "- Number of leaf nodes: " << q.size() << std::endl;

      // Step 3: 自底向上合并集合
      while (!q.empty())
      {
         IdxType current = q.front();
         q.pop();

         for (auto parent : _label_nav_graph->in_neighbors[current])
         {
            // 将当前节点的集合合并到父节点中
            _label_nav_graph->covered_sets[parent].insert(
                _label_nav_graph->covered_sets[current].begin(),
                _label_nav_graph->covered_sets[current].end());

            // 减少父节点剩余未处理的子节点数
            out_degree[parent]--;
            if (out_degree[parent] == 0)
            {
               q.push(parent);
            }
         }
      }

      // Step 4: 计算最终覆盖率
      _label_nav_graph->coverage_ratio.resize(_num_groups + 1, 0.0);
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         size_t covered_count = _label_nav_graph->covered_sets[group_id].size();
         _label_nav_graph->coverage_ratio[group_id] = static_cast<double>(covered_count) / _num_points;
      }

      // Step 5: 输出时间
      _cal_coverage_ratio_time = std::chrono::duration<double, std::milli>(
                                     std::chrono::high_resolution_clock::now() - start_time)
                                     .count();
      std::cout << "- Finish in " << _cal_coverage_ratio_time << " ms" << std::endl;
   }
   // =====================================end LNG中每个f覆盖率计算=========================================

   // =====================================begin 计算LNG中后代的个数=========================================
   // fxy_add：使用广度优先搜索（BFS）来确保所有后代都被找到
   void UniNavGraph::get_descendants_info()
   {
      std::cout << "Calculating descendants info (using corrected BFS method)..." << std::endl;
      using PairType = std::pair<IdxType, int>;

      std::vector<PairType> descendants_num(_num_groups + 1);
      std::vector<std::unordered_set<IdxType>> descendants_set(_num_groups + 1);

      auto start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         std::vector<bool> visited(_num_groups + 1, false);
         std::queue<IdxType> q;
         std::unordered_set<IdxType> temp_set;

         visited[group_id] = true;
         q.push(group_id);

         while (!q.empty())
         {
            IdxType u = q.front();
            q.pop();

            for (const auto &v : _label_nav_graph->out_neighbors[u])
            {
               if (!visited[v])
               {
                  visited[v] = true;
                  temp_set.insert(v);
                  q.push(v);
               }
            }
         }
         descendants_num[group_id] = PairType(group_id, temp_set.size());
         descendants_set[group_id] = std::move(temp_set);
      }

      // 单线程写入类成员变量
      _label_nav_graph->_lng_descendants_num = std::move(descendants_num);
      _label_nav_graph->_lng_descendants = std::move(descendants_set);

      // 计算平均后代个数
      double total_descendants = 0;
      for (const auto &pair : _label_nav_graph->_lng_descendants_num)
      {
         total_descendants += pair.second;
      }
      _label_nav_graph->avg_descendants = _num_groups > 0 ? total_descendants / _num_groups : 0.0;
      _cal_descendants_time = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count();
      std::cout << "- Finish in " << _cal_descendants_time << " ms" << std::endl;
      std::cout << "- Number of groups: " << _num_groups << std::endl;
      std::cout << "- Average number of descendants per group: " << _label_nav_graph->avg_descendants << std::endl;
   }

   // =====================================end 计算LNG中后代的个数=========================================

   // =====================================begin 添加新的跨组边=========================================
   void UniNavGraph::finalize_intra_group_graphs()
   {
      std::cout << "Finalizing intra-group graphs by converting neighbor IDs to global..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      add_offset_for_uni_nav_graph();
      auto duration = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
      std::cout << "- Finished in " << duration << " ms" << std::endl;
   }

   void UniNavGraph::add_new_distance_oriented_edges(
       const std::string &dataset,
       uint32_t num_threads,
       ANNS::AcornInUng new_cross_edge) // MODIFIED: 参数类型改为 const std::string&
   {
      // 防御性断言
      assert(_base_storage != nullptr && "Error: _base_storage is not initialized!");
      assert(_label_nav_graph != nullptr && "Error: _label_nav_graph is not initialized! Did you call build_label_nav_graph()?");
      assert(!_vamana_instances.empty() && "Error: _vamana_instances is empty! Did you call build_graph_for_all_groups()?");
      assert(_vamana_instances.size() == _num_groups + 1 && "Error: _vamana_instances has incorrect size!");

      // 根据传入的字符串策略决定执行哪些步骤
      bool add_acorn_edges = (new_cross_edge.new_edge_policy == "acorn_and_guarantee" || new_cross_edge.new_edge_policy == "just_acorn");
      bool add_guarantee_edges = (new_cross_edge.new_edge_policy == "acorn_and_guarantee" || new_cross_edge.new_edge_policy == "just_guarantee");
      auto all_start_time = std::chrono::high_resolution_clock::now();

      // 提前清空新边集合，无论执行哪个分支都需要
      _my_new_edges_set.clear();

      // ==========================================================================================
      // PART 1 - ACORN 边构建逻辑 (Steps 1-3)
      // ==========================================================================================
      if (add_acorn_edges)
      {
         std::cout << "\n========================================================" << std::endl;
         std::cout << "--- Augmenting Graph with New Distance-Oriented Edges (ACORN) ---" << std::endl;
         std::cout << "========================================================" << std::endl;
         auto acorn_part_start_time = std::chrono::high_resolution_clock::now();

         const size_t MAX_ID_SAMPLES = 15;
         const size_t SAMPLING_THRESHOLD = 100000;

         // FXY_ADD: Step 1 (识别、采样) 计时开始
         std::cout << "\n--- [Cross-Edge Sub-step 1: Identifying & Sampling Query Vectors] ---" << std::endl;
         auto step1_start_time = std::chrono::high_resolution_clock::now();

         // --- Step 0: 构建反向ID映射 ---
         std::cout << "[Step 0] Building reverse ID map for safe lookup..." << std::endl;
         std::unordered_map<IdxType, IdxType> old_to_new_map;
         old_to_new_map.reserve(_num_points);
         for (IdxType new_id = 0; new_id < _num_points; ++new_id)
         {
            old_to_new_map[_new_to_old_vec_ids[new_id]] = new_id;
         }
         std::cout << "  - Reverse ID map built successfully." << std::endl;

         // --- Step 1.1: 识别“概念根属性” ---
         std::cout << "[Step 1.1] Discovering conceptual root attributes via global coverage..." << std::endl;
         std::unordered_map<LabelType, size_t> attribute_counts;
         for (IdxType i = 0; i < _num_points; ++i)
         {
            for (const auto &label : _base_storage->get_label_set(i))
            {
               attribute_counts[label]++;
            }
         }
         std::unordered_set<LabelType> conceptual_root_labels;
         for (const auto &pair : attribute_counts)
         {
            if (static_cast<float>(pair.second) / _num_points >= new_cross_edge.root_coverage_threshold)
            {
               conceptual_root_labels.insert(pair.first);
            }
         }
         if (conceptual_root_labels.empty())
         {
            std::cerr << "  - WARNING: No conceptual root attributes found. Aborting edge addition." << std::endl;
            return;
         }
         std::cout << "  - Discovered " << conceptual_root_labels.size() << " conceptual root attributes." << std::endl;

         std::cout << "  	- Conceptual Root Labels are: { ";
         for (const auto &label : conceptual_root_labels)
         {
            std::cout << label << " ";
         }
         std::cout << "}" << std::endl;

         // --- Step 1.2: 根据层级比例识别候选组 ---
         std::cout << "[Step 1.2] Precisely identifying candidate groups using layer depth ratio..." << std::endl;
         size_t max_observed_depth = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            if (_group_id_to_label_set[group_id].size() > max_observed_depth)
            {
               max_observed_depth = _group_id_to_label_set[group_id].size();
            }
         }
         int target_max_depth = std::max(1, static_cast<int>(max_observed_depth * new_cross_edge.layer_depth_retio));
         std::cout << "  - Max observed layer depth in dataset: " << max_observed_depth << std::endl;
         std::cout << "  - Using layer_depth_ratio=" << new_cross_edge.layer_depth_retio << ", target max depth is " << target_max_depth << std::endl;
         std::unordered_set<IdxType> candidate_groups;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            const auto &label_set = _group_id_to_label_set[group_id];
            if (label_set.empty() || label_set.size() > target_max_depth)
               continue;
            for (const auto &label : label_set)
            {
               if (conceptual_root_labels.count(label))
               {
                  candidate_groups.insert(group_id);
                  break;
               }
            }
         }
         std::unordered_set<IdxType> topmost_group_ids;
         for (IdxType group_id : candidate_groups)
         {
            bool is_topmost = true;
            for (IdxType parent_id : _label_nav_graph->in_neighbors[group_id])
            {
               if (candidate_groups.count(parent_id))
               {
                  is_topmost = false;
                  break;
               }
            }
            if (is_topmost)
            {
               topmost_group_ids.insert(group_id);
            }
         }
         std::cout << "  - Identified " << candidate_groups.size() << " total candidate groups within target depth." << std::endl;
         std::cout << "  - Identified " << topmost_group_ids.size() << " true topmost groups among them." << std::endl;

         // --- 步骤 1.3: 将向量分入两个桶 ---
         std::cout << "[Step 1.3] Separating vectors into 'topmost' and 'subsequent' buckets..." << std::endl;
         std::vector<IdxType> topmost_vec_ids, subsequent_vec_ids;
         std::vector<float *> topmost_vec_data, subsequent_vec_data;
         std::vector<std::vector<LabelType>> topmost_vec_labels, subsequent_vec_labels;
         for (IdxType group_id : candidate_groups)
         {
            const auto &range = _group_id_to_range[group_id];
            const auto &label_set = _group_id_to_label_set[group_id];
            if (topmost_group_ids.count(group_id))
            {
               for (IdxType u = range.first; u < range.second; ++u)
               {
                  topmost_vec_ids.push_back(u);
                  topmost_vec_data.push_back(const_cast<float *>(reinterpret_cast<const float *>(_base_storage->get_vector(u))));
                  topmost_vec_labels.push_back(label_set);
               }
            }
            else
            {
               for (IdxType u = range.first; u < range.second; ++u)
               {
                  subsequent_vec_ids.push_back(u);
                  subsequent_vec_data.push_back(const_cast<float *>(reinterpret_cast<const float *>(_base_storage->get_vector(u))));
                  subsequent_vec_labels.push_back(label_set);
               }
            }
         }
         std::cout << "  - Topmost-layer bucket size: " << topmost_vec_ids.size() << " vectors." << std::endl;
         std::cout << "  - Subsequent-layer bucket size: " << subsequent_vec_ids.size() << " vectors." << std::endl;

         // --- 步骤 1.4: 应用分层抽样逻辑 ---
         std::cout << "[Step 1.4] Applying stratified sampling logic..." << std::endl;
         std::vector<IdxType> final_query_vec_ids;
         std::vector<float *> final_query_vec_data;
         std::vector<std::vector<LabelType>> final_query_vec_labels;
         size_t total_candidates_vecs = topmost_vec_ids.size() + subsequent_vec_ids.size();
         const auto original_topmost_vec_ids = topmost_vec_ids;

         if (total_candidates_vecs < SAMPLING_THRESHOLD || new_cross_edge.query_vector_ratio >= 1.0f)
         {
            std::cout << "  - Decision: Using ALL candidates without sampling." << std::endl;
            final_query_vec_ids = std::move(topmost_vec_ids);
            final_query_vec_ids.insert(final_query_vec_ids.end(), subsequent_vec_ids.begin(), subsequent_vec_ids.end());
            final_query_vec_data = std::move(topmost_vec_data);
            final_query_vec_data.insert(final_query_vec_data.end(), subsequent_vec_data.begin(), subsequent_vec_data.end());
            final_query_vec_labels = std::move(topmost_vec_labels);
            final_query_vec_labels.insert(final_query_vec_labels.end(), subsequent_vec_labels.begin(), subsequent_vec_labels.end());
         }
         else
         {
            std::cout << "  - Decision: Performing stratified sampling." << std::endl;
            size_t target_total_count = static_cast<size_t>(total_candidates_vecs * new_cross_edge.query_vector_ratio);
            target_total_count = std::max(target_total_count, topmost_vec_ids.size());

            final_query_vec_ids = std::move(topmost_vec_ids);
            final_query_vec_data = std::move(topmost_vec_data);
            final_query_vec_labels = std::move(topmost_vec_labels);

            size_t num_to_sample = target_total_count > final_query_vec_ids.size() ? target_total_count - final_query_vec_ids.size() : 0;
            if (num_to_sample > 0 && !subsequent_vec_ids.empty())
            {
               num_to_sample = std::min(num_to_sample, subsequent_vec_ids.size());
               std::vector<size_t> indices(subsequent_vec_ids.size());
               std::iota(indices.begin(), indices.end(), 0);
               std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

               for (size_t i = 0; i < num_to_sample; ++i)
               {
                  size_t original_idx = indices[i];
                  final_query_vec_ids.push_back(subsequent_vec_ids[original_idx]);
                  final_query_vec_data.push_back(subsequent_vec_data[original_idx]);
                  final_query_vec_labels.push_back(subsequent_vec_labels[original_idx]);
               }
            }
         }
         std::cout << "  - Final number of query vectors for edge augmentation: " << final_query_vec_ids.size() << std::endl;

         if (final_query_vec_ids.empty())
         {
            std::cout << "  - No vectors to process. Skipping edge addition." << std::endl;
            return;
         }

         // FXY_ADD: Step 1 计时结束
         _cross_edge_step1_time_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::high_resolution_clock::now() - step1_start_time)
                                         .count();
         std::cout << "--- [Finished Sub-step 1 in " << _cross_edge_step1_time_ms << " ms] ---" << std::endl;

         // --- 步骤 1.5: 执行 ACORN ---
         // FXY_ADD: Step 2 (执行ACORN) 计时开始
         std::cout << "\n--- [Cross-Edge Sub-step 2: Executing ACORN Script] ---" << std::endl;
         auto step2_start_time = std::chrono::high_resolution_clock::now();
         std::cout << "[Step 1.5] Preparing input files and executing ACORN script..." << std::endl;
         std::string query_vec_path = new_cross_edge.acorn_in_ung_output_path + "/acorn_query_vectors.fvecs";
         std::string query_attr_path = new_cross_edge.acorn_in_ung_output_path + "/acorn_query_attrs.txt";
         write_fvecs(query_vec_path, final_query_vec_data, _base_storage->get_dim());
         write_labels_txt(query_attr_path, final_query_vec_labels);
         std::string output_neighbors_file = new_cross_edge.acorn_in_ung_output_path + "/R_neighbors_with_vectors.txt";
         std::stringstream cmd_ss;
         cmd_ss << "/home/fengxiaoyao/FilterVector/FilterVectorCode/ACORN/run_acorn_for_ung.sh "
                << dataset << " " << new_cross_edge.R_in_add_new_edge << " " << _base_storage->get_num_points() << " "
                << new_cross_edge.M << " " << new_cross_edge.M_beta << " " << new_cross_edge.gamma << " " << new_cross_edge.efs << " "
                << num_threads << " "
                << query_vec_path << " " << query_attr_path << " " << new_cross_edge.acorn_in_ung_output_path << " 1";
         std::string cmd = cmd_ss.str();
         std::cout << "  - Executing command: " << cmd << std::endl;
         int ret = system(cmd.c_str());
         if (ret != 0)
         {
            std::cerr << "  - FATAL ERROR: ACORN script execution failed! Check logs in " << new_cross_edge.acorn_in_ung_output_path << std::endl;
            return;
         }
         std::cout << "  - ACORN script finished successfully." << std::endl;

         // FXY_ADD: Step 2 计时结束
         _cross_edge_step2_acorn_time_ms = std::chrono::duration<double, std::milli>(
                                               std::chrono::high_resolution_clock::now() - step2_start_time)
                                               .count();
         std::cout << "--- [Finished Sub-step 2 in " << _cross_edge_step2_acorn_time_ms << " ms] ---" << std::endl;

         // --- 步骤 1.6 & 2: 解析、过滤并添加距离驱动边 ---
         // FXY_ADD: Step 3 (添加距离驱动边) 计时开始
         std::cout << "\n--- [Cross-Edge Sub-step 3: Adding Distance-Oriented Edges] ---" << std::endl;
         auto step3_start_time = std::chrono::high_resolution_clock::now();

         std::cout << "[Step 1.6] Parsing ACORN output and generating candidates..." << std::endl;
         std::vector<NewEdgeCandidate> candidates;
         std::ifstream neighbor_file(output_neighbors_file);
         if (!neighbor_file.is_open())
         {
            std::cerr << "  - FATAL ERROR: Cannot open ACORN output file: " << output_neighbors_file << std::endl;
            return;
         }
         std::string line;
         long long current_query_local_idx = -1;
         IdxType u_id_new = 0;
         size_t intra_group_edges_skipped = 0;
         while (std::getline(neighbor_file, line))
         {
            if (line.rfind("Query: ", 0) == 0)
            {
               current_query_local_idx = std::stoll(line.substr(7));
               if (current_query_local_idx >= 0 && current_query_local_idx < final_query_vec_ids.size())
               {
                  u_id_new = final_query_vec_ids[current_query_local_idx];
               }
            }
            else if (line.rfind("NeighborID: ", 0) == 0)
            {
               std::stringstream line_ss(line);
               std::string token;
               IdxType v_id_old;
               line_ss >> token >> v_id_old;
               if (v_id_old < 0)
                  continue;
               auto it = old_to_new_map.find(v_id_old);
               if (it == old_to_new_map.end())
                  continue;
               IdxType v_id_new = it->second;
               IdxType group_u = _new_vec_id_to_group_id[u_id_new];
               IdxType group_v = _new_vec_id_to_group_id[v_id_new];
               if (group_u != group_v)
               {
                  float dist = _distance_handler->compute(
                      _base_storage->get_vector(u_id_new),
                      _base_storage->get_vector(v_id_new),
                      _base_storage->get_dim());
                  candidates.push_back({u_id_new, v_id_new, dist});
               }
               else
               {
                  intra_group_edges_skipped++;
               }
            }
         }
         neighbor_file.close();
         std::cout << "  - Generated " << candidates.size() << " initial cross-group candidates from ACORN results." << std::endl;
         std::cout << "  - Skipped " << intra_group_edges_skipped << " potential edges because they were intra-group." << std::endl;

         // ============================================================================================
         // FXY_FIX: 这是策略一的最终修复版本
         // ============================================================================================
         std::cout << "[Step 2] Adding new edges with smart selection logic and in-degree control (M=" << new_cross_edge.M_in_add_new_edge << ")..." << std::endl;

         // 1. 按“查询点”对所有候选邻居进行分组
         std::unordered_map<IdxType, std::vector<NewEdgeCandidate>> query_to_candidates;
         for (const auto &cand : candidates)
         {
            query_to_candidates[cand.from].push_back(cand);
         }

         _num_distance_oriented_edges = 0;

         // FXY_FIX: 1. 重新引入全局的入度计数器
         std::vector<std::atomic<IdxType>> new_edge_in_degree(_num_points);
         for (size_t i = 0; i < _num_points; ++i)
         {
            new_edge_in_degree[i] = 0;
         }

         // 2. 遍历每个查询点，独立地为其选择最佳出边
         for (auto const &[query_id, cand_list] : query_to_candidates)
         {
            std::vector<NewEdgeCandidate> hierarchical_candidates;
            std::vector<NewEdgeCandidate> non_hierarchical_candidates;

            IdxType from_group = _new_vec_id_to_group_id[query_id];

            // 3. 将候选邻居分为“层级邻居”和“普通邻居”两个桶
            for (const auto &cand : cand_list)
            {
               IdxType to_group = _new_vec_id_to_group_id[cand.to];
               const auto &children = _label_nav_graph->out_neighbors[from_group];
               bool is_hierarchical = false;
               for (const auto &true_child : children)
               {
                  if (to_group == true_child)
                  {
                     is_hierarchical = true;
                     break;
                  }
               }
               if (is_hierarchical)
               {
                  hierarchical_candidates.push_back(cand);
               }
               else
               {
                  non_hierarchical_candidates.push_back(cand);
               }
            }

            // 4. 对两个桶内的候选者分别按距离排序
            std::sort(hierarchical_candidates.begin(), hierarchical_candidates.end(),
                      [](const auto &a, const auto &b)
                      { return a.distance < b.distance; });
            std::sort(non_hierarchical_candidates.begin(), non_hierarchical_candidates.end(),
                      [](const auto &a, const auto &b)
                      { return a.distance < b.distance; });

            // 5. 智能添加边：同时考虑出度、入度限制
            size_t edges_added_for_this_query = 0;
            const size_t max_out_degree = new_cross_edge.M_in_add_new_edge;
            const size_t max_in_degree = new_cross_edge.M_in_add_new_edge; // 通常入度和出度限制相同

            // 5.1. 优先添加层级邻居
            for (const auto &cand : hierarchical_candidates)
            {
               if (edges_added_for_this_query >= max_out_degree)
                  break;

               // FXY_FIX: 2. 在添加边时检查目标节点的入度
               if (new_edge_in_degree[cand.to].load() < max_in_degree)
               {
                  uint64_t edge_key = (uint64_t)cand.from << 32 | cand.to;
                  if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
                  {
                     _graph->neighbors[cand.from].push_back(cand.to);
                     _num_distance_oriented_edges++;
                     edges_added_for_this_query++;
                     // FXY_FIX: 3. 成功添加后，才增加目标节点的入度计数
                     new_edge_in_degree[cand.to]++;
                     _my_new_edges_set.insert(edge_key);
                  }
               }
            }

            // 5.2. 如果出度名额还有剩余，用普通邻居来填充
            for (const auto &cand : non_hierarchical_candidates)
            {
               if (edges_added_for_this_query >= max_out_degree)
                  break;

               // FXY_FIX: 2. 同样检查目标节点的入度
               if (new_edge_in_degree[cand.to].load() < max_in_degree)
               {
                  uint64_t edge_key = (uint64_t)cand.from << 32 | cand.to;
                  if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
                  {
                     _graph->neighbors[cand.from].push_back(cand.to);
                     _num_distance_oriented_edges++;
                     edges_added_for_this_query++;
                     // FXY_FIX: 3. 成功添加后，才增加目标节点的入度计数
                     new_edge_in_degree[cand.to]++;
                     _my_new_edges_set.insert(edge_key);
                  }
               }

            }
         }

         std::cout << "  - Added a total of " << _num_distance_oriented_edges << " new distance-oriented edges to the graph with smart selection and in-degree control." << std::endl;

         // 诊断分析部分
         std::set<std::pair<IdxType, IdxType>> acorn_connected_group_pairs;
         for (const auto &edge_key : _my_new_edges_set)
         {
            IdxType from_id = edge_key >> 32;
            IdxType to_id = edge_key & 0xFFFFFFFF;
            acorn_connected_group_pairs.insert({_new_vec_id_to_group_id[from_id], _new_vec_id_to_group_id[to_id]});
         }

         std::cout << "\n--- [Cross-Edge Sub-step 3.5: ACORN Edge Connectivity Analysis] ---" << std::endl;
         // ... (诊断分析的 cout 代码完全不变) ...
         size_t total_required_hierarchical_links = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            total_required_hierarchical_links += _label_nav_graph->out_neighbors[group_id].size();
         }
         std::cout << "  - Total parent->child links in hierarchy: " << total_required_hierarchical_links << std::endl;
         std::cout << "  - ACORN edges created " << acorn_connected_group_pairs.size() << " unique group-to-group connections." << std::endl;

         size_t satisfied_hierarchical_links = 0;
         for (const auto &group_pair : acorn_connected_group_pairs)
         {
            IdxType parent_group = group_pair.first;
            IdxType child_group = group_pair.second;
            const auto &children = _label_nav_graph->out_neighbors[parent_group];
            bool is_hierarchical = false;
            for (const auto &true_child : children)
            {
               if (child_group == true_child)
               {
                  is_hierarchical = true;
                  break;
               }
            }
            if (is_hierarchical)
            {
               satisfied_hierarchical_links++;
            }
         }
         std::cout << "  - Among them, " << satisfied_hierarchical_links << " connections are valid parent->child links." << std::endl;
         double satisfaction_rate = total_required_hierarchical_links > 0 ? (double)satisfied_hierarchical_links / total_required_hierarchical_links * 100.0 : 0.0;
         std::cout << "  - Satisfaction rate of hierarchical links by ACORN: " << satisfaction_rate << "%" << std::endl;
         std::cout << "--- [Finished ACORN Edge Analysis] ---\n"
                   << std::endl;

         // FXY_ADD: Step 3 计时结束
         _cross_edge_step3_add_dist_edges_time_ms = std::chrono::duration<double, std::milli>(
                                                        std::chrono::high_resolution_clock::now() - step3_start_time)
                                                        .count();
         std::cout << "--- [Finished Sub-step 3 in " << _cross_edge_step3_add_dist_edges_time_ms << " ms] ---" << std::endl;

         // 计算并保存总的跨组边构建时间
         _build_cross_edges_time = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - acorn_part_start_time)
                                       .count();
      }

      // ==========================================================================================
      // PART 2 - 层级保障边修复逻辑 (Step 4)
      // ==========================================================================================
      if (add_guarantee_edges)
      {
         std::cout << "\n--- [Cross-Edge Sub-step 4: Adding Targeted Edges for Unconnected Children] ---" << std::endl;
         auto step4_start_time = std::chrono::high_resolution_clock::now();

         // 1. 初始化，注意SearchCacheList仅用于获取临时的搜索队列
         size_t max_group_size = 0;
         if (_num_groups > 0)
         {
            for (auto group_id = 1; group_id <= _num_groups; ++group_id)
               max_group_size = std::max(max_group_size, static_cast<size_t>(_group_id_to_range[group_id].second - _group_id_to_range[group_id].first));
         }
         SearchCacheList search_cache_list(1, max_group_size, _Lbuild);

         // 2. 准备容器存储补充边
         std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges_hierarchical(_num_groups + 1);

         // 3. 以单线程模式遍历所有组
         std::cout << "  - Checking for and repairing broken parent->child links (in single-thread mode)..." << std::endl;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            const auto &cur_range = _group_id_to_range[group_id];
            if (cur_range.first >= cur_range.second || _label_nav_graph->out_neighbors[group_id].empty())
            {
               continue;
            }

            std::unordered_set<IdxType> connected_groups;
            for (IdxType i = cur_range.first; i < cur_range.second; ++i)
            {
               for (const auto &neighbor : _graph->neighbors[i])
                  connected_groups.insert(_new_vec_id_to_group_id[neighbor]);
            }

            for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
            {
               if (connected_groups.find(out_group_id) == connected_groups.end())
               {
                  const auto &out_range = _group_id_to_range[out_group_id];
                  size_t child_group_size = out_range.second - out_range.first;
                  if (child_group_size == 0)
                     continue;
                  if (child_group_size <= _max_degree)
                  { // 小规模组逻辑（蛮力）
                     IdxType query_vec_id = _group_entry_points[group_id];
                     const char *query = _base_storage->get_vector(query_vec_id);
                     float min_dist = std::numeric_limits<float>::max();
                     IdxType best_neighbor_id = -1;
                     for (IdxType i = out_range.first; i < out_range.second; ++i)
                     {
                        float dist = _distance_handler->compute(query, _base_storage->get_vector(i), _base_storage->get_dim());
                        if (dist < min_dist)
                        {
                           min_dist = dist;
                           best_neighbor_id = i;
                        }
                     }
                     if (best_neighbor_id != -1)
                     {
                        additional_edges_hierarchical[group_id].emplace_back(query_vec_id, best_neighbor_id);
                     }
                  }
                  else
                  { // 大规模组逻辑：使用手动图搜索，并配合正确的“已访问”集合
                     IdxType cnt = 0;
                     for (auto vec_id = cur_range.first; vec_id < cur_range.second && cnt < _num_cross_edges; ++vec_id)
                     {
                        const char *query_vec = _base_storage->get_vector(vec_id);
                        auto search_cache = search_cache_list.get_free_cache();
                        search_cache->search_queue.clear();
                        std::unordered_set<IdxType> visited_nodes_global;
                        IdxType child_group_entry_point = _group_entry_points[out_group_id];
                        float dist = _distance_handler->compute(query_vec, _base_storage->get_vector(child_group_entry_point), _base_storage->get_dim());

                        visited_nodes_global.insert(child_group_entry_point);
                        search_cache->search_queue.insert(child_group_entry_point, dist);
                        while (search_cache->search_queue.has_unexpanded_node())
                        {
                           const Candidate &cur = search_cache->search_queue.get_closest_unexpanded();
                           for (auto neighbor_id : _graph->neighbors[cur.id])
                           {
                              if (neighbor_id >= out_range.first && neighbor_id < out_range.second)
                              {
                                 if (visited_nodes_global.find(neighbor_id) == visited_nodes_global.end())
                                 {
                                    visited_nodes_global.insert(neighbor_id);
                                    float d = _distance_handler->compute(query_vec, _base_storage->get_vector(neighbor_id), _base_storage->get_dim());
                                    search_cache->search_queue.insert(neighbor_id, d);
                                 }
                              }
                           }
                        }
                        for (size_t k = 0; k < search_cache->search_queue.size() && k < _num_cross_edges / 2; ++k)
                        {
                           additional_edges_hierarchical[group_id].emplace_back(vec_id, search_cache->search_queue[k].id);
                           cnt++;
                        }
                        search_cache_list.release_cache(search_cache);
                     }
                  }

                  // // 不再区分大小组，总是执行以下逻辑
                  // IdxType query_vec_id = _group_entry_points[group_id];
                  // const char* query = _base_storage->get_vector(query_vec_id);
                  // float min_dist = std::numeric_limits<float>::max();
                  // IdxType best_neighbor_id = -1;

                  // for (IdxType i = out_range.first; i < out_range.second; ++i) {
                  //    float dist = _distance_handler->compute(query, _base_storage->get_vector(i), _base_storage->get_dim());
                  //    if (dist < min_dist) { min_dist = dist; best_neighbor_id = i; }
                  // }

                  // if (best_neighbor_id != -1) {
                  //    additional_edges_hierarchical[group_id].emplace_back(query_vec_id, best_neighbor_id);
                  // }
               }
            }
         }
         std::cout << std::endl;

         // 4. 单线程合并所有补充边...
         std::cout << "  - Merging newly created fill-in edges..." << std::endl;
         size_t num_added_hierarchical_edges = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            for (const auto &[from_id, to_id] : additional_edges_hierarchical[group_id])
            {
               uint64_t edge_key = (uint64_t)from_id << 32 | to_id;
               if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
               {
                  _graph->neighbors[from_id].push_back(to_id);
                  _my_new_edges_set.insert(edge_key);
                  num_added_hierarchical_edges++;
               }
            }
         }
         std::cout << "  - Merged " << num_added_hierarchical_edges << " new hierarchical edges to repair the graph." << std::endl;

         _cross_edge_step4_add_hierarchy_edges_time_ms = std::chrono::duration<double, std::milli>(
                                                             std::chrono::high_resolution_clock::now() - step4_start_time)
                                                             .count();
         std::cout << "--- [Finished Sub-step 4 in " << _cross_edge_step4_add_hierarchy_edges_time_ms << " ms] ---" << std::endl;
      }

      auto total_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - all_start_time).count();
      std::cout << "\n--- Finished augmenting graph in " << total_time_ms << " ms. ---" << std::endl;
      std::cout << "========================================================" << std::endl;
   }

   // =====================================begin 添加新的跨组边=========================================

//    void UniNavGraph::build_label_nav_graph()
//    {
//       std::cout << "Building label navigation graph... " << std::endl;
//       auto start_time = std::chrono::high_resolution_clock::now();
//       _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
//       omp_set_num_threads(_num_threads);

// // obtain out-neighbors
// #pragma omp parallel for schedule(dynamic, 256)
//       for (auto group_id = 1; group_id <= _num_groups; ++group_id)
//       {
//          // if (group_id % 100 == 0)
//          //    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
//          std::vector<IdxType> min_super_set_ids;
//          get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
//          _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;
//       }

//       // obtain in-neighbors
//       for (auto group_id = 1; group_id <= _num_groups; ++group_id)
//          for (auto each : _label_nav_graph->out_neighbors[group_id])
//             _label_nav_graph->in_neighbors[each].emplace_back(group_id);

//       _build_LNG_time = std::chrono::duration<double, std::milli>(
//                             std::chrono::high_resolution_clock::now() - start_time)
//                             .count();
//       std::cout << "\r- Finished in " << _build_LNG_time << " ms" << std::endl;
//    }

   // fxy_add : 打印信息的build_label_nav_graph
   void UniNavGraph::build_label_nav_graph()
   {
      std::cout << "Building label navigation graph... " << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
      omp_set_num_threads(_num_threads);

      // 监控变量
      std::atomic<size_t> processed_count{0};
      size_t total_groups = _num_groups;
      size_t log_interval = 100000; 

      // ===========================================================
      // Phase 1: Obtain Out-Neighbors (并行)
      // ===========================================================
      std::cout << "--- Phase 1: Calculating Out-Neighbors (Parallel) ---" << std::endl;
#pragma omp parallel for schedule(dynamic, 256)
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         std::vector<IdxType> min_super_set_ids;
         // 核心计算
         get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
         _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;

         // 进度监控
         size_t current = ++processed_count;
         
         // 逻辑：
         // 1. 正常阶段：每 10 万打印一次
         // 2. 冲刺阶段（最后 1% 或 最后 10万）：每 1000 个打印一次，看到它在动
         bool normal_log = (current % log_interval == 0);
         bool final_sprint = (current > total_groups - 100000) && (current % 1000 == 0); 

         if (normal_log || final_sprint || current == total_groups)
         {
            #pragma omp critical 
            {
               double percentage = (double)current / total_groups * 100.0;
               std::cout << "[Phase 1] Processed " << current << " / " << total_groups 
                         << " (" << std::fixed << std::setprecision(4) << percentage << "%)" // 精度提高到4位
                         << std::endl;
            }
         }
      }

      // ===========================================================
      // Phase 2: Obtain In-Neighbors (串行)
      // ===========================================================
      std::cout << "--- Phase 2: Calculating In-Neighbors (Serial) ---" << std::endl;
      
      processed_count = 0; 
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         for (auto each : _label_nav_graph->out_neighbors[group_id])
            _label_nav_graph->in_neighbors[each].emplace_back(group_id);

         size_t current = ++processed_count;
         if (current % log_interval == 0 || current == total_groups)
         {
             double percentage = (double)current / total_groups * 100.0;
             std::cout << "[Phase 2] Processed " << current << " / " << total_groups 
                       << " (" << std::fixed << std::setprecision(1) << percentage << "%)" 
                       << std::endl;
         }
      }

      _build_LNG_time = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      std::cout << "- Finished building LNG in " << _build_LNG_time << " ms" << std::endl;
   }


   // fxy_add:初始化求flag的几个数据结构
   void UniNavGraph::initialize_lng_descendants_coverage_bitsets()
   {
      std::cout << "Initializing LNG descendants and coverage bitsets..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _lng_descendants_bits.resize(_num_groups + 1);
      _covered_sets_bits.resize(_num_groups + 1);

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // 初始化大小
         _lng_descendants_bits[group_id].resize(_num_groups);
         _covered_sets_bits[group_id].resize(_num_points);

         // 填充后代的group的集合
         const auto &descendants = _label_nav_graph->_lng_descendants[group_id];
         for (auto id : descendants)
         {
            _lng_descendants_bits[group_id].set(id);
         }

         // 填充覆盖的向量的集合
         const auto &coverage = _label_nav_graph->covered_sets[group_id];
         for (auto id : coverage)
         {
            _covered_sets_bits[group_id].set(id);
         }
      }
   }

   // fxy_add:初始化求flag的几个数据结构
   void UniNavGraph::initialize_roaring_bitsets()
   {
      std::cout << "enter initialize_roaring_bitsets" << std::endl;

      // 打印 _num_groups 的值，确保它是一个预期的、合理的值
      std::cout << "_num_groups = " << _num_groups << std::endl;

      // 检查指针是否为空
      if (!_label_nav_graph)
      {
         std::cerr << "Error: _label_nav_graph is a null pointer!" << std::endl;
         return; // 或者抛出异常
      }

      _lng_descendants_rb.resize(_num_groups + 1);
      _covered_sets_rb.resize(_num_groups + 1);

      // 关键诊断：打印所有容器的大小
      std::cout << "_lng_descendants_rb.size() = " << _lng_descendants_rb.size() << std::endl;
      std::cout << "_covered_sets_rb.size() = " << _covered_sets_rb.size() << std::endl;
      std::cout << "_label_nav_graph->_lng_descendants.size() = " << _label_nav_graph->_lng_descendants.size() << std::endl;
      std::cout << "_label_nav_graph->covered_sets.size() = " << _label_nav_graph->covered_sets.size() << std::endl;

      // std::cout << "begin for " << std::endl;

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // std::cout << "group_id: " << group_id << std::endl;
         const auto &descendants = _label_nav_graph->_lng_descendants[group_id];
         const auto &coverage = _label_nav_graph->covered_sets[group_id];

         // 初始化后代 bitset
         for (auto id : descendants)
         {
            _lng_descendants_rb[group_id].add(id);
         }

         // 初始化覆盖 bitset
         for (auto id : coverage)
         {
            _covered_sets_rb[group_id].add(id);
         }
      }

      std::cout << "Roaring bitsets initialized." << std::endl;
   }

   // 将分组内的局部索引转换为全局索引
   void UniNavGraph::add_offset_for_uni_nav_graph()
   {
      omp_set_num_threads(_num_threads);
#pragma omp parallel for schedule(dynamic, 4096)
      for (auto i = 0; i < _num_points; ++i)
         for (auto &neighbor : _graph->neighbors[i])
            neighbor += _group_id_to_range[_new_vec_id_to_group_id[i]].first;
   }

   void UniNavGraph::build_cross_group_edges()
   {
      std::cout << "Building cross-group edges ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // allocate memory for storaging cross-group neighbors
      std::vector<SearchQueue> cross_group_neighbors;
      cross_group_neighbors.resize(_num_points);
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         cross_group_neighbors[point_id].reserve(_num_cross_edges);

      // allocate memory for search caches
      size_t max_group_size = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         max_group_size = std::max(max_group_size, _group_id_to_vec_ids[group_id].size());
      SearchCacheList search_cache_list(_num_threads, max_group_size, _Lbuild);
      omp_set_num_threads(_num_threads);

      // for each group
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (group_id % 10000 == 0 || group_id == _num_groups)
         {
            std::cout << "- [进度] 已处理子组 " << group_id << " / " << _num_groups << std::endl;
         }
         if (_label_nav_graph->in_neighbors[group_id].size() > 0)
         {
            // if (group_id % 100 == 0)
            //    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
            IdxType offset = _group_id_to_range[group_id].first;

            // query vamana index
            if (_index_name == "Vamana")
            {
               auto index = _vamana_instances[group_id];
               if (_num_cross_edges > _Lbuild)
               {
                  std::cerr << "Error: num_cross_edges should be less than or equal to Lbuild" << std::endl;
                  exit(-1);
               }

               // for each in-neighbor group
               for (auto in_group_id : _label_nav_graph->in_neighbors[group_id])
               {
                  const auto &range = _group_id_to_range[in_group_id];

// take each vector in the group as the query
#pragma omp parallel for schedule(dynamic, 1)
                  for (auto vec_id = range.first; vec_id < range.second; ++vec_id)
                  {
                     const char *query = _base_storage->get_vector(vec_id);
                     auto search_cache = search_cache_list.get_free_cache();
                     index->iterate_to_fixed_point(query, search_cache);

                     // update the cross-group edges for vec_id
                     for (auto k = 0; k < search_cache->search_queue.size(); ++k)
                        cross_group_neighbors[vec_id].insert(search_cache->search_queue[k].id + offset,
                                                             search_cache->search_queue[k].distance);
                     search_cache_list.release_cache(search_cache);
                  }
               }

               // if none of the above
            }
            else
            {
               std::cerr << "Error: invalid index name " << _index_name << std::endl;
               exit(-1);
            }
         }
      }

      // add additional edges
      std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges(_num_groups + 1);
#pragma omp parallel for schedule(dynamic, 256)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &cur_range = _group_id_to_range[group_id];
         std::unordered_set<IdxType> connected_groups;

         // obtain connected groups
         for (IdxType i = cur_range.first; i < cur_range.second; ++i)
            for (IdxType j = 0; j < cross_group_neighbors[i].size(); ++j)
               connected_groups.insert(_new_vec_id_to_group_id[cross_group_neighbors[i][j].id]);

         // add additional cross-group edges for unconnected groups
         for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
            if (connected_groups.find(out_group_id) == connected_groups.end())
            {
               IdxType cnt = 0;
               for (auto vec_id = cur_range.first; vec_id < cur_range.second && cnt < _num_cross_edges; ++vec_id)
               {
                  auto search_cache = search_cache_list.get_free_cache();
                  _vamana_instances[out_group_id]->iterate_to_fixed_point(_base_storage->get_vector(vec_id), search_cache);

                  for (auto k = 0; k < search_cache->search_queue.size() && k < _num_cross_edges / 2; ++k)
                  {
                     additional_edges[group_id].emplace_back(vec_id,
                                                             search_cache->search_queue[k].id + _group_id_to_range[out_group_id].first);
                     cnt += 1;
                  }
                  search_cache_list.release_cache(search_cache);
               }
            }
      }

      // add offset for uni-nav graph
      add_offset_for_uni_nav_graph();

// merge cross-group edges
#pragma omp parallel for schedule(dynamic, 4096)
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         for (auto k = 0; k < cross_group_neighbors[point_id].size(); ++k)
            _graph->neighbors[point_id].emplace_back(cross_group_neighbors[point_id][k].id);

// merge additional cross-group edges
#pragma omp parallel for schedule(dynamic, 256)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         for (const auto &[from_id, to_id] : additional_edges[group_id])
            _graph->neighbors[from_id].emplace_back(to_id);
      }

      _build_cross_edges_time = std::chrono::duration<double, std::milli>(
                                    std::chrono::high_resolution_clock::now() - start_time)
                                    .count();
      std::cout << "\r- Finish in " << _build_cross_edges_time << " ms" << std::endl;
   }

   /*void UniNavGraph::search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler,
                            uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                            IdxType K, std::pair<IdxType, float> *results, std::vector<float> &num_cmps,
                            std::vector<std::bitset<10000001>> &bitmap)
   {
      auto num_queries = query_storage->get_num_points();
      _query_storage = query_storage;
      _distance_handler = distance_handler;
      _scenario = scenario;

      // preparation
      if (K > Lsearch)
      {
         std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
         exit(-1);
      }
      SearchCacheList search_cache_list(num_threads, _num_points, Lsearch);

      // run queries
      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
      for (auto id = 0; id < num_queries; ++id)
      {
         auto search_cache = search_cache_list.get_free_cache();
         const char *query = _query_storage->get_vector(id);
         SearchQueue cur_result;

         // for overlap or nofilter scenario
         if (scenario == "overlap" || scenario == "nofilter")
         {
            num_cmps[id] = 0;
            search_cache->visited_set.clear();
            cur_result.reserve(K);

            // obtain entry group
            std::vector<IdxType> entry_group_ids;
            if (scenario == "overlap")
               get_min_super_sets(_query_storage->get_label_set(id), entry_group_ids, false, false);
            else
               get_min_super_sets({}, entry_group_ids, true, true);

            // for each entry group
            for (const auto &group_id : entry_group_ids)
            {
               std::vector<IdxType> entry_points;
               get_entry_points_given_group_id(num_entry_points, search_cache->visited_set, group_id, entry_points);

               // graph search and dump to current result
               num_cmps[id] += iterate_to_fixed_point(query, search_cache, id, entry_points, true, false);
               for (auto k = 0; k < search_cache->search_queue.size() && k < K; ++k)
                  cur_result.insert(search_cache->search_queue[k].id, search_cache->search_queue[k].distance);
            }

            // for the other scenarios: containment, equality
         }
         else
         {

            // obtain entry points
            auto entry_points = get_entry_points(_query_storage->get_label_set(id), num_entry_points, search_cache->visited_set);
            if (entry_points.empty())
            {
               num_cmps[id] = 0;
               for (auto k = 0; k < K; ++k)
                  results[id * K + k].first = -1;
               continue;
            }

            // graph search
            num_cmps[id] = iterate_to_fixed_point(query, search_cache, id, entry_points);
            cur_result = search_cache->search_queue;
         }

         // write results
         for (auto k = 0; k < K; ++k)
         {
            if (k < cur_result.size())
            {
               results[id * K + k].first = _new_to_old_vec_ids[cur_result[k].id];
               results[id * K + k].second = cur_result[k].distance;
            }
            else
               results[id * K + k].first = -1;
         }

         // clean
         search_cache_list.release_cache(search_cache);
      }
   }*/

   //fxy_add
   void UniNavGraph::warmup_selectors(uint32_t num_threads)
   {
      // 预热 Idea1 的 Trie 方法选择器 (3个特征)
      if (_trie_method_selector != nullptr)
      {
         std::cout << "- Warming up Trie Method Selector (Idea1)..." << std::endl;
         std::vector<float> dummy_features1(3, 0.0f);
         // 去除多线程，仅由单线程执行 1 次即可完成预热
         _trie_method_selector->predict(dummy_features1);
         std::cout << "- Trie Method Selector is warm." << std::endl;
      }

      // 预热 SODA selector (GlobalPpass, NumDescendants, QuerySize)
      if (_smart_route_selector) {
         std::cout << "- Warming up Smart Route Selector..." << std::endl;
         std::vector<float> dummy(3, 0.0f);
         _smart_route_selector->predict(dummy);
         std::cout << "- Smart Route Selector is warm." << std::endl;
      }
      
      // --- 预热单层 Fast Route Selector (4个特征) ---
      if (_fast_route_single_selector) {
         std::cout << "- Warming up Fast Route Single Selector..." << std::endl;
         std::vector<float> dummy4(4, 0.0f);
         // 去除多线程
         _fast_route_single_selector->predict(dummy4);
         std::cout << "- Fast Route Single Selector is warm." << std::endl;
      }

      // --- 预热 Smartroute-revised 路由 (2个特征) ---
      if (_fast_route_revised_selector) {
         std::cout << "- Warming up Smartroute-revised Selector..." << std::endl;
         std::vector<float> dummy2(2, 0.0f);
         _fast_route_revised_selector->predict(dummy2);
         std::cout << "- Smartroute-revised Selector is warm." << std::endl;
      }
   }

   // fxy_add
#include <future>   
#include <chrono>   
#include <ctime>    
#include <iomanip>  
#include <atomic>   
#include <algorithm> 

// fxy_add
void UniNavGraph::calculate_query_features_only(
    std::shared_ptr<IStorage> query_storage,
    uint32_t num_threads,
    const std::string &output_csv_path,
    bool is_new_trie_method,
    bool is_rec_more_start)
{
    std::cout << "Starting unified feature calculation (Monitor: Label [1] Completion)..." << std::endl;
    auto num_queries = query_storage->get_num_points();
    std::vector<QueryStats> query_stats(num_queries);

    // 设置慢查询监控阈值
    const double SLOW_QUERY_THRESHOLD_MS = 500000.0; 

    // 设置进度打印间隔
    size_t log_interval = std::max((size_t)1, (size_t)(num_queries * 0.05)); 
    std::atomic<size_t> processed_count{0}; 

    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (auto id = 0; id < num_queries; ++id)
    {
        auto &stats = query_stats[id];
        const auto &query_labels = query_storage->get_label_set(id);

        // --- 计算 Idea1 特征 ---
        stats.query_length = query_labels.size(); 
        if (!query_labels.empty())
        {
            stats.candidate_set_size = _trie_index.get_candidate_count_for_label(query_labels.back());
            stats.trie_I_lmax = stats.candidate_set_size;
        }
        const auto &trie_metrics = _trie_static_metrics;
        stats.trie_total_nodes = trie_metrics.total_nodes;
        stats.trie_label_cardinality = trie_metrics.label_cardinality;
        stats.trie_avg_path_length = trie_metrics.avg_path_length;
        stats.trie_avg_branching_factor = trie_metrics.avg_branching_factor;

        // --- 计算 Idea2 特征 (带实时监控) ---
        std::vector<IdxType> entry_group_ids;
        static std::atomic<int> temp_counter{0}; 

        // 1. 启动异步计算
        auto future_result = std::async(std::launch::async, [&]() {
            get_min_super_sets_debug(query_labels, entry_group_ids, false, true, 
                                     temp_counter, is_new_trie_method, is_rec_more_start, stats, false);
        });

        // 2. 监控超时打印
        std::future_status status = future_result.wait_for(std::chrono::milliseconds((long long)SLOW_QUERY_THRESHOLD_MS));
        if (status == std::future_status::timeout)
        {
            #pragma omp critical
            {
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                std::cout << "\n[SLOW QUERY DETECTED - STILL RUNNING] " 
                          << std::put_time(std::localtime(&now_c), "%H:%M:%S") 
                          << " | ID: " << id << std::endl;
                
                std::cout << "  - Labels (" << query_labels.size() << "): [ ";
                for (size_t i = 0; i < query_labels.size(); ++i) {
                    std::cout << query_labels[i] << (i < query_labels.size() - 1 ? ", " : "");
                    if (i >= 10) { std::cout << "... "; break; } 
                }
                std::cout << " ]" << std::endl;
            }
        }

        // 3. 等待任务彻底完成
        future_result.get(); 

        // ============================================================
        // 特定标签 (Label 1) 完成时的监控打印
        // ============================================================
        if (query_labels.size() == 1 && query_labels[0] == 1)
        {
            #pragma omp critical
            {
                 // 获取当前时间
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                
                std::cout << "[Monitor] \033[1;32mTarget Query Finished\033[0m: ID " << id 
                          << " with Label [1] at " << std::put_time(std::localtime(&now_c), "%H:%M:%S") 
                          << " (Entry Groups: " << entry_group_ids.size() << ")"
                          << std::endl;
            }
        }
        // ============================================================

        // --- 后续 Bitmap 计算 ---
        if (!entry_group_ids.empty())
        {
            stats.num_entry_points = entry_group_ids.size();
            stats.num_lng_descendants = [&]
            {
                roaring::Roaring desc;
                for (auto gid : entry_group_ids)
                    if (gid > 0 && gid <= _num_groups)
                        desc |= _lng_descendants_rb[gid];
                return desc.cardinality();
            }(); 

            stats.entry_group_total_coverage = [&]
            {
                roaring::Roaring cov;
                for (auto gid : entry_group_ids)
                    if (gid > 0 && gid <= _num_groups)
                        cov |= _covered_sets_rb[gid];
                return static_cast<float>(cov.cardinality()) / _num_points;
            }(); 
        }
        else
        {
            stats.num_entry_points = 0;
            stats.num_lng_descendants = 0;
            stats.entry_group_total_coverage = 0.0f;
        }

        // 进度打印
        size_t current_processed = processed_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (current_processed % log_interval == 0 || current_processed == num_queries)
        {
            #pragma omp critical
            {
                float progress = (float)current_processed / num_queries * 100.0f;
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                std::cout << "[Progress " << std::put_time(std::localtime(&now_c), "%H:%M:%S") << "] " 
                          << current_processed << " / " << num_queries 
                          << " (" << std::fixed << std::setprecision(1) << progress << "%)" 
                          << std::endl;
            }
        }
    }
    
    std::cout << "In-memory feature calculation finished for " << num_queries << " queries." << std::endl;

    // =================================================================
    // 阶段二: 写入 CSV
    // =================================================================
    std::cout << "Now writing all features to CSV file: " << output_csv_path << std::endl;
    std::ofstream outfile(output_csv_path);
    if (!outfile.is_open())
    {
        std::cerr << "FATAL ERROR: Could not open file to save features: " << output_csv_path << std::endl;
        return; 
    }

    outfile << "QueryID,QuerySize,CandSize,TrieTotalNodes,TrieLabelCardinality,TrieAvgBranchingFactor,"
            << "NumEntries,NumDescendants,TotalCoverage\n";

    for (size_t i = 0; i < query_stats.size(); ++i)
    {
        const auto &stats = query_stats[i];

        outfile << i << ","
                << stats.query_length << ","
                << stats.candidate_set_size << ","
                << stats.trie_total_nodes << ","
                << stats.trie_label_cardinality << ","
                << stats.trie_avg_branching_factor << ","
                << stats.num_entry_points << ","
                << stats.num_lng_descendants << ","
                << stats.entry_group_total_coverage << "\n";
    }

    outfile.close();
    std::cout << "Successfully saved all features to " << output_csv_path << std::endl;
}


   //fxy_add: 在idea2之前，看看入口组个数，决定是否强制执行acorn
   std::optional<bool> UniNavGraph::check_idea2_heuristic_override(const std::string& dataset_name, size_t num_entry_groups) const
   {
      size_t entry_max= 1000;
      if (dataset_name == "AllNews") entry_max = 1000; 
         
      if (num_entry_groups > entry_max)
         {
               return true; // 返回 true = 强制使用 ACORN
         }
      return std::nullopt; 
   }

   //fxy_add:决定是否跳过 Entry Group 计算和 idea 决策
   std::optional<bool> UniNavGraph::check_pre_trie_heuristic(const std::string& dataset_name, size_t query_length, size_t candidate_set_size) const
   {

      if (query_length <= 1)
      {
         return true; // 强制使用 ACORN
      }

      if (dataset_name == "Russian" && query_length <=2)
      {
          return true; 
      }
      
      return std::nullopt; // 没有命中任何规则，返回空，让 AI 来决策
   }

   size_t UniNavGraph::get_candidate_count_for_label(LabelType label) const
   {
      // 步骤1: 检查标签ID是否在 _label_to_nodes 向量的有效范围内
      if (label >= _trie_index._label_to_nodes.size())
      {
         return 0; // 标签越界，不可能有对应的候选集
      }
      // 步骤2: 直接通过索引访问并返回内部向量的大小
      return _trie_index._label_to_nodes[label].size();
   }

// =======================begin: 计算Fpass======================
   // fxy_add
   void UniNavGraph::build_group_inverted_indices() {
      std::cout << "\n--- Building Group-level Inverted Indices for Fpass Benchmark ---" << std::endl;
      
      _group_attr_adj_list.clear();
      _group_attr_adj_list.resize(_num_attributes);
      
      _group_attr_roaring_inv.clear();
      _group_attr_roaring_inv.resize(_num_attributes);

      for (IdxType gid = 1; gid <= _num_groups; ++gid) {
         const auto& label_set = _group_id_to_label_set[gid];
         for (auto label : label_set) {
               auto it = _attr_to_id.find(label);
               if (it != _attr_to_id.end()) {
                  AtrType attr_id = it->second;
                  _group_attr_adj_list[attr_id].push_back(gid);
                  _group_attr_roaring_inv[attr_id].add(gid);
               }
         }
      }
      std::cout << "- Adjacency List & Roaring Inverted Index built." << std::endl;
   }

   // fxy_add
   void UniNavGraph::evaluate_fpass_methods(std::shared_ptr<IStorage> query_storage, const std::string& output_csv_path) {
      auto num_queries = query_storage->get_num_points();
      std::cout << "\n========================================================" << std::endl;
      std::cout << "--- Starting 5-Method Fpass Benchmark ---" << std::endl;
      
      build_group_inverted_indices();

      std::ofstream outfile(output_csv_path);
      if (!outfile.is_open()) {
         std::cerr << "Failed to open " << output_csv_path << " for writing!" << std::endl;
         return;
      }
      
      // 写入 CSV 表头
      outfile << "QueryID,QuerySize,M1_Enum,M2_BitsetInv,M3_RoaringInv,M4_ELS_Insert,M5_ELS_Roaring,"
               << "M1_Enum_T,M2_BitsetInv_T,M3_RoaringInv_T,"
               << "ELS_Search_T,"
               << "M4_InsertOnly_T,M4_Total_T,"
               << "M5_RoaringUnionOnly_T,M5_Total_T\n";

      // 统计总耗时
      double total_t1 = 0.0;
      double total_t2 = 0.0;
      double total_t3 = 0.0;
      double total_els = 0.0;
      double total_t4_insert = 0.0;
      double total_t5_union = 0.0;
      int mismatch_count = 0;

      // static thread_local std::bitset<12000000> final_group_bitmap;
      // static thread_local std::bitset<12000000> temp_group_bitmap;

      // 安全堆分配
      static thread_local std::unique_ptr<std::bitset<12000000>> final_group_bitmap_ptr;
      static thread_local std::unique_ptr<std::bitset<12000000>> temp_group_bitmap_ptr;
      if (!final_group_bitmap_ptr) final_group_bitmap_ptr = std::make_unique<std::bitset<12000000>>();
      if (!temp_group_bitmap_ptr) temp_group_bitmap_ptr = std::make_unique<std::bitset<12000000>>();
      auto& final_group_bitmap = *final_group_bitmap_ptr;
      auto& temp_group_bitmap = *temp_group_bitmap_ptr;

      for (int id = 0; id < num_queries; ++id) {
         const auto& query_labels = query_storage->get_label_set(id);

         // ==========================================
         // 1. 挨个枚举计算 Fpass
         // ==========================================
         auto start1 = std::chrono::high_resolution_clock::now();
         size_t count1 = 0;
         if (query_labels.empty()) {
               count1 = _num_groups;
         } else {
               for (IdxType gid = 1; gid <= _num_groups; ++gid) {
                  const auto& g_labels = _group_id_to_label_set[gid];
                  bool is_superset = true;
                  for (auto ql : query_labels) {
                     if (std::find(g_labels.begin(), g_labels.end(), ql) == g_labels.end()) {
                           is_superset = false; break;
                     }
                  }
                  if (is_superset) count1++;
               }
         }
         double t1 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start1).count();
         total_t1 += t1;

         // ==========================================
         // 2. Bitset 位运算倒排交集 (类似 Ppass)
         // ==========================================
         auto start2 = std::chrono::high_resolution_clock::now();
         size_t count2 = 0;
         if (query_labels.empty()) {
               count2 = _num_groups;
         } else {
               std::vector<const std::vector<IdxType>*> valid_lists;
               valid_lists.reserve(query_labels.size());
               bool is_valid = true;

               for (LabelType attr_label : query_labels) {
                  auto it = _attr_to_id.find(attr_label);
                  if (it == _attr_to_id.end()) { is_valid = false; break; }
                  valid_lists.push_back(&_group_attr_adj_list[it->second]);
               }

               if (is_valid) {
                  std::sort(valid_lists.begin(), valid_lists.end(),
                     [](const std::vector<IdxType>* a, const std::vector<IdxType>* b) {
                           return a->size() < b->size();
                     });

                  final_group_bitmap.reset();
                  const auto& first_list = *valid_lists[0];
                  for (IdxType gid : first_list) final_group_bitmap.set(gid);

                  for (size_t i = 1; i < valid_lists.size(); ++i) {
                     if (final_group_bitmap.none()) break; 
                     const auto& vec_list = *valid_lists[i];
                     for (IdxType gid : vec_list) temp_group_bitmap.set(gid);
                     final_group_bitmap &= temp_group_bitmap;
                     for (IdxType gid : vec_list) temp_group_bitmap.reset(gid);
                  }
                  count2 = final_group_bitmap.count();
               }
         }
         double t2 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start2).count();
         total_t2 += t2;

         // ==========================================
         // 3. CRoaring 倒排交集 (完全不用 ELS)
         // ==========================================
         auto start3 = std::chrono::high_resolution_clock::now();
         size_t count3 = 0;
         if (query_labels.empty()) {
               count3 = _num_groups;
         } else if (_attr_to_id.count(query_labels[0])) {
               roaring::Roaring roar_res = _group_attr_roaring_inv[_attr_to_id.at(query_labels[0])];
               for (size_t i = 1; i < query_labels.size(); ++i) {
                  if (_attr_to_id.count(query_labels[i])) {
                     roar_res &= _group_attr_roaring_inv[_attr_to_id.at(query_labels[i])];
                  } else {
                     roar_res = roaring::Roaring(); 
                     break;
                  }
               }
               count3 = roar_res.cardinality();
         }
         double t3 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start3).count();
         total_t3 += t3;

         // ==========================================
         // --- 共享的前置步骤: ELS 搜索 ---
         // ==========================================
         auto start_els = std::chrono::high_resolution_clock::now();
         std::vector<IdxType> entry_groups;
         ANNS::QueryStats dummy_stats;
         std::atomic<int> counter{0};
         get_min_super_sets_debug(query_labels, entry_groups, false, true, counter, false, false, dummy_stats, false);
         double t_els = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_els).count();
         total_els += t_els;

         // ==========================================
         // 4. 利用 ELS + unordered_set insert 计算 Fpass
         // ==========================================
         auto start4 = std::chrono::high_resolution_clock::now();
         std::unordered_set<IdxType> naive_desc;
         for (auto gid : entry_groups) {
               if (gid > 0 && gid <= _num_groups) {
                  for (auto desc_id : _label_nav_graph->_lng_descendants[gid]) {
                     naive_desc.insert(desc_id);
                  }
               }
         }
         size_t count4 = naive_desc.size();
         double t4_insert = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start4).count();
         total_t4_insert += t4_insert;

         // ==========================================
         // 5. 利用 ELS + CRoaring |= 并集 计算 Fpass
         // ==========================================
         auto start5 = std::chrono::high_resolution_clock::now();
         roaring::Roaring desc_rb;
         for (auto gid : entry_groups) {
               if (gid > 0 && gid <= _num_groups) {
                  desc_rb |= _lng_descendants_rb[gid];
               }
         }
         size_t count5 = desc_rb.cardinality();
         double t5_union = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start5).count();
         total_t5_union += t5_union;

         // 写入 CSV
         outfile << id << "," << query_labels.size() << ","
                  << count1 << "," << count2 << "," << count3 << "," << count4 << "," << count5 << ","
                  << std::fixed << std::setprecision(4) 
                  << t1 << "," << t2 << "," << t3 << ","
                  << t_els << "," 
                  << t4_insert << "," << (t_els + t4_insert) << ","
                  << t5_union << "," << (t_els + t5_union) << "\n";

         // 一致性校验 (以方法 2 为基准，验证 3, 4, 5)
         // 注意：4 和 5 依赖 Trie，若构建非完美覆盖可能与精确倒排存在极小偏差
         if (count2 != count3 || count2 != count4 || count2 != count5) {
               mismatch_count++;
         }
      }
      outfile.close();

      // 打印总体报告
      std::cout << "\n========================================================" << std::endl;
      std::cout << "--- Fpass Benchmark Summary Report ---" << std::endl;
      std::cout << "Total Queries Processed: " << num_queries << std::endl;
      std::cout << "CSV Saved to: " << output_csv_path << std::endl;
      std::cout << "--------------------------------------------------------" << std::endl;
      std::cout << "1. Enum (Brute Force)      : " << total_t1 << " ms" << std::endl;
      std::cout << "2. Bitset Inv-Intersect    : " << total_t2 << " ms" << std::endl;
      std::cout << "3. CRoaring Inv-Intersect  : " << total_t3 << " ms" << std::endl;
      std::cout << "--------------------------------------------------------" << std::endl;
      std::cout << "[Shared] ELS Search Time   : " << total_els << " ms" << std::endl;
      std::cout << "--------------------------------------------------------" << std::endl;
      std::cout << "4. ELS + Insert Method     : " << std::endl;
      std::cout << "   -> Insert Only Time     : " << total_t4_insert << " ms" << std::endl;
      std::cout << "   -> Total (ELS + Insert) : " << (total_els + total_t4_insert) << " ms" << std::endl;
      std::cout << "5. ELS + CRoaring Method   : " << std::endl;
      std::cout << "   -> |= Union Only Time   : " << total_t5_union << " ms" << std::endl;
      std::cout << "   -> Total (ELS + Union)  : " << (total_els + total_t5_union) << " ms" << std::endl;
      std::cout << "--------------------------------------------------------" << std::endl;
      if (mismatch_count > 0) {
         std::cout << "[Warning] Mismatches detected across methods: " << mismatch_count << std::endl;
      } else {
         std::cout << "[Success] All 5 methods yielded identical Fpass values." << std::endl;
      }
      std::cout << "========================================================\n" << std::endl;
   }


// ======================end: 计算Fpass======================

   //fxy_add
   std::vector<int> UniNavGraph::load_query_algo_choices_from_csv(
      const std::string &csv_path,
      size_t expected_num_queries) const
   {
      std::vector<int> query_algo_choices(expected_num_queries, -1);
      if (csv_path.empty()) return query_algo_choices;

      std::ifstream infile(csv_path);
      if (!infile.is_open()) {
         std::cerr << "[Warning] Failed to open algo choice csv: " << csv_path << ". Fallback to original routing." << std::endl;
         return query_algo_choices;
      }

      std::string line;
      bool header_checked = false;
      size_t loaded_rows = 0;
      while (std::getline(infile, line)) {
         if (line.empty() || line == "\r") continue;
         if (!line.empty() && line.back() == '\r') line.pop_back();

         if (!header_checked) {
            header_checked = true;
            if (line == "Index,Algo_Choice") continue;
         }

         std::stringstream ss(line);
         std::string query_id_str, algo_choice_str;
         if (std::getline(ss, query_id_str, ',') && std::getline(ss, algo_choice_str)) {
            try {
               long long qid_ll = std::stoll(query_id_str);
               long long algo_ll = std::stoll(algo_choice_str);
               if (qid_ll >= 0 && static_cast<size_t>(qid_ll) < expected_num_queries) {
                  query_algo_choices[static_cast<size_t>(qid_ll)] = static_cast<int>(algo_ll);
                  loaded_rows++;
               }
            } catch (...) {}
         }
      }
      std::cout << "[Info] Loaded query algo choices from CSV: " << csv_path << " | loaded=" << loaded_rows << std::endl;
      return query_algo_choices;
   }


   //fxy_add
   int UniNavGraph::determine_routing_strategy(
      int routing_mode, 
      int baseline_alg,
      const std::vector<LabelType>& query_labels,
      ANNS::QueryStats& stats,
      std::vector<IdxType>& entry_group_ids,
      bool is_new_trie_method,
      bool is_rec_more_start)
   {

      float f_qsize = static_cast<float>(stats.query_length);
      float f_cand  = static_cast<float>(stats.candidate_set_size);
      float f_ppass = stats.global_p_pass;

      // --- 默认初始化指标 ---
      stats.is_intel_els_used = false; 
      stats.is_trie_recursive = false; 

      // --- 模式 0: Baseline ---
      if (routing_mode == 0) {
         if (baseline_alg == 0 || baseline_alg == 1 || baseline_alg == 8 || baseline_alg == 14 || baseline_alg == 15) {
               bool use_nT_true = (baseline_alg == 1);
               stats.is_trie_recursive = use_nT_true; // 记录: Baseline使用的是递归还是非递归

               auto els_start = std::chrono::high_resolution_clock::now();
               if (baseline_alg == 15) {
                  get_min_super_sets_sorted_lng(query_labels, entry_group_ids, stats);
               } else if (baseline_alg == 14 && !query_labels.empty()) {
                  // UNG++: bitmap-driven exact ELS（对拍已验证与 trie 版结果一致，无 trie 遍历）
                  get_min_super_sets_bitmap(query_labels, entry_group_ids);
               } else {
                  static std::atomic<int> counter{0};
                  // 注意这里传入 use_nT_true，直接覆盖全局的 is_new_trie_method
                  get_min_super_sets_debug(query_labels, entry_group_ids, false, true, counter, use_nT_true, is_rec_more_start, stats, false);
               }
               stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - els_start).count();
               stats.num_entry_points = entry_group_ids.size();
         }
         return baseline_alg;
      }

      // --- 模式 1: SODA ---
      if (routing_mode == 1 || routing_mode == 6) {
         // 1. 新增：计算 Fpass (NumDescendants) 用于模型推理
         auto start_fpass = std::chrono::high_resolution_clock::now();
         size_t num_descendants = 0;
         if (query_labels.empty()) {
            num_descendants = _num_groups;
         } else {
            bool is_valid = true;
            std::vector<const roaring::Roaring*> valid_rb;
            for (auto label : query_labels) {
                  if (_attr_to_id.count(label)) {
                     valid_rb.push_back(&_group_attr_roaring_inv[_attr_to_id.at(label)]);
                  } else {
                     is_valid = false; break;
                  }
            }
            if (is_valid) {
                  std::sort(valid_rb.begin(), valid_rb.end(), [](const roaring::Roaring* a, const roaring::Roaring* b){
                     return a->cardinality() < b->cardinality();
                  });
                  if (valid_rb.size() == 1) {
                     num_descendants = valid_rb[0]->cardinality();
                  } else {
                     roaring::Roaring roar_res = *valid_rb[0] & *valid_rb[1];
                     for (size_t i = 2; i < valid_rb.size(); ++i) {
                        if (roar_res.isEmpty()) break;
                        roar_res &= *valid_rb[i];
                     }
                     num_descendants = roar_res.cardinality();
                  }
            }
         }
         stats.fpass_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_fpass).count();
         stats.num_lng_descendants = num_descendants;
         // 2. 进行模型预测
         auto pred_start = std::chrono::high_resolution_clock::now();
         int router_decision = 0; 
         if (_smart_route_selector) {
               // 特征输入顺序严格对齐 Python: Ppass, Descendants, QuerySize
               std::vector<float> features = {f_ppass, static_cast<float>(num_descendants), f_qsize};
               float pred_val = _smart_route_selector->predict(features);
               router_decision = static_cast<int>(std::round(pred_val));
         }
         stats.route_pred_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - pred_start).count();

         // 3. 根据预测结果进行分流映射
         // 新语义: 0 -> 第三类算法, 1 -> UNG++, 2 -> pre-filter
         if (router_decision == 0)
            return _smart_route_target_alg_id;
         if (router_decision == 2)
            return 5;
         if (router_decision == 1) {
               auto els_start = std::chrono::high_resolution_clock::now();
               get_min_super_sets_sorted_lng(query_labels, entry_group_ids, stats);
               stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - els_start).count();
               stats.num_entry_points = entry_group_ids.size();
               
               return 15; // 映射到 UNG++-sorted-lng 算法
         }
      }

      // --- 模式 2 & 3: FastSmartRoute & FastSmartRoute+ (使用 CRoaring 算 Fpass) ---
      if (routing_mode == 2 || routing_mode == 3) {

         // Fpass：使用CRoaring倒排索引 部分
         auto start_fpass = std::chrono::high_resolution_clock::now();
         size_t num_descendants = 0;
         
         if (query_labels.empty()) {
            num_descendants = _num_groups;
         } else {
            bool is_valid = true;
            std::vector<const roaring::Roaring*> valid_rb;
            
            for (auto label : query_labels) {
                  if (_attr_to_id.count(label)) {
                     valid_rb.push_back(&_group_attr_roaring_inv[_attr_to_id.at(label)]);
                  } else {
                     is_valid = false; 
                     break;
                  }
            }
            
            if (is_valid) {
                  std::sort(valid_rb.begin(), valid_rb.end(), [](const roaring::Roaring* a, const roaring::Roaring* b){
                     return a->cardinality() < b->cardinality();
                  });
                  
                  if (valid_rb.size() == 1) {
                     // 零拷贝直接读取大小
                     num_descendants = valid_rb[0]->cardinality();
                  } else {
                     // 二元 & 强制深拷贝并计算第一步交集
                     roaring::Roaring roar_res = *valid_rb[0] & *valid_rb[1];
                     for (size_t i = 2; i < valid_rb.size(); ++i) {
                        if (roar_res.isEmpty()) break;
                        roar_res &= *valid_rb[i];
                     }
                     num_descendants = roar_res.cardinality();
                  }
            }
         }
         stats.fpass_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_fpass).count();
         stats.num_lng_descendants = num_descendants;

         // 开始预测
         auto pred_start = std::chrono::high_resolution_clock::now();
         int router_decision = 0; 
         if (_fast_route_single_selector) {
             std::vector<float> features = {f_ppass, static_cast<float>(num_descendants), f_qsize, f_cand};
             float pred_val = _fast_route_single_selector->predict(features);
             router_decision = static_cast<int>(std::round(pred_val));
         }
         stats.route_pred_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - pred_start).count();

         if (router_decision == 2) return 5; 
         if (router_decision == 1) return _single_majority_acorn_id; 

         bool use_nT_true = false;
         if (routing_mode == 3 && _trie_method_selector) {
             stats.is_intel_els_used = true; 
             auto intel_els_start = std::chrono::high_resolution_clock::now();
             std::vector<float> els_features = {f_qsize, f_cand, static_cast<float>(stats.trie_total_nodes)};
             use_nT_true = (_trie_method_selector->predict(els_features) > 0.5f);
             stats.intel_els_pred_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - intel_els_start).count();
         }

         stats.is_trie_recursive = use_nT_true;
         auto els_start = std::chrono::high_resolution_clock::now();
         static std::atomic<int> counter{0};
         get_min_super_sets_debug(query_labels, entry_group_ids, false, true, counter, use_nT_true, is_rec_more_start, stats, false);
         stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - els_start).count();
         stats.num_entry_points = entry_group_ids.size();
         
         return use_nT_true ? 1 : 0; 
      }
      
      // --- 模式 4: Smartroute-revised (2特征, 延迟计算 ELS) ---
      if (routing_mode == 4) {
         stats.is_trie_recursive = false;

         // 1. Fpass: 像 SODA 一样使用 Bitset 计算 Descendants
         auto start_fpass = std::chrono::high_resolution_clock::now();
         static thread_local std::unique_ptr<std::bitset<12000000>> final_group_bitmap_ptr;
         static thread_local std::unique_ptr<std::bitset<12000000>> temp_group_bitmap_ptr;
         if (!final_group_bitmap_ptr) final_group_bitmap_ptr = std::make_unique<std::bitset<12000000>>();
         if (!temp_group_bitmap_ptr) temp_group_bitmap_ptr = std::make_unique<std::bitset<12000000>>();
         auto& final_group_bitmap = *final_group_bitmap_ptr;
         auto& temp_group_bitmap = *temp_group_bitmap_ptr;

         size_t count2 = 0;
         if (query_labels.empty()) {
             count2 = _num_groups;
         } else {
             std::vector<const std::vector<IdxType>*> valid_lists;
             valid_lists.reserve(query_labels.size());
             bool is_valid = true;

             if (_group_attr_adj_list.empty()) {
                 std::cerr << "\n[FATAL ERROR] _group_attr_adj_list is empty!" << std::endl;
                 exit(-1);
             }

             for (LabelType attr_label : query_labels) {
                 auto it = _attr_to_id.find(attr_label);
                 if (it == _attr_to_id.end()) { is_valid = false; break; }
                 AtrType mapped_id = it->second;
                 if (mapped_id < _group_attr_adj_list.size()) {
                     valid_lists.push_back(&_group_attr_adj_list[mapped_id]);
                 } else {
                     is_valid = false; break;
                 }
             }

             if (is_valid) {
                 std::sort(valid_lists.begin(), valid_lists.end(),
                     [](const std::vector<IdxType>* a, const std::vector<IdxType>* b) { return a->size() < b->size(); });
                 final_group_bitmap.reset();
                 const auto& first_list = *valid_lists[0];
                 for (IdxType gid : first_list) final_group_bitmap.set(gid);
                 for (size_t i = 1; i < valid_lists.size(); ++i) {
                     if (final_group_bitmap.none()) break; 
                     const auto& vec_list = *valid_lists[i];
                     for (IdxType gid : vec_list) temp_group_bitmap.set(gid);
                     final_group_bitmap &= temp_group_bitmap;
                     for (IdxType gid : vec_list) temp_group_bitmap.reset(gid);
                 }
                 count2 = final_group_bitmap.count();
             }
         }
         stats.fpass_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_fpass).count();
         stats.num_lng_descendants = count2;

         // 2. 开始预测 (仅输入 f_ppass 和 num_descendants)
         auto pred_start = std::chrono::high_resolution_clock::now();
         int router_decision = 0; 
         if (_fast_route_revised_selector) {
             std::vector<float> features = {f_ppass, static_cast<float>(count2)};
             float pred_val = _fast_route_revised_selector->predict(features);
             router_decision = static_cast<int>(std::round(pred_val));
         }
         stats.route_pred_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - pred_start).count();

         // 3. Early Return: 非 UNG 类直接短路返回，不计算 ELS
         if (router_decision == 2) return 5; // 走向 pre-filter
         if (router_decision == 1) return _revised_majority_acorn_id; // 走向 ACORN 多数派

         // 4. 如果预测为 0 (UNG_family)，再计算 ELS。默认使用 nTfalse
         auto els_start = std::chrono::high_resolution_clock::now();
         static std::atomic<int> counter{0};
         get_min_super_sets_debug(query_labels, entry_group_ids, false, true, counter, false, is_rec_more_start, stats, false);
         stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - els_start).count();
         stats.num_entry_points = entry_group_ids.size();
         
         return 0; 
      }

      // --- Mode 8: ALPS-fixed (fixed thresholds over exact GlobalPpass) ---
      // >39.9%: FAVOR-HNSW; [1.2%, 39.9%]: UNG++-sorted-lng; <1.2%: pre-filter.
      if (routing_mode == 8) {
         if (f_ppass > 0.399f) {
            return 12;
         }
         if (f_ppass < 0.012f) {
            return 5;
         }

         auto els_start = std::chrono::high_resolution_clock::now();
         get_min_super_sets_sorted_lng(query_labels, entry_group_ids, stats);
         stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - els_start).count();
         stats.num_entry_points = entry_group_ids.size();
         return 15;
      }
      
      return 0; // Fallback
   }
   

// fxy_add 
   void UniNavGraph::thread_function(int id, SearchCacheList& search_cache_list,
                                    std::shared_ptr<IStorage> &query_storage,
                                   std::shared_ptr<DistanceHandler> &distance_handler,
                                   uint32_t num_threads, IdxType Lsearch,
                                   IdxType num_entry_points, std::string scenario,
                                   IdxType K, std::pair<IdxType, float> *results,
                                   std::vector<float> &num_cmps,
                                   std::vector<QueryStats> &query_stats,
                                   bool is_new_trie_method, bool is_rec_more_start,
                                   bool is_ung_more_entry,
                                   int lsearch_start, int lsearch_step,
                                   int efs_start, int efs_step_slow,int efs_step_fast,int lsearch_threshold,
                                   int routing_mode, int baseline_alg, IdxType num_queries, faiss_navix::IndexHNSWFlat* navix_index,
                                    const std::vector<IdxType> &true_query_group_ids,const std::vector<int> &query_algo_choices,
                                    bool optimize_standalone_prefilter)
   {
         omp_set_num_threads(1);
         auto &stats = query_stats[id];
         auto total_search_start_time = std::chrono::high_resolution_clock::now();
         auto search_cache = search_cache_list.get_free_cache();
         const char *query = _query_storage->get_vector(id);
         SearchQueue cur_result;
         cur_result.reserve(K);
         const auto &query_labels = _query_storage->get_label_set(id);

         // ======================= STAGE 1: DECISION MAKING =======================
         auto decision_start_time = std::chrono::high_resolution_clock::now();

      // pre-filter是否使用优化模式
      bool use_optimized = false;
      if (routing_mode != 0) {
         use_optimized = true; // SODA 等模式强制使用大优化
      } else {
         use_optimized = optimize_standalone_prefilter; // 纯 Baseline 模式，听从用户指定的参数
      }

      const std::bitset<16000000>* exact_mask_ptr = nullptr;
      const roaring::Roaring* final_roaring_ptr = nullptr;
      roaring::Roaring roar_res; 
      bool has_exact_mask = false;
      stats.bitmap_time_ms = 0.0; 

      std::vector<IdxType> entry_group_ids;
      int final_algo_choice = -1;

      // -----------------------------------------------------------------------
      // 路径 A: SODA+ / SmartRoute+++ (Mode 5 / 7, global choices are precomputed)
      // -----------------------------------------------------------------------
      if (routing_mode == 5 || routing_mode == 7) {
            final_algo_choice = query_algo_choices[id]; // 此时传进来的已经是全局算好的 choice
            stats.algo_choice = final_algo_choice;

            auto prep_start = std::chrono::high_resolution_clock::now();
            if (final_algo_choice == 0 || final_algo_choice == 1 || final_algo_choice == 8 || final_algo_choice == 14 || final_algo_choice == 15) {
               if (final_algo_choice == 15) {
                  get_min_super_sets_sorted_lng(query_labels, entry_group_ids, stats);
               } else if (final_algo_choice == 14 && !query_labels.empty()) {
                  get_min_super_sets_bitmap(query_labels, entry_group_ids);
               } else {
                  static std::atomic<int> counter{0};
                  get_min_super_sets_debug(query_labels, entry_group_ids, false, true, counter, (final_algo_choice == 1), is_rec_more_start, stats, false);
               }
            } else if (final_algo_choice == 5) {
               if (query_labels.empty()) {
                     roar_res.addRange(0, _num_points);
                     final_roaring_ptr = &roar_res;
               } else {
                     std::vector<const roaring::Roaring*> valid_rb;
                     for (auto label : query_labels) {
                        if (_attr_to_id.count(label)) valid_rb.push_back(&_vec_attr_roaring_inv[_attr_to_id.at(label)]);
                     }
                     if (!valid_rb.empty()) {
                        std::sort(valid_rb.begin(), valid_rb.end(), [](const roaring::Roaring* a, const roaring::Roaring* b){ return a->cardinality() < b->cardinality(); });
                        if (valid_rb.size() == 1) {
                           final_roaring_ptr = valid_rb[0]; // 零拷贝
                        } else {
                           roar_res = *valid_rb[0] & *valid_rb[1];
                           for (size_t i = 2; i < valid_rb.size(); ++i) {
                                 if (roar_res.isEmpty()) break;
                                 roar_res &= *valid_rb[i];
                           }
                           final_roaring_ptr = &roar_res;
                        }
                     }
               }
               has_exact_mask = true;
            }
            double prep_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - prep_start).count();
            
            // 合拢总耗时
            stats.routing_total_time_ms = stats.mask_gen_time_ms + stats.route_pred_time_ms + stats.global_sort_time_ms + prep_time;
      } 
      // -----------------------------------------------------------------------
      // 路径 B: 实时推理模式 (Mode 0~4)
      // -----------------------------------------------------------------------
      else {
         auto feature_start = std::chrono::high_resolution_clock::now();
         stats.query_length = query_labels.size();
         stats.candidate_set_size = query_labels.empty() ? 0 : get_candidate_count_for_label(query_labels.back());
         stats.trie_I_lmax = stats.candidate_set_size;
         stats.trie_total_nodes = _trie_static_metrics.total_nodes;

            if (routing_mode == 4 || (routing_mode == 0 && (baseline_alg == 5 || baseline_alg == 13))) {
               // Curator (13) shares the exact same bitset filter map (bipartite
               // graph path) and the same measured cost as the pre-filter baseline.
               auto mask_start = std::chrono::high_resolution_clock::now();
               exact_mask_ptr = &get_exact_cand_size_and_mask(query_labels, stats.exact_cand_size, use_optimized);
               stats.bitmap_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - mask_start).count();
               stats.global_p_pass = static_cast<float>(stats.exact_cand_size) / _num_points;
               has_exact_mask = true;
            } else if (routing_mode == 1 || routing_mode == 2 || routing_mode == 3 || routing_mode == 6 || routing_mode == 8) {
            auto mask_start = std::chrono::high_resolution_clock::now(); 
            if (query_labels.empty()) {
               roar_res.addRange(0, _num_points);
               final_roaring_ptr = &roar_res;
               stats.exact_cand_size = _num_points;
            } else {
               bool is_valid = true;
               std::vector<const roaring::Roaring*> valid_rb;
               for (auto label : query_labels) {
                        if (_attr_to_id.count(label)) { valid_rb.push_back(&_vec_attr_roaring_inv[_attr_to_id.at(label)]); } 
                        else { is_valid = false; break; }
               }
               if (is_valid) {
                        std::sort(valid_rb.begin(), valid_rb.end(), [](const roaring::Roaring* a, const roaring::Roaring* b){ return a->cardinality() < b->cardinality(); });
                     if (valid_rb.size() == 1) {
                        final_roaring_ptr = valid_rb[0];
                        stats.exact_cand_size = final_roaring_ptr->cardinality();
                     } else {
                        roar_res = *valid_rb[0] & *valid_rb[1]; 
                        for (size_t i = 2; i < valid_rb.size(); ++i) {
                           if (roar_res.isEmpty()) break;
                           roar_res &= *valid_rb[i];
                        }
                        final_roaring_ptr = &roar_res;
                        stats.exact_cand_size = roar_res.cardinality();
                     }
                  } else { stats.exact_cand_size = 0; }
            }
            stats.bitmap_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - mask_start).count();
            stats.global_p_pass = static_cast<float>(stats.exact_cand_size) / _num_points;
            has_exact_mask = true;
            }

            stats.feature_extract_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - feature_start).count() - stats.bitmap_time_ms;

         //  final_algo_choice = determine_routing_strategy(routing_mode, baseline_alg, query_labels, stats, entry_group_ids, is_new_trie_method, is_rec_more_start);
         //  stats.algo_choice = final_algo_choice;
         //  stats.routing_total_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - decision_start_time).count();
         
         // =========================================================================
         // 算法分配与强制接管 (Override)
         // =========================================================================
         // 1. 检查是否需要强制分配算法 (仅当 routing_mode 为 1 或 2，且 CSV 中有对应的有效值)
         bool has_override = (routing_mode == 1 || routing_mode == 2) && 
                              (id >= 0 && static_cast<size_t>(id) < query_algo_choices.size() && query_algo_choices[id] != -1);

         if (!has_override) {
            // 分支 A：没有预设强制选项，走正常的实时模型预测逻辑
            final_algo_choice = determine_routing_strategy(routing_mode, baseline_alg, query_labels, stats, entry_group_ids, is_new_trie_method, is_rec_more_start);
         } else {
            // 分支 B：有强制的 Algo Choice，直接接管
            final_algo_choice = query_algo_choices[id];

            // 2. 数据兜底：由于跳过了 determine_routing_strategy，必须在这里补全后续搜索必需的前置数据

            // 兜底 A: 如果被强制分配到 UNG 家族（0, 1, 或 8），但入口点组还没算，需要补算 ELS
            if ((final_algo_choice == 0 || final_algo_choice == 1 || final_algo_choice == 8 || final_algo_choice == 14 || final_algo_choice == 15) && entry_group_ids.empty()) {
               bool use_nT_true = (final_algo_choice == 1);
               stats.is_trie_recursive = use_nT_true;
               auto els_start = std::chrono::high_resolution_clock::now();
               if (final_algo_choice == 15) {
                  get_min_super_sets_sorted_lng(query_labels, entry_group_ids, stats);
               } else if (final_algo_choice == 14 && !query_labels.empty()) {
                  get_min_super_sets_bitmap(query_labels, entry_group_ids);
               } else {
                  static std::atomic<int> counter{0};
                  get_min_super_sets_debug(query_labels, entry_group_ids, false, true, counter, use_nT_true, is_rec_more_start, stats, false);
               }
               stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - els_start).count();
               stats.num_entry_points = entry_group_ids.size();
            }

            // 兜底 B: 如果被强制分配到 Pre-filter (5)，必须确保生成了用于暴力搜索的精确掩码 (CRoaring)
            if (final_algo_choice == 5 && !has_exact_mask) {
               auto mask_start = std::chrono::high_resolution_clock::now();
               if (query_labels.empty()) {
                  roar_res.addRange(0, _num_points);
                  final_roaring_ptr = &roar_res;
                  stats.exact_cand_size = _num_points;
               } else {
                  bool is_valid = true;
                  std::vector<const roaring::Roaring*> valid_rb;
                  if (_vec_attr_roaring_inv.empty()) {
                      std::cerr << "\n[FATAL ERROR] _vec_attr_roaring_inv is empty!" << std::endl;
                      exit(-1); 
                  }
                  for (auto label : query_labels) {
                        if (_attr_to_id.count(label)) {
                           valid_rb.push_back(&_vec_attr_roaring_inv[_attr_to_id.at(label)]);
                        } else {
                           is_valid = false; 
                           break;
                        }
                  }
                  if (is_valid) {
                        std::sort(valid_rb.begin(), valid_rb.end(), [](const roaring::Roaring* a, const roaring::Roaring* b){
                           return a->cardinality() < b->cardinality();
                        });
                        if (valid_rb.size() == 1) {
                           final_roaring_ptr = valid_rb[0];
                           stats.exact_cand_size = final_roaring_ptr->cardinality();
                        } else {
                           roar_res = *valid_rb[0] & *valid_rb[1]; 
                           for (size_t i = 2; i < valid_rb.size(); ++i) {
                              if (roar_res.isEmpty()) break;
                              roar_res &= *valid_rb[i];
                           }
                           final_roaring_ptr = &roar_res;
                           stats.exact_cand_size = roar_res.cardinality();
                        }
                  } else {
                     stats.exact_cand_size = 0;
                  }
               }
                  stats.bitmap_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - mask_start).count();
               has_exact_mask = true;
               stats.global_p_pass = static_cast<float>(stats.exact_cand_size) / _num_points;
            }
         }

         // 统一下发决策结果并计算耗时
         stats.algo_choice = final_algo_choice;
         stats.routing_total_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - decision_start_time).count();
      }


         // ======================= STAGE 2: EXECUTION STAGE =======================
         
         // Apply entry point expansion logic if needed for the UNG path
         if (is_ung_more_entry && (final_algo_choice == 0 || final_algo_choice == 1 || final_algo_choice == 8 || final_algo_choice == 14 || final_algo_choice == 15))
         {
            IdxType true_group_id = 0;
            if (id < true_query_group_ids.size())
            {
               true_group_id = true_query_group_ids[id];
            }
            const size_t extra_k = entry_group_ids.size() / 5;
            entry_group_ids = select_entry_groups(entry_group_ids, SelectionMode::SizeAndDistance, extra_k, 1.0, true_group_id);
         }
         stats.num_entry_points = entry_group_ids.size();

         if (final_algo_choice == 5) // pre-filter
         {
             auto search_time_start_ms = std::chrono::high_resolution_clock::now();
             size_t dist_calcs = 0;
             std::vector<std::pair<IdxType, float>> exact_results(K);
             ANNS::rabitq::RabitQSideIndex::QueryContext local_query_ctx;
             ANNS::rabitq::RabitQSideIndex::QueryContext *prefilter_query_ctx = nullptr;

             const bool use_rabitq_prefilter = (_ung_distance_mode == UngDistanceMode::RabitQ && _rabitq_side_index.enabled());
             if (use_rabitq_prefilter) {
                if (id >= 0 && static_cast<size_t>(id) < _rabitq_query_ctx_cache.size()) {
                   prefilter_query_ctx = _rabitq_query_ctx_cache[id].get();
                   stats.rabitq_ctx_prepare_time_ms = _rabitq_query_ctx_prepare_ms[id];
                   stats.rabitq_ctx_rotate_time_ms = _rabitq_query_ctx_rotate_ms[id];
                   stats.rabitq_ctx_q2c_time_ms = _rabitq_query_ctx_q2c_ms[id];
                   stats.rabitq_ctx_wrapper_time_ms = _rabitq_query_ctx_wrapper_ms[id];
                   stats.rabitq_ctx_reused = (prefilter_query_ctx != nullptr);
                }
                if (prefilter_query_ctx == nullptr) {
                   auto init_start = std::chrono::high_resolution_clock::now();
                   ANNS::rabitq::RabitQSideIndex::InitTiming init_timing;
                   if (_rabitq_side_index.init_query(query, local_query_ctx, &init_timing)) {
                      prefilter_query_ctx = &local_query_ctx;
                      stats.rabitq_ctx_prepare_time_ms += std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - init_start
                      ).count();
                      stats.rabitq_ctx_rotate_time_ms += init_timing.rotate_ms;
                      stats.rabitq_ctx_q2c_time_ms += init_timing.q_to_centroids_ms;
                      stats.rabitq_ctx_wrapper_time_ms += init_timing.wrapper_ms;
                      stats.rabitq_ctx_reused = false;
                   }
                }
             }

            //  if (routing_mode == 2 || routing_mode == 3) {
            //    //   search_baseline_exact_roaring(query, exact_roaring_mask.value(), K, exact_results.data(), dist_calcs);
            //      search_baseline_exact_roaring(query, *final_roaring_ptr, K, exact_results.data(), dist_calcs);
            //  } else {
            //      bool use_optimized = false; // Mode 1 依然走普通 bitset 
            //      search_baseline_exact(query, *exact_mask_ptr, K, exact_results.data(), dist_calcs, use_optimized);
            //  }

            // 优先检查是否生成了 CRoaring 掩码（兜底B 和 模式2/3 都会生成）
             if (final_roaring_ptr != nullptr) {
                 if (use_rabitq_prefilter && prefilter_query_ctx != nullptr) {
                    search_baseline_rabitq_roaring(*final_roaring_ptr, K, exact_results.data(), dist_calcs, *prefilter_query_ctx, stats);
                 } else {
                    search_baseline_exact_roaring(query, *final_roaring_ptr, K, exact_results.data(), dist_calcs);
                 }
             } 
             // 否则回退到普通 Bitset 掩码（例如 Mode 1 生成的）
             else if (exact_mask_ptr != nullptr) {
                 bool use_optimized = false;
                 if (use_rabitq_prefilter && prefilter_query_ctx != nullptr) {
                    search_baseline_rabitq(*exact_mask_ptr, K, exact_results.data(), dist_calcs, *prefilter_query_ctx, stats, use_optimized);
                 } else {
                    search_baseline_exact(query, *exact_mask_ptr, K, exact_results.data(), dist_calcs, use_optimized);
                 }
             }

             stats.num_distance_calcs = dist_calcs;
             stats.core_search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - search_time_start_ms).count();
             stats.search_time_ms = stats.core_search_time_ms;

             for (size_t i = 0; i < K; ++i) {
                 if (exact_results[i].first != -1) {
                     results[id * K + i].first = _new_to_old_vec_ids[exact_results[i].first];
                     results[id * K + i].second = exact_results[i].second;
                 } else {
                     results[id * K + i].first = -1;
                     results[id * K + i].second = std::numeric_limits<float>::max();
                 }
             }

             num_cmps[id] = dist_calcs;
             stats.time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - total_search_start_time).count();
             search_cache_list.release_cache(search_cache);
             return; 
         }
         else if (final_algo_choice == 7) // NaviX(先不看，在NaviX索引)
         {
             auto search_time_start_ms = std::chrono::high_resolution_clock::now();
             
             if (!navix_index) {
                 std::cerr << "ERROR: NaviX index not loaded for query " << id << std::endl;
                 for (auto k_idx = 0; k_idx < K; ++k_idx) results[id * K + k_idx].first = -1;
                 return;
             }

             // --- 动态对齐 efSearch 参数 ---
             int current_efs = compute_aligned_efs(
                 Lsearch, lsearch_start, lsearch_step, efs_start,
                 efs_step_slow, efs_step_fast, lsearch_threshold);
             stats.acorn_efs_used = current_efs;
             navix_index->hnsw.efSearch = current_efs;

             // =========================================================================
             // 1. 将查询标签映射为二分图中的“属性节点 ID”，并生成 Old ID 掩码
             // =========================================================================
             auto bitmap_start_time = std::chrono::high_resolution_clock::now();

             std::vector<uint32_t> mapped_query_attrs;
             bool is_possible = true;
             for (auto label : query_labels) {
                 auto it = _attr_to_id.find(label);
                 if (it == _attr_to_id.end()) {
                     is_possible = false; // 查询了底库中根本不存在的属性
                     break;
                 }
                 mapped_query_attrs.push_back(_num_points + static_cast<ANNS::IdxType>(it->second));
             }

             std::vector<char> navix_mask_vec;
             if (!is_possible) {
                 navix_mask_vec.assign(_num_points, 0); // 存在无效属性，掩码全 0
             } else {
                 navix_mask_vec = generate_single_filter_map(this->_vector_attr_graph, _num_points, mapped_query_attrs);
             }

             // 将 = 改为 +=
            stats.bitmap_time_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - bitmap_start_time).count();

             // =========================================================================
             // 2. 准备 NaviX 要求的查询容器
             // =========================================================================
             const float* query_ptr = reinterpret_cast<const float*>(query); 
             
             std::vector<faiss_navix::idx_t> labels_vec(K);
             std::vector<float> distances_vec(K);
             
             faiss_navix::VisitedTable visited(_num_points);
             faiss_navix::HNSWStats navix_stats;

             auto core_search_start_time = std::chrono::high_resolution_clock::now();

             // =========================================================================
             // 3. 执行 NaviX 检索 (直接传入 Old ID 掩码)
             // =========================================================================
             navix_index->navix_single_search(
                 query_ptr, 
                 K, 
                 distances_vec.data(), 
                 labels_vec.data(), 
                 navix_mask_vec.data(), 
                 visited, 
                 navix_stats
             );
             
             stats.core_search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - core_search_start_time).count();

             // =========================================================================
             // 4. 提取结果回填至 UNG 框架
             // =========================================================================
             cur_result.clear();
             for (int i = 0; i < K; ++i) {
                 if (labels_vec[i] != -1) { 
                     cur_result.insert(_old_to_new_vec_ids[labels_vec[i]], distances_vec[i]); 
                 }
             }

             // 5. 记录统计指标
             stats.num_distance_calcs = navix_stats.ndis;
             stats.num_nodes_visited = navix_stats.n1;
             stats.search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - search_time_start_ms).count();
         }
         else if (final_algo_choice == 9 || final_algo_choice == 10) // Milvus baseline via Knowhere (IVF/HNSW) + bitset containment filter
         {
            if (!run_milvus_knowhere_baseline(
                    id, query, query_labels, Lsearch, K, final_algo_choice, cur_result, stats, num_cmps[id])) {
               for (auto k = 0; k < K; ++k) {
                  results[id * K + k].first = -1;
               }
               return;
            }
         }
         else if (final_algo_choice == 11) // FAVOR baseline adapted to containment labels
         {
            if (!run_favor_baseline(
                    id, query, query_labels, Lsearch, K, cur_result, stats, num_cmps[id])) {
               for (auto k = 0; k < K; ++k) {
                  results[id * K + k].first = -1;
               }
               return;
            }
         }
         else if (final_algo_choice == 12) // FAVOR-HNSW baseline adapted to containment labels
         {
            const int current_efs = compute_aligned_efs(
                Lsearch, lsearch_start, lsearch_step, efs_start,
                efs_step_slow, efs_step_fast, lsearch_threshold);
            if (!run_favor_hnsw_baseline(
                    id, query, query_labels, Lsearch, current_efs, K, cur_result, stats, num_cmps[id])) {
               for (auto k = 0; k < K; ++k) {
                  results[id * K + k].first = -1;
               }
               return;
            }
         }
         else if (final_algo_choice == 13) // Curator
         {
            if (!_curator_ctx.ready) {
               std::cerr << "ERROR: Curator index not ready. Skipping query " << id << std::endl;
               for (auto k = 0; k < K; ++k) results[id * K + k].first = -1;
               return;
            }
            CuratorSearchStats cs;
            // Reuse the Stage-1 bitset filter map (same bipartite-graph path and
            // cost as pre-filter). Fall back to the wrapper's own inverted index
            // only when no mask was computed for this query.
            curator_search_query_detailed(
                _curator_ctx, id, query, query_labels, K,
                &results[id * K], cs,
                exact_mask_ptr);
            // Stage-1 mask time is already in stats.bitmap_time_ms; the wrapper
            // only adds its own phase-1 cost (mask extraction, or the fallback
            // intersection). Accumulate instead of overwriting.
            stats.bitmap_time_ms += cs.bitmap_time_ms;
            stats.core_search_time_ms = cs.search_time_ms;
            stats.search_time_ms = cs.bitmap_time_ms + cs.search_time_ms;
            if (exact_mask_ptr == nullptr) {
               stats.exact_cand_size = cs.exact_cand_size;
               stats.global_p_pass = cs.global_p_pass;
            }
            stats.num_distance_calcs = 0;
         }
         else if (final_algo_choice >= 2 && final_algo_choice <= 6 && final_algo_choice != 5) // ACORN 家族 (2,3,4,6)
         {
            auto search_time_start_ms = std::chrono::high_resolution_clock::now();
            std::shared_ptr<faiss::IndexACORNFlat> selected_acorn_index = nullptr; // <--- 使用一个临时指针
            const ANNS::rabitq::RabitQSideIndex* selected_acorn_side_index = nullptr;

            if (final_algo_choice == 4 || final_algo_choice == 6) {
               selected_acorn_index = _acorn_1_index;
               selected_acorn_side_index = &_acorn_1_rabitq_side_index;
            } else { 
               selected_acorn_index = _acorn_index;
               selected_acorn_side_index = &_acorn_rabitq_side_index;
            }

            // --- Execute ACORN Search ---
            if (!selected_acorn_index)
            {
               std::cerr << "ERROR: ACORN index not loaded for query " << id << ". Skipping." << std::endl;
               for (auto k = 0; k < K; ++k) results[id * K + k].first = -1;
               return;
            }

            // efs变化过程（可以不看）
            int current_efs = compute_aligned_efs(
               Lsearch, lsearch_start, lsearch_step, efs_start,
               efs_step_slow, efs_step_fast, lsearch_threshold);
            stats.acorn_efs_used = current_efs;
            selected_acorn_index->acorn.efSearch = current_efs;

            

            // 确定当前使用哪种 ACORN 内部搜索策略 (0: ACORN, 1: Imp, 2: NaviX)
            int current_baseline_alg = 0;
            if (final_algo_choice == 2 || final_algo_choice == 6) {
                current_baseline_alg = 0; // ACORN-gamma / ACORN-1 使用基础原版
            } else if (final_algo_choice == 3) {
                current_baseline_alg = 1; // ACORN-improved
            } else if (final_algo_choice == 4) {
                current_baseline_alg = 2; // NaviX-ACORN
            }

            const float *query_vector_float = reinterpret_cast<const float *>(query);
            std::vector<faiss::idx_t> result_original_ids(K);
            std::vector<float> result_dists(K);
            bool has_els_groups = !entry_group_ids.empty();
            bool has_exact_mask = (exact_mask_ptr != nullptr);

            // 只有在既没有 ELS，也没有在特征提取阶段算过 exact_mask 时，才使用 ACORN 内部倒排索引
            bool use_acorn_native_filter = (!has_els_groups && !has_exact_mask);
            bool use_acorn_rabitq =
                (_ung_distance_mode == UngDistanceMode::RabitQ &&
                 selected_acorn_side_index != nullptr &&
                 selected_acorn_side_index->enabled());

            // 保持 ACORN 图搜索流程一致，仅替换距离计算后端
            faiss::SearchParametersACORN acorn_search_params;
            acorn_search_params.efSearch = current_efs;
            std::shared_ptr<ACORNRabitQDistanceBackend> acorn_rabitq_backend = nullptr;
            if (use_acorn_rabitq)
            {
               acorn_rabitq_backend = std::make_shared<ACORNRabitQDistanceBackend>(selected_acorn_side_index);
               acorn_search_params.distance_backend = acorn_rabitq_backend;
            }

            if (use_acorn_native_filter)
            {
               stats.acorn_filter_type = 3; // 记录使用倒排索引
               // 分支 1：Baseline 模式下的 ACORN 家族 (没有任何前置掩码可用)
               std::vector<std::vector<int>> query_attrs_for_acorn(1);
               query_attrs_for_acorn[0].assign(query_labels.begin(), query_labels.end());

               auto core_search_start_time = std::chrono::high_resolution_clock::now();
               selected_acorn_index->search_old_bitmap(
                   1, query_vector_float, K, result_dists.data(), result_original_ids.data(), 
                   query_attrs_for_acorn, nullptr, nullptr, nullptr, current_baseline_alg, &acorn_search_params);
               stats.core_search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - core_search_start_time).count();
            }
            else
            {          
               // // 分支 2 & 3：复用前置计算好的掩码 (ELS Group 或 Exact Mask)
               // std::vector<char> filter_map(_num_points, 0);
               // auto bitmap_start_time = std::chrono::high_resolution_clock::now();
               // // 只要 exact_mask_ptr 不为空 (SODA/FastSmartRoute天然满足)，就直接用，无视 ELS 是否存在
               // if (has_exact_mask) 
               // {
               //     stats.acorn_filter_type = 2; // 标记使用了 Exact Mask
               //     for (size_t pt_id = 0; pt_id < _num_points; ++pt_id) 
               //     {
               //         if (exact_mask_ptr->test(pt_id)) 
               //         {
               //             filter_map[pt_id] = 1; 
               //         }
               //     }
               // }
               // // 如果没有 exact_mask (比如普通的 Baseline UNG 算法走到这一步)，再兜底使用 ELS
               // else if (has_els_groups) 
               // {
               //     stats.acorn_filter_type = 1; // 标记使用了 ELS
               //     roaring::Roaring current_roaring_bitmap = compute_bitmap_from_groups(entry_group_ids);
               //     for (uint32_t original_id : current_roaring_bitmap)
               //     {
               //        if (original_id < _num_points)
               //        {
               //           IdxType new_id = _old_to_new_vec_ids[original_id];
               //           filter_map[new_id] = 1;
               //        }
               //     }
               // }


               // 分支 2 & 3：复用前置计算好的掩码 (ELS Group 或 Exact Mask)
               
               // 使用 thread_local 避免每次查询都在堆上分配十几MB的内存
               static thread_local std::vector<char> filter_map;
               if (filter_map.size() < _num_points) {
                   filter_map.resize(_num_points);
               }
               
               auto bitmap_start_time = std::chrono::high_resolution_clock::now();
               std::memset(filter_map.data(), 0, _num_points);

               if (has_exact_mask) 
               {
                   stats.acorn_filter_type = 2; // 标记使用了 Exact Mask
                   if (exact_mask_ptr) {
                       // 解析 Bitset
                       for (size_t pt_id = 0; pt_id < _num_points; ++pt_id) {
                           if ((*exact_mask_ptr)[pt_id]) filter_map[pt_id] = 1; 
                       }
                   } else if (final_roaring_ptr != nullptr) {
                       // 解析 CRoaring
                       for (uint32_t pt_id : *final_roaring_ptr) {
                          filter_map[pt_id] = 1;
                     }
                   }
               }
               // 如果没有 exact_mask (比如普通的 Baseline UNG 算法走到这一步)，再兜底使用 ELS
               else if (has_els_groups) 
               {
                   stats.acorn_filter_type = 1; // 标记使用了 ELS
                   roaring::Roaring current_roaring_bitmap = compute_bitmap_from_groups(entry_group_ids);
                   for (uint32_t original_id : current_roaring_bitmap)
                   {
                      if (original_id < _num_points)
                      {
                         IdxType new_id = _old_to_new_vec_ids[original_id];
                         filter_map[new_id] = 1;
                      }
                   }
               }

               stats.bitmap_time_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - bitmap_start_time).count();
               
               auto core_search_start_time = std::chrono::high_resolution_clock::now();
               selected_acorn_index->search(
                   1, query_vector_float, K, result_dists.data(), result_original_ids.data(), 
                   filter_map.data(), nullptr, nullptr, nullptr, current_baseline_alg, &acorn_search_params);
               stats.core_search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - core_search_start_time).count();
            }

            cur_result.clear();
            for (size_t i = 0; i < K; ++i)
            {
               if (result_original_ids[i] != -1)
               {
                  cur_result.insert(result_original_ids[i], result_dists[i]);
               }
            }
            num_cmps[id] = 0;
            if (use_acorn_rabitq && acorn_rabitq_backend)
            {
               const auto rabitq_snapshot = acorn_rabitq_backend->snapshot();
               stats.rabitq_ctx_prepare_time_ms += rabitq_snapshot.ctx_prepare_time_ms;
               stats.rabitq_ctx_rotate_time_ms += rabitq_snapshot.ctx_rotate_time_ms;
               stats.rabitq_ctx_q2c_time_ms += rabitq_snapshot.ctx_q2c_time_ms;
               stats.rabitq_ctx_wrapper_time_ms += rabitq_snapshot.ctx_wrapper_time_ms;
               stats.rabitq_bin_time_ms += rabitq_snapshot.bin_time_ms;
               stats.rabitq_full_time_ms += rabitq_snapshot.full_time_ms;
               stats.rabitq_bin_calls += rabitq_snapshot.bin_calls;
               stats.rabitq_full_calls += rabitq_snapshot.full_calls;
               stats.rabitq_ctx_reused = false;
            }
            stats.search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - search_time_start_ms).count();
         }
         else if (final_algo_choice == 0 || final_algo_choice == 1 || final_algo_choice == 8 || final_algo_choice == 14 || final_algo_choice == 15)
         {
            // --- Execute UNG Search ---
            auto search_time_start_ms = std::chrono::high_resolution_clock::now();
            search_cache->visited_set.clear();
            std::vector<IdxType> entry_points;
            for (const auto &group_id : entry_group_ids)
            {
               get_entry_points_given_group_id(num_entry_points, search_cache->visited_set, group_id, entry_points);
            }

            if (entry_points.empty())
            {
               stats.num_distance_calcs = 0;
               return;
            }

            auto core_search_start_time = std::chrono::high_resolution_clock::now();
            if (_ung_distance_mode == UngDistanceMode::RabitQ && _rabitq_side_index.enabled())
            {
               ANNS::rabitq::RabitQSideIndex::QueryContext *cached_query_ctx = nullptr;
               if (id >= 0 && static_cast<size_t>(id) < _rabitq_query_ctx_cache.size())
               {
                  cached_query_ctx = _rabitq_query_ctx_cache[id].get();
                  stats.rabitq_ctx_prepare_time_ms = _rabitq_query_ctx_prepare_ms[id];
                  stats.rabitq_ctx_rotate_time_ms = _rabitq_query_ctx_rotate_ms[id];
                  stats.rabitq_ctx_q2c_time_ms = _rabitq_query_ctx_q2c_ms[id];
                  stats.rabitq_ctx_wrapper_time_ms = _rabitq_query_ctx_wrapper_ms[id];
                  stats.rabitq_ctx_reused = (cached_query_ctx != nullptr);
               }
               num_cmps[id] = iterate_to_fixed_point_rabitq(
                   query,
                   search_cache,
                   id,
                   entry_points,
                   stats.num_nodes_visited,
                   stats,
                   cached_query_ctx
               );
            }
            else
            {
               num_cmps[id] = iterate_to_fixed_point(
                   query,
                   search_cache,
                   id,
                   entry_points,
                   stats.num_nodes_visited
               );
            }
            stats.core_search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - core_search_start_time).count();

            stats.num_distance_calcs = num_cmps[id];
            cur_result = search_cache->search_queue;
            stats.search_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - search_time_start_ms).count();
         }

         // ======================= STAGE 3: FINALIZE RESULTS =======================
         // Curator and FAVOR write results directly — skip ID remapping
         if (final_algo_choice != 13 && final_algo_choice != 11 && final_algo_choice != 12) {
         for (auto k = 0; k < K; ++k)
         {
            if (k < cur_result.size())
            {
               // cur_result[k].id 无论来源是ACORN还是UNG，都统一是 "新ID".在这里一次性、正确地转换为 "原始ID"
               const auto result_new_id = cur_result[k].id;
               if (static_cast<size_t>(result_new_id) < _new_to_old_vec_ids.size())
               {
                  results[id * K + k].first = _new_to_old_vec_ids[result_new_id];
                  results[id * K + k].second = cur_result[k].distance;
               }
               else
               {
                  results[id * K + k].first = -1;
               }
            }
            else
            {
               results[id * K + k].first = -1;
            }
         }
         }

      double pure_search_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - total_search_start_time).count();
      if (routing_mode == 5 || routing_mode == 7) {
            stats.time_ms = pure_search_time + stats.mask_gen_time_ms + stats.route_pred_time_ms + stats.global_sort_time_ms;
      } else {
            stats.time_ms = pure_search_time;
      }
         search_cache_list.release_cache(search_cache);
    }
   
   // fxy_add
   void UniNavGraph::search_hybrid(std::shared_ptr<IStorage> &query_storage,
                                   std::shared_ptr<DistanceHandler> &distance_handler,
                                   uint32_t num_threads, IdxType Lsearch,
                                   IdxType num_entry_points, std::string scenario,
                                   IdxType K, std::pair<IdxType, float> *results,
                                   std::vector<float> &num_cmps,
                                   std::vector<QueryStats> &query_stats,
                                   bool is_new_trie_method, bool is_rec_more_start,
                                   bool is_ung_more_entry,
                                   int lsearch_start, int lsearch_step,
                                   int efs_start, int efs_step_slow,int efs_step_fast,int lsearch_threshold, 
                                 int routing_mode,int baseline_alg, faiss_navix::IndexHNSWFlat* navix_index, 
                                 const std::vector<IdxType> &true_query_group_ids,
                                 const std::vector<int>& query_algo_choices,
                                 std::queue<int> task_queue,bool optimize_standalone_prefilter)
   {
      auto num_queries = query_storage->get_num_points();
      _query_storage = query_storage;
      _distance_handler = distance_handler;
      _scenario = scenario;
      query_stats.resize(num_queries);

      if (K > Lsearch) {
         std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
         exit(-1);
      }

      // 在线程池外部、主线程中预先分配好全局的 SearchCache 资源池，大小等于线程数
      SearchCacheList global_search_cache_list(num_threads, _num_points, Lsearch);

      std::queue<int> Qid_595 = std::move(task_queue); 

      if (!_thread_pool) {
         _thread_pool = std::make_unique<ThreadPool>(num_threads);
      }

      std::vector<std::future<int>> tp_results;
      for (auto i = 0; i < (int)num_queries; ++i) {
         
         int target_id = Qid_595.front();
         Qid_595.pop();
         tp_results.emplace_back(
               _thread_pool->enqueue([this, target_id, &global_search_cache_list, &query_storage, &distance_handler, &num_threads, &Lsearch, &num_entry_points, &scenario, &K, results, &num_cmps, &query_stats,
                                 &is_new_trie_method, &is_rec_more_start, &is_ung_more_entry, &lsearch_start, &lsearch_step,
                                 &efs_start, &efs_step_slow, &efs_step_fast, &lsearch_threshold,
                                 &routing_mode, &baseline_alg, &num_queries, &navix_index, &true_query_group_ids, &query_algo_choices,&optimize_standalone_prefilter] { 
                  this->thread_function(target_id, global_search_cache_list, query_storage, distance_handler, num_threads, Lsearch, num_entry_points, scenario, K, results, num_cmps, query_stats,
                                 is_new_trie_method, is_rec_more_start, is_ung_more_entry, lsearch_start, lsearch_step,
                                 efs_start, efs_step_slow, efs_step_fast, lsearch_threshold,
                                 routing_mode, baseline_alg, num_queries, navix_index, true_query_group_ids, query_algo_choices,optimize_standalone_prefilter);
                  return 1;
                }));   
      }
      for (auto &&tp_result : tp_results) tp_result.get(); 
   }

   // fxy_add
   std::vector<int> UniNavGraph::global_predict_algo_choices(
      std::shared_ptr<IStorage> query_storage,
      int routing_mode,
      const std::vector<int>& csv_choices,
      uint32_t num_threads,
      std::vector<QueryStats>& out_global_stats) 
   {
      size_t num_queries = query_storage->get_num_points();
      std::vector<int> final_choices = csv_choices;
      if (final_choices.size() < num_queries) final_choices.resize(num_queries, -1);

      if (routing_mode != 5 && routing_mode != 7) return final_choices;

      std::cout << "\n[SODA" << (routing_mode == 7 ? "+++" : "+") << "] Starting Global Prediction Phase..." << std::endl;
         
      std::vector<int> target_ids;
      std::vector<int> id_to_batch_idx(num_queries, -1); 
      
      // 找出所有需要预测的 query
      for (int id = 0; id < (int)num_queries; ++id) {
         if (final_choices[id] == -1) {
               id_to_batch_idx[id] = target_ids.size();
               target_ids.push_back(id);
         }
      }
      
      // 提前为 batch 分配好整块内存 (直接给 ONNX 用)
      std::vector<std::vector<float>> batch_features(target_ids.size());

            // -------------------------------------------------------------
      // [阶段 1]：并行提取特征
      // -------------------------------------------------------------
      omp_set_num_threads(num_threads);
      #pragma omp parallel for schedule(static)
      for (int id = 0; id < (int)num_queries; ++id) {
         auto &stats = out_global_stats[id];
         const auto& query_labels = query_storage->get_label_set(id);

         auto mask_start = std::chrono::high_resolution_clock::now();
         size_t exact_cand_size = 0;
         size_t num_descendants = 0; // 新增：Fpass 初始化

         if (query_labels.empty()) {
               exact_cand_size = _num_points;
               num_descendants = _num_groups; // 新增：空标签覆盖全组
         } else {
               // 1. 计算 exact_cand_size (基于 _vec_attr_roaring_inv)
               bool is_valid = true;
               std::vector<const roaring::Roaring*> valid_rb;
               valid_rb.reserve(query_labels.size()); 
               
               for (auto label : query_labels) {
                  auto it = _attr_to_id.find(label);
                  if (it != _attr_to_id.end()) {
                     valid_rb.push_back(&_vec_attr_roaring_inv[it->second]);
                  } else {
                     is_valid = false; 
                     break;            
                  }
               }
               
               if (!is_valid) {
                  exact_cand_size = 0; 
               } else if (!valid_rb.empty()) {
                  std::sort(valid_rb.begin(), valid_rb.end(), [](const roaring::Roaring* a, const roaring::Roaring* b){
                     return a->cardinality() < b->cardinality();
                  });
                     
                  if (valid_rb.size() == 1) {
                     exact_cand_size = valid_rb[0]->cardinality();
                  } else {
                     roaring::Roaring temp_roar = *valid_rb[0];
                     for (size_t i = 1; i < valid_rb.size(); ++i) {
                           if (temp_roar.isEmpty()) break;
                           temp_roar &= *valid_rb[i];
                     }
                     exact_cand_size = temp_roar.cardinality();
                  }
               }

               // 2. 计算 num_descendants (基于 _group_attr_roaring_inv)
               bool is_valid_group = true;
               std::vector<const roaring::Roaring*> valid_group_rb;
               valid_group_rb.reserve(query_labels.size());

               for (auto label : query_labels) {
                  auto it = _attr_to_id.find(label);
                  if (it != _attr_to_id.end()) {
                     valid_group_rb.push_back(&_group_attr_roaring_inv[it->second]);
                  } else {
                     is_valid_group = false; break;
                  }
               }
               
               if (is_valid_group && !valid_group_rb.empty()) {
                  std::sort(valid_group_rb.begin(), valid_group_rb.end(), [](const roaring::Roaring* a, const roaring::Roaring* b){
                     return a->cardinality() < b->cardinality();
                  });
                  if (valid_group_rb.size() == 1) {
                     num_descendants = valid_group_rb[0]->cardinality();
                  } else {
                     roaring::Roaring temp_group_roar = *valid_group_rb[0];
                     for (size_t i = 1; i < valid_group_rb.size(); ++i) {
                           if (temp_group_roar.isEmpty()) break;
                           temp_group_roar &= *valid_group_rb[i];
                     }
                     num_descendants = temp_group_roar.cardinality();
                  }
               }
         }
         
         stats.exact_cand_size = exact_cand_size;
         stats.global_p_pass = static_cast<float>(exact_cand_size) / _num_points;
         stats.num_lng_descendants = num_descendants; // 新增：将 Fpass 存入 stats
         stats.mask_gen_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - mask_start).count();

         // 如果是需要预测的 target，填入 batch_features 的对应槽位
         int batch_idx = id_to_batch_idx[id];
         if (batch_idx != -1) {
               size_t cand_size = query_labels.empty() ? 0 : get_candidate_count_for_label(query_labels.back());
               // 添加 num_descendants 到批次特征，严格对齐 Python 顺序
               batch_features[batch_idx] = {
                   stats.global_p_pass, 
                   static_cast<float>(num_descendants), 
                   (float)query_labels.size(), 
                   (float)cand_size
               };
         }
      }

      // -------------------------------------------------------------
      // [阶段 2]：单次调用 Batch 推理
      // -------------------------------------------------------------
      auto pred_start = std::chrono::high_resolution_clock::now();
         if (!batch_features.empty()) {
         if (_smart_route_selector) {
            // The SODA selector is trained with three features. Keep the
            // fourth CandSize value in batch_features for the legacy
            // FastSmartRoute fallback, but omit it for this selector.
            std::vector<std::vector<float>> smart_batch_features;
            smart_batch_features.reserve(batch_features.size());
            for (const auto& row : batch_features) {
               smart_batch_features.emplace_back(row.begin(), row.begin() + std::min<size_t>(3, row.size()));
            }
            std::vector<float> batch_preds = _smart_route_selector->predict_batch(smart_batch_features);

            // 顺序与 target_ids 对应，直接回写
            for (size_t i = 0; i < target_ids.size(); ++i) {
                  int id = target_ids[i];
                  int router_decision = std::round(batch_preds[i]);

                  if (router_decision == 0) final_choices[id] = _smart_route_target_alg_id;
                  else if (router_decision == 2) final_choices[id] = 5;
                  else final_choices[id] = 15;
            }
         } else if (_fast_route_single_selector) {
            // SODA 模型缺失时，回退到 FastSmartRoute 单模型，避免 algo choice 保持 -1。
            std::vector<float> batch_preds = _fast_route_single_selector->predict_batch(batch_features);
            for (size_t i = 0; i < target_ids.size(); ++i) {
                  int id = target_ids[i];
                  int router_decision = std::round(batch_preds[i]);

                  if (router_decision == 2) final_choices[id] = 5;
                  else if (router_decision == 1) final_choices[id] = _single_majority_acorn_id;
                  else final_choices[id] = 8;
            }
            std::cout << "[SODA+] SODA model is unavailable, fallback to FastSmartRoute selector." << std::endl;
         } else {
            for (int id : target_ids) {
               final_choices[id] = 8;
            }
            std::cout << "[SODA+] No routing selector is available, fallback all unresolved queries to UNG+ (choice=8)." << std::endl;
         }
      }

      // 保底：不允许未决策查询残留为 -1（否则后续执行阶段不会进入任何算法分支）。
      for (int id : target_ids) {
         if (final_choices[id] == -1) {
            final_choices[id] = 8;
         }
      }
      double total_pred_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - pred_start).count();
         
      // 平摊推理时间到每个 query 上
      double avg_pred_time = num_queries > 0 ? (total_pred_time / num_queries) : 0.0;
      for (int id = 0; id < (int)num_queries; ++id) {
         out_global_stats[id].route_pred_time_ms = avg_pred_time;
      }

      std::cout << "[SODA" << (routing_mode == 7 ? "+++" : "+") << "] Global Phase completed." << std::endl;
      return final_choices;
   }
   
   // fxy_add
   std::vector<int> UniNavGraph::get_sorted_query_ids(
      std::shared_ptr<IStorage> query_storage,
      const std::vector<int>& algo_choices,
      int routing_mode) 
   {
      size_t num_queries = query_storage->get_num_points();
      std::vector<int> sorted_ids(num_queries);
      std::iota(sorted_ids.begin(), sorted_ids.end(), 0);

      if (routing_mode != 5 && routing_mode != 7) return sorted_ids;
      
      std::vector<size_t> query_hashes(num_queries);
      #pragma omp parallel for
      for (int id = 0; id < (int)num_queries; ++id) {
         const auto& labels = query_storage->get_label_set(id);
         query_hashes[id] = std::hash<std::string>{}(std::string((char*)labels.data(), labels.size() * sizeof(ANNS::LabelType)));
      }

      std::sort(sorted_ids.begin(), sorted_ids.end(), [&](int a, int b) {
         if (algo_choices[a] != algo_choices[b]) return algo_choices[a] < algo_choices[b];
         return query_hashes[a] < query_hashes[b];
      });

      return sorted_ids;
   }

   std::vector<IdxType> UniNavGraph::get_entry_points(const std::vector<LabelType> &query_label_set,
                                                      IdxType num_entry_points, VisitedSet &visited_set)
   {
      std::vector<IdxType> entry_points;
      entry_points.reserve(num_entry_points);
      visited_set.clear();

      // obtain entry points for label-equality scenario
      if (_scenario == "equality")
      {
         auto node = _trie_index.find_exact_match(query_label_set);
         if (node == nullptr)
            return entry_points;
         get_entry_points_given_group_id(num_entry_points, visited_set, node->group_id, entry_points);

         // obtain entry points for label-containment scenario
      }
      else if (_scenario == "containment")
      {
         std::vector<IdxType> min_super_set_ids;
         get_min_super_sets(query_label_set, min_super_set_ids);
         for (auto group_id : min_super_set_ids)
            get_entry_points_given_group_id(num_entry_points, visited_set, group_id, entry_points);
      }
      else
      {
         std::cerr << "Error: invalid scenario " << _scenario << std::endl;
         exit(-1);
      }

      return entry_points;
   }

   void UniNavGraph::get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet &visited_set,
                                                     IdxType group_id, std::vector<IdxType> &entry_points)
   {
      const auto &group_range = _group_id_to_range[group_id];

      // not enough entry points, use all of them
      if (group_range.second - group_range.first <= num_entry_points)
      {
         for (auto i = 0; i < group_range.second - group_range.first; ++i)
            entry_points.emplace_back(i + group_range.first);
         return;
      }

      // add the entry point of the group
      const auto &group_entry_point = _group_entry_points[group_id];
      visited_set.set(group_entry_point);
      entry_points.emplace_back(group_entry_point);

      // randomly sample the other entry points
      for (auto i = 1; i < num_entry_points; ++i)
      {
         auto entry_point = rand() % (group_range.second - group_range.first) + group_range.first;
         if (visited_set.check(entry_point) == false)
         {
            visited_set.set(entry_point);
            entry_points.emplace_back(i + group_range.first);
         }
      }
   }

   IdxType UniNavGraph::iterate_to_fixed_point(const char *query, std::shared_ptr<SearchCache> search_cache,
                                               IdxType target_id, const std::vector<IdxType> &entry_points,
                                               size_t &num_nodes_visited,
                                               bool clear_search_queue, bool clear_visited_set)
   {
      auto dim = _base_storage->get_dim();
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      std::vector<IdxType> neighbors;
      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      // entry point
      for (const auto &entry_point : entry_points)
         search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
      IdxType num_cmps = entry_points.size();

      // greedily expand closest nodes
      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();

         // iterate neighbors
         // {
         //    std::lock_guard<std::mutex> lock(_graph->neighbor_locks[cur.id]);
         //    neighbors = _graph->neighbors[cur.id];
         // }
            neighbors = _graph->neighbors[cur.id];
         for (auto i = 0; i < neighbors.size(); ++i)
         {

            // prefetch
            if (i + 1 < neighbors.size() && visited_set.check(neighbors[i + 1]) == false)
               _base_storage->prefetch_vec_by_id(neighbors[i + 1]);

            // skip if visited
            auto &neighbor = neighbors[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);

            num_nodes_visited++; // search过程中遍历的点的个数

            // push to search queue
            search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
            num_cmps++;
         }
      }
      return num_cmps;
   }

   IdxType UniNavGraph::iterate_to_fixed_point_rabitq(const char *query, std::shared_ptr<SearchCache> search_cache,
                                                      IdxType target_id, const std::vector<IdxType> &entry_points,
                                                      size_t &num_nodes_visited,
                                                      QueryStats &stats,
                                                      ANNS::rabitq::RabitQSideIndex::QueryContext *cached_query_ctx,
                                                      bool clear_search_queue, bool clear_visited_set)
   {
      (void)target_id;
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      std::vector<IdxType> neighbors;

      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      ANNS::rabitq::RabitQSideIndex::QueryContext local_query_ctx;
      ANNS::rabitq::RabitQSideIndex::QueryContext *query_ctx = cached_query_ctx;
      if (query_ctx == nullptr)
      {
         auto init_start = std::chrono::high_resolution_clock::now();
         ANNS::rabitq::RabitQSideIndex::InitTiming init_timing;
         if (!_rabitq_side_index.init_query(query, local_query_ctx, &init_timing))
         {
            return iterate_to_fixed_point(
                query,
                search_cache,
                target_id,
                entry_points,
                num_nodes_visited,
                false,
                false
            );
         }
         stats.rabitq_ctx_prepare_time_ms += std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - init_start
         ).count();
         stats.rabitq_ctx_rotate_time_ms += init_timing.rotate_ms;
         stats.rabitq_ctx_q2c_time_ms += init_timing.q_to_centroids_ms;
         stats.rabitq_ctx_wrapper_time_ms += init_timing.wrapper_ms;
         stats.rabitq_ctx_reused = false;
         query_ctx = &local_query_ctx;
      }

      for (const auto &entry_point : entry_points)
      {
         if (visited_set.check(entry_point))
         {
            continue;
         }
         visited_set.set(entry_point);
         float low_dist = 0.0F;
         auto bin_start = std::chrono::high_resolution_clock::now();
         float est_dist = _rabitq_side_index.estimate_bin(entry_point, *query_ctx, &low_dist);
         stats.rabitq_bin_time_ms += std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - bin_start
         ).count();
         stats.rabitq_bin_calls += 1;
         (void)low_dist;
         search_queue.insert(entry_point, est_dist);
      }
      IdxType num_cmps = static_cast<IdxType>(search_queue.size());

      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();
         neighbors = _graph->neighbors[cur.id];

         for (size_t i = 0; i < neighbors.size(); ++i)
         {
            if (i + 1 < neighbors.size() && visited_set.check(neighbors[i + 1]) == false)
            {
               _base_storage->prefetch_vec_by_id(neighbors[i + 1]);
            }

            auto &neighbor = neighbors[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);
            num_nodes_visited++;

            float low_dist = 0.0F;
            auto bin_start = std::chrono::high_resolution_clock::now();
            float est_dist = _rabitq_side_index.estimate_bin(neighbor, *query_ctx, &low_dist);
            stats.rabitq_bin_time_ms += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - bin_start
            ).count();
            stats.rabitq_bin_calls += 1;
            num_cmps++;

            bool should_refine = false;
            if (search_queue.size() >= search_queue.capacity() && search_queue.size() > 0)
            {
               should_refine = low_dist < search_queue[search_queue.size() - 1].distance;
            }

            float dist_to_insert = est_dist;
            if (should_refine)
            {
               float refined_low = 0.0F;
               auto full_start = std::chrono::high_resolution_clock::now();
               float refined_dist = _rabitq_side_index.estimate_full(neighbor, *query_ctx, &refined_low);
               stats.rabitq_full_time_ms += std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - full_start
               ).count();
               stats.rabitq_full_calls += 1;
               (void)refined_low;
               dist_to_insert = refined_dist;
               num_cmps++;
            }
            search_queue.insert(neighbor, dist_to_insert);
         }
      }
      return num_cmps;
   }

   // fxy_add
   IdxType UniNavGraph::iterate_to_fixed_point_global(const char *query, std::shared_ptr<SearchCache> search_cache,
                                                      IdxType target_id, const std::vector<IdxType> &entry_points,
                                                      bool clear_search_queue, bool clear_visited_set)
   {
      auto dim = _base_storage->get_dim();
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      std::vector<IdxType> neighbors;
      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      // entry point
      for (const auto &entry_point : entry_points)
         search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
      IdxType num_cmps = entry_points.size();

      // greedily expand closest nodes
      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();

         // iterate neighbors
         {
            std::lock_guard<std::mutex> lock(_global_graph->neighbor_locks[cur.id]);
            neighbors = _global_graph->neighbors[cur.id];
         }
         for (auto i = 0; i < neighbors.size(); ++i)
         {

            // prefetch
            if (i + 1 < neighbors.size() && visited_set.check(neighbors[i + 1]) == false)
               _base_storage->prefetch_vec_by_id(neighbors[i + 1]);

            // skip if visited
            auto &neighbor = neighbors[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);

            // push to search queue
            search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
            num_cmps++;
         }
      }
      return num_cmps;
   }

   void UniNavGraph::save(std::string index_path_prefix, std::string results_path_prefix)
   {
      fs::create_directories(index_path_prefix);
      auto start_time = std::chrono::high_resolution_clock::now();

      // save meta data
      std::map<std::string, std::string> meta_data;
      statistics();
      meta_data["num_points"] = std::to_string(_num_points);
      meta_data["num_groups"] = std::to_string(_num_groups);
      meta_data["index_name"] = _index_name;
      meta_data["max_degree"] = std::to_string(_max_degree);
      meta_data["Lbuild"] = std::to_string(_Lbuild);
      meta_data["alpha"] = std::to_string(_alpha);
      meta_data["build_num_threads"] = std::to_string(_num_threads);
      meta_data["scenario"] = _scenario;
      meta_data["num_cross_edges"] = std::to_string(_num_cross_edges);
      meta_data["graph_num_edges"] = std::to_string(_graph_num_edges);
      meta_data["LNG_num_edges"] = std::to_string(_LNG_num_edges);
      meta_data["index_size(MB)"] = std::to_string(_index_size);
      meta_data["_index_size_add_rb(MB)"] = std::to_string(_index_size_add_rb);
      meta_data["index_time(ms)"] = std::to_string(_index_time - _build_roaring_bitsets_time);
      meta_data["index_time_add_rb(ms)"] = std::to_string(_index_time);
      meta_data["label_processing_time(ms)"] = std::to_string(_label_processing_time);
      meta_data["build_graph_time(ms)"] = std::to_string(_build_graph_time);
      meta_data["build_vector_attr_graph_time(ms)"] = std::to_string(_build_vector_attr_graph_time);
      meta_data["cal_descendants_time(ms)"] = std::to_string(_cal_descendants_time);
      meta_data["cal_coverage_ratio_time(ms)"] = std::to_string(_cal_coverage_ratio_time);
      meta_data["build_LNG_time(ms)"] = std::to_string(_build_LNG_time);
      meta_data["build_cross_edges_time(ms)"] = std::to_string(_build_cross_edges_time);
      // FXY_ADD: 保存详细的跨组边构建时间到 meta 文件
      meta_data["cross_edge_step1_time(ms)"] = std::to_string(_cross_edge_step1_time_ms);
      meta_data["cross_edge_step2_acorn_time(ms)"] = std::to_string(_cross_edge_step2_acorn_time_ms);
      meta_data["cross_edge_step3_add_dist_edges_time(ms)"] = std::to_string(_cross_edge_step3_add_dist_edges_time_ms);
      meta_data["cross_edge_step4_add_hierarchy_edges_time(ms)"] = std::to_string(_cross_edge_step4_add_hierarchy_edges_time_ms);
      meta_data["rabitq_build_requested"] = _build_rabitq_side_index ? "1" : "0";
      meta_data["rabitq_enabled"] = _rabitq_side_index.enabled() ? "1" : "0";
      meta_data["rabitq_total_bits"] = std::to_string(_rabitq_side_index.total_bits());
      meta_data["rabitq_build_time(ms)"] = std::to_string(_rabitq_build_time_ms);
      meta_data["rabitq_side_size_bytes"] = "0";
      meta_data["favor_enabled"] = _favor_ready ? "1" : "0";
      meta_data["favor_build_time(ms)"] = std::to_string(_favor_build_time_ms);
      meta_data["favor_serialized_size_bytes"] = std::to_string(_favor_serialized_index_size_bytes);
      meta_data["index_time_without_rabitq(ms)"] = std::to_string(std::max(0.0, static_cast<double>(_index_time) - _rabitq_build_time_ms));
      meta_data["index_time_with_rabitq(ms)"] = std::to_string(_index_time);
      meta_data["index_size_without_rabitq(MB)"] = std::to_string(_index_size_add_rb);
      meta_data["index_size_with_rabitq(MB)"] = std::to_string(_index_size_add_rb);

      // FXY_ADD: 计算并保存 Trie 静态指标
      std::cout << "Calculating and saving Trie static metrics..." << std::endl;
      TrieStaticMetrics trie_metrics = _trie_index.calculate_static_metrics();
      meta_data["trie_label_cardinality"] = std::to_string(trie_metrics.label_cardinality);
      meta_data["trie_total_nodes"] = std::to_string(trie_metrics.total_nodes);
      meta_data["trie_avg_path_length"] = std::to_string(trie_metrics.avg_path_length);
      meta_data["trie_avg_branching_factor"] = std::to_string(trie_metrics.avg_branching_factor);
      // Save the detailed label frequency distribution to its own file for easier analysis
      std::string trie_freq_filename = results_path_prefix + "trie_label_frequency.csv";
      std::ofstream freq_file(trie_freq_filename);
      freq_file << "LabelID,Frequency\n";
      for (const auto &pair : trie_metrics.label_frequency)
         freq_file << pair.first << "," << pair.second << "\n";
      freq_file.close();
      std::cout << "- Trie label frequency distribution saved to " << trie_freq_filename << std::endl;

      std::string meta_filename = index_path_prefix + "meta";
      write_kv_file(meta_filename, meta_data);

      // save build_time to csv
      std::string build_time_filename = results_path_prefix + "build_time.csv";
      std::ofstream build_time_file(build_time_filename);
      build_time_file << "Index Name,Build Time (ms)\n";
      build_time_file << "index_time" << "," << _index_time << "\n";
      build_time_file << "label_processing_time" << "," << _label_processing_time << "\n";
      build_time_file << "build_graph_time" << "," << _build_graph_time << "\n";
      build_time_file << "build_vector_attr_graph_time" << "," << _build_vector_attr_graph_time << "\n";
      build_time_file << "cal_descendants_time" << "," << _cal_descendants_time << "\n";
      build_time_file << "cal_coverage_ratio_time" << "," << _cal_coverage_ratio_time << "\n";
      build_time_file << "build_LNG_time" << "," << _build_LNG_time << "\n";
      build_time_file << "build_cross_edges_time" << "," << _build_cross_edges_time << "\n";
      // FXY_ADD: 保存详细的跨组边构建时间到 CSV 文件
      build_time_file << "cross_edge_step1_time" << "," << _cross_edge_step1_time_ms << "\n";
      build_time_file << "cross_edge_step2_acorn_time" << "," << _cross_edge_step2_acorn_time_ms << "\n";
      build_time_file << "cross_edge_step3_add_dist_edges_time" << "," << _cross_edge_step3_add_dist_edges_time_ms << "\n";
      build_time_file << "cross_edge_step4_add_hierarchy_edges_time" << "," << _cross_edge_step4_add_hierarchy_edges_time_ms << "\n";
      build_time_file << "rabitq_build_time" << "," << _rabitq_build_time_ms << "\n";
      build_time_file << "favor_build_time" << "," << _favor_build_time_ms << "\n";
      build_time_file << "favor_serialized_size_bytes" << "," << _favor_serialized_index_size_bytes << "\n";
      build_time_file.close();

      // save vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage->write_to_file(bin_file, label_file);

      // save group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      write_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // save group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      write_2d_vectors(group_id_to_range_filename, _group_id_to_range);

      // save group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      write_1d_vector(group_entry_points_filename, _group_entry_points);

      // save group id to vec ids
      std::string group_id_to_vec_ids_filename = index_path_prefix + "group_id_to_vec_ids.dat";
      write_2d_vectors(group_id_to_vec_ids_filename, _group_id_to_vec_ids);
      std::cout << "group_id_to_vec_ids saved." << std::endl;

      // save new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      write_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

      // save trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.save(trie_filename);

      // save graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph->save(graph_filename);

      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph->save(global_graph_filename);

      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      write_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      // save LNG coverage ratio
      std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
      write_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);

      // save covered_sets in LNG
      std::string covered_sets_filename = index_path_prefix + "covered_sets";
      write_2d_vectors(covered_sets_filename, _label_nav_graph->covered_sets);
      std::cout << "LNG covered_sets saved." << std::endl;

      // save LNG descendant num
      std::string lng_descendants_num_filename = index_path_prefix + "lng_descendants_num";
      write_1d_pair_vector(lng_descendants_num_filename, _label_nav_graph->_lng_descendants_num);

      // save LNG descendants
      std::string lng_descendants_filename = index_path_prefix + "lng_descendants";
      write_2d_vectors(lng_descendants_filename, _label_nav_graph->_lng_descendants);

      // 保存 LNG 的核心图结构 (邻接表)
      std::string lng_out_neighbors_filename = index_path_prefix + "lng_out_neighbors.dat";
      write_2d_vectors(lng_out_neighbors_filename, _label_nav_graph->out_neighbors);
      std::cout << "LNG out_neighbors saved." << std::endl; // 增加一个打印，确认保存成功

      // save vector attr graph data
      std::string vector_attr_graph_filename = index_path_prefix + "vector_attr_graph";
      save_bipartite_graph(vector_attr_graph_filename);

      // // save _lng_descendants_bits and _covered_sets_bits
      // std::string lng_descendants_bits_filename = index_path_prefix + "lng_descendants_bits";
      // write_bitset_vector(lng_descendants_bits_filename, _lng_descendants_bits);
      // std::string covered_sets_bits_filename = index_path_prefix + "covered_sets_bits";
      // write_bitset_vector(covered_sets_bits_filename, _covered_sets_bits);

      // save lng_descendants_rb and _covered_sets_rb
      std::string lng_descendants_rb_filename = index_path_prefix + "lng_descendants_rb.bin";
      save_roaring_vector(lng_descendants_rb_filename, _lng_descendants_rb);
      std::string covered_sets_rb_filename = index_path_prefix + "covered_sets_rb.bin";
      save_roaring_vector(covered_sets_rb_filename, _covered_sets_rb);

      if (_rabitq_side_index.enabled())
      {
         const std::string rabitq_side_filename = index_path_prefix + "rabitq_side.bin";
         if (!_rabitq_side_index.save(rabitq_side_filename))
         {
            std::cerr << "[RabitQ] Failed to save side index to " << rabitq_side_filename << std::endl;
         }
         else
         {
            // Use an in-memory estimate (sizeof-based) instead of on-disk file size.
            _rabitq_side_size_bytes = _rabitq_side_index.estimated_memory_bytes(false);
         }
      }
      else
      {
         _rabitq_side_size_bytes = 0;
      }

      // save acorn index
      std::cout << "\n--- Exporting reordered data for ACORN index building ---" << std::endl;

      // 1. 导出重排后的向量
      std::string reordered_vec_path = index_path_prefix + "reordered_vecs.fvecs";
      std::cout << "- Saving reordered vectors to: " << reordered_vec_path << std::endl;
      try
      {
         uint32_t dim = static_cast<uint32_t>(_base_storage->get_dim());
         std::ofstream out(reordered_vec_path, std::ios::binary);
         if (!out)
         {
            throw std::runtime_error("Cannot open file for writing: " + reordered_vec_path);
         }

         // 遍历所有已重排的向量并按 .fvecs 格式写入
         for (ANNS::IdxType new_id = 0; new_id < _num_points; ++new_id)
         {
            const float *vec_data = reinterpret_cast<const float *>(_base_storage->get_vector(new_id));
            // 先写入维度
            out.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
            // 再写入向量数据
            out.write(reinterpret_cast<const char *>(vec_data), dim * sizeof(float));
         }
         out.close();
         std::cout << "- Reordered vectors saved successfully." << std::endl;
      }
      catch (const std::exception &e)
      {
         std::cerr << "Error while saving reordered vectors: " << e.what() << std::endl;
      }

      // 2. 导出与重排后向量顺序完全一致的标签
      std::string reordered_label_path = index_path_prefix + "reordered_labels.txt";
      std::ofstream reordered_label_file(reordered_label_path);
      if (!reordered_label_file.is_open())
      {
         std::cerr << "Error: Could not open file to save reordered labels: " << reordered_label_path << std::endl;
      }
      else
      {
         // 直接按新顺序 (0 to _num_points-1) 遍历已重排的 _base_storage
         for (ANNS::IdxType new_id = 0; new_id < _num_points; ++new_id)
         {
            const auto &label_set = _base_storage->get_label_set(new_id);
            for (size_t i = 0; i < label_set.size(); ++i)
            {
               reordered_label_file << label_set[i] << (i == label_set.size() - 1 ? "" : ",");
            }
            reordered_label_file << "\n";
         }
         reordered_label_file.close();
         std::cout << "- Reordered labels saved to: " << reordered_label_path << std::endl;
      }
      std::cout << "--- Finished exporting reordered data ---\n"
                << std::endl;
      // ========================== 修改部分结束 ==========================

      const double rabitq_side_size_mb = static_cast<double>(_rabitq_side_size_bytes) / (1024.0 * 1024.0);
      meta_data["rabitq_side_size_bytes"] = std::to_string(_rabitq_side_size_bytes);
      meta_data["index_size_with_rabitq(MB)"] = std::to_string(_index_size_add_rb + rabitq_side_size_mb);
      write_kv_file(meta_filename, meta_data);

      std::ofstream build_time_append_file(build_time_filename, std::ios::app);
      build_time_append_file << "index_time_without_rabitq" << "," << std::max(0.0, static_cast<double>(_index_time) - _rabitq_build_time_ms) << "\n";
      build_time_append_file << "index_time_with_rabitq" << "," << _index_time << "\n";
      build_time_append_file << "rabitq_side_size_bytes" << "," << _rabitq_side_size_bytes << "\n";
      build_time_append_file << "index_size_without_rabitq_MB" << "," << _index_size_add_rb << "\n";
      build_time_append_file << "index_size_with_rabitq_MB" << "," << (_index_size_add_rb + rabitq_side_size_mb) << "\n";
      build_time_append_file.close();

      // print
      std::cout << "- Index saved in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   std::unordered_map<int, std::vector<int>> load_inverted_index(const std::string &input_path)
   {
      std::ifstream in(input_path, std::ios::binary);
      if (!in)
      {
         fprintf(stderr, "Error: Cannot open file for reading inverted index: %s\n", input_path.c_str());
         exit(1);
      }

      std::unordered_map<int, std::vector<int>> inverted_index;

      // 读取文件头：属性总数
      uint64_t map_size;
      in.read(reinterpret_cast<char *>(&map_size), sizeof(map_size));
      inverted_index.reserve(map_size);

      // 依次读取每个属性及其向量列表
      for (uint64_t i = 0; i < map_size; ++i)
      {
         int attr_id;
         uint64_t list_size;

         in.read(reinterpret_cast<char *>(&attr_id), sizeof(attr_id));
         in.read(reinterpret_cast<char *>(&list_size), sizeof(list_size));

         std::vector<int> vec_list(list_size);
         in.read(reinterpret_cast<char *>(vec_list.data()), list_size * sizeof(int));

         inverted_index[attr_id] = std::move(vec_list);
      }

      in.close();
      printf("Inverted index loaded successfully.\n");
      return inverted_index;
   }

   void UniNavGraph::load(std::string index_path_prefix, std::string selector_modle_prefix, const std::string &data_type,
                          const std::string &acorn_index_path, const std::string &acorn_1_index_path,
                          const std::string &dataset, int routing_mode, int baseline_alg)
   {
      std::cout << "Loading index from " << index_path_prefix << " ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _dataset = dataset;

      // load meta data
      std::string meta_filename = index_path_prefix + "meta";
      auto meta_data = parse_kv_file(meta_filename);
      _num_points = std::stoi(meta_data["num_points"]);
      _num_groups = std::stoi(meta_data["num_groups"]);
      _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);

      // load vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage = create_storage(data_type, false);
      _base_storage->load_from_file(bin_file, label_file);

      fs::path index_files_dir(index_path_prefix);
      while (!index_files_dir.empty() &&
             (index_files_dir.filename().string().empty() || index_files_dir.filename() == ".")) {
         index_files_dir = index_files_dir.parent_path();
      }
      fs::path index_root_dir = index_files_dir;
      if (index_root_dir.filename() == "index_files") {
         index_root_dir = index_root_dir.parent_path();
      } else {
         index_root_dir = index_root_dir.parent_path();
      }
      const std::string build_time_filename = (index_root_dir / "others" / "build_time.csv").string();

      // load group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      load_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // load group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      load_2d_vectors(group_id_to_range_filename, _group_id_to_range);

      // load group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      load_1d_vector(group_entry_points_filename, _group_entry_points);

      // load group id to vec ids
      std::string group_id_to_vec_ids_filename = index_path_prefix + "group_id_to_vec_ids.dat";
      load_2d_vectors(group_id_to_vec_ids_filename, _group_id_to_vec_ids);
      std::cout << "group_id_to_vec_ids loaded." << std::endl;

      // load new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      load_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

      // fxy_add: load old to new vec ids
      std::cout << "- Building reverse ID map (original_id -> new_id)..." << std::endl;
      _old_to_new_vec_ids.resize(_num_points);
      for (IdxType new_id = 0; new_id < _num_points; ++new_id)
      {
         IdxType old_id = _new_to_old_vec_ids[new_id];
         _old_to_new_vec_ids[old_id] = new_id;
      }
      std::cout << "- Reverse ID map built successfully." << std::endl;

      // load trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.load(trie_filename);
      // --- 预计算并缓存 Trie 静态指标 ---
      std::cout << "Pre-calculating and caching Trie static metrics..." << std::endl;
      _trie_static_metrics = _trie_index.calculate_static_metrics();
      std::cout << "- Caching complete." << std::endl;

      // load graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _graph->load(graph_filename);

      // fxy_add:load global graph data
      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _global_graph->load(global_graph_filename);

      // fxy_add: load global vamana entry point
      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      load_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      // fxy_add: load LNG coverage ratio
      std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
      std::cout << "_label_nav_graph->coverage_ratio size: " << _label_nav_graph->coverage_ratio.size() << std::endl;
      load_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);
      std::cout << "LNG coverage ratio loaded." << std::endl;

      // fxy_add: load LNG descendants num
      std::string lng_descendants_num_filename = index_path_prefix + "lng_descendants_num";
      load_1d_pair_vector(lng_descendants_num_filename, _label_nav_graph->_lng_descendants_num);
      std::cout << "LNG descendants num loaded." << std::endl;

      // fxy_add: load LNG descendants
      std::string lng_descendants_filename = index_path_prefix + "lng_descendants";
      load_2d_vectors(lng_descendants_filename, _label_nav_graph->_lng_descendants);

      // fxy_add: load covered_sets in LNG
      std::string covered_sets_filename = index_path_prefix + "covered_sets";
      load_2d_vectors(covered_sets_filename, _label_nav_graph->covered_sets);
      std::cout << "LNG covered_sets loaded." << std::endl;

      // fxy_add: load LNG out_neighbors
      std::string lng_out_neighbors_filename = index_path_prefix + "lng_out_neighbors.dat";
      load_2d_vectors(lng_out_neighbors_filename, _label_nav_graph->out_neighbors);
      std::cout << "LNG out_neighbors loaded." << std::endl;

      // // fxy_add: load  _lng_descendants_bits and _covered_sets_bits
      // std::string lng_descendants_bits_filename = index_path_prefix + "lng_descendants_bits";
      // load_bitset_vector(lng_descendants_bits_filename, _lng_descendants_bits);
      // std::cout << "_lng_descendants_bits loaded." << std::endl;
      // std::string covered_sets_bits_filename = index_path_prefix + "covered_sets_bits";
      // load_bitset_vector(covered_sets_bits_filename, _covered_sets_bits);
      // std::cout << "_covered_sets_bits loaded." << std::endl;

      // fxy_add: load lng_descendants_rb and _covered_sets_rb
      std::string lng_descendants_rb_filename = index_path_prefix + "lng_descendants_rb.bin";
      load_roaring_vector(lng_descendants_rb_filename, _lng_descendants_rb);
      std::cout << "_lng_descendants_rb loaded." << std::endl;
      std::string covered_sets_rb_filename = index_path_prefix + "covered_sets_rb.bin";
      load_roaring_vector(covered_sets_rb_filename, _covered_sets_rb);
      std::cout << "_covered_sets_rb loaded." << std::endl;
      std::cout << " _label_nav_graph->out_neighbors.size() = " << _label_nav_graph->out_neighbors.size() << std::endl;
      std::cout << " _num_groups = " << _num_groups << std::endl;

      // load vector-attribute bipartite graph and prepare vector-level inverted index for containment filtering
      std::string vector_attr_graph_filename = index_path_prefix + "vector_attr_graph";
      if (fs::exists(vector_attr_graph_filename))
      {
         load_bipartite_graph(vector_attr_graph_filename);
         build_vector_inverted_indices();
      }
      else
      {
         std::cerr << "[Warning] vector_attr_graph not found: " << vector_attr_graph_filename
                   << ". Containment filter acceleration may be unavailable." << std::endl;
      }

      const bool is_smartroute_mode =
          (routing_mode == 1 || routing_mode == 2 || routing_mode == 3 ||
           routing_mode == 4 || routing_mode == 5 || routing_mode == 6 ||
           routing_mode == 7);
      std::optional<int> smart_route_target_alg_from_config = load_router_target_algo_id_from_env();
      if (!smart_route_target_alg_from_config.has_value()) {
         const auto smart_route_model_dir = resolve_smart_route_model_dir(selector_modle_prefix);
         if (smart_route_model_dir.has_value()) {
            const fs::path class_labels_path = smart_route_model_dir.value() / "class_labels.txt";
            smart_route_target_alg_from_config = load_router_target_algo_id_from_class_labels(class_labels_path);
         }
      }
      const bool need_milvus_ivf_baseline = (routing_mode == 0 && baseline_alg == 9);
      const bool need_milvus_hnsw_baseline = (routing_mode == 0 && baseline_alg == 10);
      const bool need_acorn_baseline =
          is_smartroute_mode ||
          (routing_mode == 0 && (baseline_alg == 2 || baseline_alg == 3 ||
                                 baseline_alg == 4 || baseline_alg == 6));

#ifdef ENABLE_KNOWHERE_MILVUS_BASELINE
      _milvus_knowhere_ready = false;
      _milvus_knowhere_hnsw_ready = false;
      _milvus_knowhere_is_ivf = true;
      if ((need_milvus_ivf_baseline || need_milvus_hnsw_baseline) &&
          _base_storage != nullptr && _base_storage->get_num_points() > 0)
      {
         try
         {
            // Stabilize Knowhere/Faiss runtime behavior for in-process baseline builds.
            // We keep inner thread pools conservative and let outer experiment parallelism dominate.
            static std::once_flag kh_pool_init_once;
            std::call_once(kh_pool_init_once, []() {
               knowhere::KnowhereConfig::SetBuildThreadPoolSize(1);
               knowhere::KnowhereConfig::SetSearchThreadPoolSize(1);
            });

            const int64_t rows = static_cast<int64_t>(_base_storage->get_num_points());
            const int64_t dim = static_cast<int64_t>(_base_storage->get_dim());
            const float *base_ptr = reinterpret_cast<const float *>(_base_storage->get_vector(0));
            const int kh_add_threads = std::max<int>(1, std::min<int>(64, static_cast<int>(_num_threads > 0 ? _num_threads : 32)));
            fs::path index_files_dir(index_path_prefix);
            while (!index_files_dir.empty() &&
                   (index_files_dir.filename().string().empty() || index_files_dir.filename() == ".")) {
               index_files_dir = index_files_dir.parent_path();
            }
            fs::path index_root_dir = index_files_dir;
            if (index_root_dir.filename() == "index_files") {
               index_root_dir = index_root_dir.parent_path();
            } else {
               index_root_dir = index_root_dir.parent_path();
            }
            // Keep Milvus caches under the concrete index configuration directory:
            // .../Index/<INDEX_DIR_NAME>/Milvus-IVF and .../Milvus-HNSW
            const std::string milvus_ivf_cache_dir = (index_root_dir / "Milvus-IVF").string() + "/";
            const std::string milvus_hnsw_cache_dir = (index_root_dir / "Milvus-HNSW").string() + "/";
            const std::string milvus_cache_index_file = milvus_ivf_cache_dir + "milvus_knowhere.index";
            const std::string milvus_cache_meta_file = milvus_ivf_cache_dir + "milvus_knowhere.meta";
            const std::string milvus_cache_hnsw_index_file = milvus_hnsw_cache_dir + "milvus_knowhere_hnsw.index";
            const std::string milvus_cache_hnsw_meta_file = milvus_hnsw_cache_dir + "milvus_knowhere_hnsw.meta";
            // IVF training in Knowhere/Faiss can touch shared BLAS/Faiss state during clustering.
            // Keep centroid training deterministic and single-threaded; full-vector Add/Search still
            // remain controlled by the outer experiment's num_threads/query-level parallelism.
            const int kh_train_threads = 1;
            if (base_ptr == nullptr)
            {
               std::cerr << "[Milvus-IVF-Knowhere] base vector pointer is null, skip index build." << std::endl;
            }
            else
            {
               MilvusKnowhereCacheMeta cache_meta;
               if (need_milvus_ivf_baseline &&
                   fs::exists(milvus_cache_index_file) &&
                   fs::exists(milvus_cache_meta_file) &&
                   load_milvus_knowhere_meta(milvus_cache_meta_file, cache_meta) &&
                   cache_meta.rows == rows &&
                   cache_meta.dim == dim)
               {
                  const bool cache_is_ivf = (cache_meta.index_kind == "IVF_FLAT");
                  const auto cache_index_enum = cache_is_ivf ? knowhere::IndexEnum::INDEX_FAISS_IVFFLAT
                                                             : knowhere::IndexEnum::INDEX_FAISS_IDMAP;
                  auto cached_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                      cache_index_enum,
                      knowhere::Version::GetCurrentVersion().VersionNumber());
                  if (cached_expected.has_value())
                  {
                     knowhere::Json load_json;
                     load_json[knowhere::meta::DIM] = dim;
                     load_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
                     if (cache_is_ivf) {
                        load_json[knowhere::indexparam::NLIST] = std::max(1, cache_meta.nlist);
                        load_json[knowhere::indexparam::NPROBE] = std::max(1, cache_meta.nprobe);
                     }
                     _milvus_knowhere_index = cached_expected.value();
                     const auto load_status = _milvus_knowhere_index.DeserializeFromFile(milvus_cache_index_file, load_json);
                     if (load_status == knowhere::Status::success)
                     {
                        _milvus_knowhere_ready = true;
                        _milvus_knowhere_is_ivf = cache_is_ivf;
                        _milvus_knowhere_nlist = std::max(1, cache_meta.nlist);
                        _milvus_knowhere_nprobe = std::max(1, cache_meta.nprobe);
                        std::cout << "[Milvus-IVF-Knowhere] loaded cached "
                                  << (cache_is_ivf ? "IVF" : "FLAT")
                                  << " index from " << milvus_cache_index_file << std::endl;
                        if (cache_meta.build_time_ms >= 0.0) {
                           std::cout << "[Milvus-IVF-Knowhere] cached build stats: total="
                                     << cache_meta.build_time_ms << " ms";
                           if (cache_meta.train_time_ms >= 0.0) {
                              std::cout << ", train=" << cache_meta.train_time_ms << " ms";
                           }
                           if (cache_meta.add_time_ms >= 0.0) {
                              std::cout << ", add=" << cache_meta.add_time_ms << " ms";
                           }
                           std::cout << std::endl;
                        }
                        if (cache_meta.index_sizeof_mb > 0.0 || cache_meta.serialized_index_size_mb > 0.0) {
                           std::cout << "[Milvus-IVF-Knowhere] cached size stats: sizeof="
                                     << cache_meta.index_sizeof_mb
                                     << " MB, serialized="
                                     << cache_meta.serialized_index_size_mb
                                     << " MB" << std::endl;
                        }
                     }
                     else
                     {
                        std::cerr << "[Milvus-IVF-Knowhere] cached index load failed, status="
                                  << static_cast<int>(load_status)
                                  << ". Rebuilding from base vectors." << std::endl;
                     }
                  }
               }

               if (need_milvus_ivf_baseline && !_milvus_knowhere_ready)
               {
                  auto idx_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                      knowhere::IndexEnum::INDEX_FAISS_IVFFLAT,
                      knowhere::Version::GetCurrentVersion().VersionNumber());
                  if (!idx_expected.has_value())
                  {
                     std::cerr << "[Milvus-IVF-Knowhere] failed to create INDEX_FAISS_IVFFLAT instance." << std::endl;
                  }
                  else
                  {
                     _milvus_knowhere_index = idx_expected.value();
                     _milvus_knowhere_nlist = std::max<int>(1, std::min<int>(128, static_cast<int>(std::sqrt(static_cast<double>(rows)))));
                     _milvus_knowhere_nprobe = std::max<int>(1, std::min<int>(16, _milvus_knowhere_nlist));

                     knowhere::Json build_json;
                     build_json[knowhere::meta::DIM] = dim;
                     build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
                     build_json[knowhere::indexparam::NLIST] = _milvus_knowhere_nlist;
                     build_json[knowhere::indexparam::NPROBE] = _milvus_knowhere_nprobe;
                     build_json["use_elkan"] = false;

                     knowhere::Json train_json = build_json;
                     train_json[knowhere::meta::NUM_BUILD_THREAD] = kh_train_threads;
                     knowhere::Json add_json = build_json;
                     add_json[knowhere::meta::NUM_BUILD_THREAD] = kh_add_threads;

                     const int64_t train_rows = std::min<int64_t>(rows, 30000);
                     std::unique_ptr<float[]> owned_train(new float[static_cast<size_t>(train_rows) * static_cast<size_t>(dim)]);
                     std::memcpy(owned_train.get(), base_ptr, sizeof(float) * static_cast<size_t>(train_rows) * static_cast<size_t>(dim));
                     auto train_ds = knowhere::GenResultDataSet(train_rows, dim, std::move(owned_train));
                     const auto ivf_build_begin = std::chrono::high_resolution_clock::now();
                     double ivf_train_time_ms = -1.0;
                     double ivf_add_time_ms = -1.0;
                     double ivf_effective_build_time_ms = -1.0;
                     std::cout << "[Milvus-IVF-Knowhere] building IVF_FLAT index. rows=" << rows
                               << ", dim=" << dim
                               << ", train_rows=" << train_rows
                               << ", nlist=" << _milvus_knowhere_nlist
                               << ", train_threads=" << kh_train_threads
                               << ", add_threads=" << kh_add_threads
                               << ", use_elkan=false" << std::endl;
                     const auto ivf_train_begin = std::chrono::high_resolution_clock::now();
                     auto st = _milvus_knowhere_index.Train(train_ds, train_json, false);
                     ivf_train_time_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - ivf_train_begin).count();
                     if (st == knowhere::Status::success) {
                        std::unique_ptr<float[]> owned_base(new float[static_cast<size_t>(rows) * static_cast<size_t>(dim)]);
                        std::memcpy(owned_base.get(), base_ptr, sizeof(float) * static_cast<size_t>(rows) * static_cast<size_t>(dim));
                        auto add_ds = knowhere::GenResultDataSet(rows, dim, std::move(owned_base));
                        const auto ivf_add_begin = std::chrono::high_resolution_clock::now();
                        st = _milvus_knowhere_index.Add(add_ds, add_json, false);
                        ivf_add_time_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - ivf_add_begin).count();
                     }
                     const double ivf_build_time_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - ivf_build_begin).count();
                     _milvus_knowhere_ready = (st == knowhere::Status::success);
                     if (_milvus_knowhere_ready)
                     {
                        _milvus_knowhere_is_ivf = true;
                        ivf_effective_build_time_ms = ivf_build_time_ms;
                        std::cout << "[Milvus-IVF-Knowhere] IVF index built. nlist=" << _milvus_knowhere_nlist
                                  << ", nprobe=" << _milvus_knowhere_nprobe
                                  << ", total_build_time_ms=" << ivf_build_time_ms
                                  << ", train_time_ms=" << ivf_train_time_ms
                                  << ", add_time_ms=" << ivf_add_time_ms << std::endl;
                     }
                     else
                     {
                        std::cerr << "[Milvus-IVF-Knowhere] IVF build failed, status=" << static_cast<int>(st)
                                  << ". Fallback to FLAT baseline." << std::endl;

                        auto flat_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                            knowhere::IndexEnum::INDEX_FAISS_IDMAP,
                            knowhere::Version::GetCurrentVersion().VersionNumber());
                        if (flat_expected.has_value()) {
                           _milvus_knowhere_index = flat_expected.value();
                           knowhere::Json flat_json;
                           flat_json[knowhere::meta::DIM] = dim;
                           flat_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
                           flat_json[knowhere::meta::NUM_BUILD_THREAD] = kh_add_threads;

                           std::unique_ptr<float[]> owned_base_flat(new float[static_cast<size_t>(rows) * static_cast<size_t>(dim)]);
                           std::memcpy(owned_base_flat.get(), base_ptr, sizeof(float) * static_cast<size_t>(rows) * static_cast<size_t>(dim));
                           auto flat_ds = knowhere::GenResultDataSet(rows, dim, std::move(owned_base_flat));
                           const auto flat_build_begin = std::chrono::high_resolution_clock::now();
                           const auto st_flat = _milvus_knowhere_index.Build(flat_ds, flat_json, false);
                           ivf_effective_build_time_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::high_resolution_clock::now() - flat_build_begin).count();
                           _milvus_knowhere_ready = (st_flat == knowhere::Status::success);
                           _milvus_knowhere_is_ivf = false;
                           _milvus_knowhere_nlist = 1;
                           _milvus_knowhere_nprobe = 0;
                           if (_milvus_knowhere_ready) {
                              ivf_train_time_ms = -1.0;
                              ivf_add_time_ms = -1.0;
                              std::cout << "[Milvus-IVF-Knowhere] FLAT fallback built successfully. total_build_time_ms="
                                        << ivf_effective_build_time_ms << std::endl;
                           } else {
                              std::cerr << "[Milvus-IVF-Knowhere] FLAT fallback build failed, status="
                                        << static_cast<int>(st_flat) << std::endl;
                           }
                        } else {
                           std::cerr << "[Milvus-IVF-Knowhere] failed to create INDEX_FAISS_FLAT fallback instance." << std::endl;
                        }
                     }

                     if (_milvus_knowhere_ready)
                     {
                        if (!fs::exists(milvus_ivf_cache_dir)) {
                           fs::create_directories(milvus_ivf_cache_dir);
                        }
                        const double index_sizeof_mb =
                            static_cast<double>(sizeof(_milvus_knowhere_index)) / (1024.0 * 1024.0);
                        const MilvusKnowhereCacheMeta built_meta{
                            _milvus_knowhere_is_ivf ? "IVF_FLAT" : "FLAT",
                            rows,
                            dim,
                            _milvus_knowhere_nlist,
                            _milvus_knowhere_nprobe,
                            ivf_effective_build_time_ms,
                            ivf_train_time_ms,
                            ivf_add_time_ms,
                            index_sizeof_mb,
                            0};
                        uint64_t serialized_index_size_bytes = 0;
                        if (save_milvus_knowhere_index_file(_milvus_knowhere_index, milvus_cache_index_file,
                                                            &serialized_index_size_bytes))
                        {
                           MilvusKnowhereCacheMeta meta_to_save = built_meta;
                           meta_to_save.serialized_index_size_mb =
                               static_cast<double>(serialized_index_size_bytes) / (1024.0 * 1024.0);
                           if (save_milvus_knowhere_meta(milvus_cache_meta_file, meta_to_save)) {
                              std::cout << "[Milvus-IVF-Knowhere] cached index saved to " << milvus_cache_index_file << std::endl;
                           } else {
                              std::cerr << "[Milvus-IVF-Knowhere] warning: failed to save cached meta to "
                                        << milvus_cache_meta_file << std::endl;
                           }
                        }
                        else
                        {
                           std::cerr << "[Milvus-IVF-Knowhere] warning: failed to save cached index to "
                                     << milvus_cache_index_file << std::endl;
                        }
                     }
                  }
               }

               MilvusKnowhereCacheMeta hnsw_meta;
               if (need_milvus_hnsw_baseline &&
                   fs::exists(milvus_cache_hnsw_index_file) &&
                   fs::exists(milvus_cache_hnsw_meta_file) &&
                   load_milvus_knowhere_meta(milvus_cache_hnsw_meta_file, hnsw_meta) &&
                   hnsw_meta.rows == rows &&
                   hnsw_meta.dim == dim &&
                   hnsw_meta.index_kind == "HNSW")
               {
                  auto hnsw_cached_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                      knowhere::IndexEnum::INDEX_HNSW,
                      knowhere::Version::GetCurrentVersion().VersionNumber());
                  if (hnsw_cached_expected.has_value())
                  {
                     knowhere::Json hnsw_load_json;
                     hnsw_load_json[knowhere::meta::DIM] = dim;
                     hnsw_load_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
                     hnsw_load_json[knowhere::indexparam::HNSW_M] = std::max(1, hnsw_meta.nlist);
                     hnsw_load_json[knowhere::indexparam::EF] = std::max(1, hnsw_meta.nprobe);
                     _milvus_knowhere_hnsw_index = hnsw_cached_expected.value();
                     const auto hnsw_load_status = _milvus_knowhere_hnsw_index.DeserializeFromFile(
                         milvus_cache_hnsw_index_file, hnsw_load_json);
                     if (hnsw_load_status == knowhere::Status::success)
                     {
                        _milvus_knowhere_hnsw_ready = true;
                        _milvus_knowhere_hnsw_m = std::max(1, hnsw_meta.nlist);
                        _milvus_knowhere_hnsw_ef = std::max(1, hnsw_meta.nprobe);
                        std::cout << "[Milvus-HNSW-Knowhere] loaded cached HNSW index from "
                                  << milvus_cache_hnsw_index_file << std::endl;
                        if (hnsw_meta.build_time_ms >= 0.0) {
                           std::cout << "[Milvus-HNSW-Knowhere] cached build stats: total="
                                     << hnsw_meta.build_time_ms << " ms" << std::endl;
                        }
                        if (hnsw_meta.index_sizeof_mb > 0.0 || hnsw_meta.serialized_index_size_mb > 0.0) {
                           std::cout << "[Milvus-HNSW-Knowhere] cached size stats: sizeof="
                                     << hnsw_meta.index_sizeof_mb
                                     << " MB, serialized="
                                     << hnsw_meta.serialized_index_size_mb
                                     << " MB" << std::endl;
                        }
                     }
                     else
                     {
                        std::cerr << "[Milvus-HNSW-Knowhere] cached HNSW index load failed, status="
                                  << static_cast<int>(hnsw_load_status)
                                  << ". Rebuilding from base vectors." << std::endl;
                     }
                  }
               }

               if (need_milvus_hnsw_baseline && !_milvus_knowhere_hnsw_ready)
               {
                  auto hnsw_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                      knowhere::IndexEnum::INDEX_HNSW,
                      knowhere::Version::GetCurrentVersion().VersionNumber());
                  if (!hnsw_expected.has_value())
                  {
                     std::cerr << "[Milvus-HNSW-Knowhere] failed to create INDEX_HNSW instance." << std::endl;
                  }
                  else
                  {
                     _milvus_knowhere_hnsw_index = hnsw_expected.value();
                     _milvus_knowhere_hnsw_m = 32;
                     _milvus_knowhere_hnsw_efc = 200;
                     _milvus_knowhere_hnsw_ef = 100;

                     knowhere::Json hnsw_build_json;
                     hnsw_build_json[knowhere::meta::DIM] = dim;
                     hnsw_build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
                     hnsw_build_json[knowhere::indexparam::HNSW_M] = _milvus_knowhere_hnsw_m;
                     hnsw_build_json[knowhere::indexparam::EFCONSTRUCTION] = _milvus_knowhere_hnsw_efc;
                     hnsw_build_json[knowhere::indexparam::EF] = _milvus_knowhere_hnsw_ef;
                     hnsw_build_json[knowhere::meta::NUM_BUILD_THREAD] = kh_add_threads;

                     std::unique_ptr<float[]> owned_base_hnsw(new float[static_cast<size_t>(rows) * static_cast<size_t>(dim)]);
                     std::memcpy(owned_base_hnsw.get(), base_ptr,
                                 sizeof(float) * static_cast<size_t>(rows) * static_cast<size_t>(dim));
                     auto hnsw_ds = knowhere::GenResultDataSet(rows, dim, std::move(owned_base_hnsw));
                     const auto hnsw_build_begin = std::chrono::high_resolution_clock::now();
                     std::cout << "[Milvus-HNSW-Knowhere] building HNSW index. rows=" << rows
                               << ", dim=" << dim
                               << ", M=" << _milvus_knowhere_hnsw_m
                               << ", efConstruction=" << _milvus_knowhere_hnsw_efc
                               << ", efSearch=" << _milvus_knowhere_hnsw_ef
                               << ", build_threads=" << kh_add_threads << std::endl;
                     const auto st_hnsw = _milvus_knowhere_hnsw_index.Build(hnsw_ds, hnsw_build_json, false);
                     const double hnsw_build_time_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - hnsw_build_begin).count();
                     _milvus_knowhere_hnsw_ready = (st_hnsw == knowhere::Status::success);
                     if (_milvus_knowhere_hnsw_ready)
                     {
                        std::cout << "[Milvus-HNSW-Knowhere] HNSW index built. total_build_time_ms="
                                  << hnsw_build_time_ms << std::endl;
                        if (!fs::exists(milvus_hnsw_cache_dir)) {
                           fs::create_directories(milvus_hnsw_cache_dir);
                        }
                        const double index_sizeof_mb =
                            static_cast<double>(sizeof(_milvus_knowhere_hnsw_index)) / (1024.0 * 1024.0);
                        const MilvusKnowhereCacheMeta built_hnsw_meta{
                            "HNSW",
                            rows,
                            dim,
                            _milvus_knowhere_hnsw_m,
                            _milvus_knowhere_hnsw_ef,
                            hnsw_build_time_ms,
                            -1.0,
                            -1.0,
                            index_sizeof_mb,
                            0};
                        uint64_t serialized_index_size_bytes = 0;
                        if (save_milvus_knowhere_index_file(_milvus_knowhere_hnsw_index, milvus_cache_hnsw_index_file,
                                                            &serialized_index_size_bytes))
                        {
                           MilvusKnowhereCacheMeta meta_to_save = built_hnsw_meta;
                           meta_to_save.serialized_index_size_mb =
                               static_cast<double>(serialized_index_size_bytes) / (1024.0 * 1024.0);
                           if (save_milvus_knowhere_meta(milvus_cache_hnsw_meta_file, meta_to_save)) {
                              std::cout << "[Milvus-HNSW-Knowhere] cached index saved to "
                                        << milvus_cache_hnsw_index_file << std::endl;
                           } else {
                              std::cerr << "[Milvus-HNSW-Knowhere] warning: failed to save cached meta to "
                                        << milvus_cache_hnsw_meta_file << std::endl;
                           }
                        }
                        else
                        {
                           std::cerr << "[Milvus-HNSW-Knowhere] warning: failed to save cached index to "
                                     << milvus_cache_hnsw_index_file << std::endl;
                        }
                     }
                     else
                     {
                        std::cerr << "[Milvus-HNSW-Knowhere] HNSW build failed, status="
                                  << static_cast<int>(st_hnsw) << std::endl;
                     }
                  }
               }
            }
         }
         catch (const std::exception &e)
         {
            std::cerr << "[Milvus-IVF-Knowhere] exception during build: " << e.what() << std::endl;
            _milvus_knowhere_ready = false;
         }
      }
#endif

      // FAVOR is built by build_hybrid.sh as a sibling of index_files.  Unlike
      // the UNG files above, it is not part of index_path_prefix itself, so use
      // the already-resolved index root to find and deserialize it here.
      _favor_ready = false;
      _favor_index.reset();
      _favor_space.reset();
      _favor_build_time_ms = -1.0;
      _favor_serialized_index_size_bytes = 0;
      uint64_t favor_index_sizeof_bytes = 0;
      const bool need_favor_index =
          is_smartroute_mode || routing_mode == 8 ||
          (routing_mode == 0 && (baseline_alg == 11 || baseline_alg == 12));
      const fs::path favor_dir = index_root_dir / "FAVOR";
      const fs::path favor_index_file = favor_dir / "favor.index";

      if (need_favor_index) {
         if (_base_storage == nullptr || _base_storage->get_dim() == 0) {
            std::cerr << "[FAVOR] Cannot load index: base storage is unavailable." << std::endl;
         } else if (!fs::exists(favor_index_file)) {
            std::cerr << "[FAVOR] Index file not found: " << favor_index_file.string() << std::endl;
         } else {
            try {
               const size_t dim = static_cast<size_t>(_base_storage->get_dim());
               const size_t max_elements = static_cast<size_t>(_base_storage->get_num_points());
               _favor_space = std::make_unique<hnswlib::L2Space>(dim);
               _favor_index = std::make_unique<favor::FAVOR<float>>(
                   _favor_space.get(), favor_index_file.string(), false, max_elements, 100);

               if (_favor_index->cur_element_count != max_elements) {
                  std::cerr << "[FAVOR] Loaded index count mismatch: index="
                            << _favor_index->cur_element_count << ", base=" << max_elements
                            << ". FAVOR is disabled." << std::endl;
                  _favor_index.reset();
                  _favor_space.reset();
               } else {
                  _favor_ready = true;
                  _favor_serialized_index_size_bytes =
                      static_cast<uint64_t>(fs::file_size(favor_index_file));
                  favor_index_sizeof_bytes = estimate_favor_logical_size_bytes(*_favor_index);

                  FavorCacheMeta favor_meta;
                  if (load_favor_meta((favor_dir / "favor.meta").string(), favor_meta)) {
                     _favor_build_time_ms = favor_meta.build_time_ms;
                  }

                  std::cout << "[FAVOR] Loaded index from " << favor_index_file.string()
                            << " (elements=" << _favor_index->cur_element_count
                            << ", dim=" << dim << ")" << std::endl;
               }
            } catch (const std::exception& e) {
               std::cerr << "[FAVOR] Failed to load index from " << favor_index_file.string()
                         << ": " << e.what() << std::endl;
               _favor_index.reset();
               _favor_space.reset();
            }
         }
      }

      meta_data["favor_enabled"] = _favor_ready ? "1" : "0";
      meta_data["favor_build_time(ms)"] = std::to_string(_favor_build_time_ms);
      meta_data["favor_index_sizeof_bytes"] = std::to_string(favor_index_sizeof_bytes);
      meta_data["favor_index_sizeof_mb"] =
          std::to_string(static_cast<double>(favor_index_sizeof_bytes) / (1024.0 * 1024.0));
      meta_data["favor_serialized_size_bytes"] = std::to_string(_favor_serialized_index_size_bytes);
      write_kv_file(meta_filename, meta_data);
      upsert_build_time_metric(build_time_filename, "favor_build_time", std::to_string(_favor_build_time_ms));
      upsert_build_time_metric(build_time_filename, "favor_index_sizeof_bytes", std::to_string(favor_index_sizeof_bytes));
      upsert_build_time_metric(
          build_time_filename,
          "favor_index_sizeof_mb",
          std::to_string(static_cast<double>(favor_index_sizeof_bytes) / (1024.0 * 1024.0)));
      upsert_build_time_metric(build_time_filename, "favor_serialized_size_bytes", std::to_string(_favor_serialized_index_size_bytes));

      _build_rabitq_side_index = false;
      _rabitq_total_bits = 4;
      auto rabitq_bits_it = meta_data.find("rabitq_total_bits");
      if (rabitq_bits_it != meta_data.end())
      {
         _rabitq_total_bits = static_cast<size_t>(std::stoul(rabitq_bits_it->second));
      }
      auto rabitq_enabled_it = meta_data.find("rabitq_enabled");
      if (rabitq_enabled_it != meta_data.end() && rabitq_enabled_it->second == "1")
      {
         const std::string rabitq_side_filename = index_path_prefix + "rabitq_side.bin";
         if (_rabitq_side_index.load(rabitq_side_filename))
         {
            std::cout << "[RabitQ] Side index loaded from " << rabitq_side_filename << std::endl;
         }
         else
         {
            std::cerr << "[RabitQ] Failed to load side index. Exact mode remains available." << std::endl;
         }
      }

      // load idea1 selector
      std::string model_path = selector_modle_prefix + "/intelElS/idea1_selector_model_final.onnx";
      std::cout << "Loading Trie method selector model from " << model_path << " ..." << std::endl;
      try
      {
         if (fs::exists(model_path))
         {
            _trie_method_selector = std::make_unique<MethodSelector>(model_path);
            std::cout << "- Model loaded successfully." << std::endl;
         }
         else
         {
            std::cerr << "- WARNING: Model file not found. Will use default method passed by command line." << std::endl;
            _trie_method_selector = nullptr;
         }
      }
      catch (const std::exception &e)
      {
         std::cerr << "- ERROR: Failed to load model: " << e.what() << ". Will use default method." << std::endl;
         _trie_method_selector = nullptr;
      }

      // --- 加载 SODA 模型 ---
      _smart_route_target_alg_id = 2;
      const auto smart_route_model_dir = resolve_smart_route_model_dir(selector_modle_prefix);
      if (smart_route_model_dir.has_value()) {
         const std::string sr_model_path = (smart_route_model_dir.value() / "router.onnx").string();
         _smart_route_selector = std::make_unique<MethodSelector>(sr_model_path);
         std::cout << "- SODA Model loaded from: " << sr_model_path << std::endl;

         if (const auto routed_alg_id = load_router_target_algo_id_from_env()) {
            _smart_route_target_alg_id = *routed_alg_id;
            std::cout << "- SODA class 0 target algorithm loaded from ROUTER_ZERO_ALGORITHM: "
                      << _smart_route_target_alg_id << std::endl;
         } else {
            const fs::path class_labels_path = smart_route_model_dir.value() / "class_labels.txt";
            if (const auto routed_alg_id = load_router_target_algo_id_from_class_labels(class_labels_path)) {
               _smart_route_target_alg_id = *routed_alg_id;
               std::cout << "- SODA class 0 target algorithm loaded from class_labels.txt: "
                         << _smart_route_target_alg_id << std::endl;
            } else {
               std::cout << "- [Warning] SODA class_labels.txt missing or unsupported at: "
                         << class_labels_path.string()
                         << ". Fallback target algorithm id: " << _smart_route_target_alg_id << std::endl;
            }
         }
      } else {
         _smart_route_selector = nullptr;
         std::cout << "- [Warning] SODA Model not found under: " << selector_modle_prefix << std::endl;
      }

      // --- 加载 FastSmartRoute 单层模型 (4特征) ---
      std::string fsr_single_model_path = selector_modle_prefix + "/fast_smart_route/router.onnx";
      if (fs::exists(fsr_single_model_path)) {
         _fast_route_single_selector = std::make_unique<MethodSelector>(fsr_single_model_path);
         std::cout << "- FastSmartRoute Single Model loaded." << std::endl;

         std::string majority_id_path = selector_modle_prefix + "/fast_smart_route/majority_acorn_id.txt";
         if (fs::exists(majority_id_path)) {
             std::ifstream infile(majority_id_path);
             if (infile >> _single_majority_acorn_id) {
                 std::cout << "- FastSmartRoute majority ACORN ID loaded: " << _single_majority_acorn_id << std::endl;
             }
         } else {
             std::cout << "- [Warning] majority_acorn_id.txt not found, using default: " << _single_majority_acorn_id << std::endl;
         }
      } else {
         _fast_route_single_selector = nullptr;
      }

      // --- 加载 Smartroute-revised 模型 (2特征) ---
      std::string fsr_revised_model_path = selector_modle_prefix + "/fast_smart_revised_route/router.onnx";
      if (fs::exists(fsr_revised_model_path)) {
         _fast_route_revised_selector = std::make_unique<MethodSelector>(fsr_revised_model_path);
         std::cout << "- Smartroute-revised Model loaded." << std::endl;

         std::string revised_majority_id_path = selector_modle_prefix + "/fast_smart_revised_route/majority_acorn_id.txt";
         if (fs::exists(revised_majority_id_path)) {
             std::ifstream infile(revised_majority_id_path);
             if (infile >> _revised_majority_acorn_id) {
                 std::cout << "- Smartroute-revised majority ACORN ID loaded: " << _revised_majority_acorn_id << std::endl;
             }
         } else {
             std::cout << "- [Warning] majority_acorn_id.txt not found, using default: " << _revised_majority_acorn_id << std::endl;
         }
      } else {
         _fast_route_revised_selector = nullptr;
      }

      // === load ACORN index ===
      // Declare a shared inverted index and a flag to track if it's loaded.
      std::unordered_map<int, std::vector<int>> shared_inverted_index;
      bool inverted_index_loaded = false;

      if (need_acorn_baseline && !acorn_index_path.empty() && fs::exists(acorn_index_path))
      {
         std::cout << "Loading REORDERED ACORN index from " << acorn_index_path << " ..." << std::endl;
         try
         {
            faiss::Index *raw_index = faiss::read_index(acorn_index_path.c_str());
            _acorn_index = std::shared_ptr<faiss::IndexACORNFlat>(dynamic_cast<faiss::IndexACORNFlat *>(raw_index));

            if (_acorn_index)
            {
               std::cout << "- Associating metadata directly (reordered)..." << std::endl;
               std::vector<std::vector<int>> metadata(_num_points);
               for (size_t new_id = 0; new_id < _num_points; ++new_id)
               {
                  const auto &label_set = _base_storage->get_label_set(new_id);
                  metadata[new_id].assign(label_set.begin(), label_set.end());
                  std::sort(metadata[new_id].begin(), metadata[new_id].end());
               }
               _acorn_index->set_metadata(metadata);
               std::cout << "- ACORN index loaded and metadata associated successfully." << std::endl;

               std::string acorn_rabitq_side_path = acorn_index_path + ".rabitq_side.bin";
               if (fs::exists(acorn_rabitq_side_path))
               {
                  if (_acorn_rabitq_side_index.load(acorn_rabitq_side_path))
                  {
                     std::cout << "- ACORN RabitQ side index loaded from: " << acorn_rabitq_side_path << std::endl;
                  }
                  else
                  {
                     std::cerr << "  - [WARNING] Failed to load ACORN RabitQ side index from: "
                               << acorn_rabitq_side_path << std::endl;
                  }
               }
               else
               {
                  std::cout << "- ACORN RabitQ side index not found: " << acorn_rabitq_side_path << std::endl;
               }

               std::string inverted_index_path = acorn_index_path + ".inverted_index";
               if (fs::exists(inverted_index_path))
               {
                  std::cout << "- Loading inverted index from: " << inverted_index_path << std::endl;
                  shared_inverted_index = load_inverted_index(inverted_index_path);
                  _acorn_index->set_inverted_index(shared_inverted_index);
                  inverted_index_loaded = true;
                  std::cout << "- Inverted index loaded and set successfully for ACORN." << std::endl;
               }
               else
               {
                  std::cerr << "  - [WARNING] Inverted index file not found at: " << inverted_index_path
                            << ". ACORN's original filtering (routing_mode=3) will not work." << std::endl;
               }
            }
            else
            {
               std::cerr << "ERROR: Failed to cast loaded index to faiss::IndexACORNFlat." << std::endl;
               delete raw_index;
            }
         }
         catch (const std::exception &e)
         {
            std::cerr << "ERROR: Exception caught while loading ACORN index: " << e.what() << std::endl;
            _acorn_index = nullptr;
         }
      }
      else if (need_acorn_baseline && !acorn_index_path.empty())
      {
         std::cerr << "Warning: ACORN index path provided, but file not found at: " << acorn_index_path << std::endl;
      }

      // --- load ACORN-1 index ---
      if (need_acorn_baseline && !acorn_1_index_path.empty() && fs::exists(acorn_1_index_path))
      {
         std::cout << "Loading REORDERED ACORN-1 index from " << acorn_1_index_path << " ..." << std::endl;
         try
         {
            faiss::Index *raw_index_1 = faiss::read_index(acorn_1_index_path.c_str());
            _acorn_1_index = std::shared_ptr<faiss::IndexACORNFlat>(dynamic_cast<faiss::IndexACORNFlat *>(raw_index_1));

            if (_acorn_1_index)
            {
               std::cout << "- Associating metadata directly with ACORN-1 index (reordered)..." << std::endl;
               std::vector<std::vector<int>> metadata(_num_points);
               for (size_t new_id = 0; new_id < _num_points; ++new_id)
               {
                  const auto &label_set = _base_storage->get_label_set(new_id);
                  metadata[new_id].assign(label_set.begin(), label_set.end());
                  std::sort(metadata[new_id].begin(), metadata[new_id].end());
               }
               _acorn_1_index->set_metadata(metadata);
               std::cout << "- ACORN-1 index loaded and metadata associated successfully." << std::endl;

               std::string acorn_1_rabitq_side_path = acorn_1_index_path + ".rabitq_side.bin";
               if (fs::exists(acorn_1_rabitq_side_path))
               {
                  if (_acorn_1_rabitq_side_index.load(acorn_1_rabitq_side_path))
                  {
                     std::cout << "- ACORN-1 RabitQ side index loaded from: " << acorn_1_rabitq_side_path << std::endl;
                  }
                  else
                  {
                     std::cerr << "  - [WARNING] Failed to load ACORN-1 RabitQ side index from: "
                               << acorn_1_rabitq_side_path << std::endl;
                  }
               }
               else
               {
                  std::cout << "- ACORN-1 RabitQ side index not found: " << acorn_1_rabitq_side_path << std::endl;
               }

               // Replace the old loading logic to reuse the inverted index.
               if (inverted_index_loaded)
               {
                  std::cout << "- Reusing the loaded inverted index for ACORN-1." << std::endl;
                  _acorn_1_index->set_inverted_index(shared_inverted_index);
                  std::cout << "- Inverted index set successfully for ACORN-1." << std::endl;
               }
               else
               {
                  std::cerr << "  - [WARNING] Inverted index could not be set for ACORN-1 because the primary inverted index file was not found." << std::endl;
               }
            }
            else
            {
               std::cerr << "ERROR: Failed to cast loaded index to faiss::IndexACORNFlat from " << acorn_1_index_path << std::endl;
               delete raw_index_1;
            }
         }
         catch (const std::exception &e)
         {
            std::cerr << "ERROR: Exception caught while loading ACORN-1 index: " << e.what() << std::endl;
            _acorn_1_index = nullptr;
         }
      }
      else if (need_acorn_baseline && !acorn_1_index_path.empty())
      {
         std::cerr << "Warning: ACORN-1 index file not found at: " << acorn_1_index_path << std::endl;
      }

      // print
      std::cout << "- Index loaded in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   /*void UniNavGraph::statistics()
   {
      // number of edges in the unified navigating graph
      _graph_num_edges = 0;
      for (IdxType i = 0; i < _num_points; ++i)
         _graph_num_edges += _graph->neighbors[i].size();

      // number of edges in the label navigating graph
      _LNG_num_edges = 0;
      if (_label_nav_graph != nullptr)
         for (IdxType i = 1; i <= _num_groups; ++i)
            _LNG_num_edges += _label_nav_graph->out_neighbors[i].size();

      // index size
      _index_size = 0;
      for (IdxType i = 1; i <= _num_groups; ++i)
         _index_size += _group_id_to_label_set[i].size() * sizeof(LabelType);
      _index_size += _group_id_to_range.size() * sizeof(IdxType) * 2;
      _index_size += _group_entry_points.size() * sizeof(IdxType);
      _index_size += _new_to_old_vec_ids.size() * sizeof(IdxType);
      _index_size += _trie_index.get_index_size();
      _index_size += _graph->get_index_size();

      _index_size_add_rb = _index_size; // 记录不含 Roaring Bitmap 部分的索引大小
      // fxy_add:计算为混合搜索策略预处理的 Roaring Bitmaps 的大小
      if (!_lng_descendants_rb.empty())
      {
         for (const auto &rb : _lng_descendants_rb)
         {
            _index_size_add_rb += rb.getSizeInBytes();
         }
      }
      if (!_covered_sets_rb.empty())
      {
         for (const auto &rb : _covered_sets_rb)
         {
            _index_size_add_rb += rb.getSizeInBytes();
         }
      }

      // return as MB
      _index_size /= 1024 * 1024;
      _index_size_add_rb /= 1024 * 1024;
   }*/
   void UniNavGraph::statistics()
   {
      // number of edges in the unified navigating graph
      _graph_num_edges = 0;
      for (IdxType i = 0; i < _num_points; ++i)
         _graph_num_edges += _graph->neighbors[i].size();

      // number of edges in the label navigating graph
      _LNG_num_edges = 0;
      if (_label_nav_graph != nullptr)
         for (IdxType i = 1; i <= _num_groups; ++i)
               _LNG_num_edges += _label_nav_graph->out_neighbors[i].size();

      // index size
      _index_size = 0;
      for (IdxType i = 1; i <= _num_groups; ++i)
         _index_size += _group_id_to_label_set[i].size() * sizeof(LabelType);
      _index_size += _group_id_to_range.size() * sizeof(IdxType) * 2;
      _index_size += _group_entry_points.size() * sizeof(IdxType);
      _index_size += _new_to_old_vec_ids.size() * sizeof(IdxType);
      _index_size += _trie_index.get_index_size();
      _index_size += _graph->get_index_size();

      // --- 统计 _label_nav_graph (LNG) 的大小 ---
      size_t lng_graph_size = 0;
      if (_label_nav_graph != nullptr)
      {
         // 1. 统计 out_neighbors (基于 .size())
         for (const auto &neighbors : _label_nav_graph->out_neighbors)
         {
               lng_graph_size += neighbors.size() * sizeof(ANNS::IdxType);
         }
         // 2. 统计 in_neighbors (基于 .size())
         for (const auto &neighbors : _label_nav_graph->in_neighbors)
         {
               lng_graph_size += neighbors.size() * sizeof(ANNS::IdxType);
         }
      }
      _index_size += lng_graph_size;

      // --- 统计 _group_id_to_vec_ids 的大小 ---
      size_t group_to_vec_size = 0;
      for (const auto &vec_ids : _group_id_to_vec_ids)
      {
         group_to_vec_size += vec_ids.size() * sizeof(ANNS::IdxType);
      }
      _index_size += group_to_vec_size;

      // fxy_add:计算为混合搜索策略预处理的 Roaring Bitmaps 的大小
      _index_size_add_rb = _index_size; 
      
      if (!_lng_descendants_rb.empty())
      {
         for (const auto &rb : _lng_descendants_rb)
         {
               _index_size_add_rb += rb.getSizeInBytes();
         }
      }
      if (!_covered_sets_rb.empty())
      {
         for (const auto &rb : _covered_sets_rb)
         {
               _index_size_add_rb += rb.getSizeInBytes();
         }
      }

      // return as MB
      _index_size /= 1024 * 1024;
      _index_size_add_rb /= 1024 * 1024;
   }

   // =========================================================================
   // Curator build (thin wrapper, like FAVOR)
   // =========================================================================

   void UniNavGraph::configure_curator(int nlist, int nprobe, int max_leaf_size, int search_ef, int beam_size) {
       _curator_ctx.nlist = nlist;
       _curator_ctx.nprobe = nprobe;
       _curator_ctx.max_leaf_size = max_leaf_size;
       _curator_ctx.search_ef = search_ef;
       _curator_ctx.beam_size = beam_size;
   }

   void UniNavGraph::build_curator(const std::string& save_path) {
       if (_curator_ctx.ready) return;

       // Curator-only mode skips index.load(), so the framework structures the
       // Stage-1 bitset filter map depends on (_num_points, _attr_to_id,
       // _vector_attr_graph) may not exist yet. Build them here (index-build
       // phase, not counted as query time) — same bipartite-graph path as the
       // pre-filter baseline.
       if (_vector_attr_graph.empty()) {
           _num_points = _base_storage->get_num_points();
           build_vector_and_attr_graph();
       }

       if (!save_path.empty()) {
           curator_load_index(_curator_ctx, _base_storage, save_path);
       } else {
           curator_build_index(_curator_ctx, _base_storage);
       }
   }
}
