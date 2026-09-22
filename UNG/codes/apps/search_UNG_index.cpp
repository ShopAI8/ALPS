#include <chrono>
#include <fstream>
#include <memory>
#include <numeric>
#include <iostream>
#include <bitset>
#include <cstdlib>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "uni_nav_graph.h"
#include "utils.h"
#include <roaring/roaring.h>
#include <roaring/roaring.hh>
#include <faiss_navix/IndexHNSW.h>
#include <faiss_navix/index_io.h>
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

// Helper: compute recall for a single query, including distance-tie handling.
float calculate_single_query_recall(const std::pair<ANNS::IdxType, float> *gt,
                                    const std::pair<ANNS::IdxType, float> *results,
                                    ANNS::IdxType K)
{
   // 1. Read the distance threshold of the K-th GT result.
   // GT is sorted, and `pair.second` stores the distance.
   // If GT contains fewer than K valid results, fall back to the last valid one.
   float gt_threshold = 0.0f;
   if (K > 0 && gt[K - 1].first != -1) {
       gt_threshold = gt[K - 1].second;
   } else {
       // If GT is shorter than K, use the last valid distance instead.
       for(int i = K - 1; i >= 0; i--) {
           if(gt[i].first != -1) {
               gt_threshold = gt[i].second;
               break;
           }
       }
   }
   
   // Allow a small floating-point tolerance.
   float epsilon = 1e-6; 

   std::unordered_set<ANNS::IdxType> gt_set;
   for (int i = 0; i < K; ++i)
   {
      if (gt[i].first != -1)
      {
         gt_set.insert(gt[i].first);
      }
   }
   
   if (gt_set.empty()) return 1.0f; // Empty GT is treated as full recall.

   int correct = 0;
   for (int i = 0; i < K; ++i)
   {
      if (results[i].first == -1) continue;

      // Condition 1: exact ID match.
      if (gt_set.count(results[i].first))
      {
         correct++;
      }
      // Condition 2: ID differs, but the distance is still within the GT tie threshold.
      // This assumes `results[i].second` is a valid distance value.
      else if (results[i].second <= gt_threshold + epsilon)
      {
         correct++;
      }
   }

   // Debug: print first query
   static int dbg_cnt = 0;
   if (dbg_cnt++ < 2) {
       std::cout << "[RECALL DEBUG] gt_threshold=" << gt_threshold
                 << " gt_set_size=" << gt_set.size() << " correct=" << correct
                 << " recall=" << static_cast<float>(correct) / gt_set.size() << std::endl;
       std::cout << "[RECALL DEBUG] GT top-3: ";
       for (int i = 0; i < std::min((int)K, 3); ++i)
           std::cout << "(" << gt[i].first << "," << gt[i].second << ") ";
       std::cout << std::endl;
       std::cout << "[RECALL DEBUG] RES top-3: ";
       for (int i = 0; i < std::min((int)K, 3); ++i)
           std::cout << "(" << results[i].first << "," << results[i].second << ") ";
       std::cout << std::endl;
   }

   return static_cast<float>(correct) / gt_set.size();
}

bool export_acorn_level0_graph(const std::string &acorn_index_path,
                               const std::string &output_path)
{
   if (acorn_index_path.empty())
   {
      std::cerr << "ERROR: --acorn_index_path is required for --export_acorn_level0_graph" << std::endl;
      return false;
   }
   if (output_path.empty())
   {
      std::cerr << "ERROR: --export_acorn_level0_graph output path is empty" << std::endl;
      return false;
   }

   std::cout << "[Export] Loading ACORN index from: " << acorn_index_path << std::endl;
   faiss::Index *raw_index = nullptr;
   try
   {
      raw_index = faiss::read_index(acorn_index_path.c_str());
   }
   catch (const std::exception &e)
   {
      std::cerr << "ERROR: Failed to read ACORN index: " << e.what() << std::endl;
      return false;
   }

   std::unique_ptr<faiss::Index> index_guard(raw_index);
   auto *acorn_index = dynamic_cast<faiss::IndexACORN *>(raw_index);
   if (!acorn_index)
   {
      std::cerr << "ERROR: The loaded index is not faiss::IndexACORN-compatible." << std::endl;
      return false;
   }

   std::ofstream out(output_path);
   if (!out)
   {
      std::cerr << "ERROR: Cannot open output file: " << output_path << std::endl;
      return false;
   }

   const faiss::idx_t n = acorn_index->ntotal;
   size_t edge_count = 0;
   for (faiss::idx_t i = 0; i < n; ++i)
   {
      size_t begin = 0;
      size_t end = 0;
      acorn_index->acorn.neighbor_range(i, 0, &begin, &end);

      out << i << ":";
      bool first = true;
      for (size_t j = begin; j < end; ++j)
      {
         const int v = acorn_index->acorn.neighbors[j];
         if (v < 0 || v >= n)
         {
            continue;
         }
         out << (first ? " " : ",") << v;
         first = false;
         ++edge_count;
      }
      out << "\n";
   }

   std::cout << "[Export] ACORN level-0 graph exported successfully." << std::endl;
   std::cout << "[Export] nodes=" << n << ", edges=" << edge_count << std::endl;
   std::cout << "[Export] output=" << output_path << std::endl;
   return true;
}


int main(int argc, char **argv)
{
   std::string data_type, dist_fn, scenario;
   std::string base_bin_file, query_bin_file, base_label_file, query_label_file, gt_file, index_path_prefix, result_path_prefix, selector_modle_prefix, query_group_id_file;
   std::string acorn_index_path, acorn_1_index_path, navix_index_path, algo_choice_csv_path;
   std::string curator_index_path;
   std::string export_acorn_level0_graph_path;
   ANNS::IdxType K, num_entry_points;
   std::vector<ANNS::IdxType> Lsearch_list;
   uint32_t num_threads;
   bool is_new_method = false;                                 // true: use new method
   bool is_new_trie_method = false, is_rec_more_start = false; // false: original UNG trie method; true: recursive variant. false: default root strategy.
   bool is_ung_more_entry = false;                             // false: original UNG entry-point selection; true: allow more entry points.
   // bool is_bfs_filter = true;                               // true: original ACORN; false: improved variant.
   int baseline_alg = 0; // 0/1/8=UNG, 2/3/4/6=ACORN, 5=pre-filter, 7=NaviX, 9=Milvus-IVF, 10=Milvus-HNSW, 11=FAVOR, 12=FAVOR-HNSW, 13=Curator
   // Curator parameters (defaults from original curator-v2 paper)
   int curator_nlist = 32;
   int curator_nprobe = 1200;        // matches original Curator default
   int curator_max_leaf_size = 256;  // matches original max_sl_size default
   int curator_search_ef = 128;      // middle of paper's [32,64,128,256,512] sweep
   int curator_beam_size = 1;
   int num_repeats = 1;                                        // Run one repeat by default.
   int routing_mode = 0;                                      // 0: auto, 1: UNG (nT=false), 2: UNG-nTtrue, 3: ACORN
   int lsearch_start, lsearch_step;
   int efs_start, efs_step_slow, efs_step_fast, lsearch_threshold;
   std::string dataset; 
   std::string ung_distance_mode = "exact";
   bool optimize_standalone_prefilter = false; // Keep standalone pre-filter unoptimized by default; enable it for large-query runs if needed.

   try
   {
      po::options_description desc{"Arguments"};
      desc.add_options()("help,h", "Print information on arguments");
      desc.add_options()("dataset", po::value<std::string>(&dataset)->required(),
                         "dataset");
      desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                         "data type <int8/uint8/float>");
      desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                         "distance function <L2/IP/cosine>");
      desc.add_options()("base_bin_file", po::value<std::string>(&base_bin_file)->required(),
                         "File containing the base vectors in binary format");
      desc.add_options()("query_bin_file", po::value<std::string>(&query_bin_file)->required(),
                         "File containing the query vectors in binary format");
      desc.add_options()("base_label_file", po::value<std::string>(&base_label_file)->default_value(""),
                         "Base label file in txt format");
      desc.add_options()("query_label_file", po::value<std::string>(&query_label_file)->default_value(""),
                         "Query label file in txt format");
      desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                         "Filename for the computed ground truth in binary format");
      desc.add_options()("K", po::value<ANNS::IdxType>(&K)->required(),
                         "Number of ground truth nearest neighbors to compute");
      desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(ANNS::default_paras::NUM_THREADS),
                         "Number of threads to use");
      desc.add_options()("result_path_prefix", po::value<std::string>(&result_path_prefix)->required(),
                         "Path to save the querying result file");
      desc.add_options()("selector_modle_prefix", po::value<std::string>(&selector_modle_prefix)->required(),
                         "Path to selector_modle_prefix");
      desc.add_options()("query_group_id_file", po::value<std::string>(&query_group_id_file)->default_value(""),
                         "Optional per-query source-group ID file");
      desc.add_options()("acorn_index_path", po::value<std::string>(&acorn_index_path)->default_value(""),
                         "acorn_index_path");
      desc.add_options()("acorn_1_index_path", po::value<std::string>(&acorn_1_index_path)->default_value(""),
                         "acorn_1_index_path");
      desc.add_options()("export_acorn_level0_graph", po::value<std::string>(&export_acorn_level0_graph_path)->default_value(""),
                         "Export ACORN level-0 adjacency list to the given path and exit without running search");

      // graph search parameters
      desc.add_options()("scenario", po::value<std::string>(&scenario)->default_value("containment"),
                         "Scenario for building UniNavGraph, <equality/containment/overlap/nofilter>");
      desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                         "Prefix of the path to load the index");
      desc.add_options()("num_entry_points", po::value<ANNS::IdxType>(&num_entry_points)->default_value(ANNS::default_paras::NUM_ENTRY_POINTS),
                         "Number of entry points in each entry group");
      desc.add_options()("Lsearch", po::value<std::vector<ANNS::IdxType>>(&Lsearch_list)->multitoken()->required(),
                         "Number of candidates to search in the graph");
      desc.add_options()("is_new_method", po::value<bool>(&is_new_method)->required(),
                         "is_new_method");
      desc.add_options()("is_new_trie_method", po::value<bool>(&is_new_trie_method)->required(),
                         "is_new_trie_method");
      desc.add_options()("is_rec_more_start", po::value<bool>(&is_rec_more_start)->required(),
                         "is_rec_more_start");
      // desc.add_options()("is_bfs_filter", po::value<bool>(&is_bfs_filter)->default_value(true), "Whether to use BFS filter in ACORN");
      desc.add_options()("baseline_alg", po::value<int>(&baseline_alg)->default_value(0), "Algorithm selector: 0/1/8=UNG family, 2/3/4/6=ACORN family, 5=pre-filter, 7=NaviX, 9=Milvus-IVF, 10=Milvus-HNSW, 11=FAVOR, 12=FAVOR-HNSW, 14=UNG++(bitmap ELS), 15=UNG++(sorted-LNG ELS)");
      desc.add_options()("num_repeats", po::value<int>(&num_repeats)->default_value(1),
                         "Number of repeats for each Lsearch value");
      desc.add_options()("routing_mode", po::value<int>(&routing_mode)->required(),
                         "0: auto, 1: SODA, 2: FastSmartRoute, 3: FastSmartRoute+, 5: SODA+, 6: SmartRoute++, 7: SmartRoute+++, 8: ALPS-fixed");
      desc.add_options()("algo_choice_csv", po::value<std::string>(&algo_choice_csv_path)->default_value(""),
                         "Optional CSV path for per-query algorithm override. Format: QueryID,Algo_Choice");
      desc.add_options()("lsearch_start", po::value<int>(&lsearch_start)->required(), "Lsearch start value");
      desc.add_options()("lsearch_step", po::value<int>(&lsearch_step)->required(), "Lsearch step value");
      desc.add_options()("efs_start", po::value<int>(&efs_start)->required(), "ACORN efs start value");
      desc.add_options()("efs_step_slow", po::value<int>(&efs_step_slow)->required(), "ACORN efs step value");
      desc.add_options()("efs_step_fast", po::value<int>(&efs_step_fast)->required(), "ACORN efs step value");
      desc.add_options()("lsearch_threshold", po::value<int>(&lsearch_threshold)->required(), "lsearch_threshold");
      desc.add_options()("ung_distance_mode", po::value<std::string>(&ung_distance_mode)->default_value("exact"),
                         "UNG distance mode: exact or rabitq");

      // NaviX
      desc.add_options()("navix_index_path", po::value<std::string>(&navix_index_path)->default_value(""), "Path to NaviX index");

      desc.add_options()("optimize_standalone_prefilter", po::value<bool>(&optimize_standalone_prefilter)->default_value(false),
                   "Whether to fully optimize pre-filter when running as standalone baseline");

      // Curator
      desc.add_options()("curator_nlist", po::value<int>(&curator_nlist)->default_value(32),
                   "Curator: number of clusters (<=64)");
      desc.add_options()("curator_nprobe", po::value<int>(&curator_nprobe)->default_value(16),
                   "Curator: number of probes for unfiltered search");
      desc.add_options()("curator_max_leaf_size", po::value<int>(&curator_max_leaf_size)->default_value(128),
                   "Curator: max vectors per leaf");
      desc.add_options()("curator_search_ef", po::value<int>(&curator_search_ef)->default_value(10),
                   "Curator: search expansion factor (must be >= 1)");
      desc.add_options()("curator_beam_size", po::value<int>(&curator_beam_size)->default_value(1),
                   "Curator: beam search width");
      desc.add_options()("curator_index_path", po::value<std::string>(&curator_index_path)->default_value(""),
                   "Curator: path to save/load curator index");

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      if (vm.count("help"))
      {
         std::cout << desc;
         return 0;
      }
      if (vm.count("export_acorn_level0_graph") &&
          !vm["export_acorn_level0_graph"].as<std::string>().empty())
      {
         const std::string export_path = vm["export_acorn_level0_graph"].as<std::string>();
         const std::string export_acorn_path =
             vm.count("acorn_index_path") ? vm["acorn_index_path"].as<std::string>() : "";
         return export_acorn_level0_graph(export_acorn_path, export_path) ? 0 : -1;
      }
      po::notify(vm);
   }
   catch (const std::exception &ex)
   {
      std::cerr << ex.what() << std::endl;
      return -1;
   }

   // check scenario
   if (scenario != "containment" && scenario != "equality" && scenario != "overlap")
   {
      std::cerr << "Invalid scenario: " << scenario << std::endl;
      return -1;
   }

   // load query data
   std::shared_ptr<ANNS::IStorage> query_storage = ANNS::create_storage(data_type);
   query_storage->load_from_file(query_bin_file, query_label_file);

   // load index
   ANNS::UniNavGraph index(query_storage->get_num_points());

   if (routing_mode == 0 && baseline_alg == 13) {
       // Curator-only mode: load base data directly, skip UNG graph/trie
       std::cout << "[SODA] Curator-only mode: loading base data, skipping UNG index" << std::endl;
       auto base_storage = ANNS::create_storage(data_type);
       base_storage->load_from_file(base_bin_file, base_label_file);
       index.set_base_storage(base_storage);
       index.configure_curator(curator_nlist, curator_nprobe, curator_max_leaf_size, curator_search_ef, curator_beam_size);
       if (curator_index_path.empty()) {
           curator_index_path = std::string(index_path_prefix) + "curator_index.bin";
       }
       index.build_curator(curator_index_path);
   } else {
       index.load(index_path_prefix, selector_modle_prefix, data_type, acorn_index_path, acorn_1_index_path,
                  dataset, routing_mode, baseline_alg);
       index.set_ung_distance_mode(ung_distance_mode);
       index.prepare_rabitq_query_contexts(query_storage, query_bin_file);
       index.build_group_inverted_indices();
   }


   // Naxiv
   const bool navix_needed = (routing_mode == 0 && baseline_alg == 7) || (routing_mode != 0);
   faiss_navix::IndexHNSWFlat* navix_index = nullptr;
   if (!navix_index_path.empty() && fs::exists(navix_index_path)) {
      std::cout << "[SODA] Loading NaviX index from: " << navix_index_path << std::endl;
      faiss_navix::Index* raw_navix = faiss_navix::read_index(navix_index_path.c_str());
      navix_index = dynamic_cast<faiss_navix::IndexHNSWFlat*>(raw_navix);
      if (!navix_index) {
         std::cerr << "ERROR: Failed to cast loaded NaviX index to faiss_navix::IndexHNSWFlat" << std::endl;
         delete raw_navix;
      }
   } else if (navix_needed) {
      std::cout << "[Warning] NaviX index path is empty or does not exist. NaviX routing will fail." << std::endl;
   }

   // Load the per-query source-group id file.
   std::vector<ANNS::IdxType> true_query_group_ids;
   std::ifstream source_group_file;
   if (!query_group_id_file.empty())
   {
      source_group_file.open(query_group_id_file);
   }
   if (source_group_file.is_open())
   {
      ANNS::IdxType group_id;
      while (source_group_file >> group_id)
      {
         true_query_group_ids.push_back(group_id);
      }
      source_group_file.close();
      std::cout << "成功加载 " << true_query_group_ids.size() << " 个查询的来源组ID。" << std::endl;
   }
   else if (!query_group_id_file.empty()) // The run can continue without it, but that optimization path stays disabled.
   {
      std::cerr << "警告：未找到查询来源组ID文件: " << query_group_id_file << std::endl;
   }

   // preparation
   auto num_queries = query_storage->get_num_points();

   std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);
   auto gt = new std::pair<ANNS::IdxType, float>[num_queries * K];
   ANNS::load_gt_file(gt_file, gt, num_queries, K);
   auto results = new std::pair<ANNS::IdxType, float>[num_queries * K];
   std::vector<int> query_algo_choices = index.load_query_algo_choices_from_csv(algo_choice_csv_path, num_queries);

   // ==================== [SODA+/SmartRoute+++ global preprocessing (run once)] ====================
   std::vector<ANNS::QueryStats> global_pred_stats(num_queries);
   double total_global_pred_time = 0.0;
   double total_global_sort_time = 0.0;
   std::vector<int> final_global_choices = query_algo_choices;
   std::vector<int> sorted_query_ids(num_queries);
   std::iota(sorted_query_ids.begin(), sorted_query_ids.end(), 0);

   if (routing_mode == 5 || routing_mode == 7) {
       // --- Run several trials and keep the shortest global-prediction time ---
       double min_pred_time = std::numeric_limits<double>::max();
       std::vector<int> best_choices;
       std::vector<ANNS::QueryStats> best_stats;
       
       int num_trials = 3; // Number of timing trials.
       for (int trial = 0; trial < num_trials; ++trial) {
           auto global_pred_start = std::chrono::high_resolution_clock::now();
           std::vector<ANNS::QueryStats> trial_stats(num_queries);
           auto trial_choices = index.global_predict_algo_choices(
               query_storage, routing_mode, query_algo_choices, num_threads, trial_stats
           );
           double trial_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - global_pred_start).count();
           
           std::cout << "[SODA" << (routing_mode == 7 ? "+++" : "+") << "] Trial " << trial + 1 << " Prediction Time: " << trial_time << " ms" << std::endl;
           
           if (trial_time < min_pred_time) {
               min_pred_time = trial_time;
               best_choices = std::move(trial_choices);
               best_stats = std::move(trial_stats);
           }
       }
       
       // Keep the best run.
       final_global_choices = std::move(best_choices);
       global_pred_stats = std::move(best_stats);
       total_global_pred_time = min_pred_time;
       std::cout << "\n[SODA" << (routing_mode == 7 ? "+++" : "+") << "] Best Global Prediction Time: " << total_global_pred_time << " ms" << std::endl;

       // 2. Global cache-friendly sorting.
       auto global_sort_start = std::chrono::high_resolution_clock::now();
       sorted_query_ids = index.get_sorted_query_ids(query_storage, final_global_choices, routing_mode);
       total_global_sort_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - global_sort_start).count();
       std::cout << "[SODA" << (routing_mode == 7 ? "+++" : "+") << "] Global Sort Time: " << total_global_sort_time << " ms" << std::endl;

       double avg_sort_ms = total_global_sort_time / num_queries;
       for (int i = 0; i < (int)num_queries; ++i) {
           global_pred_stats[i].global_sort_time_ms = avg_sort_ms;
       }
       std::cout << "[SODA" << (routing_mode == 7 ? "+++" : "+") << "] Total Preprocessing Time: " << (total_global_pred_time + total_global_sort_time) << " ms" << std::endl;
   }
   // ====================================================================================

//    // Precompute and store entry-group ids for all queries.
//    std::cout << "\n--- Step 1: Pre-computing Entry Group IDs (Measuring Entry Cost) ---" << std::endl;
//    std::vector<std::vector<ANNS::IdxType>> all_entry_group_ids(num_queries);
//    auto entry_cost_start_time = std::chrono::high_resolution_clock::now();
// #pragma omp parallel for
//    for (int id = 0; id < num_queries; ++id)
//    {
//       const auto &query_labels = query_storage->get_label_set(id);
//       ANNS::QueryStats dummy_stats;
//       static std::atomic<int> trie_debug_print_counter{0};

//       // Compute entry groups and store them in `all_entry_group_ids`.
//       const_cast<ANNS::UniNavGraph &>(index).get_min_super_sets_debug(
//           query_labels,
//           all_entry_group_ids[id], // Write the result into the new container.
//           false, true,
//           trie_debug_print_counter,
//           false,
//           false,
//           dummy_stats,
//           false);
//    }
//    auto entry_cost_total_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - entry_cost_start_time).count();
//    std::cout << "Total time for finding all entry groups (Entry Cost): " << entry_cost_total_time << " ms\n"
//              << std::endl;
//    std::cout << "--- Step 2: Starting Fair Bitmap Computation Comparison ---" << std::endl;
//    double ung_bitmap_total_time = 0.0;
//    double attr_bitmap_total_time = 0.0;
//    // --- Evaluation A: UNG path (build bitmaps from known groups) ---
//    {
//       std::cout << "  -> Testing UNG method (compute_bitmap_from_groups)..." << std::endl;
//       std::vector<roaring::Roaring> ung_bitmaps(num_queries);
//       auto start_time = std::chrono::high_resolution_clock::now();
// #pragma omp parallel for
//       for (int id = 0; id < num_queries; ++id)
//       {
//          ung_bitmaps[id] = index.compute_bitmap_from_groups(all_entry_group_ids[id]);
//       }
//       ung_bitmap_total_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
//       std::cout << "  -> UNG bitmap generation time: " << ung_bitmap_total_time << " ms" << std::endl;
//    }
//    // --- Evaluation B: inverted-index path (`compute_attribute_bitmap`) ---
//    {
//       std::cout << "  -> Testing Attribute method (compute_attribute_bitmap)..." << std::endl;
//       std::vector<std::bitset<10000001>> attr_bitmaps(num_queries);
//       auto start_time = std::chrono::high_resolution_clock::now();
// #pragma omp parallel for
//       for (int id = 0; id < num_queries; id++)
//       {
//          // `compute_attribute_bitmap` returns a pair; we only need the bitmap.
//          attr_bitmaps[id] = index.compute_attribute_bitmap(query_storage->get_label_set(id)).first;
//       }
//       attr_bitmap_total_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
//       std::cout << "  -> Attribute bitmap generation time: " << attr_bitmap_total_time << " ms" << std::endl;
//    }
//    std::cout << "--- Fair Comparison Finished ---\n"
//              << std::endl;
//    auto bitmap_total_time = attr_bitmap_total_time; // Default to the inverted-index path.


      

   // 5-Method Fpass Benchmark
   // std::cout << "\n--- Step: Benchmarking 5 Fpass Calculation Methods ---" << std::endl;
   // std::string fpass_csv_path = result_path_prefix + "fpass_benchmark_5methods.csv";
   // index.evaluate_fpass_methods(query_storage, fpass_csv_path);

   // Warm-up selector
   std::cout << "\n--- Starting Warm-up Phase ---" << std::endl;
   index.warmup_selectors(num_threads);
   std::cout << "--- Warm-up Finished ---"<< std::endl;

   if (baseline_alg == 8 || routing_mode == 1 || routing_mode == 5 || routing_mode == 6 || routing_mode == 7) {
    index.skip_els_filter = true;
    std::cout << "[UNG+] Mode Enabled: ELS filtering will be skipped." << std::endl;
   }

   // if (routing_mode == 0){
   // // calculate query features and save to CSV
   // std::string features_csv_path = result_path_prefix + "query_features.csv";
   // index.calculate_query_features_only(
   //    query_storage,
   //    num_threads,       
   //    features_csv_path, 
   //    true,              // is_new_trie_method
   //    true               // is_rec_more_start
   // );
   // }
   // init query stats
   std::vector<std::vector<std::vector<ANNS::QueryStats>>> query_stats(num_repeats, std::vector<std::vector<ANNS::QueryStats>>(Lsearch_list.size(), std::vector<ANNS::QueryStats>(num_queries))); //(repeat,Lsearch,queryID)

   // Store detailed timing for each run.
   struct SearchTimeLog
   {
      int repeat;
      ANNS::IdxType l_search;
      int efs;
      double time_ms;
      float avg_recall;

      // ELS-specific timing fields.
      double els_trie_avg;
      double els_sort_avg;
      double els_filter_avg;
      double els_total_avg;

      double mask_gen_avg;
      double global_sort_avg;
   };
   std::vector<SearchTimeLog> detailed_times;                      // All detailed timing records.
   std::map<ANNS::IdxType, std::vector<double>> time_per_lsearch;  // Group repeat timings by Lsearch for later averaging.
   std::map<ANNS::IdxType, std::vector<float>> recall_per_lsearch; // Recall values grouped by Lsearch.
   std::map<ANNS::IdxType, std::vector<int>> efs_per_lsearch;      // EFS values grouped by Lsearch.

   for (int repeat = 0; repeat < num_repeats; ++repeat)
   {
      std::cout << "\n=== Repeat " << (repeat + 1) << "/" << num_repeats << " ===" << std::endl;

      // search
      std::vector<float> all_cmps, all_qpss, all_recalls;
      std::vector<float> all_time_ms, all_flag_time, all_bitmap_time, all_entry_points, all_lng_descendants, all_entry_group_coverage;
      std::vector<float> all_is_global_search; // Keep this only if we want the global-search ratio.

      std::cout << "Start querying ..." << std::endl;
      for (int LsearchId = 0; LsearchId < Lsearch_list.size(); LsearchId++)
      {
         ANNS::IdxType current_Lsearch = Lsearch_list[LsearchId];
         std::vector<float> num_cmps(num_queries);

         // --- 1. Build this round's query queue and pass down global preprocessing data ---
         std::queue<int> task_queue;
         for (int id : sorted_query_ids) {
             task_queue.push(id);
         }

         if (routing_mode == 5 || routing_mode == 7) {
             for (int i = 0; i < (int)num_queries; ++i) {
                 query_stats[repeat][LsearchId][i].mask_gen_time_ms = global_pred_stats[i].mask_gen_time_ms;
                 query_stats[repeat][LsearchId][i].route_pred_time_ms = global_pred_stats[i].route_pred_time_ms;
                 query_stats[repeat][LsearchId][i].global_sort_time_ms = global_pred_stats[i].global_sort_time_ms;
                 query_stats[repeat][LsearchId][i].exact_cand_size = global_pred_stats[i].exact_cand_size;
                 query_stats[repeat][LsearchId][i].global_p_pass = global_pred_stats[i].global_p_pass;
             }
         }

         // --- 2. For Curator: repurpose Lsearch as search_ef sweep parameter ---
         if (routing_mode == 0 && baseline_alg == 13) {
             ANNS::curator_update_search_ef(index.get_curator_context(), static_cast<int>(current_Lsearch));
         }

         // --- 3. Time and execute the search ---
         auto start_time = std::chrono::high_resolution_clock::now();
         if (!is_new_method)
         {
            // index.search(...);
         }
         else
         {
             index.search_hybrid(query_storage, distance_handler, num_threads, current_Lsearch,
                                num_entry_points, scenario, K, results, num_cmps, query_stats[repeat][LsearchId],is_new_trie_method, is_rec_more_start, is_ung_more_entry, lsearch_start, lsearch_step, efs_start, efs_step_slow,efs_step_fast,lsearch_threshold,routing_mode, baseline_alg ,navix_index, true_query_group_ids, final_global_choices, task_queue, optimize_standalone_prefilter);
         }
         double pure_search_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
         
         // --- 3. Apply batch-time compensation ---
         double time_cost = pure_search_time;
         if (routing_mode == 5 || routing_mode == 7) {
             time_cost += (total_global_pred_time + total_global_sort_time);
         }

         // --- 4. Compute recall for each individual query ---
         for (int i = 0; i < num_queries; ++i)
            query_stats[repeat][LsearchId][i].recall = calculate_single_query_recall(gt + i * K, results + i * K, K);

         // --- 5. Compute average recall for the current batch (`LsearchId`) ---
         double total_recall_for_batch = 0.0;
         int total_efs_for_batch = 0;
         double total_ndc_for_batch = 0.0;
         double sum_trie = 0.0;
         double sum_sort = 0.0;
         double sum_filter = 0.0;
         double sum_total = 0.0;
         double sum_mask_gen = 0.0;
         double sum_global_sort = 0.0;

         for (int i = 0; i < num_queries; ++i){
            total_recall_for_batch += query_stats[repeat][LsearchId][i].recall;
            total_efs_for_batch += query_stats[repeat][LsearchId][i].acorn_efs_used;
            total_ndc_for_batch += query_stats[repeat][LsearchId][i].num_distance_calcs;

            // Accumulate timing across ELS stages.
            const auto &s = query_stats[repeat][LsearchId][i];
            sum_trie += s.els_trie_time;
            sum_sort += s.els_sort_time;
            sum_filter += s.els_filter_time;
            sum_total += s.els_total_time;

            sum_mask_gen += query_stats[repeat][LsearchId][i].mask_gen_time_ms;
            sum_global_sort += query_stats[repeat][LsearchId][i].global_sort_time_ms;
         }
         float avg_recall_for_batch = (num_queries > 0) ? (static_cast<float>(total_recall_for_batch) / num_queries) : 0.0f;
         int efs_for_batch = (num_queries > 0) ? (total_efs_for_batch / num_queries) : 0;
         double avg_ndc_for_batch = (num_queries > 0) ? (total_ndc_for_batch / num_queries) : 0.0;
         // Compute average time for each ELS stage.
         double avg_trie = (num_queries > 0) ? sum_trie / num_queries : 0.0;
         double avg_sort = (num_queries > 0) ? sum_sort / num_queries : 0.0;
         double avg_filter = (num_queries > 0) ? sum_filter / num_queries : 0.0;
         double avg_total = (num_queries > 0) ? sum_total / num_queries : 0.0;

         double avg_mask_gen = (num_queries > 0) ? sum_mask_gen / num_queries : 0.0;
         double avg_global_sort = (num_queries > 0) ? sum_global_sort / num_queries : 0.0;

         // Store batch timing and average recall in the reporting structures.
         // a. Save into `detailed_times` for `search_time_details.csv`.
         detailed_times.push_back({repeat, current_Lsearch, efs_for_batch, time_cost, avg_recall_for_batch, avg_trie, avg_sort, avg_filter, avg_total, avg_mask_gen, avg_global_sort});

         // b. Group by Lsearch for final averaging in `search_time_summary.csv`.
         time_per_lsearch[current_Lsearch].push_back(time_cost);
         recall_per_lsearch[current_Lsearch].push_back(avg_recall_for_batch);
         efs_per_lsearch[current_Lsearch].push_back(efs_for_batch);

         std::cout << "  Lsearch=" << current_Lsearch << ", efs=" << efs_per_lsearch[current_Lsearch][0] 
                   << ", global pred time=" << total_global_pred_time << "ms"
                   << ", global sort time=" << total_global_sort_time << "ms"
                   << ", pure search time=" << pure_search_time << "ms"
                   << ", time=" << time_cost << "ms" << ", avg_recall=" << avg_recall_for_batch << std::endl;

         /*// Print recall, ground truth, and retrieved neighbors for each query.
         std::cout << "  --- K-NN Results for Lsearch=" << current_Lsearch << " ---" << std::endl;
         for (int id = 0; id < std::min((int)num_queries, 5); ++id)
         {
            std::cout << "    Query " << id << ":" << std::endl;

            // --- Print this query's recall and exact candidate-set size ---
            float single_query_recall = query_stats[repeat][LsearchId][id].recall;
            size_t exact_cand_size = query_stats[repeat][LsearchId][id].exact_cand_size; // Exact candidate-set size.
            
            std::cout << "      Recall for this query: " << single_query_recall << std::endl;
            std::cout << "      Exact matching vectors in entire DB: " << exact_cand_size << std::endl; // Print the exact candidate count.

            // --- Print the ground truth for comparison ---
            std::cout << "      Ground Truth Neighbors:" << std::endl;
            for (int i = 0; i < K; ++i)
            {
               const auto &gt_pair = gt[id * K + i];
               if (gt_pair.first != -1)
               {
                     std::cout << "        - ID=" << gt_pair.first << ", Distance=" << gt_pair.second << std::endl;
               }
               else
               {
                     break;
               }
            }

            // --- Print the neighbors returned by the algorithm ---
            std::cout << "      Algorithm's Found Neighbors:" << std::endl;
            for (int i = 0; i < K; ++i)
            {
               const auto &result_pair = results[id * K + i];
               if (result_pair.first != -1)
               {
                     std::cout << "        - Rank " << i + 1
                              << ": ID=" << result_pair.first
                              << ", Distance=" << result_pair.second << std::endl;
               }
               else
               {
                     std::cout << "        - Rank " << i + 1 << ": (No more valid results)" << std::endl;
                     break;
               }
            }
            std::cout << "    ------------------------------------" << std::endl;
         }
         std::cout << "  --- End of K-NN Results ---" << std::endl;*/

      }
   }

   // save search_time_details.csv
   std::string details_file_path = result_path_prefix + "search_time_details.csv";
   std::ofstream details_out(details_file_path);
   if (details_out.is_open())
   {
      details_out << "Repeat,Lsearch,efs,Time_ms,Avg_Recall,Avg_Trie,Avg_Sort,Avg_Filter,Avg_Total,Avg_MaskGen,Avg_GlobalSort\n";
      for (const auto &log : detailed_times)
      {
         details_out << log.repeat << "," << log.l_search <<","<< log.efs << "," << log.time_ms << "," << log.avg_recall << "," 
                     << log.els_trie_avg << "," << log.els_sort_avg << "," << log.els_filter_avg << "," << log.els_total_avg << ","
                     << log.mask_gen_avg << "," << log.global_sort_avg << "\n";
      }
      details_out.close();
      std::cout << "\n详细的搜索耗时已保存到: " << details_file_path << std::endl;
   }
   else
   {
      std::cerr << "错误：无法打开文件 " << details_file_path << " 进行写入" << std::endl;
   }

   // save search_time_summary.csv
   std::string summary_file_path = result_path_prefix + "search_time_summary.csv";
   std::ofstream summary_out(summary_file_path);
   if (summary_out.is_open())
   {
      // 1. Extend the header with the `Average_Efs` column.
      summary_out << "Lsearch,Average_Efs,Average_Time_ms,Average_Recall\n";
      for (auto const &[l_search, times] : time_per_lsearch)
      {
         if (!times.empty())
         {
            // Compute average latency.
            double sum_time = std::accumulate(times.begin(), times.end(), 0.0);
            double avg_time = sum_time / times.size();

            // Compute average recall.
            const auto &recalls = recall_per_lsearch.at(l_search);
            double sum_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0f);
            double avg_recall = sum_recall / recalls.size();

            // 2. Compute average EFS.
            const auto &efs_values = efs_per_lsearch.at(l_search);
            double sum_efs = std::accumulate(efs_values.begin(), efs_values.end(), 0.0);
            double avg_efs = sum_efs / efs_values.size();

            // 3. Write `avg_efs` to the output file.
            summary_out << l_search << "," << avg_efs << "," << avg_time << "," << avg_recall << "\n";
         }
      }
      summary_out.close();
      std::cout << "性能汇总 (平均efs/耗时/召回率) 已保存到: " << summary_file_path << std::endl;
   }
   else
   {
      std::cerr << "错误：无法打开文件 " << summary_file_path << " 进行写入" << std::endl;
   }

   // save query details
   std::ofstream detail_out(result_path_prefix + "query_details_repeat" + std::to_string(num_repeats) + ".csv");
   detail_out << "repeat,Lsearch,efs,QueryID,Time_ms,search_time_ms,core_search_time_ms,Recall,"         
              << "Algo_Choice,IsIntelElsUsed,IsTrieRec,"                                                         
              << "DistCalcs,NumNodeVisited,"                                                             
              << "MinSupersetT_ms,"
              << "ELS_TrieT_ms,ELS_SortT_ms,ELS_FilterT_ms,ELS_TotalT_ms,"
            //   << "IntelELS_PredT_ms,Route_PredT_ms,FpassT_ms,Routing_TotalT_ms,BitmapT_new_ms,FeatureT_ms," 
              << "IntelELS_PredT_ms,Route_PredT_ms,Mask_GenT_ms,Global_SortT_ms,FpassT_ms,Routing_TotalT_ms,BitmapT_new_ms,FeatureT_ms,"
              << "RabitQ_CtxPrepareT_ms,RabitQ_CtxRotateT_ms,RabitQ_CtxQ2CentroidsT_ms,RabitQ_CtxWrapperT_ms,"
              << "RabitQ_BinT_ms,RabitQ_FullT_ms,RabitQ_BinCalls,RabitQ_FullCalls,RabitQ_CtxReused,"
              << "AcornFilterType,"
              << "QuerySize,CandSize,ExactCandSize,GlobalPpass,"
              << "NumEntries,NumDescendants,"
              << "Trie_I_lmax,ELS_C_trie,ELS_A_all,ELS_M_min,"
              << "ELS_ContainmentCalls,ELS_LabelSteps,ELS_TrieNodes,"
              << "ELSPP_Scan,ELSPP_Skip,ELSPP_Selected,ELSPP_LNGNodes,ELSPP_LNGEdges,"
              << "ELSPP_MarkTests,ELSPP_NewMarks,ELSPP_DuplicateMarks,"
              << "ELSPP_BitmapT_ms,ELSPP_SortT_ms,ELSPP_ScanT_ms,ELSPP_LNGT_ms,ELSPP_TotalT_ms"
              << "\n";
   for (int repeat = 0; repeat < num_repeats; repeat++)
   {
      for (int LsearchId = 0; LsearchId < Lsearch_list.size(); LsearchId++)
      {
         for (int i = 0; i < num_queries; ++i)
         {
            const auto &stats = query_stats[repeat][LsearchId][i];
            detail_out << repeat << ","
                       << Lsearch_list[LsearchId] << ","
                       << stats.acorn_efs_used << ","
                       << i << ","
                       << stats.time_ms << ","
                       << stats.search_time_ms << ","
                       << stats.core_search_time_ms << ","
                       << stats.recall << ","
                       << stats.algo_choice << ","
                       << stats.is_intel_els_used << "," // <-- IntelELS 模型是否真正被调用
                       << stats.is_trie_recursive << "," // <-- 究竟使用了 nTtrue(1) 还是 nTfalse(0)
                       << stats.num_distance_calcs << ","
                       << stats.num_nodes_visited << ","
                       << stats.get_min_super_sets_time_ms << ","
                       << stats.els_trie_time << ","
                        << stats.els_sort_time << ","
                        << stats.els_filter_time << ","
                        << stats.els_total_time << ","
                       << stats.intel_els_pred_time_ms << ","
                       << stats.route_pred_time_ms << ","
                       << stats.mask_gen_time_ms << ","       
                       << stats.global_sort_time_ms << ","    
                       << stats.fpass_time_ms << ","
                       << stats.routing_total_time_ms << ","
                       << stats.bitmap_time_ms << ","
                       << stats.feature_extract_time_ms << ","
                       << stats.rabitq_ctx_prepare_time_ms << ","
                       << stats.rabitq_ctx_rotate_time_ms << ","
                       << stats.rabitq_ctx_q2c_time_ms << ","
                       << stats.rabitq_ctx_wrapper_time_ms << ","
                       << stats.rabitq_bin_time_ms << ","
                       << stats.rabitq_full_time_ms << ","
                       << stats.rabitq_bin_calls << ","
                       << stats.rabitq_full_calls << ","
                       << stats.rabitq_ctx_reused << ","
                       << stats.acorn_filter_type << ","
                       // Idea1 and trie-related features.
                       << stats.query_length << ","
                       << stats.candidate_set_size << ","
                       << stats.exact_cand_size << ","  
                       << stats.global_p_pass << ","
                       << stats.num_entry_points << ","
                       << stats.num_lng_descendants << ","
                       << stats.trie_I_lmax << ","
                       << stats.els_C_trie << ","
                       << stats.els_A_all << ","
                       << stats.els_M_min << ","
                       << stats.els_containment_calls << ","
                       << stats.els_label_steps << ","
                       << stats.els_trie_nodes << ","
                       << stats.els_scan_count << ","
                       << stats.els_skip_count << ","
                       << stats.els_selected_count << ","
                       << stats.els_lng_nodes << ","
                       << stats.els_lng_edges << ","
                       << stats.els_mark_tests << ","
                       << stats.els_new_marks << ","
                       << stats.els_duplicate_marks << ","
                       << stats.els_bitmap_time << ","
                       << stats.elspp_sort_time << ","
                       << stats.elspp_scan_time << ","
                       << stats.elspp_lng_time << ","
                       << stats.elspp_total_time << "\n";
         }
      }
   }
   detail_out.close();
   
   if (navix_index) {
       delete navix_index;
   }

   std::cout << "- all done" << std::endl;
   return 0;
}
