#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "uni_nav_graph.h"

namespace {

class QueryStorage final : public ANNS::IStorage {
 public:
  explicit QueryStorage(std::vector<std::vector<ANNS::LabelType>> labels)
      : labels_(std::move(labels)) {}

  void load_from_file(const std::string&, const std::string&, ANNS::IdxType) override {}
  void write_to_file(const std::string&, const std::string&) override {}
  void reorder_data(const std::vector<ANNS::IdxType>&) override {}
  ANNS::DataType get_data_type() const override { return ANNS::DataType::FLOAT; }
  ANNS::IdxType get_num_points() const override {
    return static_cast<ANNS::IdxType>(labels_.size());
  }
  ANNS::IdxType get_dim() const override { return 0; }
  std::vector<ANNS::LabelType>* get_offseted_label_sets(ANNS::IdxType idx) override {
    return labels_.data() + idx;
  }
  char* get_vector(ANNS::IdxType) override { return nullptr; }
  std::vector<ANNS::LabelType>& get_label_set(ANNS::IdxType idx) override {
    return labels_[idx];
  }
  void prefetch_vec_by_id(ANNS::IdxType) const override {}
  ANNS::IdxType choose_medoid(
      uint32_t, std::shared_ptr<ANNS::DistanceHandler>) override {
    return 0;
  }
  void clean() override {}

 private:
  std::vector<std::vector<ANNS::LabelType>> labels_;
};

bool expect_equal(
    const std::vector<int>& actual,
    const std::vector<int>& expected,
    const std::string& case_name) {
  if (actual == expected) return true;

  std::cerr << case_name << " failed. actual:";
  for (int id : actual) std::cerr << ' ' << id;
  std::cerr << "; expected:";
  for (int id : expected) std::cerr << ' ' << id;
  std::cerr << '\n';
  return false;
}

}  // namespace

int main() {
  std::shared_ptr<ANNS::IStorage> storage = std::make_shared<QueryStorage>(
      std::vector<std::vector<ANNS::LabelType>>{
          {2},       // 0: algorithm 15
          {1, 3},    // 1: algorithm 12
          {1, 2},    // 2: algorithm 12
          {1, 2},    // 3: identical labels; query id breaks the tie
          {},        // 4: algorithm 5
          {10},      // 5: algorithm 15
          {2},       // 6: algorithm 12
          {1},       // 7: prefix of {1, 2}
      });
  const std::vector<int> choices{15, 12, 12, 12, 5, 15, 12, 12};

  ANNS::UniNavGraph index;
  const auto alps_plus_order = index.get_sorted_query_ids(storage, choices, 5);
  if (!expect_equal(alps_plus_order, {4, 7, 2, 3, 1, 6, 0, 5}, "ALPS+ order")) {
    return EXIT_FAILURE;
  }

  const auto alps_order = index.get_sorted_query_ids(storage, choices, 1);
  if (!expect_equal(alps_order, {0, 1, 2, 3, 4, 5, 6, 7}, "ALPS unchanged order")) {
    return EXIT_FAILURE;
  }

  std::cout << "ALPS+ query sorting tests passed\n";
  return EXIT_SUCCESS;
}
