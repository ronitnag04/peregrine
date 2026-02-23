#ifndef PARAMS_H
#define PARAMS_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace analytical {
#define PARAM_STEP StepType::BASE_2

////////////////////////////////////////////////////////////////////////////
// Parameters
////////////////////////////////////////////////////////////////////////////
enum class ParamType : uint8_t {
  ROB_SIZE,
  COMMIT_WIDTH,
  LOAD_QUEUE_SIZE,
  STORE_QUEUE_SIZE,
  ALU_ISSUE_WIDTH,
  ALU_MUL_ISSUE_WIDTH,
  ALU_DIV_ISSUE_WIDTH,
  FP_ISSUE_WIDTH,
  LS_ISSUE_WIDTH,
  NUM_LS_PIPES,
  NUM_LOAD_PIPES,
  FETCH_WIDTH,
  DECODE_WIDTH,
  RENAME_WIDTH,
  NUM_FETCH_BUFFERS,
  MAX_ICACHE_FILLS,
  BRANCH_PREDICTOR,
  MISPREDICTION_PERCENT,
  L1D_CACHE_KB,
  L1I_CACHE_KB,
  L2_CACHE_KB,
  L1D_STRIDE_PREFETCH,
  COUNT
};

////////////////////////////////////////////////////////////////////////////
// Range Sweeping
////////////////////////////////////////////////////////////////////////////
enum class StepType : uint8_t { LINEAR, BASE_2 };

struct ParamRange {
  uint16_t min;
  uint16_t max;
  StepType step_type;

  constexpr ParamRange(uint16_t min_val, uint16_t max_val,
                       StepType step = StepType::BASE_2)
      : min(min_val), max(max_val), step_type(step) {}

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = uint16_t;
    using difference_type = std::ptrdiff_t;
    using pointer = uint16_t*;
    using reference = uint16_t&;

    uint16_t current;
    uint16_t max;
    StepType step_type;

    Iterator(uint16_t c, uint16_t m, StepType st)
        : current(c), max(m), step_type(st) {}

    uint16_t operator*() const { return current; }

    Iterator& operator++() {
      if (step_type == StepType::LINEAR) {
        ++current;
      } else {  // BASE_2
        if (current == 0) {
          current = 1;
        } else {
          current *= 2;
        }
      }
      return *this;
    }

    bool operator==(const Iterator& other) const {
      return current == other.current;
    }

    bool operator!=(const Iterator& other) const { return !(*this == other); }
  };

  Iterator begin() const { return Iterator(min, max, step_type); }
  Iterator end() const {
    uint16_t end_val = min;
    if (step_type == StepType::BASE_2) {
      end_val = (min == 0) ? 1 : min;
      while (end_val <= max) end_val *= 2;
    } else {
      end_val = max + 1;
    }
    return Iterator(end_val, max, step_type);
  }
};

inline constexpr std::array<ParamRange, (size_t)ParamType::COUNT> PARAM_RANGES =
    {{
        {1, 1024, PARAM_STEP},       // ROB_SIZE
        {1, 12, PARAM_STEP},         // COMMIT_WIDTH
        {1, 256, PARAM_STEP},        // LOAD_QUEUE_SIZE
        {1, 256, PARAM_STEP},        // STORE_QUEUE_SIZE
        {1, 8, PARAM_STEP},          // ALU_ISSUE_WIDTH
        {1, 8, PARAM_STEP},          // ALU_MUL_ISSUE_WIDTH
        {1, 8, PARAM_STEP},          // ALU_DIV_ISSUE_WIDTH
        {1, 8, PARAM_STEP},          // FP_ISSUE_WIDTH
        {1, 8, PARAM_STEP},          // LS_ISSUE_WIDTH
        {1, 8, PARAM_STEP},          // NUM_LS_PIPES
        {0, 8, PARAM_STEP},          // NUM_LOAD_PIPES
        {1, 12, PARAM_STEP},         // FETCH_WIDTH
        {1, 12, PARAM_STEP},         // DECODE_WIDTH
        {1, 12, PARAM_STEP},         // RENAME_WIDTH
        {1, 8, PARAM_STEP},          // NUM_FETCH_BUFFERS
        {1, 32, PARAM_STEP},         // MAX_ICACHE_FILLS
        {0, 1, StepType::LINEAR},    // BRANCH_PREDICTOR
        {0, 100, StepType::LINEAR},  // MISPREDICTION_PERCENT
        {16, 256, PARAM_STEP},       // L1D_CACHE_KB
        {16, 256, PARAM_STEP},       // L1I_CACHE_KB
        {512, 4096, PARAM_STEP},     // L2_CACHE_KB
        {0, 1, StepType::LINEAR}     // L1D_STRIDE_PREFETCH
    }};

inline constexpr ParamRange get_param_range(ParamType type) {
  return PARAM_RANGES[static_cast<size_t>(type)];
}

////////////////////////////////////////////////////////////////////////////
// Range Sweeping with parameter sets
////////////////////////////////////////////////////////////////////////////
struct ParamSweep {
  std::vector<ParamType> param_types;

  ParamSweep(std::initializer_list<ParamType> types) : param_types(types) {}

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::vector<uint16_t>;
    using difference_type = std::ptrdiff_t;
    using pointer = const std::vector<uint16_t>*;
    using reference = const std::vector<uint16_t>&;

    const std::vector<ParamType>* param_types;
    std::vector<ParamRange::Iterator> range_iters;
    std::vector<ParamRange::Iterator> range_ends;
    bool is_end;
    mutable std::vector<uint16_t> current;

    Iterator(const std::vector<ParamType>* types, bool end)
        : param_types(types), is_end(end) {
      if (!is_end && !param_types->empty()) {
        for (ParamType pt : *param_types) {
          auto range = get_param_range(pt);
          range_iters.push_back(range.begin());
          range_ends.push_back(range.end());
        }
        update_current();
      }
    }

    void update_current() {
      current.clear();
      for (const auto& iter : range_iters) {
        current.push_back(*iter);
      }
    }

    const std::vector<uint16_t>& operator*() const { return current; }

    Iterator& operator++() {
      if (is_end || param_types->empty()) return *this;

      for (int i = range_iters.size() - 1; i >= 0; --i) {
        ++range_iters[i];
        if (range_iters[i] != range_ends[i]) {
          update_current();
          return *this;
        }
        auto range = get_param_range((*param_types)[i]);
        range_iters[i] = range.begin();
      }

      is_end = true;
      return *this;
    }

    bool operator==(const Iterator& other) const {
      if (is_end && other.is_end) return true;
      if (is_end != other.is_end) return false;
      return range_iters == other.range_iters;
    }

    bool operator!=(const Iterator& other) const { return !(*this == other); }
  };

  Iterator begin() const { return Iterator(&param_types, false); }
  Iterator end() const { return Iterator(&param_types, true); }
};

}  // namespace analytical

#endif  // PARAMS_H