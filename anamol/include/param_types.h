#ifndef PARAM_TYPES_H
#define PARAM_TYPES_H

// param_types.h — manual infrastructure for parameter sweeping.
// Contains only StepType and ParamRange.
//
// ParamSweep, ParamType, PARAM_RANGES, and get_param_range() live in
// params_gen.h (auto-generated from registry.yaml), which includes this file.

#include <cstddef>
#include <cstdint>
#include <iterator>

namespace analytical {

////////////////////////////////////////////////////////////////////////////
// Step type
////////////////////////////////////////////////////////////////////////////
enum class StepType : uint8_t { LINEAR, BASE_2 };

////////////////////////////////////////////////////////////////////////////
// ParamRange — iterable range over one parameter's sweep values
////////////////////////////////////////////////////////////////////////////
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

}  // namespace analytical

#endif  // PARAM_TYPES_H
