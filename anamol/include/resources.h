#ifndef RESOURCES_H
#define RESOURCES_H

#include <cstdint>

namespace analytical {

enum class Resource : uint8_t {
  ROB,
  LOAD_QUEUE,
  STORE_QUEUE,
  ALU_ISSUE,
  ALU_MUL_ISSUE,
  ALU_DIV_ISSUE,
  FP_ISSUE,
  LS_ISSUE,
  LOAD_LS_PIPES_LOWER,
  LOAD_LS_PIPES_UPPER,
  ICACHE_FILLS,
  FETCH_BUFFERS,
  COUNT
};

}  // namespace analytical

#endif  // RESOURCES_H