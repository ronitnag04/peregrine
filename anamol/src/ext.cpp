#include "ext.h"

// Simple response cycle model
// Returns the cycle when an instruction completes
uint32_t resp_cycle(const uint32_t start_cycle, const uint64_t IP) {
  // Simple model: most instructions take 1 cycle
  // This can be enhanced with actual latency modeling based on instruction type
  return start_cycle + 1;
}