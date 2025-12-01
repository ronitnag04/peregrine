#ifndef PARSE_CSV_H
#define PARSE_CSV_H

#include <string>
#include <vector>

#include "instr.h"

namespace tracing {

// Main function to parse CSV file containing instruction trace data
// Returns a vector of instruction_data_t structs, one per line (excluding
// header)
std::vector<instr_trace_t> parse_trace_csv(const std::string& filename);

}  // namespace tracing

#endif  // PARSE_CSV_H