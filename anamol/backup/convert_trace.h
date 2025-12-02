#ifndef CONVERT_TRACE_H
#define CONVERT_TRACE_H

#include <string>
#include <vector>

#include "instr.h"
#include "parse_csv.h"

namespace analytical {

// Convert parsed trace data to analytical instruction format
std::vector<Instr> convert_trace(
    const std::vector<tracing::instr_trace_t>& trace_data);

std::vector<Instr> read_trace(const std::string& filename);

}  // namespace analytical

#endif  // CONVERT_TRACE_H