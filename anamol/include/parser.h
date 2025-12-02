#ifndef PARSER_H
#define PARSER_H

#pragma once

#include <string>
#include <vector>

#include "instr.h"

namespace analytical {

// Step 1: CSV → instr_trace_t
std::vector<tracing::instr_trace_t> parse_csv(const std::string& csv_path);

// Step 2: instr_trace_t → Instr
Instr convert_to_instr(const tracing::instr_trace_t& inst, instr_id_t id);

// Helper: Convert entire trace
std::vector<Instr> convert_trace(
    const std::vector<tracing::instr_trace_t>& instructions);

// Complete pipeline: CSV → Instr
std::vector<Instr> parse_and_convert(const std::string& csv_path);

}  // namespace analytical

#endif  // PARSER_H