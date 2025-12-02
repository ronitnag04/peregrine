#include <iostream>
#include <string>

#include "params.h"

using namespace analytical;

std::string param_to_string(ParamType p) {
  switch (p) {
    case ParamType::ROB_SIZE:
      return "ROB_SIZE";
    case ParamType::COMMIT_WIDTH:
      return "COMMIT_WIDTH";
    case ParamType::LOAD_QUEUE_SIZE:
      return "LOAD_QUEUE_SIZE";
    case ParamType::STORE_QUEUE_SIZE:
      return "STORE_QUEUE_SIZE";
    case ParamType::ALU_ISSUE_WIDTH:
      return "ALU_ISSUE_WIDTH";
    case ParamType::FP_ISSUE_WIDTH:
      return "FP_ISSUE_WIDTH";
    case ParamType::LS_ISSUE_WIDTH:
      return "LS_ISSUE_WIDTH";
    case ParamType::NUM_LS_PIPES:
      return "NUM_LS_PIPES";
    case ParamType::NUM_LOAD_PIPES:
      return "NUM_LOAD_PIPES";
    case ParamType::FETCH_WIDTH:
      return "FETCH_WIDTH";
    case ParamType::DECODE_WIDTH:
      return "DECODE_WIDTH";
    case ParamType::RENAME_WIDTH:
      return "RENAME_WIDTH";
    case ParamType::NUM_FETCH_BUFFERS:
      return "NUM_FETCH_BUFFERS";
    case ParamType::MAX_ICACHE_FILLS:
      return "MAX_ICACHE_FILLS";
    case ParamType::BRANCH_PREDICTOR:
      return "BRANCH_PREDICTOR";
    case ParamType::MISPREDICTION_PERCENT:
      return "MISPREDICTION_PERCENT";
    case ParamType::L1D_CACHE_KB:
      return "L1D_CACHE_KB";
    case ParamType::L1I_CACHE_KB:
      return "L1I_CACHE_KB";
    case ParamType::L2_CACHE_KB:
      return "L2_CACHE_KB";
    case ParamType::L1D_STRIDE_PREFETCH:
      return "L1D_STRIDE_PREFETCH";
    default:
      return "UNKNOWN";
  }
}

int main() {
  std::cout << "=== Testing Params with LINEAR StepType ===" << std::endl;

  // LINEAR step type parameters (only these ones)
  ParamType linear_params[] = {ParamType::BRANCH_PREDICTOR,
                               ParamType::MISPREDICTION_PERCENT,
                               ParamType::L1D_STRIDE_PREFETCH};

  for (auto param : linear_params) {
    ParamRange range = get_param_range(param);  // Changed
    std::cout << "\n" << param_to_string(param) << " (LINEAR):" << std::endl;
    std::cout << "  Range: [" << range.min << ", " << range.max << "]"
              << std::endl;
    std::cout << "  Values: ";
    for (auto val : range) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }

  // All other parameters - show both LINEAR and BASE_2
  ParamType other_params[] = {
      ParamType::ROB_SIZE,          ParamType::COMMIT_WIDTH,
      ParamType::LOAD_QUEUE_SIZE,   ParamType::STORE_QUEUE_SIZE,
      ParamType::ALU_ISSUE_WIDTH,   ParamType::FP_ISSUE_WIDTH,
      ParamType::LS_ISSUE_WIDTH,    ParamType::NUM_LS_PIPES,
      ParamType::NUM_LOAD_PIPES,    ParamType::FETCH_WIDTH,
      ParamType::DECODE_WIDTH,      ParamType::RENAME_WIDTH,
      ParamType::NUM_FETCH_BUFFERS, ParamType::MAX_ICACHE_FILLS,
      ParamType::L1D_CACHE_KB,      ParamType::L1I_CACHE_KB,
      ParamType::L2_CACHE_KB};

  std::cout << "\n=== Testing Other Params with LINEAR StepType ==="
            << std::endl;

  for (auto param : other_params) {
    ParamRange range = get_param_range(param);  // Changed
    // Create a linear version
    ParamRange linear_range(range.min, range.max, StepType::LINEAR);

    std::cout << "\n" << param_to_string(param) << " (LINEAR):" << std::endl;
    std::cout << "  Range: [" << linear_range.min << ", " << linear_range.max
              << "]" << std::endl;
    std::cout << "  Values: ";
    for (auto val : linear_range) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "\n=== Testing Other Params with BASE_2 StepType ==="
            << std::endl;

  for (auto param : other_params) {
    ParamRange range = get_param_range(param);  // Changed
    std::cout << "\n" << param_to_string(param) << " (BASE_2):" << std::endl;
    std::cout << "  Range: [" << range.min << ", " << range.max << "]"
              << std::endl;
    std::cout << "  Values: ";
    for (auto val : range) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }

  // New ParamSweep tests
  std::cout << "\n\n=== Testing ParamSweep ===" << std::endl;

  // Test 1: Single parameter sweep
  std::cout << "\n--- Single Parameter: ROB_SIZE ---" << std::endl;
  ParamSweep sweep1{ParamType::ROB_SIZE};
  int count1 = 0;
  std::cout << "First 10 combinations:" << std::endl;
  for (const auto& params : sweep1) {
    if (count1 < 10) {
      std::cout << "  [" << params[0] << "]" << std::endl;
    }
    count1++;
  }
  std::cout << "Total combinations: " << count1 << std::endl;

  // Test 2: Two parameter sweep (small range for readability)
  std::cout
      << "\n--- Two Parameters: BRANCH_PREDICTOR x L1D_STRIDE_PREFETCH ---"
      << std::endl;
  ParamSweep sweep2{ParamType::BRANCH_PREDICTOR,
                    ParamType::L1D_STRIDE_PREFETCH};
  int count2 = 0;
  std::cout << "All combinations:" << std::endl;
  for (const auto& params : sweep2) {
    std::cout << "  [" << params[0] << ", " << params[1] << "]" << std::endl;
    count2++;
  }
  std::cout << "Total combinations: " << count2 << std::endl;

  // Test 3: Three parameter sweep
  std::cout << "\n--- Three Parameters: BRANCH_PREDICTOR x L1D_STRIDE_PREFETCH "
               "x NUM_FETCH_BUFFERS ---"
            << std::endl;
  ParamSweep sweep3{ParamType::BRANCH_PREDICTOR, ParamType::L1D_STRIDE_PREFETCH,
                    ParamType::NUM_FETCH_BUFFERS};
  int count3 = 0;
  std::cout << "First 20 combinations:" << std::endl;
  for (const auto& params : sweep3) {
    if (count3 < 20) {
      std::cout << "  [" << params[0] << ", " << params[1] << ", " << params[2]
                << "]" << std::endl;
    }
    count3++;
  }
  std::cout << "Total combinations: " << count3 << std::endl;

  // Test 4: Load/Store pipes sweep (NUM_LS_PIPES x NUM_LOAD_PIPES)
  std::cout << "\n--- Two Parameters: NUM_LS_PIPES x NUM_LOAD_PIPES ---"
            << std::endl;
  ParamSweep sweep4{ParamType::NUM_LS_PIPES, ParamType::NUM_LOAD_PIPES};
  int count4 = 0;
  std::cout << "First 30 combinations:" << std::endl;
  for (const auto& params : sweep4) {
    if (count4 < 30) {
      std::cout << "  [" << params[0] << ", " << params[1] << "]" << std::endl;
    }
    count4++;
  }
  std::cout << "Total combinations: " << count4 << std::endl;

  // Test 5: Verify expected counts
  std::cout << "\n--- Verification of Cartesian Product Sizes ---" << std::endl;

  ParamSweep verify1{ParamType::BRANCH_PREDICTOR};  // 2 values (0, 1)
  int v1 = 0;
  for (const auto& p : verify1) v1++;
  std::cout << "BRANCH_PREDICTOR: " << v1 << " (expected: 2)" << std::endl;

  ParamSweep verify2{ParamType::BRANCH_PREDICTOR,
                     ParamType::L1D_STRIDE_PREFETCH};  // 2 x 2 = 4
  int v2 = 0;
  for (const auto& p : verify2) v2++;
  std::cout << "BRANCH_PREDICTOR x L1D_STRIDE_PREFETCH: " << v2
            << " (expected: 4)" << std::endl;

  // With PARAM_STEP = LINEAR:
  // NUM_LS_PIPES: [1..8] = 8 values
  // NUM_LOAD_PIPES: [0..8] = 9 values
  // Total: 8 x 9 = 72
  ParamSweep verify3{ParamType::NUM_LS_PIPES, ParamType::NUM_LOAD_PIPES};
  int v3 = 0;
  for (const auto& p : verify3) v3++;
  std::cout << "NUM_LS_PIPES x NUM_LOAD_PIPES: " << v3
            << " (expected: 72 with LINEAR, 4 x 5 = 20 with BASE_2)"
            << std::endl;

  // Additional verification for ROB_SIZE
  ParamSweep verify4{ParamType::ROB_SIZE};
  int v4 = 0;
  for (const auto& p : verify4) v4++;
  std::cout << "ROB_SIZE: " << v4
            << " (expected: 1024 with LINEAR, 11 with BASE_2: "
               "1,2,4,8,16,32,64,128,256,512,1024)"
            << std::endl;

  // Show what PARAM_STEP is currently set to
  std::cout << "\n--- Current PARAM_STEP Setting ---" << std::endl;
  auto rob_range = get_param_range(ParamType::ROB_SIZE);  // Changed
  std::cout << "PARAM_STEP is currently: "
            << (rob_range.step_type == StepType::LINEAR ? "LINEAR" : "BASE_2")
            << std::endl;

  return 0;
}