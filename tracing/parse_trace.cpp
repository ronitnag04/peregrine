#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <algorithm>

// Memory access structure
typedef struct {
    unsigned long addr;
    unsigned int size;
} mem_access_t;

// Instruction data structure matching the original
typedef struct {
    unsigned long ip;
    std::string assembly;
    std::string category;
    std::string opcode;
    std::string branch_type;
    bool branch_taken;
    unsigned long branch_target_addr;
    bool inst_sync;
    std::vector<std::string> read_registers;
    std::vector<std::string> write_registers;
    std::vector<unsigned long> reg_dependent_ips;
    std::vector<mem_access_t> read_addresses;
    std::vector<mem_access_t> write_addresses;
    std::vector<unsigned long> mem_dependent_ips;
} instruction_data_t;

// Helper function to trim whitespace
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

// Helper function to split a string by delimiter
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        token = trim(token);
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

// Helper function to parse hex address
unsigned long parse_hex(const std::string& hex_str) {
    return std::strtoul(hex_str.c_str(), nullptr, 16);
}

// Helper function to parse memory address in format "0xADDR(SIZE)"
mem_access_t parse_mem_address(const std::string& mem_str) {
    mem_access_t mem;
    mem.addr = 0;
    mem.size = 0;
    
    // Find the opening parenthesis
    size_t paren_pos = mem_str.find('(');
    if (paren_pos == std::string::npos) {
        // Try to parse as just an address
        mem.addr = parse_hex(mem_str);
        return mem;
    }
    
    // Extract address part (before parenthesis)
    std::string addr_str = trim(mem_str.substr(0, paren_pos));
    mem.addr = parse_hex(addr_str);
    
    // Extract size part (between parentheses)
    size_t close_paren = mem_str.find(')', paren_pos);
    if (close_paren != std::string::npos) {
        std::string size_str = trim(mem_str.substr(paren_pos + 1, close_paren - paren_pos - 1));
        mem.size = std::stoul(size_str, nullptr, 10);
    }
    
    return mem;
}

// Helper function to parse CSV field, handling quoted fields
std::string parse_csv_field(const std::string& line, size_t& pos) {
    std::string field;
    bool in_quotes = false;
    
    while (pos < line.length()) {
        char c = line[pos];
        
        if (c == '"') {
            if (in_quotes && pos + 1 < line.length() && line[pos + 1] == '"') {
                // Escaped quote
                field += '"';
                pos += 2;
            } else {
                // Toggle quote state
                in_quotes = !in_quotes;
                pos++;
            }
        } else if (c == ',' && !in_quotes) {
            // End of field
            pos++;
            break;
        } else {
            field += c;
            pos++;
        }
    }
    
    return trim(field);
}

// Parse a single CSV line into an instruction_data_t struct
instruction_data_t parse_csv_line(const std::string& line) {
    instruction_data_t inst;
    
    size_t pos = 0;
    
    // IP (column 0)
    std::string ip_str = parse_csv_field(line, pos);
    inst.ip = parse_hex(ip_str);
    
    // Assembly (column 1)
    inst.assembly = parse_csv_field(line, pos);
    
    // Category (column 2)
    inst.category = parse_csv_field(line, pos);
    
    // Opcode (column 3)
    inst.opcode = parse_csv_field(line, pos);
    
    // Branch Type (column 4)
    inst.branch_type = parse_csv_field(line, pos);
    
    // Branch Taken (column 5)
    std::string branch_taken_str = parse_csv_field(line, pos);
    if (!branch_taken_str.empty()) {
        inst.branch_taken = (branch_taken_str == "true");
    } else {
        inst.branch_taken = false;
    }
    
    // Branch Target Address (column 6)
    std::string branch_target_str = parse_csv_field(line, pos);
    if (!branch_target_str.empty()) {
        inst.branch_target_addr = parse_hex(branch_target_str);
    } else {
        inst.branch_target_addr = 0;
    }
    
    // Instruction Sync (column 7)
    std::string sync_str = parse_csv_field(line, pos);
    inst.inst_sync = (sync_str == "true");
    
    // Read Registers (column 8)
    std::string read_regs_str = parse_csv_field(line, pos);
    if (!read_regs_str.empty()) {
        inst.read_registers = split(read_regs_str, ';');
    }
    
    // Write Registers (column 9)
    std::string write_regs_str = parse_csv_field(line, pos);
    if (!write_regs_str.empty()) {
        inst.write_registers = split(write_regs_str, ';');
    }
    
    // Register Dependent IPs (column 10)
    std::string reg_dep_ips_str = parse_csv_field(line, pos);
    if (!reg_dep_ips_str.empty()) {
        std::vector<std::string> ip_strs = split(reg_dep_ips_str, ';');
        for (const auto& ip_str : ip_strs) {
            inst.reg_dependent_ips.push_back(parse_hex(ip_str));
        }
    }
    
    // Read Addresses (column 11)
    std::string read_addrs_str = parse_csv_field(line, pos);
    if (!read_addrs_str.empty()) {
        std::vector<std::string> addr_strs = split(read_addrs_str, ';');
        for (const auto& addr_str : addr_strs) {
            inst.read_addresses.push_back(parse_mem_address(addr_str));
        }
    }
    
    // Write Addresses (column 12)
    std::string write_addrs_str = parse_csv_field(line, pos);
    if (!write_addrs_str.empty()) {
        std::vector<std::string> addr_strs = split(write_addrs_str, ';');
        for (const auto& addr_str : addr_strs) {
            inst.write_addresses.push_back(parse_mem_address(addr_str));
        }
    }
    
    // Memory Dependent IPs (column 13)
    std::string mem_dep_ips_str = parse_csv_field(line, pos);
    if (!mem_dep_ips_str.empty()) {
        std::vector<std::string> ip_strs = split(mem_dep_ips_str, ';');
        for (const auto& ip_str : ip_strs) {
            inst.mem_dependent_ips.push_back(parse_hex(ip_str));
        }
    }
    
    return inst;
}

// Main function to parse CSV file
std::vector<instruction_data_t> parse_trace_csv(const std::string& filename) {
    std::vector<instruction_data_t> instructions;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return instructions;
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        // Skip header line
        if (first_line) {
            first_line = false;
            continue;
        }
        
        // Skip empty lines
        if (trim(line).empty()) {
            continue;
        }
        
        // Parse the line
        instruction_data_t inst = parse_csv_line(line);
        instructions.push_back(inst);
    }
    
    file.close();
    return instructions;
}

// Print usage information
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [CSV_FILE] [OPTIONS]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --range START END     Print instructions from START to END (inclusive)" << std::endl;
    std::cout << "  --start START --num NUM   Print NUM instructions starting from START" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " trace.csv --range 0 10" << std::endl;
    std::cout << "  " << program_name << " trace.csv --start 100 --num 5" << std::endl;
    std::cout << "  " << program_name << " trace.csv  (prints first 5 instructions by default)" << std::endl;
}

// Print a single instruction
void print_instruction(const instruction_data_t& inst, int index) {
    std::cout << "\nInstruction " << index << ":" << std::endl;
    std::cout << "  IP: 0x" << std::hex << inst.ip << std::dec << std::endl;
    std::cout << "  Assembly: " << inst.assembly << std::endl;
    std::cout << "  Category: " << inst.category << std::endl;
    std::cout << "  Opcode: " << inst.opcode << std::endl;
    std::cout << "  Branch Type: " << (inst.branch_type.empty() ? "(none)" : inst.branch_type) << std::endl;
    if (!inst.branch_type.empty()) {
        std::cout << "  Branch Taken: " << (inst.branch_taken ? "true" : "false") << std::endl;
        std::cout << "  Branch Target Address: 0x" << std::hex << inst.branch_target_addr << std::dec << std::endl;
    } else {
        std::cout << "  Branch Taken: (none)" << std::endl;
        std::cout << "  Branch Target Address: (none)" << std::endl;
    }
    std::cout << "  Instruction Sync: " << (inst.inst_sync ? "true" : "false") << std::endl;
    
    // Read Registers
    std::cout << "  Read Registers: ";
    if (inst.read_registers.empty()) {
        std::cout << "(none)";
    } else {
        for (size_t i = 0; i < inst.read_registers.size(); i++) {
            if (i > 0) std::cout << "; ";
            std::cout << inst.read_registers[i];
        }
    }
    std::cout << std::endl;
    
    // Write Registers
    std::cout << "  Write Registers: ";
    if (inst.write_registers.empty()) {
        std::cout << "(none)";
    } else {
        for (size_t i = 0; i < inst.write_registers.size(); i++) {
            if (i > 0) std::cout << "; ";
            std::cout << inst.write_registers[i];
        }
    }
    std::cout << std::endl;
    
    // Register Dependent IPs
    std::cout << "  Register Dependent IPs: ";
    if (inst.reg_dependent_ips.empty()) {
        std::cout << "(none)";
    } else {
        for (size_t i = 0; i < inst.reg_dependent_ips.size(); i++) {
            if (i > 0) std::cout << "; ";
            std::cout << "0x" << std::hex << inst.reg_dependent_ips[i] << std::dec;
        }
    }
    std::cout << std::endl;
    
    // Read Addresses
    std::cout << "  Read Addresses: ";
    if (inst.read_addresses.empty()) {
        std::cout << "(none)";
    } else {
        for (size_t i = 0; i < inst.read_addresses.size(); i++) {
            if (i > 0) std::cout << "; ";
            std::cout << "0x" << std::hex << inst.read_addresses[i].addr << std::dec;
            if (inst.read_addresses[i].size > 0) {
                std::cout << "(" << inst.read_addresses[i].size << ")";
            }
        }
    }
    std::cout << std::endl;
    
    // Write Addresses
    std::cout << "  Write Addresses: ";
    if (inst.write_addresses.empty()) {
        std::cout << "(none)";
    } else {
        for (size_t i = 0; i < inst.write_addresses.size(); i++) {
            if (i > 0) std::cout << "; ";
            std::cout << "0x" << std::hex << inst.write_addresses[i].addr << std::dec;
            if (inst.write_addresses[i].size > 0) {
                std::cout << "(" << inst.write_addresses[i].size << ")";
            }
        }
    }
    std::cout << std::endl;
    
    // Memory Dependent IPs
    std::cout << "  Memory Dependent IPs: ";
    if (inst.mem_dependent_ips.empty()) {
        std::cout << "(none)";
    } else {
        for (size_t i = 0; i < inst.mem_dependent_ips.size(); i++) {
            if (i > 0) std::cout << "; ";
            std::cout << "0x" << std::hex << inst.mem_dependent_ips[i] << std::dec;
        }
    }
    std::cout << std::endl;
}

// Example usage and demonstration
int main(int argc, char* argv[]) {
    std::string csv_filename = "trace.csv";
    int start_idx = -1;
    int end_idx = -1;
    int num_instructions = -1;
    bool use_range = false;
    bool use_start_num = false;
    
    // Parse command-line arguments
    int arg_idx = 1;
    
    // First argument might be the filename
    if (argc > 1 && argv[arg_idx][0] != '-') {
        csv_filename = argv[arg_idx];
        arg_idx++;
    }
    
    // Parse options
    while (arg_idx < argc) {
        std::string arg = argv[arg_idx];
        
        if (arg == "--range") {
            if (arg_idx + 2 >= argc) {
                std::cerr << "Error: --range requires two arguments (START END)" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
            start_idx = std::stoi(argv[arg_idx + 1]);
            end_idx = std::stoi(argv[arg_idx + 2]);
            use_range = true;
            arg_idx += 3;
        } else if (arg == "--start") {
            if (arg_idx + 1 >= argc) {
                std::cerr << "Error: --start requires an argument" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
            start_idx = std::stoi(argv[arg_idx + 1]);
            use_start_num = true;
            arg_idx += 2;
        } else if (arg == "--num") {
            if (arg_idx + 1 >= argc) {
                std::cerr << "Error: --num requires an argument" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
            num_instructions = std::stoi(argv[arg_idx + 1]);
            use_start_num = true;
            arg_idx += 2;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate that we have either both range values or both start/num
    if (use_range && use_start_num) {
        std::cerr << "Error: Cannot use both --range and --start/--num together" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    if (use_start_num && (start_idx == -1 || num_instructions == -1)) {
        std::cerr << "Error: --start and --num must both be specified together" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    std::cout << "Parsing CSV file: " << csv_filename << std::endl;
    
    // Parse the CSV file
    std::vector<instruction_data_t> instructions = parse_trace_csv(csv_filename);
    
    std::cout << "Parsed " << instructions.size() << " instructions" << std::endl;
    
    // Determine which instructions to print
    int print_start = 0;
    int print_end = 0;
    
    if (use_range) {
        print_start = std::max(0, start_idx);
        print_end = std::min((int)instructions.size() - 1, end_idx);
        if (print_start > print_end) {
            std::cerr << "Error: Invalid range: start (" << start_idx << ") > end (" << end_idx << ")" << std::endl;
            return 1;
        }
        std::cout << "\nPrinting instructions " << print_start << " to " << print_end << " (inclusive):" << std::endl;
    } else if (use_start_num) {
        print_start = std::max(0, start_idx);
        print_end = std::min((int)instructions.size() - 1, print_start + num_instructions - 1);
        std::cout << "\nPrinting " << num_instructions << " instructions starting from " << print_start << ":" << std::endl;
    } else {
        // Default: print first 5 instructions
        print_start = 0;
        print_end = std::min(4, (int)instructions.size() - 1);
        std::cout << "\nPrinting first " << (print_end - print_start + 1) << " instructions (default):" << std::endl;
    }
    
    std::cout << "===========================================" << std::endl;
    
    for (int i = print_start; i <= print_end; i++) {
        print_instruction(instructions[i], i);
    }
    
    // The instructions vector is now available for further analysis
    // You can add your analysis code here
    
    return 0;
}

