"""
Entry point
"""
import sys
import csv

from humanfriendly import parse_size

from evantrace.parser import Parser
from evantrace.writer import Writer
from evantrace.caches import Cache
from evantrace.sim import Sim
from evantrace.x86.instructions import Instruction
from evantrace.branch_predictor import LocalBranchPredictor, TAGEBranchPredictor

def process_row(row: dict[str, str], trace: list[Instruction]):
    filename = (
        f"output/"
        f"{row['branch_predictor']}_"
        f"{row['commit_width']}_"
        f"{row['decode_width']}_"
        f"{row['fetch_width']}_"
        f"{row['fp_mult_div_issue_width']}_"
        f"{row['fp_reg_issue_width']}_"
        f"{row['int_mult_div_issue_width']}_"
        f"{row['int_reg_issue_width']}_"
        f"{row['l1d_size']}_"
        f"{row['l1i_size']}_"
        f"{row['l2_size']}_"
        f"{row['lq_entries']}_"
        f"{row['max_icache_fills']}_"
        f"{row['rdwr_port_issue_width']}_"
        f"{row['read_port_issue_width']}_"
        f"{row['rename_width']}_"
        f"{row['rob_size']}_"
        f"{row['simd_unit_issue_width']}_"
        f"{row['sq_entries']}_"
        f"{row['stride_prefetcher_degree']}_"
        f"{row['wb_width']}"
        f".csv"
    )
    
    writer = Writer(filename)
    
    l2cache = Cache(
        associativity=4,
        total_size=parse_size(row['l2_size']),
        read_latency=12
    )
    
    icache = Cache(
        associativity=4,
        total_size=parse_size(row['l1i_size']),
        read_latency=4
    )
    
    dcache = Cache(
        associativity=4,
        total_size=parse_size(row['l1d_size']),
        read_latency=4
    )
    
    if row['branch_predictor'] == 'local':
        branch_predictor = LocalBranchPredictor(local_predictor_size=2048, local_ctr_bits=2)
    else:
        branch_predictor = TAGEBranchPredictor()
        
    sim = Sim(
        trace=trace, 
        icache=icache, 
        dcache=dcache, 
        l2cache=l2cache, 
        branch_predictor=branch_predictor
    )
    sim.run()
    
    writer.write(trace)
        

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <trace.csv> <sweep.csv>")
        sys.exit(1)
        
    trace_filename = sys.argv[1]
    parser = Parser(trace_filename)
    instructions = parser.parse()
    
    sweep_filename = sys.argv[2]
    try:
        with open(sweep_filename, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                process_row(row, instructions)
    except FileNotFoundError:
        print(f"Error: File '{sweep_filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
