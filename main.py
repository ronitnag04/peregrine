"""
Entry point
"""
import sys
from evantrace.parser import Parser
from evantrace.writer import Writer
from evantrace.caches import Cache
from evantrace.sim import Sim

def main():
    print("Hello from evantrace! Beginning simulation...")
    if len(sys.argv) < 2:
        print("Need an input and an output file.")
        
    parser = Parser(sys.argv[1])
    instructions = parser.parse()
    
    l2cache = Cache(
        associativity=4,
        total_size=524_288,
        read_latency=12
    )
    
    icache = Cache(
        associativity=4,
        parent=l2cache
    )
    
    dcache = Cache(
        associativity=4,
        parent=l2cache
    )
    
    sim = Sim(trace=instructions, icache=icache, dcache=dcache, l2cache=l2cache)
    sim.run()
    
    writer = Writer(sys.argv[2])
    writer.write(instructions)

if __name__ == "__main__":
    main()
