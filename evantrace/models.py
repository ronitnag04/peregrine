from .instructions import Instruction

def is_load(instruction: Instruction):
    # TODO
    return False
    
def get_constant_latency(instruction: Instruction):
    # TODO
    return 0

"""
Recreation of the resp_cycle function from
Concorde. Refines the predictions of load
execution times from initial in-order
simulation
"""
def resp_cycle(
    req_cycle: int,
    instruction: Instruction,
    last_req_cycle: dict[int, int],
    last_resp_cycle: dict[int, int],
    access_count: dict[int, int],
    exec_time: dict[int, list[int]]
):
    cache_line = instruction.dcache_line
    # enforce that req_cycle must be non-decreasing for requests to the same cache line
    if cache_line and cache_line in last_req_cycle and last_req_cycle[cache_line] < req_cycle:
        raise ValueError("Request cycles for the same cache line should be non-decreasing.")
        
    if is_load(instruction):
        prev_resp_cycle = last_resp_cycle[cache_line]
        access_number = access_count[cache_line]
        exec_time = exec_time[cache_line][access_number]
        resp_cycle = max(req_cycle + exec_time, prev_resp_cycle)
        last_resp_cycle[cache_line] = resp_cycle
        access_count[cache_line] += 1
    else:
        resp_cycle = req_cycle + get_constant_latency(instruction)