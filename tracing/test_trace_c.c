// C wrapper to call the assembly test function
// Compile with: gcc -o test_trace_c test_trace_c.c test_trace_c.s

extern void test_trace_function(void);

int main() {
    test_trace_function();
    return 0;
}

