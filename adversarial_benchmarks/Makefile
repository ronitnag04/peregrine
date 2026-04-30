# Adversarial benchmarks for Ronamol
#
# All generated artifacts (binaries, generated headers, object files)
# land in build/. The source tree stays clean.

CC      ?= gcc
CFLAGS  ?= -O2 -g -Wall -Wextra -fno-inline
LDFLAGS ?=

BUILD   := build
GEN     := $(BUILD)/gen
OBJ     := $(BUILD)/obj
BIN     := $(BUILD)/bin

# Executables
BINS := \
    $(BIN)/pow2_stride_thrash \
    $(BIN)/pow2_stride_benign \
    $(BIN)/icache_blast \
    $(BIN)/ptrchase_rand \
    $(BIN)/serial_mul_chain \
    $(BIN)/many_pages_streaming \
    $(BIN)/adversarial_branches \
    $(BIN)/stlf_misalign

# Generated sources
GEN_HEADERS := $(GEN)/icache_blast_gen.h

.PHONY: all clean dirs
all: dirs $(BINS)

dirs:
	@mkdir -p $(GEN) $(OBJ) $(BIN)

# ---- Generated headers ----------------------------------------------------

$(GEN)/icache_blast_gen.h: gen_icache_blast.py | dirs
	python3 $< > $@

# ---- Two variants of the stride benchmark from one source ----------------

$(OBJ)/pow2_stride_thrash.o: pow2_stride_thrash.c | dirs
	$(CC) $(CFLAGS) -DSTRIDE_LOG2=15 -c -o $@ $<

$(OBJ)/pow2_stride_benign.o: pow2_stride_thrash.c | dirs
	$(CC) $(CFLAGS) -DSTRIDE_LOG2=6  -c -o $@ $<

$(BIN)/pow2_stride_thrash: $(OBJ)/pow2_stride_thrash.o | dirs
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN)/pow2_stride_benign: $(OBJ)/pow2_stride_benign.o | dirs
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ---- icache_blast needs the generated header ------------------------------

$(OBJ)/icache_blast.o: icache_blast.c $(GEN)/icache_blast_gen.h | dirs
	$(CC) $(CFLAGS) -I$(GEN) -c -o $@ $<

$(BIN)/icache_blast: $(OBJ)/icache_blast.o | dirs
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ---- Generic rule for the single-file benchmarks -------------------------

$(OBJ)/%.o: %.c | dirs
	$(CC) $(CFLAGS) -c -o $@ $<

$(BIN)/%: $(OBJ)/%.o | dirs
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ---------------------------------------------------------------------------

clean:
	rm -rf $(BUILD)
