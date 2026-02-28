CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra
INCLUDE := -I include -I src

# OpenMP usage: set OMP=0 to disable OpenMP (e.g., make OMP=0)
OMP ?= 1
ifeq ($(OMP),1)
OMP_FLAG := -fopenmp
else
OMP_FLAG :=
endif

SRC_DIR := src
TOOLS_DIR := $(SRC_DIR)/tools
TESTS_DIR := tests
TRACE_DIR := traces

BUILD := build
OBJDIR := $(BUILD)/obj

# ---- Parameters ----
WINDOW_SIZE ?= 400
PARAM_STEP ?= BASE_2

# Trace file
TRACE_CSV ?= $(TRACE_DIR)/trace.csv

# ---- Registry code generation ----
# Edit registry.yaml to add/remove/enable/disable resources and params, then run make.
REGISTRY_CONFIG ?= registry.yaml
GENERATED_HEADERS := include/resources_gen.h include/params_gen.h include/resource_registry.h include/models_decl_gen.h
GENERATED_PY := python/config_gen.py

$(GENERATED_HEADERS) $(GENERATED_PY): $(REGISTRY_CONFIG) python/gen_registry.py
	$(PYTHON) python/gen_registry.py --config $(REGISTRY_CONFIG)

# ---- Common sources ----
SRCS_MODELS := $(SRC_DIR)/models.cpp

# Serial-only models (no OpenMP), used for benchmark no-OMP binary
SRCS_MODELS_SERIAL := $(SRC_DIR)/models.cpp
OBJS_MODELS_SERIAL := $(patsubst $(SRC_DIR)/%.cpp,$(OBJDIR)/serial_%.o,$(SRCS_MODELS_SERIAL))

# RapidCSV-based parsing
SRCS_PARSER := \
  $(SRC_DIR)/parser.cpp

# ---- Executables ----
SRCS_TEST_PARAMS := \
  $(TESTS_DIR)/test_params.cpp

SRCS_TEST_MODELS := \
  $(TESTS_DIR)/test_models.cpp \
  $(SRCS_PARSER) \
  $(SRCS_MODELS)

SRCS_MAIN := \
  $(SRC_DIR)/main.cpp \
  $(SRCS_PARSER) \
  $(SRCS_MODELS)

# ---- Objects ----
OBJS_MAIN := $(patsubst $(SRC_DIR)/%.cpp,$(OBJDIR)/%.o,$(filter $(SRC_DIR)/%,$(SRCS_MAIN)))

OBJS_TEST_PARAMS := $(patsubst $(TESTS_DIR)/%.cpp,$(OBJDIR)/tests/%.o,$(SRCS_TEST_PARAMS))

OBJS_TEST_MODELS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJDIR)/%.o,$(filter $(SRC_DIR)/%,$(SRCS_TEST_MODELS))) \
					$(patsubst $(TESTS_DIR)/%.cpp,$(OBJDIR)/tests/%.o,$(filter $(TESTS_DIR)/%,$(SRCS_TEST_MODELS)))

OBJS_PARSER := $(patsubst $(SRC_DIR)/%.cpp,$(OBJDIR)/standalone_%.o,$(SRCS_PARSER))

# ---- Default ----
all: $(BUILD)/main $(BUILD)/test_params $(BUILD)/test_models_omp $(BUILD)/test_models_noomp $(BUILD)/parser

# Add a named target for the main executable
anamol: $(BUILD)/main
	@cp $(BUILD)/main anamol || true

# Ensure build directories exist
prep:
	@mkdir -p $(BUILD) $(OBJDIR) $(OBJDIR)/tools $(OBJDIR)/tests

# ---- Compile rules ----
# All objects depend on generated headers so a registry.yaml change triggers a rebuild.
$(OBJDIR)/%.o: $(SRC_DIR)/%.cpp $(GENERATED_HEADERS) | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) -c $< -o $@

$(OBJDIR)/tests/%.o: $(TESTS_DIR)/%.cpp $(GENERATED_HEADERS) | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) -c $< -o $@

# Standalone executables with STANDALONE_PARSER flag
$(OBJDIR)/standalone_%.o: $(SRC_DIR)/%.cpp $(GENERATED_HEADERS) | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) -DSTANDALONE_PARSER -c $< -o $@

# Serial (no OpenMP) objects for benchmark-only binary
$(OBJDIR)/serial_%.o: $(SRC_DIR)/%.cpp $(GENERATED_HEADERS) | prep
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# ---- Link executables ----
$(BUILD)/main: $(OBJS_MAIN)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) $^ -o $@

$(BUILD)/parser: $(OBJS_PARSER)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ -o $@

$(BUILD)/test_params: $(OBJS_TEST_PARAMS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ -o $@

# OpenMP version (parallel)
$(BUILD)/test_models_omp: $(OBJS_TEST_MODELS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OMP_FLAG) $^ -o $@

# Non-OpenMP version (serial, for benchmark)
$(BUILD)/test_models_noomp: $(filter-out $(OBJDIR)/models.o,$(OBJS_TEST_MODELS)) $(OBJS_MODELS_SERIAL)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ -o $@

# Keep backward compatibility
$(BUILD)/test_models: $(BUILD)/test_models_omp
	cp $< $@

# ---- Run targets ----

# Python interpreter
PYTHON := python3
# Output directory to pass to the visualizer (defaults to 'output')
THR_DIR ?= output

main.run: $(BUILD)/main
	$< -t $(TRACE_CSV) -w $(WINDOW_SIZE)

# Convenience run target for anamol
anamol.run: main.run visualize.run
	@true

# Visualizer run target (passes --output-dir)
visualize.run:
	$(PYTHON) python/visualize.py --output-dir $(THR_DIR)

parser.run: $(BUILD)/parser
	$< $(TRACE_CSV)

test_parser.run: $(BUILD)/parser
	$< $(TRACE_CSV)

test_params.run: $(BUILD)/test_params
	$<

test_models.run: $(BUILD)/test_models
	$< $(TRACE_CSV) $(WINDOW_SIZE)

test_models_omp.run: $(BUILD)/test_models_omp
	$< $(TRACE_CSV) $(WINDOW_SIZE)

test_models_noomp.run: $(BUILD)/test_models_noomp
	$< $(TRACE_CSV) $(WINDOW_SIZE)

# ---- Performance comparison ----
benchmark: $(BUILD)/test_models_omp $(BUILD)/test_models_noomp
	@echo "========================================="
	@echo "Performance Comparison"
	@echo "Trace: $(TRACE_CSV)"
	@echo "Window size: $(WINDOW_SIZE)"
	@echo "========================================="
	@echo ""
	@echo "Running SERIAL (no OpenMP)..."
	@/usr/bin/time -l $(BUILD)/test_models_noomp $(TRACE_CSV) $(WINDOW_SIZE) 2>&1 | grep -E "real|user"
	@echo ""
	@echo "Running PARALLEL with 1 thread..."
	@OMP_NUM_THREADS=1 /usr/bin/time -l $(BUILD)/test_models_omp $(TRACE_CSV) $(WINDOW_SIZE) 2>&1 | grep -E "real|user"
	@echo ""
	@echo "Running PARALLEL with 4 threads..."
	@OMP_NUM_THREADS=4 /usr/bin/time -l $(BUILD)/test_models_omp $(TRACE_CSV) $(WINDOW_SIZE) 2>&1 | grep -E "real|user"
	@echo ""
	@echo "Running PARALLEL with 8 threads..."
	@OMP_NUM_THREADS=8 /usr/bin/time -l $(BUILD)/test_models_omp $(TRACE_CSV) $(WINDOW_SIZE) 2>&1 | grep -E "real|user"
	@echo ""
	@echo "========================================="

# ---- Clean ----
clean:
	rm -rf $(BUILD)

clean-gen:
	rm -f $(GENERATED_HEADERS) $(GENERATED_PY)

.PHONY: all clean clean-gen prep main.run parser.run test_parser.run \
		test_params.run test_models.run test_models_omp.run test_models_noomp.run \
		benchmark anamol anamol.run