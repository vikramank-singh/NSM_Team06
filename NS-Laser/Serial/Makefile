# Standardized build for the NS-Laser serial solver.
#
# Targets:
#   make            -> release build (bin/A-NS)
#   make debug      -> -O0 -g build for correctness debugging (bin/A-NS_debug)
#   make vecreport   -> -O3 build with vectorization-report output (bin/A-NS_vecreport)
#   make small       -> alias for release, intended to be invoked with a
#                        small grid at runtime for fast Advisor turnaround:
#                        ./bin/A-NS 51 51 10 50 --no-output
#   make clean
#
# Compiler selection: CXX defaults to icpx if present on PATH (matches the
# oneAPI toolchain Advisor expects symbols from), else falls back to g++.
# Override with `make CXX=g++` / `make CXX=icpx` explicitly.

CXX_AUTO := $(shell command -v icpx 2>/dev/null || command -v g++)
CXX      ?= $(CXX_AUTO)

FFTW_ROOT ?=
ifeq ($(FFTW_ROOT),)
  FFTW_CFLAGS := $(shell pkg-config --cflags fftw3 2>/dev/null)
  FFTW_LIBS   := $(shell pkg-config --libs fftw3 2>/dev/null)
  ifeq ($(FFTW_LIBS),)
    FFTW_LIBS := -lfftw3
  endif
else
  FFTW_CFLAGS := -I$(FFTW_ROOT)/include
  FFTW_LIBS   := -L$(FFTW_ROOT)/lib -Wl,-rpath,$(FFTW_ROOT)/lib -lfftw3
endif

WARN     := -Wall -Wextra
COMMON   := -std=c++17 $(WARN) $(FFTW_CFLAGS)
RELFLAGS := -O3 -march=native -funroll-loops -DNDEBUG
DBGFLAGS := -O0 -g

ifeq ($(findstring icpx,$(CXX)),icpx)
  VECREPORT_FLAGS := -O3 -march=native -qopt-report=3 -qopt-report-phase=vec
else ifeq ($(findstring icpc,$(CXX)),icpc)
  VECREPORT_FLAGS := -O3 -march=native -qopt-report=3 -qopt-report-phase=vec
else
  VECREPORT_FLAGS := -O3 -march=native -fopt-info-vec-optimized -fopt-info-vec-missed
endif

BIN := bin
SRC := A-NS.cpp
HDR := functions.h

.PHONY: all debug vecreport small clean
all: $(BIN)/A-NS

small: all

$(BIN):
	mkdir -p $(BIN)

$(BIN)/A-NS: $(SRC) $(HDR) | $(BIN)
	$(CXX) $(COMMON) $(RELFLAGS) -o $@ $(SRC) $(FFTW_LIBS)

debug: $(BIN)/A-NS_debug
$(BIN)/A-NS_debug: $(SRC) $(HDR) | $(BIN)
	$(CXX) $(COMMON) $(DBGFLAGS) -o $@ $(SRC) $(FFTW_LIBS)

vecreport: $(BIN)/A-NS_vecreport
$(BIN)/A-NS_vecreport: $(SRC) $(HDR) | $(BIN)
	$(CXX) $(COMMON) $(VECREPORT_FLAGS) -o $@ $(SRC) $(FFTW_LIBS) 2> $(BIN)/vecreport.txt
	@echo "Vectorization report written to $(BIN)/vecreport.txt"

clean:
	rm -rf $(BIN)
