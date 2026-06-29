# ===========================================================================
# OpenMP target-offload builds for the coupled thermal-laser solver.
#
#   FULL GPU (NS + cuFFT laser, A100):   make -f Makefile.gpu gpu-cufft
#   Partial (NS on GPU, laser on host):  make -f Makefile.gpu gpu
#   Host-fallback validation (CPU/FFTW): make -f Makefile.gpu host-cufft / host
#
# CRITICAL - CUDA VERSION MUST MATCH THE DRIVER:
#   This node's driver (550.54.14) supports CUDA 12.4. NVHPC 25.11 defaults to a
#   newer bundled CUDA, which compiles kernels the driver cannot launch -> every
#   kernel segfaults (exit 139). We pin -gpu=...,cuda12.4 to fix it.
#   Verify 12.4 is installed in NVHPC:
#       ls $(dirname $(which nvc++))/../../cuda
#   If there is no 12.4 (lowest is 12.6+), load an older NVHPC that ships <=12.4,
#   e.g.:  spack load nvhpc@24.5     (then rebuild; also update the SLURM script)
#
# FFTW: nvc++ CANNOT link an Intel-built FFTW. Point FFTW_ROOT at a GCC build.
# ===========================================================================

HDR_GPU  := functions_gpu.h functions.h
HDR_FULL := functions_gpu.h functions_laser_gpu.h functions.h

# (keep comments on their OWN lines so no value picks up trailing spaces)
FFTW_ROOT ?= $(HOME)/fftw-3.3.10-gcc
# A100 compute capability
CCAP ?= cc80
# CUDA toolkit version; MUST be <= the driver's CUDA and installed in NVHPC
CUDA_VER ?= cuda12.9

# strip guards against any stray whitespace sneaking into the -gpu switch
GPUARCH := $(strip $(CCAP)),$(strip $(CUDA_VER))
INCS := -I$(FFTW_ROOT)/include
LIBS := -L$(FFTW_ROOT)/lib -lfftw3 -lm

.PHONY: all gpu-cufft gpu host-cufft host clean
all: gpu

# ---- FULL GPU: NS stencils + cuFFT BPM laser, zero per-step transfers -------
gpu-cufft: A-NS-gpu-cufft.cpp $(HDR_FULL)
        nvc++ -std=c++17 -O3 -mp=gpu -gpu=$(GPUARCH) -cuda -cudalib=cufft -Minfo=mp \
              -DUSE_CUFFT $(INCS) -o A-NS-gpu-cufft A-NS-gpu-cufft.cpp -lcufft $(LIBS)

# ---- partial: NS on GPU, laser on host (FFTW) ------------------------------
gpu: A-NS-gpu.cpp $(HDR_GPU)
        nvc++ -std=c++17 -O3 -mp=gpu -gpu=$(GPUARCH) -Minfo=mp \
              $(INCS) -o A-NS-gpu A-NS-gpu.cpp $(LIBS)

# ---- host-fallback validation (FFTW, target regions on CPU) ----------------
host-cufft: A-NS-gpu-cufft.cpp $(HDR_FULL)
        g++ -std=c++17 -O2 -fopenmp $(INCS) -o A-NS-gpu-cufft A-NS-gpu-cufft.cpp $(LIBS)

host: A-NS-gpu.cpp $(HDR_GPU)
        g++ -std=c++17 -O2 -fopenmp $(INCS) -o A-NS-gpu A-NS-gpu.cpp $(LIBS)

clean:
        rm -f A-NS-gpu A-NS-gpu-cufft gpu_smoke
