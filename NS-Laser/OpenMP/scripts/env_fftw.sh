#!/bin/bash
# Configure FFTW3 environment for building/running the NS-Laser solver.
# Source this, don't execute it: `source scripts/env_fftw.sh`
#
# Override FFTW_ROOT before sourcing to point at a different install, e.g.:
#   FFTW_ROOT=/home/apps/fftw-3.3.10 source scripts/env_fftw.sh

FFTW_ROOT="${FFTW_ROOT:-/home/samirs/source/samir/spack/opt/spack/linux-cascadelake/fftw-3.3.10-r7kcrecj6okcxddpaiju2djxqorncvbp}"

if [ ! -d "$FFTW_ROOT" ]; then
	echo "env_fftw.sh: FFTW_ROOT not found: $FFTW_ROOT" >&2
	echo "env_fftw.sh: set FFTW_ROOT to a valid FFTW3 install before sourcing" >&2
	return 1 2>/dev/null || exit 1
fi

export FFTW_ROOT
export FFTW_DIR=${FFTW_ROOT}
export FFTW_INC=${FFTW_ROOT}/include
export FFTW_LIB=${FFTW_ROOT}/lib

export PATH=${FFTW_ROOT}/bin:${PATH:-}
export LD_LIBRARY_PATH=${FFTW_ROOT}/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${FFTW_ROOT}/lib:${LIBRARY_PATH:-}
export CPATH=${FFTW_ROOT}/include:${CPATH:-}
export PKG_CONFIG_PATH=${FFTW_ROOT}/lib/pkgconfig:${PKG_CONFIG_PATH:-}

echo "FFTW configured from: ${FFTW_ROOT}"
