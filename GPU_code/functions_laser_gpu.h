#ifndef FUNCTIONS_LASER_GPU_H
#define FUNCTIONS_LASER_GPU_H

#include <cmath>
#include <fftw3.h>
#include "functions.h"   // dummy_variables, PoissonMethod, host sqr, init/io

#ifdef USE_CUFFT
  #include <cufft.h>
  #include <cuda_runtime.h>
#endif

// ===========================================================================
// On-device split-step BPM laser march. Reproduces the FFTW march in
// functions.h step for step, with every element-wise stage an OpenMP target
// kernel and the 2D transform pluggable:
//
//   * GPU build  (-DUSE_CUFFT): cuFFT on the device pointers of the mapped
//     in/out buffers. Ar, Ai, T, phase tables and FFT scratch stay on the GPU,
//     so the laser adds ZERO host<->device traffic.
//   * Host-fallback (no USE_CUFFT): same pipeline with FFTW, regions on CPU.
//
// Every kernel carries an explicit map clause for the arrays it touches. The
// arrays are already present (driver `target enter data`), so these are no-op
// transfers, but they force nvc++ to address the device copies through the
// inline-function boundary instead of dereferencing a host pointer.
//
// Complex FFT buffers are interleaved doubles (len 2*fftN): [2*id]=Re,
// [2*id+1]=Im -- binary-compatible with fftw_complex and cufftDoubleComplex.
// ===========================================================================

struct LaserFFT {
    int nx1, ny1, fftN;
#ifdef USE_CUFFT
    cufftHandle plan;
#else
    fftw_plan ff, fb;
#endif
};

inline void laser_fft_init(LaserFFT& h, int nx, int ny, double* in_buf, double* out_buf)
{
    h.nx1 = nx-1; h.ny1 = ny-1; h.fftN = h.nx1*h.ny1;
#ifdef USE_CUFFT
    (void)in_buf; (void)out_buf;
    cufftPlan2d(&h.plan, h.nx1, h.ny1, CUFFT_Z2Z);   // same axes as fftw_plan_dft_2d(nx-1,ny-1)
#else
    h.ff = fftw_plan_dft_2d(h.nx1, h.ny1, (fftw_complex*)in_buf,
                            (fftw_complex*)out_buf, FFTW_FORWARD,  FFTW_MEASURE);
    h.fb = fftw_plan_dft_2d(h.nx1, h.ny1, (fftw_complex*)in_buf,
                            (fftw_complex*)out_buf, FFTW_BACKWARD, FFTW_MEASURE);
#endif
}

inline void laser_fft_destroy(LaserFFT& h)
{
#ifdef USE_CUFFT
    cufftDestroy(h.plan);
#else
    fftw_destroy_plan(h.ff); fftw_destroy_plan(h.fb);
#endif
}

inline void laser_fft_exec(LaserFFT& h, double* in_buf, double* out_buf, int forward)
{
#ifdef USE_CUFFT
    #pragma omp target data use_device_ptr(in_buf, out_buf)
    {
        cufftExecZ2Z(h.plan,
                     reinterpret_cast<cufftDoubleComplex*>(in_buf),
                     reinterpret_cast<cufftDoubleComplex*>(out_buf),
                     forward ? CUFFT_FORWARD : CUFFT_INVERSE);
    }
    cudaDeviceSynchronize();   // cuFFT runs on its own stream; finish before next kernel
#else
    if (forward) fftw_execute(h.ff); else fftw_execute(h.fb);
    (void)in_buf; (void)out_buf;
#endif
}

inline void A_compute_ss_gpu(double* Ar, double* Ai, const double* T,
                             const double* phase_c, const double* phase_s,
                             double* in_buf, double* out_buf, LaserFFT& fft,
                             int nx, int ny, int nz,
                             double beta, double wn, double C, double dz)
{
    const int nx1 = nx-1, ny1 = ny-1;
    const long long plane = (long long)nx*ny;
    const long long N = plane*nz;
    const int fftN = nx1*ny1;
    const long long Nc = 2LL*fftN;
    const double atten = std::exp(-beta*dz/2.0);
    const double inv_fftN = 1.0/(double)fftN;
    const double tps = 0.5*C*wn*dz;

    for (int k = 1; k < nz; k++) {
        const long long prev = (long long)(k-1)*plane;
        const long long curr = (long long)k*plane;

        // (1) half-step attenuation + temperature phase -> in_buf
        #pragma omp target teams distribute parallel for collapse(2) \
            map(to: Ar[0:N], Ai[0:N], T[0:N]) map(tofrom: in_buf[0:Nc])
        for (int j = 0; j < ny1; j++)
            for (int i = 0; i < nx1; i++) {
                const int fid = j*nx1 + i;
                const long long gid = prev + (long long)j*nx + i;
                const double ca = tps*T[gid];
                const double s = sin(ca), c = cos(ca);
                in_buf[2*fid]   = atten*(Ar[gid]*c - Ai[gid]*s);
                in_buf[2*fid+1] = atten*(Ar[gid]*s + Ai[gid]*c);
            }

        laser_fft_exec(fft, in_buf, out_buf, 1);          // forward

        // (3) diffraction phase multiply -> in_buf
        #pragma omp target teams distribute parallel for collapse(2) \
            map(to: out_buf[0:Nc], phase_c[0:fftN], phase_s[0:fftN]) map(tofrom: in_buf[0:Nc])
        for (int j = 0; j < ny1; j++)
            for (int i = 0; i < nx1; i++) {
                const int fid = j*nx1 + i;
                const double c = phase_c[fid], s = phase_s[fid];
                const double re = out_buf[2*fid], im = out_buf[2*fid+1];
                in_buf[2*fid]   = re*c - im*s;
                in_buf[2*fid+1] = re*s + im*c;
            }

        laser_fft_exec(fft, in_buf, out_buf, 0);          // inverse

        // (5) normalize + store into plane k
        #pragma omp target teams distribute parallel for collapse(2) \
            map(to: out_buf[0:Nc]) map(tofrom: Ar[0:N], Ai[0:N])
        for (int j = 0; j < ny1; j++)
            for (int i = 0; i < nx1; i++) {
                const int fid = j*nx1 + i;
                const long long gid = curr + (long long)j*nx + i;
                Ar[gid] = out_buf[2*fid]*inv_fftN;
                Ai[gid] = out_buf[2*fid+1]*inv_fftN;
            }

        // (6) periodic edge wrap (row ny-1 <- row 0, col nx-1 <- col 0)
        #pragma omp target teams distribute parallel for map(tofrom: Ar[0:N], Ai[0:N])
        for (int i = 0; i < nx; i++) {
            Ar[curr + (long long)(ny-1)*nx + i] = Ar[curr + i];
            Ai[curr + (long long)(ny-1)*nx + i] = Ai[curr + i];
        }
        #pragma omp target teams distribute parallel for map(tofrom: Ar[0:N], Ai[0:N])
        for (int j = 0; j < ny; j++) {
            Ar[curr + (long long)j*nx + nx-1] = Ar[curr + (long long)j*nx];
            Ai[curr + (long long)j*nx + nx-1] = Ai[curr + (long long)j*nx];
        }
    }
}

inline void intensity_gpu(double* I, const double* Ar, const double* Ai, long long N)
{
    #pragma omp target teams distribute parallel for \
        map(tofrom: I[0:N]) map(to: Ar[0:N], Ai[0:N])
    for (long long i = 0; i < N; i++) I[i] = Ar[i]*Ar[i] + Ai[i]*Ai[i];
}

#endif
