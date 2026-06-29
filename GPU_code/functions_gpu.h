#ifndef FUNCTIONS_GPU_H
#define FUNCTIONS_GPU_H

#include <omp.h>
#include <algorithm>
#include <utility>
#include "functions.h"   // host BPM march, init/io, PoissonMethod, dummy_variables, host sqr

// ===========================================================================
// OpenMP target-offload kernels for the coupled thermal-laser solver.
//
//   - Field arrays stay resident on the device for the whole time loop (the
//     driver issues one `target enter data`). Each kernel ALSO lists its arrays
//     in a map clause: since they are already present, that is a no-op transfer
//     (refcount bump only), but it makes nvc++ resolve every pointer to its
//     device copy -- without it, NVHPC passes the host pointer through the
//     inline-function boundary and the kernel segfaults on the device.
//   - Arithmetic is identical to the serial routines in functions.h, so the
//     host-fallback build reproduces the serial result bit-for-bit.
// ===========================================================================

#pragma omp declare target
namespace gpu {
static inline double dmax(double a, double b) { return a > b ? a : b; }
static inline double dmin(double a, double b) { return a < b ? a : b; }
static inline double sqr(double x) { return x * x; }

static inline double advection(double a, double b, double c,
                               const double* f, int nx, int ny, long long id,
                               double idx1, double idy1, double idz1)
{
    return (dmax(a,0.0)*(f[id]-f[id-1])  + dmin(a,0.0)*(f[id+1]-f[id])) *idx1
         + (dmax(b,0.0)*(f[id]-f[id-nx]) + dmin(b,0.0)*(f[id+nx]-f[id]))*idy1
         + (dmax(c,0.0)*(f[id]-f[id-(long long)nx*ny])
            + dmin(c,0.0)*(f[id+(long long)nx*ny]-f[id]))*idz1;
}
static inline double diffusion(const double* f, int nx, int ny, long long id,
                               double idx2, double idy2, double idz2)
{
    return (f[id+1]+f[id-1]-2.0*f[id])*idx2
         + (f[id+nx]+f[id-nx]-2.0*f[id])*idy2
         + (f[id+(long long)nx*ny]+f[id-(long long)nx*ny]-2.0*f[id])*idz2;
}
} // namespace gpu
#pragma omp end declare target

// ---- small full-array device utilities -------------------------------------
inline void zero_field_gpu(double* a, long long N)
{
    #pragma omp target teams distribute parallel for map(tofrom: a[0:N])
    for (long long i = 0; i < N; i++) a[i] = 0.0;
}
inline void scale_field_gpu(double* dst, const double* src, double s, long long N)
{
    #pragma omp target teams distribute parallel for map(tofrom: dst[0:N]) map(to: src[0:N])
    for (long long i = 0; i < N; i++) dst[i] = src[i] * s;
}
inline void axpy_inplace_gpu(double* acc, const double* add, long long N)
{
    #pragma omp target teams distribute parallel for map(tofrom: acc[0:N]) map(to: add[0:N])
    for (long long i = 0; i < N; i++) acc[i] += add[i];
}
inline void copy_field_gpu(double* dst, const double* src, long long N)
{
    #pragma omp target teams distribute parallel for map(tofrom: dst[0:N]) map(to: src[0:N])
    for (long long i = 0; i < N; i++) dst[i] = src[i];
}

// ---- non-periodic boundary conditions (sequenced z, y, x) ------------------
inline void Neumann_bc_gpu(double* f, int nx, int ny, int nz)
{
    const long long nxny = (long long)nx*ny;
    const long long N = nxny*nz;
    #pragma omp target teams distribute parallel for collapse(2) map(tofrom: f[0:N])
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
            f[(long long)j*nx + i]                 = f[nxny + (long long)j*nx + i];
            f[(long long)(nz-1)*nxny + j*nx + i]   = f[(long long)(nz-2)*nxny + j*nx + i];
        }
    #pragma omp target teams distribute parallel for collapse(2) map(tofrom: f[0:N])
    for (int k = 0; k < nz; k++)
        for (int i = 0; i < nx; i++) {
            f[(long long)k*nxny + i]               = f[(long long)k*nxny + nx + i];
            f[(long long)k*nxny + (ny-1)*nx + i]   = f[(long long)k*nxny + (ny-2)*nx + i];
        }
    #pragma omp target teams distribute parallel for collapse(2) map(tofrom: f[0:N])
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++) {
            f[(long long)k*nxny + j*nx]            = f[(long long)k*nxny + j*nx + 1];
            f[(long long)k*nxny + j*nx + nx-1]     = f[(long long)k*nxny + j*nx + nx-2];
        }
}

inline void Dirichlet_bc_gpu(double* f, int nx, int ny, int nz)
{
    const long long nxny = (long long)nx*ny;
    const long long N = nxny*nz;
    #pragma omp target teams distribute parallel for collapse(2) map(tofrom: f[0:N])
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
            f[(long long)j*nx + i]               = 0.0;
            f[(long long)(nz-1)*nxny + j*nx + i] = 0.0;
        }
    #pragma omp target teams distribute parallel for collapse(2) map(tofrom: f[0:N])
    for (int k = 0; k < nz; k++)
        for (int i = 0; i < nx; i++) {
            f[(long long)k*nxny + i]             = 0.0;
            f[(long long)k*nxny + (ny-1)*nx + i] = 0.0;
        }
    #pragma omp target teams distribute parallel for collapse(2) map(tofrom: f[0:N])
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++) {
            f[(long long)k*nxny + j*nx]          = 0.0;
            f[(long long)k*nxny + j*nx + nx-1]   = 0.0;
        }
}

// ---- NS stencils -----------------------------------------------------------
inline void prediction_gpu(double* us, double* vs, double* ws,
                           const double* u, const double* v, const double* w,
                           const double* p, const double* T, double Pr,
                           int nx, int ny, int nz, double dx, double dy, double dz, double dt)
{
    const double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy), idz2=1.0/(dz*dz);
    const double idx1=1.0/dx, idy1=1.0/dy, idz1=1.0/dz;
    const double hx=0.5/dx, hy=0.5/dy, hz=0.5/dz;
    const long long nxny = (long long)nx*ny;
    const long long N = nxny*nz;

    #pragma omp target teams distribute parallel for collapse(3) \
        map(tofrom: us[0:N], vs[0:N], ws[0:N]) \
        map(to: u[0:N], v[0:N], w[0:N], p[0:N], T[0:N])
    for (int k = 1; k < nz-1; k++)
        for (int j = 1; j < ny-1; j++)
            for (int i = 1; i < nx-1; i++) {
                const long long id = (long long)k*nxny + (long long)j*nx + i;
                double adv  = gpu::advection(u[id],v[id],w[id], u, nx,ny,id, idx1,idy1,idz1);
                double diff = Pr*gpu::diffusion(u, nx,ny,id, idx2,idy2,idz2);
                double src  = -(p[id+1]-p[id-1])*hx;
                us[id] = u[id] + dt*(-adv + diff + src);

                adv  = gpu::advection(u[id],v[id],w[id], v, nx,ny,id, idx1,idy1,idz1);
                diff = Pr*gpu::diffusion(v, nx,ny,id, idx2,idy2,idz2);
                src  = -(p[id+nx]-p[id-nx])*hy + Pr*T[id];
                vs[id] = v[id] + dt*(-adv + diff + src);

                adv  = gpu::advection(u[id],v[id],w[id], w, nx,ny,id, idx1,idy1,idz1);
                diff = Pr*gpu::diffusion(w, nx,ny,id, idx2,idy2,idz2);
                src  = -(p[id+nxny]-p[id-nxny])*hz;
                ws[id] = w[id] + dt*(-adv + diff + src);
            }
}

inline void divergence_gpu(double* f, const double* u, const double* v, const double* w,
                           int nx, int ny, int nz, double dx, double dy, double dz)
{
    const double hx=0.5/dx, hy=0.5/dy, hz=0.5/dz;
    const long long nxny = (long long)nx*ny;
    const long long N = nxny*nz;
    #pragma omp target teams distribute parallel for collapse(3) \
        map(tofrom: f[0:N]) map(to: u[0:N], v[0:N], w[0:N])
    for (int k = 1; k < nz-1; k++)
        for (int j = 1; j < ny-1; j++)
            for (int i = 1; i < nx-1; i++) {
                const long long id = (long long)k*nxny + (long long)j*nx + i;
                f[id] = (u[id+1]-u[id-1])*hx
                      + (v[id+nx]-v[id-nx])*hy
                      + (w[id+nxny]-w[id-nxny])*hz;
            }
}

inline void correction_gpu(const double* us, const double* vs, const double* ws,
                           double* u, double* v, double* w, const double* pp,
                           int nx, int ny, int nz, double dx, double dy, double dz, double dt)
{
    const double dtx=dt*0.5/dx, dty=dt*0.5/dy, dtz=dt*0.5/dz;
    const long long nxny = (long long)nx*ny;
    const long long N = nxny*nz;
    #pragma omp target teams distribute parallel for collapse(3) \
        map(to: us[0:N], vs[0:N], ws[0:N], pp[0:N]) \
        map(tofrom: u[0:N], v[0:N], w[0:N])
    for (int k = 1; k < nz-1; k++)
        for (int j = 1; j < ny-1; j++)
            for (int i = 1; i < nx-1; i++) {
                const long long id = (long long)k*nxny + (long long)j*nx + i;
                u[id] = us[id] - (pp[id+1]-pp[id-1])*dtx;
                v[id] = vs[id] - (pp[id+nx]-pp[id-nx])*dty;
                w[id] = ws[id] - (pp[id+nxny]-pp[id-nxny])*dtz;
            }
}

inline void T_compute_gpu(const double* u, const double* v, const double* w,
                          double* T, const double* I, double Ra, double* T_new,
                          int nx, int ny, int nz, double dx, double dy, double dz, double dt)
{
    const double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy), idz2=1.0/(dz*dz);
    const double idx1=1.0/dx, idy1=1.0/dy, idz1=1.0/dz;
    const long long nxny = (long long)nx*ny;
    const long long N = nxny*nz;

    #pragma omp target teams distribute parallel for collapse(3) \
        map(to: u[0:N], v[0:N], w[0:N], T[0:N], I[0:N]) map(tofrom: T_new[0:N])
    for (int k = 1; k < nz-1; k++)
        for (int j = 1; j < ny-1; j++)
            for (int i = 1; i < nx-1; i++) {
                const long long id = (long long)k*nxny + (long long)j*nx + i;
                const double adv  = gpu::advection(u[id],v[id],w[id], T, nx,ny,id, idx1,idy1,idz1);
                const double diff = gpu::diffusion(T, nx,ny,id, idx2,idy2,idz2);
                T_new[id] = T[id] + dt*(-adv + diff + Ra*I[id]);
            }
    #pragma omp target teams distribute parallel for map(tofrom: T[0:N]) map(to: T_new[0:N])
    for (long long i = 0; i < N; i++) T[i] = T_new[i];
}

// ---- Poisson (device): Jacobi ping-pong and Red-Black GS -------------------
inline void poisson_jacobi_gpu(double* u, const double* f, double* u_new,
                               int nx, int ny, int nz, double dx, double dy, double dz, int nitr)
{
    const double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy), idz2=1.0/(dz*dz);
    const double factor = 1.0/(2.0*(idx2+idy2+idz2));
    const long long nxny = (long long)nx*ny;
    const long long N = nxny*nz;
    double* cur = u;
    double* next = u_new;

    for (int it = 0; it < nitr; it++) {
        #pragma omp target teams distribute parallel for collapse(3) \
            map(tofrom: next[0:N]) map(to: cur[0:N], f[0:N])
        for (int k = 1; k < nz-1; k++)
            for (int j = 1; j < ny-1; j++)
                for (int i = 1; i < nx-1; i++) {
                    const long long id = (long long)k*nxny + (long long)j*nx + i;
                    next[id] = ((cur[id+1]+cur[id-1])*idx2
                              + (cur[id+nx]+cur[id-nx])*idy2
                              + (cur[id+nxny]+cur[id-nxny])*idz2
                              - f[id]) * factor;
                }
        Neumann_bc_gpu(next, nx, ny, nz);
        std::swap(cur, next);
    }
    if (cur != u) copy_field_gpu(u, cur, N);
}

inline void poisson_rbgs_gpu(double* u, const double* f,
                             int nx, int ny, int nz, double dx, double dy, double dz, int nitr)
{
    const double idx2=1.0/(dx*dx), idy2=1.0/(dy*dy), idz2=1.0/(dz*dz);
    const double factor = 1.0/(2.0*(idx2+idy2+idz2));
    const long long nxny = (long long)nx*ny;
    const long long N = nxny*nz;

    for (int it = 0; it < nitr; it++) {
        for (int color = 0; color < 2; color++) {
            #pragma omp target teams distribute parallel for collapse(3) \
                map(tofrom: u[0:N]) map(to: f[0:N])
            for (int k = 1; k < nz-1; k++)
                for (int j = 1; j < ny-1; j++)
                    for (int i = 1; i < nx-1; i++)
                        if ((((i + j + k) & 1) == color)) {
                            const long long id = (long long)k*nxny + (long long)j*nx + i;
                            u[id] = ((u[id+1]+u[id-1])*idx2
                                   + (u[id+nx]+u[id-nx])*idy2
                                   + (u[id+nxny]+u[id-nxny])*idz2
                                   - f[id]) * factor;
                        }
        }
        Neumann_bc_gpu(u, nx, ny, nz);
    }
}

inline void poisson_solve_gpu(double* u, const double* f, double* u_new, PoissonMethod m,
                              int nx, int ny, int nz, double dx, double dy, double dz, int nitr)
{
    if (m == PoissonMethod::RedBlackGS) poisson_rbgs_gpu(u, f, nx, ny, nz, dx, dy, dz, nitr);
    else                                poisson_jacobi_gpu(u, f, u_new, nx, ny, nz, dx, dy, dz, nitr);
}

// ---- one coupled NS step, entirely on the device ---------------------------
// All arrays are plain top-level pointers mapped by name in the driver. dv is
// NOT passed onto the device (struct-member mapping needs the struct attached,
// which segfaults); scalars (method, sweeps) are passed by value.
inline void ns_compute_gpu(double* u, double* v, double* w, double* p, double* T, double* I,
                           double* us, double* vs, double* ws, double* pp,
                           double* div, double* div_dt, double* Tn, double* unew,
                           double Pr, double Ra, PoissonMethod method, int piters,
                           int nx, int ny, int nz, double dx, double dy, double dz, double dt)
{
    const long long N = (long long)nx*ny*nz;

    zero_field_gpu(pp, N);

    prediction_gpu(us, vs, ws, u, v, w, p, T, Pr, nx, ny, nz, dx, dy, dz, dt);
    Dirichlet_bc_gpu(us, nx, ny, nz);
    Dirichlet_bc_gpu(vs, nx, ny, nz);
    Dirichlet_bc_gpu(ws, nx, ny, nz);

    divergence_gpu(div, us, vs, ws, nx, ny, nz, dx, dy, dz);
    scale_field_gpu(div_dt, div, 1.0/dt, N);

    poisson_solve_gpu(pp, div_dt, unew, method, nx, ny, nz, dx, dy, dz, piters);

    correction_gpu(us, vs, ws, u, v, w, pp, nx, ny, nz, dx, dy, dz, dt);
    Dirichlet_bc_gpu(u, nx, ny, nz);
    Dirichlet_bc_gpu(v, nx, ny, nz);
    Dirichlet_bc_gpu(w, nx, ny, nz);

    axpy_inplace_gpu(p, pp, N);
    Neumann_bc_gpu(p, nx, ny, nz);

    T_compute_gpu(u, v, w, T, I, Ra, Tn, nx, ny, nz, dx, dy, dz, dt);
    Dirichlet_bc_gpu(T, nx, ny, nz);
}

#endif
