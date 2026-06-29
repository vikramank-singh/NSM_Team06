#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <fftw3.h>
#include <omp.h>

const double pi = 3.14159265358;

inline double sqr(const double x)
{
    return x*x;
}

// Selectable Poisson relaxation method.
enum class PoissonMethod { Jacobi, RedBlackGS };

struct dummy_variables {
    double *u_star = nullptr;
    double *v_star = nullptr;
    double *w_star = nullptr;
    double *p_prime = nullptr;
    double *div = nullptr;
    double *div_dt = nullptr;
    double *T_new = nullptr;
    double *u_new = nullptr;

    // Runtime-selected Poisson solver (set from the command line in main()).
    PoissonMethod poisson_method = PoissonMethod::Jacobi;

    // Poisson relaxation sweeps per projection step. Default matches the
    // original hardcoded value; override via POISSON_SWEEPS env var. Use this
    // to run RBGS at ~half the Jacobi sweeps for an equal-residual comparison.
    int poisson_iters = 150;
};

void initialize_wavnumbers(double *kx, double *ky, int nx, int ny,
                           double lx, double ly, double dx, double dy)
{
    (void)dx;
    (void)dy;

    for (int j=0; j<ny-1; j++) {
        const double KY = (j <= (ny-1)/2) ? j : j-(ny-1);
        for (int i=0; i<nx-1; i++) {
            const double KX = (i <= (nx-1)/2) ? i : i-(nx-1);
            kx[j*(nx-1) + i] = (2.0*pi/lx)*KX;
            ky[j*(nx-1) + i] = (2.0*pi/ly)*KY;
        }
    }
}

void initialize_fft_phase(double *phase_c, double *phase_s, const double *kx,
                          const double *ky, int nx, int ny, double dz,
                          double wn)
{
    const int nx1 = nx-1;
    const int ny1 = ny-1;
    const double phase_scale = dz/(2.0*wn);

    #pragma omp parallel for schedule(static)
    for (int id=0; id<nx1*ny1; id++) {
        const double k2 = sqr(kx[id]) + sqr(ky[id]);
        const double ca = k2*phase_scale;
        phase_c[id] = std::cos(ca);
        phase_s[id] = std::sin(ca);
    }
}

void initialization(double *Ar, double *Ai, double *u, double *v, double *w,
                    double *I, double *T, double *p, double beta, int nx,
                    int ny, int nz, double lx, double ly, double dx,
                    double dy, double dz)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k=0; k<nz; k++) {
        for (int j=0; j<ny; j++) {
            for (int i=0; i<nx; i++) {
                const int idx = k*nx*ny + j*nx + i;
                const double x = i*dx;
                const double y = j*dy;
                const double z = k*dz;
                const double ar0 =
                    std::exp(-2.0*(sqr(x-lx/2.0) + sqr(y-ly/2.0)));

                Ar[idx] = ar0;
                Ai[idx] = 0.0;
                I[idx] = std::exp(-beta*z)*ar0*ar0;
                T[idx] = 0.0;
                u[idx] = 0.0;
                v[idx] = 0.0;
                w[idx] = 0.0;
                p[idx] = 0.0;
            }
        }
    }
}

inline double advection(const double a, const double b, const double c,
                        const double* __restrict f, const int nx,
                        const int ny, const int idx, const double idx1,
                        const double idy1, const double idz1)
{
    return
        (std::max(a,0.0)*(f[idx]-f[idx-1]) +
         std::min(a,0.0)*(f[idx+1]-f[idx]))*idx1 +
        (std::max(b,0.0)*(f[idx]-f[idx-nx]) +
         std::min(b,0.0)*(f[idx+nx]-f[idx]))*idy1 +
        (std::max(c,0.0)*(f[idx]-f[idx-nx*ny]) +
         std::min(c,0.0)*(f[idx+nx*ny]-f[idx]))*idz1;
}

inline double diffusion(const double* __restrict f, const int nx,
                        const int ny, const int idx, const double idx2,
                        const double idy2, const double idz2)
{
    return
        (f[idx+1]     + f[idx-1]     - 2.0*f[idx])*idx2 +
        (f[idx+nx]    + f[idx-nx]    - 2.0*f[idx])*idy2 +
        (f[idx+nx*ny] + f[idx-nx*ny] - 2.0*f[idx])*idz2;
}

void divergence(double* __restrict f, const double* __restrict u,
                const double* __restrict v, const double* __restrict w,
                int nx, int ny, int nz, double dx, double dy, double dz)
{
    const double idx = 0.5/dx;
    const double idy = 0.5/dy;
    const double idz = 0.5/dz;

    #pragma omp for collapse(2) schedule(static)
    for (int k=1; k<nz-1; k++) {
        for (int j=1; j<ny-1; j++) {
            #pragma omp simd aligned(f, u, v, w: 64)
            for (int i=1; i<nx-1; i++) {
                const int id = k*ny*nx + j*nx + i;
                f[id] = (u[id+1] - u[id-1])*idx +
                        (v[id+nx] - v[id-nx])*idy +
                        (w[id+nx*ny] - w[id-nx*ny])*idz;
            }
        }
    }
}

// Parallelized Neumann Boundaries
inline void Neumann_bc_parallel(double* __restrict f, int nx, int ny, int nz)
{
    #pragma omp for schedule(static) nowait
    for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
            f[j*nx + i] = f[nx*ny + j*nx + i];
            f[(nz-1)*nx*ny + j*nx + i] = f[(nz-2)*nx*ny + j*nx + i];
        }
    }
    #pragma omp for schedule(static) nowait
    for (int k=0; k<nz; k++) {
        for (int i=0; i<nx; i++) {
            f[k*nx*ny + i] = f[k*nx*ny + nx + i];
            f[k*nx*ny + (ny-1)*nx + i] = f[k*nx*ny + (ny-2)*nx + i];
        }
    }
    #pragma omp for schedule(static)
    for (int k=0; k<nz; k++) {
        for (int j=0; j<ny; j++) {
            f[k*nx*ny + j*nx] = f[k*nx*ny + j*nx + 1];
            f[k*nx*ny + j*nx + nx-1] = f[k*nx*ny + j*nx + nx-2];
        }
    }
}

// Parallelized Dirichlet Boundaries
inline void Dirichlet_bc_parallel(double* __restrict f, int nx, int ny, int nz)
{
    #pragma omp for schedule(static) nowait
    for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
            f[j*nx + i] = 0.0;
            f[(nz-1)*nx*ny + j*nx + i] = 0.0;
        }
    }
    #pragma omp for schedule(static) nowait
    for (int k=0; k<nz; k++) {
        for (int i=0; i<nx; i++) {
            f[k*nx*ny + i] = 0.0;
            f[k*nx*ny + (ny-1)*nx + i] = 0.0;
        }
    }
    #pragma omp for schedule(static)
    for (int k=0; k<nz; k++) {
        for (int j=0; j<ny; j++) {
            f[k*nx*ny + j*nx] = 0.0;
            f[k*nx*ny + j*nx + nx-1] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Poisson solvers (Neumann BCs). Both are collective OpenMP routines:
// they MUST be called from inside an already-active parallel region.
// ---------------------------------------------------------------------------

// Jacobi relaxation (original). Reads from `cur`, writes to scratch dv.u_new,
// then ping-pongs. Vectorizes cleanly (unit-stride inner loop) but converges
// half as fast as red-black GS and needs the extra scratch buffer.
void poisson_solver_Neumann_bc_jacobi(double* __restrict u,
                                      const double* __restrict f,
                                      int nx, int ny, int nz, double dx,
                                      double dy, double dz, int nitr,
                                      dummy_variables& dv)
{
    const double idx2 = 1.0/sqr(dx);
    const double idy2 = 1.0/sqr(dy);
    const double idz2 = 1.0/sqr(dz);
    const double factor = 1.0/(2.0*(idx2+idy2+idz2));
    const long long N = static_cast<long long>(nx)*ny*nz;
    double *cur = u;
    double *next = dv.u_new;

    for (int itr=0; itr<nitr; itr++) {
        #pragma omp for collapse(2) schedule(static)
        for (int k=1; k<nz-1; k++) {
            for (int j=1; j<ny-1; j++) {
                #pragma omp simd aligned(cur, next, f: 64)
                for (int i=1; i<nx-1; i++) {
                    const int id = k*ny*nx + j*nx + i;
                    next[id] =
                        ((cur[id+1]     + cur[id-1])     * idx2 +
                         (cur[id+nx]    + cur[id-nx])    * idy2 +
                         (cur[id+nx*ny] + cur[id-nx*ny]) * idz2 -
                         f[id]) * factor;
                }
            }
        }

        Neumann_bc_parallel(next, nx, ny, nz);

        double *tmp = cur;
        cur = next;
        next = tmp;
    }

    if (cur != u) {
        #pragma omp for schedule(static)
        for (long long i=0; i<N; i++) {
            u[i] = cur[i];
        }
    }
}

// Red-Black Gauss-Seidel relaxation. Updates `u` in place (no scratch buffer).
// Each sweep does a red pass (i+j+k even) then a black pass (i+j+k odd); the
// black pass reads the just-updated red neighbours, which is the GS speed-up
// (~2x faster asymptotic convergence than Jacobi -> roughly half the sweeps
// for the same residual). Trade-off: the stride-2 inner loop does NOT
// vectorize like the Jacobi sweep, so per-sweep cost is higher; the net win
// comes from needing fewer sweeps and no ping-pong copy.
//
// Barriers per sweep: 1 (red) + 1 (black) + 1 (Neumann BC) = 3, vs Jacobi's 2.
// With ~half the sweeps this is still fewer total barriers than Jacobi.
void poisson_solver_Neumann_bc_rbgs(double* __restrict u,
                                    const double* __restrict f,
                                    int nx, int ny, int nz, double dx,
                                    double dy, double dz, int nitr)
{
    const double idx2 = 1.0/sqr(dx);
    const double idy2 = 1.0/sqr(dy);
    const double idz2 = 1.0/sqr(dz);
    const double factor = 1.0/(2.0*(idx2+idy2+idz2));

    for (int itr=0; itr<nitr; itr++) {
        // color 0 -> red (i+j+k even), color 1 -> black (i+j+k odd).
        for (int color=0; color<2; color++) {
            #pragma omp for collapse(2) schedule(static)
            for (int k=1; k<nz-1; k++) {
                for (int j=1; j<ny-1; j++) {
                    // Smallest interior i whose (i+j+k) parity matches `color`.
                    const int start = 2 - ((j + k + color) & 1);
                    for (int i=start; i<nx-1; i+=2) {
                        const int id = k*ny*nx + j*nx + i;
                        u[id] =
                            ((u[id+1]     + u[id-1])     * idx2 +
                             (u[id+nx]    + u[id-nx])    * idy2 +
                             (u[id+nx*ny] + u[id-nx*ny]) * idz2 -
                             f[id]) * factor;
                    }
                }
            }
            // implicit barrier at end of the omp for: the black pass sees the
            // updated red values, and vice versa on the next sweep.
        }

        Neumann_bc_parallel(u, nx, ny, nz);
    }
}

// Dispatcher: picks the relaxation method selected at runtime. Keeps the same
// signature as the original solver, so call sites stay unchanged. Collective:
// call from inside an active parallel region.
void poisson_solver_Neumann_bc(double* __restrict u, const double* __restrict f,
                               int nx, int ny, int nz, double dx, double dy,
                               double dz, int nitr, dummy_variables& dv)
{
    if (dv.poisson_method == PoissonMethod::RedBlackGS) {
        poisson_solver_Neumann_bc_rbgs(u, f, nx, ny, nz, dx, dy, dz, nitr);
    } else {
        poisson_solver_Neumann_bc_jacobi(u, f, nx, ny, nz,
                                         dx, dy, dz, nitr, dv);
    }
}

void prediction(double* __restrict u_star, double* __restrict v_star,
                double* __restrict w_star, const double* __restrict u,
                const double* __restrict v, const double* __restrict w,
                const double* __restrict p, const double* __restrict T,
                double Pr, int nx, int ny, int nz, double dx, double dy,
                double dz, double dt)
{
    const double idx2 = 1.0/sqr(dx);
    const double idy2 = 1.0/sqr(dy);
    const double idz2 = 1.0/sqr(dz);
    const double idx1 = 1.0/dx;
    const double idy1 = 1.0/dy;
    const double idz1 = 1.0/dz;
    const double idx = 0.5/dx;
    const double idy = 0.5/dy;
    const double idz = 0.5/dz;

    #pragma omp for collapse(2) schedule(static)
    for (int k=1; k<nz-1; k++) {
        for (int j=1; j<ny-1; j++) {
            #pragma omp simd aligned(u_star, v_star, w_star, u, v, w, p, T: 64)
            for (int i=1; i<nx-1; i++) {
                const int id = k*ny*nx + j*nx + i;

                double adv = advection(u[id], v[id], w[id], u,
                                       nx, ny, id, idx1, idy1, idz1);
                double diff = Pr*diffusion(u, nx, ny, id, idx2, idy2, idz2);
                double source = -(p[id+1] - p[id-1])*idx;
                u_star[id] = u[id] + dt*(-adv + diff + source);

                adv = advection(u[id], v[id], w[id], v,
                                nx, ny, id, idx1, idy1, idz1);
                diff = Pr*diffusion(v, nx, ny, id, idx2, idy2, idz2);
                source = -(p[id+nx] - p[id-nx])*idy + Pr*T[id];
                v_star[id] = v[id] + dt*(-adv + diff + source);

                adv = advection(u[id], v[id], w[id], w,
                                nx, ny, id, idx1, idy1, idz1);
                diff = Pr*diffusion(w, nx, ny, id, idx2, idy2, idz2);
                source = -(p[id+nx*ny] - p[id-nx*ny])*idz;
                w_star[id] = w[id] + dt*(-adv + diff + source);
            }
        }
    }
}

void correction(const double* __restrict u_star, const double* __restrict v_star,
                const double* __restrict w_star, double* __restrict u,
                double* __restrict v, double* __restrict w,
                const double* __restrict p_prime, int nx, int ny,
                int nz, double dx, double dy, double dz, double dt)
{
    const double idx = 0.5/dx;
    const double idy = 0.5/dy;
    const double idz = 0.5/dz;
    const double dt_idx = dt*idx;
    const double dt_idy = dt*idy;
    const double dt_idz = dt*idz;

    #pragma omp for collapse(2) schedule(static)
    for (int k=1; k<nz-1; k++) {
        for (int j=1; j<ny-1; j++) {
            #pragma omp simd aligned(u, v, w, u_star, v_star, w_star, p_prime: 64)
            for (int i=1; i<nx-1; i++) {
                const int id = k*ny*nx + j*nx + i;
                u[id] = u_star[id] - (p_prime[id+1] - p_prime[id-1])*dt_idx;
                v[id] = v_star[id] - (p_prime[id+nx] - p_prime[id-nx])*dt_idy;
                w[id] = w_star[id] -
                        (p_prime[id+nx*ny] - p_prime[id-nx*ny])*dt_idz;
            }
        }
    }
}

void T_compute(const double* __restrict u, const double* __restrict v,
               const double* __restrict w, double* __restrict T,
               const double* __restrict I, double Ra, int nx, int ny, int nz,
               double dx, double dy, double dz, double dt, dummy_variables& dv)
{
    const double idx2 = 1.0/sqr(dx);
    const double idy2 = 1.0/sqr(dy);
    const double idz2 = 1.0/sqr(dz);
    const double idx1 = 1.0/dx;
    const double idy1 = 1.0/dy;
    const double idz1 = 1.0/dz;
    const int N = nx*ny*nz;

    double* __restrict T_new = dv.T_new;

    #pragma omp for collapse(2) schedule(static)
    for (int k=1; k<nz-1; k++) {
        for (int j=1; j<ny-1; j++) {
            #pragma omp simd aligned(u, v, w, T, I, T_new: 64)
            for (int i=1; i<nx-1; i++) {
                const int id = k*ny*nx + j*nx + i;
                const double adv = advection(u[id], v[id], w[id], T,
                                             nx, ny, id, idx1, idy1, idz1);
                const double diff = diffusion(T, nx, ny, id, idx2, idy2, idz2);
                const double source = Ra*I[id];
                T_new[id] = T[id] + dt*(-adv + diff + source);
            }
        }
    }

    #pragma omp for schedule(static)
    for (int i=0; i<N; i++) {
        T[i] = T_new[i];
    }
}

// Collective OpenMP routine: call from inside an active parallel region.
void ns_compute_parallel(double *u, double *v, double *w, double *p, double *T,
                         double *I, double Pr, double Ra, int nx, int ny,
                         int nz, double dx, double dy, double dz, double dt,
                         dummy_variables& dv)
{
    const int N = nx*ny*nz;
    const double inv_dt = 1.0/dt;

    #pragma omp for schedule(static)
    for (int i=0; i<N; i++) {
        dv.p_prime[i] = 0.0;
    }

    prediction(dv.u_star, dv.v_star, dv.w_star, u, v, w, p, T,
               Pr, nx, ny, nz, dx, dy, dz, dt);

    Dirichlet_bc_parallel(dv.u_star, nx, ny, nz);
    Dirichlet_bc_parallel(dv.v_star, nx, ny, nz);
    Dirichlet_bc_parallel(dv.w_star, nx, ny, nz);

    divergence(dv.div, dv.u_star, dv.v_star, dv.w_star,
               nx, ny, nz, dx, dy, dz);

    #pragma omp for schedule(static)
    for (int i=0; i<N; i++) {
        dv.div_dt[i] = dv.div[i]*inv_dt;
    }

    poisson_solver_Neumann_bc(dv.p_prime, dv.div_dt, nx, ny, nz,
                              dx, dy, dz, dv.poisson_iters, dv);

    correction(dv.u_star, dv.v_star, dv.w_star, u, v, w, dv.p_prime,
               nx, ny, nz, dx, dy, dz, dt);

    Dirichlet_bc_parallel(u, nx, ny, nz);
    Dirichlet_bc_parallel(v, nx, ny, nz);
    Dirichlet_bc_parallel(w, nx, ny, nz);

    #pragma omp for schedule(static)
    for (int i=0; i<N; i++) {
        p[i] += dv.p_prime[i];
    }

    Neumann_bc_parallel(p, nx, ny, nz);

    T_compute(u, v, w, T, I, Ra, nx, ny, nz, dx, dy, dz, dt, dv);

    Dirichlet_bc_parallel(T, nx, ny, nz);
}

void ns_compute(double *u, double *v, double *w, double *p, double *T,
                double *I, double Pr, double Ra, int nx, int ny, int nz,
                double dx, double dy, double dz, double dt,
                dummy_variables& dv)
{
    #pragma omp parallel
    {
        ns_compute_parallel(u, v, w, p, T, I, Pr, Ra, nx, ny, nz,
                            dx, dy, dz, dt, dv);
    }
}

// Mathematically Correct: Preserves Z-Marching Physics.
// Must be called from inside a parallel region.
void A_compute_ss_parallel(double* __restrict Ar, double* __restrict Ai,
                           const double* __restrict T,
                           const double* __restrict phase_c,
                           const double* __restrict phase_s,
                           fftw_complex *in, fftw_complex *out,
                           fftw_plan ff, fftw_plan fb, int nx, int ny,
                           int nz, double beta, double wn, double C,
                           double dz)
{
    const int nx1 = nx-1;
    const int ny1 = ny-1;
    const int plane = nx*ny;
    const int fftN = nx1*ny1;
    const double atten = std::exp(-beta*dz/2.0);
    const double inv_fftN = 1.0/static_cast<double>(fftN);
    const double temp_phase_scale = 0.5*C*wn*dz;

    // NO #pragma omp for on the Z-loop. Must march sequentially!
    for (int k=1; k<nz; k++) {
        const int prev = (k-1)*plane;
        const int curr = k*plane;

        // Threads share the work of setting up the X-Y plane
        #pragma omp for collapse(2) schedule(static)
        for (int j=0; j<ny1; j++) {
            for (int i=0; i<nx1; i++) {
                const int fft_id = j*nx1 + i;
                const int grid_id = prev + j*nx + i;
                const double ca = temp_phase_scale*T[grid_id];
                const double s = std::sin(ca);
                const double c = std::cos(ca);

                in[fft_id][0] = atten*(Ar[grid_id]*c - Ai[grid_id]*s);
                in[fft_id][1] = atten*(Ar[grid_id]*s + Ai[grid_id]*c);
            }
        }

        // Threads MUST wait while one thread performs the 2D FFT
        #pragma omp single
        {
            fftw_execute(ff);
        }

        // Threads share the phase shift multiplication
        #pragma omp for collapse(2) schedule(static)
        for (int j=0; j<ny1; j++) {
            for (int i=0; i<nx1; i++) {
                const int fft_id = j*nx1 + i;
                const double c = phase_c[fft_id];
                const double s = phase_s[fft_id];
                const double re = out[fft_id][0];
                const double im = out[fft_id][1];

                in[fft_id][0] = re*c - im*s;
                in[fft_id][1] = re*s + im*c;
            }
        }

        // Threads MUST wait again
        #pragma omp single
        {
            fftw_execute(fb);
        }

        // Threads share writing the output back to the main Ar/Ai grid
        #pragma omp for collapse(2) schedule(static)
        for (int j=0; j<ny1; j++) {
            for (int i=0; i<nx1; i++) {
                const int fft_id = j*nx1 + i;
                const int grid_id = curr + j*nx + i;
                Ar[grid_id] = out[fft_id][0]*inv_fftN;
                Ai[grid_id] = out[fft_id][1]*inv_fftN;
            }
        }

        #pragma omp for schedule(static)
        for (int i=0; i<nx; i++) {
            Ar[curr + (ny-1)*nx + i] = Ar[curr + i];
            Ai[curr + (ny-1)*nx + i] = Ai[curr + i];
        }

        #pragma omp for schedule(static)
        for (int j=0; j<ny; j++) {
            Ar[curr + j*nx + nx-1] = Ar[curr + j*nx];
            Ai[curr + j*nx + nx-1] = Ai[curr + j*nx];
        }
    }
}

void A_compute_ss(double *Ar, double *Ai, double *T,
                  double *phase_c, double *phase_s,
                  fftw_complex *in, fftw_complex *out, fftw_plan ff,
                  fftw_plan fb, int nx, int ny, int nz, double beta,
                  double wn, double C, double dz)
{
    #pragma omp parallel
    {
        A_compute_ss_parallel(Ar, Ai, T, phase_c, phase_s, in, out, ff, fb,
                              nx, ny, nz, beta, wn, C, dz);
    }
}

void data_write(const std::string& filename, double *X, int nx, int ny, int nz)
{
    std::ofstream File(filename);
    const long long N = static_cast<long long>(nx) * ny * nz;
    for (long long i = 0; i < N; i++) {
        File << X[i] << '\n';
      }
}

#endif
