cat A-NS-gpu.cpp
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <omp.h>
#include "functions_gpu.h"

namespace {
int    iarg(char** v, int i, int d) { return v[i] ? std::atoi(v[i]) : d; }
double darg(char** v, int i, double d) { return v[i] ? std::atof(v[i]) : d; }

PoissonMethod parse_method(const char* s)
{
    if (!s) return PoissonMethod::Jacobi;
    if (!std::strcmp(s,"rbgs") || !std::strcmp(s,"rb") ||
        !std::strcmp(s,"gs")   || !std::strcmp(s,"1"))
        return PoissonMethod::RedBlackGS;
    return PoissonMethod::Jacobi;
}
const char* mname(PoissonMethod m)
{ return m==PoissonMethod::RedBlackGS ? "red-black-gs" : "jacobi"; }

double* calloc_d(long long n) { return (double*)std::calloc((size_t)n, sizeof(double)); }
}

int main(int argc, char** argv)
{
    const bool has_method = (argc==2 || argc==6 || argc==9);
    const int  ga = has_method ? argc-1 : argc;
    const int nx   = (ga>=5) ? iarg(argv,1,301) : 301;
    const int ny   = (ga>=5) ? iarg(argv,2,301) : 301;
    const int nz   = (ga>=5) ? iarg(argv,3,20)  : 20;
    const int Nitr = (ga>=5) ? iarg(argv,4,100) : 100;
    const double lx = (ga==8) ? darg(argv,5,5.0)   : 5.0;
    const double ly = (ga==8) ? darg(argv,6,5.0)   : 5.0;
    const double lz = (ga==8) ? darg(argv,7,200.0) : 200.0;
    const PoissonMethod method = has_method ? parse_method(argv[argc-1]) : PoissonMethod::Jacobi;

    const double dx=lx/(nx-1), dy=ly/(ny-1), dz=lz/(nz-1);
    const double Pr=0.7, Ra=10000.0, beta=5.5e-7, C=2.2e-8, wn=60000.0;
    const double dt = 0.5*sqr(dx)/4.0;
    const int fftN = (nx-1)*(ny-1);
    const long long N = (long long)nx*ny*nz;

    int piters = 150;
    if (const char* s = std::getenv("POISSON_SWEEPS")) { int vv=std::atoi(s); if (vv>0) piters=vv; }

    // ---- device-resident fields + scratch: all plain top-level pointers ----
    double *u=calloc_d(N), *v=calloc_d(N), *w=calloc_d(N), *p=calloc_d(N), *T=calloc_d(N), *I=calloc_d(N);
    double *us=calloc_d(N), *vs=calloc_d(N), *ws=calloc_d(N), *pp=calloc_d(N);
    double *dvg=calloc_d(N), *div_dt=calloc_d(N), *Tn=calloc_d(N), *unew=calloc_d(N);

    // laser (host) buffers
    double *Ar=calloc_d(N), *Ai=calloc_d(N);
    double *kx=calloc_d(fftN), *ky=calloc_d(fftN), *phase_c=calloc_d(fftN), *phase_s=calloc_d(fftN);
    fftw_complex *in  = (fftw_complex*)fftw_malloc(fftN*sizeof(fftw_complex));
    fftw_complex *out = (fftw_complex*)fftw_malloc(fftN*sizeof(fftw_complex));
    fftw_plan ff = fftw_plan_dft_2d(nx-1, ny-1, in, out, FFTW_FORWARD,  FFTW_MEASURE);
    fftw_plan fb = fftw_plan_dft_2d(nx-1, ny-1, in, out, FFTW_BACKWARD, FFTW_MEASURE);

    // ---- initialization (host) ----
    for (int k=0;k<nz;k++)
        for (int j=0;j<ny;j++)
            for (int i=0;i<nx;i++) {
                const double x=i*dx, y=j*dy, z=k*dz;
                const double ar0 = std::exp(-2.0*(sqr(x-lx/2.0)+sqr(y-ly/2.0)));
                const long long id=(long long)k*nx*ny+(long long)j*nx+i;
                u[id]=0.0; v[id]=0.0; w[id]=0.0; p[id]=0.0; T[id]=0.0;
                Ar[id]=ar0; Ai[id]=0.0;
                I[id]=std::exp(-beta*z)*ar0*ar0;
            }
    initialize_wavnumbers(kx, ky, nx, ny, lx, ly, dx, dy);
    initialize_fft_phase(phase_c, phase_s, kx, ky, nx, ny, dz, wn);

    const int ndev = omp_get_num_devices();
    std::cout << "omp_get_num_devices()=" << ndev << "  (0 => host-fallback build)" << std::endl;

    const double t0 = omp_get_wtime();

    #pragma omp target enter data map(to: u[0:N], v[0:N], w[0:N], p[0:N], T[0:N], I[0:N], \
        us[0:N], vs[0:N], ws[0:N], pp[0:N], dvg[0:N], div_dt[0:N], Tn[0:N], unew[0:N])
    {
        for (int itr = 0; itr < Nitr; itr++) {
            ns_compute_gpu(u, v, w, p, T, I, us, vs, ws, pp, dvg, div_dt, Tn, unew,
                           Pr, Ra, method, piters, nx, ny, nz, dx, dy, dz, dt);

            // laser BPM stays on the host: bring T down, run FFT march, push I up.
            #pragma omp target update from(T[0:N])
            A_compute_ss(Ar, Ai, T, phase_c, phase_s, in, out, ff, fb, nx, ny, nz, beta, wn, C, dz);
            for (long long i = 0; i < N; i++) I[i] = Ar[i]*Ar[i] + Ai[i]*Ai[i];
            #pragma omp target update to(I[0:N])
        }
        #pragma omp target update from(u[0:N], v[0:N], w[0:N], p[0:N], T[0:N], I[0:N])
    }
    #pragma omp target exit data map(release: u[0:N], v[0:N], w[0:N], p[0:N], T[0:N], I[0:N], \
        us[0:N], vs[0:N], ws[0:N], pp[0:N], dvg[0:N], div_dt[0:N], Tn[0:N], unew[0:N])

    const double t1 = omp_get_wtime();
    std::cout << "device=" << (ndev>0?"GPU":"host") << " nx=" << nx << " ny=" << ny
              << " nz=" << nz << " iters=" << Nitr << " poisson=" << mname(method)
              << " sweeps=" << piters << " elapsed=" << (t1-t0) << "s" << std::endl;

    data_write("T.txt", T, nx, ny, nz);
    data_write("u.txt", u, nx, ny, nz);
    data_write("v.txt", v, nx, ny, nz);
    data_write("w.txt", w, nx, ny, nz);
    data_write("I.txt", I, nx, ny, nz);
    data_write("p.txt", p, nx, ny, nz);

    std::free(u); std::free(v); std::free(w); std::free(p); std::free(T); std::free(I);
    std::free(us); std::free(vs); std::free(ws); std::free(pp);
    std::free(dvg); std::free(div_dt); std::free(Tn); std::free(unew);
    std::free(Ar); std::free(Ai); std::free(kx); std::free(ky); std::free(phase_c); std::free(phase_s);
    fftw_destroy_plan(ff); fftw_destroy_plan(fb); fftw_free(in); fftw_free(out);
    return 0;
}
