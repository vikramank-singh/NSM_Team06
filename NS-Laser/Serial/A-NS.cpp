#include<iostream>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<fftw3.h>
#include"functions.h"

// Coupled Navier-Stokes / paraxial-laser solver.
//
// Usage: ./A-NS [nx ny nz n_iters] [--no-output]
//   With no arguments the original production grid (201x201x20, 1500 iters)
//   and full data_write() dump are used, so existing post-processing
//   (plot.py) keeps working unchanged.
//   A small grid + --no-output is the fast-turnaround configuration for
//   Intel Advisor / vectorization-report runs, e.g.:
//     ./A-NS 51 51 10 50 --no-output

int main(int argc, char **argv){

int nx = 201;
int ny = 201;
int nz = 20;
int n_iters = 1500;
bool write_output = true;

int posarg = 0;
for (int a = 1; a < argc; a++){
	if (std::strcmp(argv[a], "--no-output") == 0){
		write_output = false;
	} else {
		switch (posarg){
			case 0: nx = std::atoi(argv[a]); break;
			case 1: ny = std::atoi(argv[a]); break;
			case 2: nz = std::atoi(argv[a]); break;
			case 3: n_iters = std::atoi(argv[a]); break;
		}
		posarg++;
	}
}

double lx = 5.0;
double ly = 5.0;
double lz = 200.0;
double dx = lx/(nx-1);
double dy = ly/(ny-1);
double dz = lz/(nz-1);

double Pr = 0.7;
double Ra = 10000.0;
double beta = 5.5e-7;
double C = 2.2e-8;
double wn = 60000.0; // wavenuber k

double dt = 0.5*dx*dx/4;

const int n = nx*ny*nz;

double *Ar = alloc_field(n);
double *Ai = alloc_field(n);
double *T  = alloc_field(n);
double *u  = alloc_field(n);
double *v  = alloc_field(n);
double *w  = alloc_field(n);
double *p  = alloc_field(n);
double *I  = alloc_field(n);

double *kx = alloc_field((nx-1)*(ny-1));
double *ky = alloc_field((nx-1)*(ny-1));

fftw_complex* in  = new fftw_complex[(nx-1)*(ny-1)];
fftw_complex* out = new fftw_complex[(nx-1)*(ny-1)];

fftw_plan ff = fftw_plan_dft_2d(nx-1, ny-1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
fftw_plan fb = fftw_plan_dft_2d(nx-1, ny-1, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

dummy_variables dv;

dv.u_star = alloc_field(n);
dv.v_star = alloc_field(n);
dv.w_star = alloc_field(n);
dv.p_prime = alloc_field(n);
dv.div = alloc_field(n);
dv.div_dt = alloc_field(n);
dv.T_new = alloc_field(n);

dv.dct_buf = alloc_field((nx-2)*(ny-2)*(nz-2));
dv.lam_x = alloc_field(nx-2);
dv.lam_y = alloc_field(ny-2);
dv.lam_z = alloc_field(nz-2);
dct_eigs(dv.lam_x, nx-2, dx);
dct_eigs(dv.lam_y, ny-2, dy);
dct_eigs(dv.lam_z, nz-2, dz);
dv.dct_fwd = fftw_plan_r2r_3d(nz-2, ny-2, nx-2, dv.dct_buf, dv.dct_buf,
	FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
dv.dct_bwd = fftw_plan_r2r_3d(nz-2, ny-2, nx-2, dv.dct_buf, dv.dct_buf,
	FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE);


initialization(Ar, Ai, u, v, w, I, T, p, beta, nx, ny, nz, lx, ly, dx, dy, dz);
initialize_wavnumbers(kx, ky, nx, ny, lx, ly, dx, dy);

for (int itr=0; itr<n_iters; itr++){
	ns_compute(u, v, w, p, T, I, Pr, Ra, nx, ny, nz, dx, dy, dz, dt, dv);
	A_compute_ss(Ar, Ai, T, kx, ky, in, out, ff, fb, nx, ny, nz, beta, wn, C, dz);
	for(int i=0; i<n; i++) I[i] = Ar[i]*Ar[i] + Ai[i]*Ai[i];
	std::cout<<"itr = "<<itr<<std::endl;
	}

std::cout<<"Time ="<<n_iters*dt<<std::endl;

if (write_output){
	data_write("T.txt", T, nx, ny, nz);
	data_write("u.txt", u, nx, ny, nz);
	data_write("v.txt", v, nx, ny, nz);
	data_write("w.txt", w, nx, ny, nz);
	data_write("I.txt", I, nx, ny, nz);
	data_write("p.txt", p, nx, ny, nz);
}


free(dv.u_star);
free(dv.v_star);
free(dv.w_star);
free(dv.p_prime);
free(dv.div);
free(dv.div_dt);
free(dv.T_new);
free(dv.dct_buf);
free(dv.lam_x);
free(dv.lam_y);
free(dv.lam_z);
fftw_destroy_plan(dv.dct_fwd);
fftw_destroy_plan(dv.dct_bwd);

free(Ar); free(Ai); free(T); free(u); free(v); free(w); free(p); free(I);
free(kx); free(ky);
delete[] in; delete[] out;
fftw_destroy_plan(ff); fftw_destroy_plan(fb);

return 0;
}
