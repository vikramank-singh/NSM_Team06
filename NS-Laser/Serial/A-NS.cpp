#include<iostream>
#include<cmath>
#include<fftw3.h>
#include"functions.h"



int nx = 201;
int ny = 201;
int nz = 20;

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

double *Ar = new double[nx*ny*nz];
double *Ai = new double[nx*ny*nz];
double *T  = new double[nx*ny*nz];
double *u  = new double[nx*ny*nz];
double *v  = new double[nx*ny*nz];
double *w  = new double[nx*ny*nz];
double *p  = new double[nx*ny*nz];
double *I = new double[nx*ny*nz];

double error;

double *kx = new double[(nx-1)*(ny-1)];
double *ky = new double[(nx-1)*(ny-1)];

fftw_complex* in  = new fftw_complex[(nx-1)*(ny-1)];
fftw_complex* out = new fftw_complex[(nx-1)*(ny-1)];

fftw_plan ff = fftw_plan_dft_2d(nx-1, ny-1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
fftw_plan fb = fftw_plan_dft_2d(nx-1, ny-1, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);


// ********************************************************** \\

int main(){

dummy_variables dv;

dv.u_star = new double[nx*ny*nz];
dv.v_star = new double[nx*ny*nz];
dv.w_star = new double[nx*ny*nz];
dv.p_prime = new double[nx*ny*nz];
dv.div = new double[nx*ny*nz];
dv.div_dt = new double[nx*ny*nz];
dv.T_new = new double[nx*ny*nz];


initialization(Ar, Ai, u, v, w, I, T, p, beta, nx, ny, nz, lx, ly, dx, dy, dz);
initialize_wavnumbers(kx, ky, nx, ny, lx, ly, dx, dy);

for (int itr=0; itr<1500; itr++){
	ns_compute(u, v, w, p, T, I, Pr, Ra, nx, ny, nz, dx, dy, dz, dt, dv);
	A_compute_ss(Ar, Ai, T, kx, ky, in, out, ff, fb, nx, ny, nz, beta, wn, C, dz);
	for(int i=0; i<nx*ny*nz; i++) I[i] = Ar[i]*Ar[i] + Ai[i]*Ai[i];
	std::cout<<"itr = "<<itr<<std::endl;
	}

std::cout<<"Time ="<<1500*dt;
	
data_write("T.txt", T, nx, ny, nz);
data_write("u.txt", u, nx, ny, nz);
data_write("v.txt", v, nx, ny, nz);
data_write("w.txt", w, nx, ny, nz);	
data_write("I.txt", I, nx, ny, nz);
data_write("p.txt", p, nx, ny, nz);


delete[] dv.u_star;
delete[] dv.v_star; 
delete[] dv.w_star;
delete[] dv.p_prime;
delete[] dv.div; 
delete[] dv.div_dt;
delete[] dv.div_dt;



return 0;
}
