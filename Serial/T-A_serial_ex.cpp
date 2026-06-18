#include<iostream>
#include<cmath>
#include<fftw3.h>
#include"functions.h"


double pi = 3.14159265358;

int nx = 121;
int ny = 121;
int nz = 20;

double lx = 0.7/100;
double ly = 0.7/100;
double lz = 2.5/100;
double dx = lx/(nx-1);
double dy = ly/(ny-1);
double dz = lz/(nz-1);

double n0 = 1.63;
double d  = 2.8/1000;
double P  = 11;
double lambda = 10.6e-6;
double wn  = 2*pi*n0/lambda;
double alpha = 1.25e-7;
double beta = 25.2;
double rho = 1260;
double cp = 950;
double u = 0.9/100;
double A0 = sqrt(4*P/(pi*d*d));
double dn_dT = -0.79e-3;

double dt = 0.5*std::min(dx/u, dx*dx/(4*alpha));

double *Ar = new double[nx*ny*nz];
double *Ai = new double[nx*ny*nz];
double *T  = new double[nx*ny*nz];
double *Tnew  = new double[nx*ny*nz];
double *I = new double[nx*ny*nz];

double error;

double *kx = new double[(nx-1)*(ny-1)];
double *ky = new double[(nx-1)*(ny-1)];

fftw_complex* in  = new fftw_complex[(nx-1)*(ny-1)];
fftw_complex* out = new fftw_complex[(nx-1)*(ny-1)];

fftw_plan ff = fftw_plan_dft_2d(nx-1, ny-1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
fftw_plan fb = fftw_plan_dft_2d(nx-1, ny-1, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);


// **************************************************************************** //

int main()
{

// Wavenumbers kx, ky
for(int j=0; j<ny-1; j++){ 
	double KY = j*(j <= (ny-1)/2) + (j-(ny-1))*(j > (ny-1)/2);
	for(int i=0; i<nx-1; i++){	
		double KX = i*(i <= (nx-1)/2) + (i-(nx-1))*(i > (nx-1)/2);
		kx[j*(nx-1) + i] = (2*pi/lx)*KX;
		ky[j*(nx-1) + i] = (2*pi/ly)*KY;
	}
}


// Initial Conditions
for(int k=0; k<nz; k++){
	double z = k*dz;
	for(int i=0; i<nx; i++){ 
		double x = i*dx;
		for(int j=0; j<ny; j++){ 
			double y = j*dy;
			Ar[k*nx*ny + j*nx + i] = A0*exp(-2*((x-lx/2)*(x-lx/2) + (y-ly/2)*(y-ly/2))/pow(d,2));
			Ai[k*nx*ny + j*nx + i] = 0.0;
			I[k*nx*ny + j*nx +i] = exp(-beta*z)*pow(Ar[j*nx + i],2);
			T[k*nx*ny + j*nx + i] = 300.0;
			Tnew[k*nx*ny + j*nx + i] = 300.0;
		}
	}
}



// Main loop
for(int itr=0; itr<200; itr++){
	T_compute(T, Tnew, I, nx, ny, nz, beta, alpha, rho, cp, u, dx, dy, dz, dt);  // Temperature
	bcT_compute(T, Tnew, nx, ny, nz);                      // Temperature BC
	A_compute_ss(Ar, Ai, T, kx, ky, in, out, ff, fb, nx, ny, nz, beta, wn, dn_dT, dz, n0);     // Laser
	//A_compute_im(Ar, Ai, T, nx, ny, nz, 0, wn, dn_dT, dz, n0, dx, dy, 300);
	error = L_inf_norm(T, Tnew, nx*ny*nz);
	for(int i=0; i<nx*ny*nz; i++){
		I[i] = Ar[i]*Ar[i] + Ai[i]*Ai[i];
		T[i] = Tnew[i];
	}
}
		

std::cout<<"L infinite norm of dT/dt: "<<error/dt;
 
 for(int i=0; i<nx*ny*nz; i++) I[i] /= A0*A0;

data_write("I.txt", I, nx, ny, nz);
data_write("T.txt", T, nx, ny, nz);


return 0;
}




