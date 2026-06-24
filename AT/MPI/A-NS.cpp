#include<iostream>
#include<cmath>
#include<fftw3.h>
#include"functions.h" 
#include"mpi.h"


double pi = 3.14159265358;

int nx = 121;
int ny = 121;
int nz = 80;

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

double error;

// **************************************************************************** //

int main(int argc, char **argv)
{

int rank, size;

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

int local_nz = nz/size;
int i, j, k;

double *Ar = nullptr;
double *Ai = nullptr;
double *kx = nullptr;
double *ky = nullptr;
double *Tfull = nullptr;
double *Ifull = nullptr;

double *T  = new double[nx*ny*local_nz];
double *Tnew  = new double[nx*ny*local_nz];
double *I = new double[nx*ny*local_nz];
double *ghostL = new double[nx*ny];
double *ghostR = new double[nx*ny];

fftw_complex* in  = nullptr;
fftw_complex* out = nullptr;
fftw_plan ff = nullptr;
fftw_plan fb = nullptr;


// Initial Conditions
for(int k=0; k<local_nz; k++){
	for(int i=0; i<nx; i++){ 
		for(int j=0; j<ny; j++){ 
			I[k*nx*ny + j*nx +i] = 0.0;
			T[k*nx*ny + j*nx + i] = 300.0;
			Tnew[k*nx*ny + j*nx + i] = 300.0;
		}
	}
}

if(rank==0){
	kx = new double[(nx-1)*(ny-1)];
	ky = new double[(nx-1)*(ny-1)];
	// Wavenumbers kx, ky
	for(int j=0; j<ny-1; j++){ 
		double KY = j*(j <= (ny-1)/2) + (j-(ny-1))*(j > (ny-1)/2);
		for(int i=0; i<nx-1; i++){	
			double KX = i*(i <= (nx-1)/2) + (i-(nx-1))*(i > (nx-1)/2);
			kx[j*(nx-1) + i] = (2*pi/lx)*KX;
			ky[j*(nx-1) + i] = (2*pi/ly)*KY;
		}
	}
	Ar = new double[(nx-1)*(ny-1)];
	Ai = new double[(nx-1)*(ny-1)];
	Ifull = new double[(nx-1)*(ny-1)];
	Tfull = new double[(nx-1)*(ny-1)];
	
	fftw_complex* in  = new fftw_complex[(nx-1)*(ny-1)];
	fftw_complex* out = new fftw_complex[(nx-1)*(ny-1)];
	fftw_plan ff = fftw_plan_dft_2d(nx-1, ny-1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan fb = fftw_plan_dft_2d(nx-1, ny-1, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
	
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
}

// main loop
for(int itr=0; itr<100; itr++){
	T_compute(T, Tnew, I, nx, ny, local_nz, beta, alpha, rho, cp, u, dx, dy, dz, dt);
	bcT_compute(T, Tnew, nx, ny, local_nz, rank, size);
	if(rank>0) MPI_Sendrecv(&T[0], nx*ny, MPI_DOUBLE, rank-1, 0, &ghostL[0], nx*ny, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	if(rank<size-1) MPI_Sendrecv(&T[(local_nz-1)*nx*ny], nx*ny, MPI_DOUBLE, rank+1, 1, &ghostR[0], nx*ny, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	interfaceT_compute(T, Tnew, I, ghostL, ghostR, nx, ny, local_nz, beta, alpha,  rho, cp, u, dx, dy, dz, dt, rank, size);
	for(int i=0; i<nx*ny*local_nz; i++) T[i] = Tnew[i];

	if(rank!=0) MPI_Send(&T[0], nx*ny*local_nz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	if(rank==0){
		for(k=0; k<local_nz; k++){
			for(j=0; j<ny; j++){
				for(i=0; i<nx; i++){
					Tfull[k*nx*ny + j*nx + i] = T[k*nx*ny + j*nx + i];
					}
				}
			}
		for(i=1; i<size; i++) MPI_Recv(&Tfull[(i*local_nz)*nx*ny], nx*ny*local_nz, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}	
	MPI_Barrier(MPI_COMM_WORLD);
	// This step should not procees till above lines are executed on all PEs	
	if(rank==0){
		A_compute(Ar, Ai, Tfull, kx, ky, in, out, ff, fb, nx, ny, nz, beta, wn, dn_dT, dz, n0); 
		for(i=0; i<nx*ny*nz; i++) Ifull[i] = Ar[i]*Ar[i] + Ai[i]*Ai[i];
		for(k=0; k<local_nz; k++){
			for(j=0; j<ny; j++){
				for(i=0; i<nx; i++){
					I[k*nx*ny + j*nx + i] = Ifull[k*nx*ny + j*nx + i];
					}
				}
			}
		for(i=1; i<size; i++) MPI_Send(&Ifull[(i*local_nz)*nx*ny], nx*ny*local_nz, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank!=0) MPI_Recv(&I[0], nx*ny*local_nz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

		
 
if(rank==0){
	data_write("I.txt", Ifull, nx, ny, nz);
	data_write("T.txt", Tfull, nx, ny, nz);
}

MPI_Finalize();

return 0;
}





