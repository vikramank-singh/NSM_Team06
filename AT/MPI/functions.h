#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <fstream>
#include <string>


// *************************************************************************** //
void print(double *X, int nx, int ny){
	for(int j=0; j<ny; j++){
		for(int i=0; i<nx; i++){
		std::cout<<X[j*nx + i]<<" ";
		}
		std::cout<<std::endl;
	}
}

// *************************************************************************** //

void T_compute(double *T, double *Tnew, double *I, int nx, int ny, int nz, double beta, double alpha, double rho, double cp, double u, double dx, double dy, double dz, double dt){
	int i, j, k;  double conv, diff, source;
	for(k=1; k<nz-1; k++){
		for(j=1; j<ny-1; j++){
			for(i=1; i<nx-1; i++){
				conv = u*(T[k*nx*ny + j*nx +i] - T[k*nx*ny + j*nx +i-1])/dx;
				diff = alpha*((T[k*nx*ny + j*nx +i+1]+T[k*nx*ny + j*nx +i-1]-2*T[k*nx*ny + j*nx +i])/pow(dx,2) + (T[k*nx*ny + (j+1)*nx +i]+T[k*nx*ny + (j-1)*nx +i]-2*T[k*nx*ny + j*nx +i])/pow(dy,2) + (T[(k+1)*nx*ny + j*nx +i]+T[(k-1)*nx*ny + j*nx +i]-2*T[k*nx*ny + j*nx +i])/pow(dz,2));
				source = (beta/(rho*cp))*I[k*nx*ny + j*nx +i];
				Tnew[k*nx*ny + j*nx +i] = T[k*nx*ny + j*nx +i] + dt*(diff + source - conv);
			}
		}
	}
}

// *************************************************************************** //

void bcT_compute(double *T, double *Tnew, int nx, int ny, int nz, int rank, int size){
	int i, j, k;
	for(k=0; k<nz; k++){
		for(j=0; j<ny; j++){
			Tnew[k*nx*ny + j*nx + nx-1] = T[k*nx*ny + j*nx + nx-2];
		}
	}
	for(k=0; k<nz; k++){
		for(i=0; i<nx; i++){
			Tnew[k*nx*ny + 0*nx + i] = T[k*nx*ny + 1*nx + i];
			Tnew[k*nx*ny + (ny-1)*nx + i] = T[k*nx*ny + (ny-2)*nx + i];
		}
	}
	if(rank==0){
		for(j=0; j<ny; j++){
			for(i=0; i<ny; i++)
				Tnew[0*nx*ny + j*nx + i] = T[1*nx*ny + j*nx + i];
		}
	}
	if(rank==size-1){
		for(j=0; j<ny; j++){
			for(i=0; i<ny; i++)
				Tnew[(nz-1)*nx*ny + j*nx + i] = T[(nz-2)*nx*ny + j*nx + i];
		}
	}
}

// ************************************************************ //

void interfaceT_compute(double *T, double *Tnew, double *I, double *ghostL, double *ghostR, int nx, int ny, int local_nz, double beta, double alpha, double rho, double cp, double u, double dx, double dy, double dz, double dt, int rank, int size){
	int i, j, k;  double conv, diff, source;
	if(rank>0){
		for(j=1; j<ny-1; j++){
			for(i=1; i<nx-1; i++){
				conv = u*(T[0*nx*ny + j*nx +i] - T[0*nx*ny + j*nx +i-1])/dx;
				diff = alpha*((T[0*nx*ny + j*nx +i+1]+T[0*nx*ny + j*nx +i-1]-2*T[0*nx*ny + j*nx +i])/pow(dx,2) + (T[0*nx*ny + (j+1)*nx +i]+T[0*nx*ny + (j-1)*nx +i]-2*T[0*nx*ny + j*nx +i])/pow(dy,2) + (T[1*nx*ny + j*nx +i]+ghostL[j*nx +i]-2*T[0*nx*ny + j*nx +i])/pow(dz,2));
				source = (beta/(rho*cp))*I[0*nx*ny + j*nx +i];
				Tnew[0*nx*ny + j*nx +i] = T[0*nx*ny + j*nx +i] + dt*(diff + source -conv);
				}
			}
	}
	if(rank<size-1){
		for(j=1; j<ny-1; j++){
			for(i=1; i<nx-1; i++){
				conv = u*(T[(local_nz-1)*nx*ny + j*nx +i] - T[(local_nz-1)*nx*ny + j*nx +i-1])/dx;
				diff = alpha*((T[(local_nz-1)*nx*ny + j*nx +i+1]+T[(local_nz-1)*nx*ny + j*nx +i-1]-2*T[(local_nz-1)*nx*ny + j*nx +i])/pow(dx,2) + (T[(local_nz-1)*nx*ny + (j+1)*nx +i]+T[(local_nz-1)*nx*ny + (j-1)*nx +i]-2*T[(local_nz-1)*nx*ny + j*nx +i])/pow(dy,2) + (ghostR[j*nx +i]+T[(local_nz-2)*nx*ny + j*nx +i]-2*T[(local_nz-1)*nx*ny + j*nx +i])/pow(dz,2));
				source = (beta/(rho*cp))*I[(local_nz-1)*nx*ny + j*nx +i];
				Tnew[(local_nz-1)*nx*ny + j*nx +i] = T[(local_nz-1)*nx*ny + j*nx +i] + dt*(diff + source -conv);
			}
		}
	}
}



// *************************************************************************** //

void A_compute(double *Ar, double *Ai, double *T, double *kx, double *ky, fftw_complex *in, fftw_complex *out, fftw_plan ff, fftw_plan fb, int nx, int ny, int nz, double beta, double wn, double dn_dT, double dz, double n0){
	int i, j, k; double ca;
	for(k=1; k<nz; k++){
		for(j=0; j<ny-1; j++){
			for(i=0; i<nx-1; i++){
			ca = wn*dn_dT*dz*(T[(k-1)*nx*ny + j*nx + i]- 300)/n0;
			in[j*(nx-1) + i][0] = exp(-beta*dz/2)*(Ar[(k-1)*nx*ny + j*nx + i]*cos(ca) + Ai[(k-1)*nx*ny + j*nx + i]*sin(ca));
			in[j*(nx-1) + i][1] = exp(-beta*dz/2)*(-Ar[(k-1)*nx*ny + j*nx + i]*sin(ca) + Ai[(k-1)*nx*ny + j*nx + i]*cos(ca));
			}
		}
		fftw_execute(ff);
		for(j=0; j<ny-1; j++){
			for(i=0; i<nx-1; i++){
				ca = (pow(kx[j*(nx-1) + i],2) + pow(ky[j*(nx-1) + i],2))*dz/(2*wn);
				in[j*(nx-1) + i][0] = out[j*(nx-1) + i][0]*cos(ca) - out[j*(nx-1) + i][1]*sin(ca);
				in[j*(nx-1) + i][1] = out[j*(nx-1) + i][0]*sin(ca) + out[j*(nx-1) + i][1]*cos(ca);
			}
		}
		fftw_execute(fb);
		for(i=0; i<(nx-1)*(ny-1); i++){
			out[i][0] /= (nx-1)*(ny-1);
			out[i][1] /= (nx-1)*(ny-1);
		}
		for(j=0; j<ny-1; j++){
			for(i=0; i<nx-1; i++){
				Ar[k*nx*ny + j*nx + i] = out[j*(nx-1) + i][0];
				Ai[k*nx*ny + j*nx + i] = out[j*(nx-1) + i][1];
			}
		}
		for(i=0; i<nx; i++){
			Ar[k*nx*ny + (ny-1)*nx + i] = Ar[k*nx*ny + 0*nx + i];
			Ai[k*nx*ny + (ny-1)*nx + i] = Ai[k*nx*ny + 0*nx + i];
		}
		for(j=0; j<ny; j++){
			Ar[k*nx*ny + j*nx + nx-1] = Ar[k*nx*ny + j*nx + 0];
			Ai[k*nx*ny + j*nx + nx-1] = Ai[k*nx*ny + j*nx + 0];
		}
		
	}
}

// *************************************************************************** //

double L_inf_norm(double *T, double *T_new, int N){
	double error = 0.0;
	for(int i=1; i<N; i++) 
	error = std::max(error,T[i]-T_new[i]);
	return error;
}

// *************************************************************************** //

void data_write(const std::string& filename, double *X, int nx, int ny, int nz){
	std::ofstream File(filename);
	for(int k=0; k<nz; k++){
		for(int j=0; j<ny; j++){
			for(int i=0; i<nx; i++){
				File<<X[k*nx*ny +j*nx + i]<<std::endl;
			}
		}
	}
	File.close();
}
	
// *************************************************************************** //
	
#endif
