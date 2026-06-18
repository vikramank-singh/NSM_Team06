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
	int i, j, k;  double conv, diff, source, ca;
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

void bcT_compute(double *T, double *Tnew, int nx, int ny, int nz){
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
	for(j=0; j<ny; j++){
		for(i=0; i<ny; i++){
			Tnew[0*nx*ny + j*nx + i] = T[1*nx*ny + j*nx + i];
			Tnew[(nz-1)*nx*ny + j*nx + i] = T[(nz-2)*nx*ny + j*nx + i];
		}
	}
}

// *************************************************************************** //

void T_compute_implicit(double *T, double *I, int nx, int ny, int nz, double beta, double alpha, double rho, double cp, double u, double dx, double dy, double dz, double dt){
        int i, j, k, itr;  double conv, diff, source;
        double *Told = new double [nx*ny*nz];
        double *Tnew = new double [nx*ny*nz];
        for (int i=0; i<nx*ny*nz; i++) Told[i] = T[i];
        for (itr=0; itr<100; itr++){
                for(k=1; k<nz-1; k++){
                        for(j=1; j<ny-1; j++){
                                for(i=1; i<nx-1; i++){
                                        diff = alpha*((T[k*nx*ny + j*nx +i+1]+T[k*nx*ny + j*nx +i-1])/pow(dx,2) + (T[k*nx*ny + (j+1)*nx +i]+T[k*nx*ny + (j-1)*nx +i])/pow(dy,2) + (T[(k+1)*nx*ny + j*nx +i]+T[(k-1)*nx*ny + j*nx +i])/pow(dz,2));
                                        source = (beta/(rho*cp))*I[k*nx*ny + j*nx +i] + Told[k*nx*ny + j*nx + i]/dt;
                                        conv = -u * T[k*nx*ny + j*nx + i-1]/dx;
                                        Tnew[k*nx*ny + j*nx +i] = (diff + source - conv) / ( 1/dt + u/dx + 2*alpha*(1/pow(dx,2) + 1/pow(dy,2) + 1/pow(dz,2)) );
                                }
                        }
                }
                for (int i=0; i<nx*ny*nz; i++) T[i] = Tnew[i];
                bcT_compute(T, Tnew, nx, ny, nz);
        }
}

// *************************************************************************** //

void A_compute_ss(double *Ar, double *Ai, double *T, double *kx, double *ky, fftw_complex *in, fftw_complex *out, fftw_plan ff, fftw_plan fb, int nx, int ny, int nz, double beta, double wn, double dn_dT, double dz, double n0){
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

void A_compute_im(double *Ar, double *Ai, double *T, int nx, int ny, int nz, double beta, double wn, double dn_dT, double dz, double n0, double dx, double dy, double T0){
	int i, j, k, l, itr, id;
	double dr, di, D, rr;
	double idx2 = 1.0 / (dx*dx);
	double c1 = wn*wn * 2.0 * dn_dT / n0;
	double c2 = (2*wn/dz) + (wn*beta);
	double c3 = 2*wn/dz;
	double *Arnew = new double [nx*ny];
	double *Ainew = new double [nx*ny];    
	for (i=0; i<nx*ny; i++) { Arnew[i] = 0.0; Ainew[i] = 0.0; }
	for(k=1; k<nz; k++){
		for (itr=0; itr<100; itr++){
			for(j=1; j<ny-1; j++){
				for(i=1; i<nx-1; i++){
					id = k*nx*ny + j*nx + i;
					dr = (Ar[id+1]+Ar[id-1] + Ar[id+nx]+Ar[id-nx]) * idx2;
					di = (Ai[id+1]+Ai[id-1] + Ai[id+nx]+Ai[id-nx]) * idx2;
					rr =  c1 * (T[id]-T0);
					D = pow(4.0*idx2-rr,2) + pow(c3 + wn*beta,2);
					Arnew[j*nx + i] = ( (4.0*idx2-rr) * (dr - c3*Ai[id-nx*ny]) + c2 * (di + c3*Ar[id-nx*ny]) ) / D;
					Ainew[j*nx + i] = ( (4.0*idx2-rr) * (di + c3*Ar[id-nx*ny]) - c2 * (dr - c3*Ai[id-nx*ny]) ) / D;					
				}	
			}
		for (i=0; i<nx*ny; i++) { Ar[k*nx*ny + i] = Arnew[i]; Ai[k*nx*ny + i] = Ainew[i]; }
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
