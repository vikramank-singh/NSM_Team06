#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include<fstream>
#include<chrono>

const double pi = 3.14159265358;

struct dummy_variables {
	double *u_star = nullptr;
	double *v_star = nullptr;
	double *w_star = nullptr;
	double *p_prime = nullptr;
	double *div = nullptr;
	double *div_dt = nullptr;
	double *T_new = nullptr;
};

// ********************************************************* \\

void initialize_wavnumbers(double *kx, double *ky, double *ksq, int nx, int ny, int nz, double lx, double ly, double lz, double dx, double dy){
	for(int j=0; j<ny-1; j++){ 
		double KY = j*(j <= (ny-1)/2) + (j-(ny-1))*(j > (ny-1)/2);
		for(int i=0; i<nx-1; i++){	
			double KX = i*(i <= (nx-1)/2) + (i-(nx-1))*(i > (nx-1)/2);
			kx[j*(nx-1) + i] = (2*pi/lx)*KX;
			ky[j*(nx-1) + i] = (2*pi/ly)*KY;
		}}
	for (int k=0; k<nz; k++){ 
		for (int j=0; j<ny; j++){
			for (int i=0; i<nx; i++){ 
				ksq[k*ny*nx + j*nx + i] = pow(pi,2)*(pow(i/lx,2) + pow(j/ly,2) + pow(k/lz,2));
				}}}
}

// ************************************************************ \\

void initialization(double *Ar, double *Ai, double *u, double *v, double *w, double *I, double *T, double *p, double beta, int nx, int ny, int nz, double lx, double ly, double dx, double dy, double dz){
	for(int k=0; k<nz; k++){
		double z = k*dz;
		for(int i=0; i<nx; i++){ 
			double x = i*dx;
			for(int j=0; j<ny; j++){ 
				double y = j*dy;
				Ar[k*nx*ny + j*nx + i] = exp(-2*((x-lx/2)*(x-lx/2) + (y-ly/2)*(y-ly/2)));
				Ai[k*nx*ny + j*nx + i] = 0.0;
				I[k*nx*ny + j*nx +i] = exp(-beta*z)*pow(Ar[j*nx + i],2);
				T[k*nx*ny + j*nx + i] = 0.0;
				u[k*nx*ny + j*nx + i] = 0.0;
				v[k*nx*ny + j*nx + i] = 0.0;
				w[k*nx*ny + j*nx + i] = 0.0;
				p[k*nx*ny + j*nx + i] = 0.0;
			}
		}
	}
}

// ************************************************************ \\

double advection(double a, double b, double c, double *f, int nx, int ny, int idx, double dx,  double dy,  double dz){
	return (std::max(a,0.0)*(f[idx]-f[idx-1]) + std::min(a,0.0)*(f[idx+1]-f[idx]))/dx + (std::max(b,0.0)*(f[idx]-f[idx-nx]) + std::min(b,0.0)*(f[idx+nx]-f[idx]))/dy + (std::max(c,0.0)*(f[idx]-f[idx-(nx*ny)]) + std::min(c,0.0)*(f[idx+(nx*ny)]-f[idx]))/dz;
	}
	
// ************************************************************ \\

double diffusion(double *f, int nx, int ny, int idx, double dx,  double dy,  double dz){
	return (f[idx+1]+f[idx-1]-2*f[idx])/pow(dx,2) + (f[idx+nx]+f[idx-nx]-2*f[idx])/pow(dy,2) + (f[idx+(nx*ny)]+f[idx-(nx*ny)]-2*f[idx])/pow(dz,2);
}

// ************************************************************ \\
	
void divergence(double *f, double *u, double *v, double *w, int nx, int ny, int nz, double dx,  double dy,  double dz){
	for (int k=1; k<nz-1; k++){ for (int j=1; j<ny-1; j++){ for (int i=1; i<nx-1; i++){ int idx = k*ny*nx + j*nx + i;
		f[idx] = 0.5*(u[idx+1]-u[idx-1])/dx + 0.5*(v[idx+nx]-v[idx-nx])/dy + 0.5*(w[idx+(nx*ny)]-w[idx-(nx*ny)])/dz;
		}}}
}

// ************************************************************ \\

void Neumann_bc(double *f, int nx, int ny, int nz){
	for (int j=0; j<ny; j++){ // front & back
		for (int i=0; i<nx; i++){
			f[j*nx + i] = f[nx*ny + j*nx + i];
			f[(nz-1)*nx*ny + j*nx + i] = f[(nz-2)*nx*ny + j*nx + i];
		}}
	for (int k=0; k<nz; k++){ //  bottom & top
		for (int i=0; i<nx; i++){	
			f[k*nx*ny + i] = f[k*nx*ny + nx + i];
			f[k*nx*ny + (ny-1)*nx + i] = f[k*nx*ny + (ny-2)*nx + i];
		}}
	for (int k=0; k<nz; k++){ // left & right
		for (int j=0; j<ny; j++){
			f[k*nx*ny + j*nx] = f[k*nx*ny + j*nx + 1];
			f[k*nx*ny + j*nx + nx-1] = f[k*nx*ny + j*nx + nx-2];
		}}
}
	
// ************************************************************ \\
	
void poisson_solver_dct(double *u, double *f, double *ksq, double *in, double *out, fftw_plan df, fftw_plan di, int nx, int ny, int nz) { // ∇² f = u
	for (int i=0; i<nx*ny*nz; i++) in[i] = u[i];
	fftw_execute(df);
	for (int i=1; i<nx*ny*nz; i++) in[i] = - out[i] / ksq[i];
	in[0] = 0.0;
	fftw_execute(di);
	for (int i=0; i<nx*ny*nz; i++) f[i] = out[i] / (8*nx*ny*nz);
}

// ************************************************************ \\

void Dirichlet_bc(double *f, int nx, int ny, int nz){
	for (int j=0; j<ny; j++){ // front & back
		for (int i=0; i<nx; i++){
			f[j*nx + i] = 0.0;
			f[(nz-1)*nx*ny + j*nx + i] = 0.0;
		}}
	for (int k=0; k<nz; k++){ //  bottom & top
		for (int i=0; i<nx; i++){	
			f[k*nx*ny + i] = 0.0;
			f[k*nx*ny + (ny-1)*nx + i] = 0.0;
		}}
	for (int k=0; k<nz; k++){ // left & right
		for (int j=0; j<ny; j++){
			f[k*nx*ny + j*nx] = 0.0;
			f[k*nx*ny + j*nx + nx-1] = 0.0;
		}}
}


// ********************************************************* \\

void prediction(double *u_star, double *v_star, double *w_star, double *u, double *v, double *w, double *p, double *T, double Pr, int nx, int ny,  int nz,  double dx,  double dy,  double dz, double dt){
	int i, j, k, idx, itr;
	double adv, diff, source;
	for (k=1; k<nz-1; k++){
		for (j=1; j<ny-1; j++){
			for (i=1; i<nx-1; i++){
				idx = k*ny*nx + j*nx + i;
				adv = advection(u[idx], v[idx], w[idx], u, nx, ny, idx, dx, dy, dz);
				diff = Pr * diffusion(u, nx, ny, idx, dx, dy, dz) ;
				source = -(p[idx+1]-p[idx-1]) / (2*dx);
				u_star[idx] = u[idx] + dt * (-adv + diff + source); 
				
				adv = advection(u[idx], v[idx], w[idx], v, nx, ny, idx, dx, dy, dz);
				diff = Pr * diffusion(v, nx, ny, idx, dx, dy, dz) ;
				source = -(p[idx+nx]-p[idx-nx]) / (2*dy) + Pr*T[idx];
				v_star[idx] = v[idx] + dt * (-adv + diff + source); 

				adv = advection(u[idx], v[idx], w[idx], w, nx, ny, idx, dx, dy, dz);
				diff = Pr * diffusion(w, nx, ny, idx, dx, dy, dz) ;
				source = -(p[idx+(nx*ny)]-p[idx-(nx*ny)]) / (2*dz);
				w_star[idx] = w[idx] + dt * (-adv + diff + source); 	
			}}}	
}

// ********************************************************* \\

void correction(double *u_star, double *v_star, double *w_star, double *u, double *v, double *w, double *p_prime, int nx, int ny,  int nz,  double dx,  double dy,  double dz, double dt){
	for (int k=1; k<nz-1; k++){
		for (int j=1; j<ny-1; j++){
			for (int i=1; i<nx-1; i++){
				int idx = k*ny*nx + j*nx + i;
				u[idx] = u_star[idx] - dt * (p_prime[idx+1]-p_prime[idx-1]) / (2*dx);
				v[idx] = v_star[idx] - dt * (p_prime[idx+nx]-p_prime[idx-nx]) / (2*dy);
				w[idx] = w_star[idx] - dt * (p_prime[idx+(nx*ny)]-p_prime[idx-(nx*ny)]) / (2*dz);
			}}}
}


// ********************************************************* \\

void T_compute(double *u, double *v, double *w, double *T, double *I, double Ra, int nx, int ny,  int nz,  double dx,  double dy,  double dz, double dt, dummy_variables& dv){
	int i, j, k, idx;
	double adv, diff, source;
	for (int k=1; k<nz-1; k++){
		for (int j=1; j<ny-1; j++){
			for (int i=1; i<nx-1; i++){	
				idx = k*ny*nx + j*nx + i;
				adv = advection(u[idx], v[idx], w[idx], T, nx, ny, idx, dx, dy, dz);
				diff = diffusion(T, nx, ny, idx, dx, dy, dz);
				source = Ra * I[idx];
				dv.T_new[idx] = T[idx] + dt * (-adv + diff + source); 
			}}}
	for (int i=0; i<nx*ny*nz; i++) T[i] = dv.T_new[i]; 
}

// ********************************************************* \\

void ns_compute(double *u, double *v, double *w, double *p, double *T, double *I, double *ksq, double *in, double *out, fftw_plan df, fftw_plan di, double Pr, double Ra, int nx, int ny,  int nz, double Lx, double Ly, double Lz, double dx,  double dy,  double dz, double dt, dummy_variables& dv){	
	for (int i=0; i<nx*ny*nz; i++) dv.p_prime[i] = 0.0; 
	
	prediction(dv.u_star, dv.v_star, dv.w_star, u, v, w, p, T, Pr, nx, ny, nz, dx, dy, dz, dt);
	Dirichlet_bc(dv.u_star, nx, ny, nz); Dirichlet_bc(dv.v_star, nx, ny, nz); Dirichlet_bc(dv.w_star, nx, ny, nz);
	 
	divergence(dv.div, dv.u_star, dv.v_star, dv.w_star, nx, ny, nz, dx, dy, dz);
	for (int i=0; i<nx*ny*nz; i++) {dv.div_dt[i] = dv.div[i] / dt; }
	
	//auto start = std::chrono::high_resolution_clock::now();
	poisson_solver_dct(dv.div_dt, dv.p_prime, ksq, in, out, df, di, nx, ny, nz);
	//auto end = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
	//std::cout<<" Poission computee time = "<<duration.count()/1000.0<< std::endl;
	correction(dv.u_star, dv.v_star, dv.w_star, u, v, w, dv.p_prime, nx, ny, nz, dx, dy, dz, dt);	
	Dirichlet_bc(u, nx, ny, nz);
	Dirichlet_bc(v, nx, ny, nz);
	Dirichlet_bc(w, nx, ny, nz);	
	for (int i=0; i<nx*ny*nz; i++) {p[i] += dv.p_prime[i]; }
	Neumann_bc(p, nx, ny, nz);
	
	T_compute(u, v, w, T, I, Ra, nx, ny, nz, dx, dy, dz, dt, dv);
	Dirichlet_bc(T, nx, ny, nz);						
}

// ********************************************************* \\

void A_compute_ss(double *Ar, double *Ai, double *T, double *kx, double *ky, fftw_complex *in, fftw_complex *out, fftw_plan ff, fftw_plan fb, int nx, int ny, int nz, double beta, double wn, double C, double dz){
	int i, j, k; double ca;
	for(k=1; k<nz; k++){
		for(j=0; j<ny-1; j++){
			for(i=0; i<nx-1; i++){
			ca = C*wn*dz*T[(k-1)*nx*ny + j*nx + i]/2.0;
			in[j*(nx-1) + i][0] = exp(-beta*dz/2)*(Ar[(k-1)*nx*ny + j*nx + i]*cos(ca) - Ai[(k-1)*nx*ny + j*nx + i]*sin(ca));
			in[j*(nx-1) + i][1] = exp(-beta*dz/2)*(Ar[(k-1)*nx*ny + j*nx + i]*sin(ca) + Ai[(k-1)*nx*ny + j*nx + i]*cos(ca));
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

// ********************************************************* \\

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



#endif
