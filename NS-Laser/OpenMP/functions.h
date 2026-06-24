#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include<fstream>
#include<cstdlib>
#include<new>
#include<omp.h>

const double pi = 3.14159265358;

struct dummy_variables {
	double *u_star = nullptr;
	double *v_star = nullptr;
	double *w_star = nullptr;
	double *p_prime = nullptr;
	double *div = nullptr;
	double *div_dt = nullptr;
	double *T_new = nullptr;
	double *dct_buf = nullptr;
	double *lam_x = nullptr;
	double *lam_y = nullptr;
	double *lam_z = nullptr;
	fftw_plan dct_fwd = nullptr;
	fftw_plan dct_bwd = nullptr;
};

// ********************************************************* \\
// 64B-aligned allocation so the vectorizer can assume aligned loads/stores
// on the nx*ny*nz field arrays instead of emitting peel/remainder loops.

inline double* alloc_field(std::size_t n){
	void *p = nullptr;
	if (posix_memalign(&p, 64, n*sizeof(double)) != 0) throw std::bad_alloc();
	return static_cast<double*>(p);
}

// ********************************************************* \\

void initialize_wavnumbers(double *__restrict kx, double *__restrict ky, int nx, int ny, double lx, double ly, double dx, double dy){
	const double two_pi_over_lx = (2*pi)/lx;
	const double two_pi_over_ly = (2*pi)/ly;
	const int ny_half = (ny-1)/2;
	const int nx_half = (nx-1)/2;
	for(int j=0; j<ny-1; j++){
		double KY = j*(j <= ny_half) + (j-(ny-1))*(j > ny_half);
		for(int i=0; i<nx-1; i++){
			double KX = i*(i <= nx_half) + (i-(nx-1))*(i > nx_half);
			kx[j*(nx-1) + i] = two_pi_over_lx*KX;
			ky[j*(nx-1) + i] = two_pi_over_ly*KY;
		}
	}
}

// ************************************************************ \\

void initialization(double *__restrict Ar, double *__restrict Ai, double *__restrict u, double *__restrict v, double *__restrict w, double *__restrict I, double *__restrict T, double *__restrict p, double beta, int nx, int ny, int nz, double lx, double ly, double dx, double dy, double dz){
	for(int k=0; k<nz; k++){
		double z = k*dz;
		double atten = exp(-beta*z);
		for(int i=0; i<nx; i++){
			double x = i*dx;
			for(int j=0; j<ny; j++){
				double y = j*dy;
				double dxr = x-lx/2, dyr = y-ly/2;
				double ar = exp(-2*(dxr*dxr + dyr*dyr));
				Ar[k*nx*ny + j*nx + i] = ar;
				Ai[k*nx*ny + j*nx + i] = 0.0;
				I[k*nx*ny + j*nx +i] = atten*ar*ar;
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
// advection/diffusion are marked inline with restrict-qualified pointers so
// the compiler can fuse them into the caller's i-loop body instead of
// emitting an out-of-line scalar call per grid point (which was blocking
// vectorization of prediction()/T_compute()).

inline double advection(double a, double b, double c, const double *__restrict f, int nx, int ny, int idx, double inv_dx, double inv_dy, double inv_dz){
	return (std::max(a,0.0)*(f[idx]-f[idx-1]) + std::min(a,0.0)*(f[idx+1]-f[idx]))*inv_dx
	     + (std::max(b,0.0)*(f[idx]-f[idx-nx]) + std::min(b,0.0)*(f[idx+nx]-f[idx]))*inv_dy
	     + (std::max(c,0.0)*(f[idx]-f[idx-(nx*ny)]) + std::min(c,0.0)*(f[idx+(nx*ny)]-f[idx]))*inv_dz;
}

// ************************************************************ \\

inline double diffusion(const double *__restrict f, int nx, int ny, int idx, double inv_dx2, double inv_dy2, double inv_dz2){
	return (f[idx+1]+f[idx-1]-2*f[idx])*inv_dx2 + (f[idx+nx]+f[idx-nx]-2*f[idx])*inv_dy2 + (f[idx+(nx*ny)]+f[idx-(nx*ny)]-2*f[idx])*inv_dz2;
}

// ************************************************************ \\

void divergence(double *__restrict f, const double *__restrict u, const double *__restrict v, const double *__restrict w, int nx, int ny, int nz, double dx,  double dy,  double dz){
	const double inv_2dx = 0.5/dx, inv_2dy = 0.5/dy, inv_2dz = 0.5/dz;
	#pragma omp for collapse(2)
	for (int k=1; k<nz-1; k++){ for (int j=1; j<ny-1; j++){ for (int i=1; i<nx-1; i++){ int idx = k*ny*nx + j*nx + i;
		f[idx] = (u[idx+1]-u[idx-1])*inv_2dx + (v[idx+nx]-v[idx-nx])*inv_2dy + (w[idx+(nx*ny)]-w[idx-(nx*ny)])*inv_2dz;
		}}}
}

// ************************************************************ \\

void Neumann_bc(double *__restrict f, int nx, int ny, int nz){
	// collapse(2) on every face-pair: the bottom/top and left/right loops are
	// driven by nz (e.g. 20 for the production grid), smaller than the
	// thread count -- without collapsing in the nx/ny dimension, most
	// threads get 0 work items. Collapsing gives every face-pair nx*ny or
	// ny*nz work items regardless of nz.
	#pragma omp for collapse(2)
	for (int j=0; j<ny; j++){ // front & back
		for (int i=0; i<nx; i++){
			f[j*nx + i] = f[nx*ny + j*nx + i];
			f[(nz-1)*nx*ny + j*nx + i] = f[(nz-2)*nx*ny + j*nx + i];
		}}
	#pragma omp for collapse(2)
	for (int k=0; k<nz; k++){ //  bottom & top
		for (int i=0; i<nx; i++){
			f[k*nx*ny + i] = f[k*nx*ny + nx + i];
			f[k*nx*ny + (ny-1)*nx + i] = f[k*nx*ny + (ny-2)*nx + i];
		}}
	#pragma omp for collapse(2)
	for (int k=0; k<nz; k++){ // left & right
		for (int j=0; j<ny; j++){
			f[k*nx*ny + j*nx] = f[k*nx*ny + j*nx + 1];
			f[k*nx*ny + j*nx + nx-1] = f[k*nx*ny + j*nx + nx-2];
		}}
}

// ************************************************************ \\
// Eigenvalues of the 3-point Laplacian under half-sample symmetric
// (DCT-II/III, Neumann) boundary conditions, length N = interior point
// count along that axis. Used by poisson_solver_Neumann_fft below.

void dct_eigs(double *__restrict lam, int N, double h){
	const double scale = 4.0 / (h*h);
	for (int m = 0; m < N; m++){
		double s = std::sin(pi*m/(2.0*N));
		lam[m] = -scale * s*s;
	}
}

// ************************************************************ \\
// Direct Poisson solve via 3D DCT (replaces Gauss-Seidel relaxation): the
// boundary convention in Neumann_bc (boundary plane = nearest interior
// plane) matches FFTW's half-sample symmetric REDFT10/REDFT01, so solving
// on the interior subgrid and refilling boundaries via Neumann_bc reproduces
// the same discrete BC without an iterative sweep.

void poisson_solver_Neumann_fft(double *__restrict u, const double *__restrict f, int nx, int ny, int nz, double *__restrict buf, const double *__restrict lam_x, const double *__restrict lam_y, const double *__restrict lam_z, fftw_plan fwd, fftw_plan bwd){
	const int Nx = nx-2, Ny = ny-2, Nz = nz-2;
	#pragma omp for collapse(2)
	for (int k=0; k<Nz; k++) for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++)
		buf[(k*Ny+j)*Nx+i] = f[(k+1)*nx*ny + (j+1)*nx + (i+1)];

	// fftw_execute is not safe to call concurrently from multiple threads on
	// the same plan/buffer -- one thread runs it, the rest wait (implicit
	// barrier), since the whole SPMD-style ns_compute body is inside a
	// single #pragma omp parallel region.
	#pragma omp single
	{
		fftw_execute(fwd);
		buf[0] = 0.0; // null-space / mean mode: pressure is defined up to a constant
	}
	#pragma omp for collapse(2)
	for (int k=0; k<Nz; k++) for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++){
		if (k==0 && j==0 && i==0) continue;
		buf[(k*Ny+j)*Nx+i] /= (lam_x[i]+lam_y[j]+lam_z[k]);
	}

	#pragma omp single
	fftw_execute(bwd);
	const double inv_norm = 1.0/(8.0*Nx*Ny*Nz); // FFTW r2r unnormalized-transform scale

	#pragma omp for collapse(2)
	for (int k=0; k<Nz; k++) for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++)
		u[(k+1)*nx*ny + (j+1)*nx + (i+1)] = buf[(k*Ny+j)*Nx+i] * inv_norm;

	Neumann_bc(u, nx, ny, nz);
}

// ************************************************************ \\

void Dirichlet_bc(double *__restrict f, int nx, int ny, int nz){
	#pragma omp for collapse(2)
	for (int j=0; j<ny; j++){ // front & back
		for (int i=0; i<nx; i++){
			f[j*nx + i] = 0.0;
			f[(nz-1)*nx*ny + j*nx + i] = 0.0;
		}}
	#pragma omp for collapse(2)
	for (int k=0; k<nz; k++){ //  bottom & top
		for (int i=0; i<nx; i++){
			f[k*nx*ny + i] = 0.0;
			f[k*nx*ny + (ny-1)*nx + i] = 0.0;
		}}
	#pragma omp for collapse(2)
	for (int k=0; k<nz; k++){ // left & right
		for (int j=0; j<ny; j++){
			f[k*nx*ny + j*nx] = 0.0;
			f[k*nx*ny + j*nx + nx-1] = 0.0;
		}}
}

// ************************************************************ \\
// Same as Dirichlet_bc but applied to three independent arrays in one set of
// work-sharing loops instead of three separate Dirichlet_bc() calls. Each
// #pragma omp for carries an implicit barrier; VTune showed barrier/fork-join
// overhead (libgomp), not compute, dominating runtime once kernels were
// merged into a single thread team per ns_compute() call -- fusing the
// repeated per-array BC calls cuts the remaining barrier count further.

void Dirichlet_bc3(double *__restrict a, double *__restrict b, double *__restrict c, int nx, int ny, int nz){
	#pragma omp for collapse(2)
	for (int j=0; j<ny; j++){ // front & back
		for (int i=0; i<nx; i++){
			a[j*nx + i] = 0.0; b[j*nx + i] = 0.0; c[j*nx + i] = 0.0;
			a[(nz-1)*nx*ny + j*nx + i] = 0.0; b[(nz-1)*nx*ny + j*nx + i] = 0.0; c[(nz-1)*nx*ny + j*nx + i] = 0.0;
		}}
	#pragma omp for collapse(2)
	for (int k=0; k<nz; k++){ //  bottom & top
		for (int i=0; i<nx; i++){
			a[k*nx*ny + i] = 0.0; b[k*nx*ny + i] = 0.0; c[k*nx*ny + i] = 0.0;
			a[k*nx*ny + (ny-1)*nx + i] = 0.0; b[k*nx*ny + (ny-1)*nx + i] = 0.0; c[k*nx*ny + (ny-1)*nx + i] = 0.0;
		}}
	#pragma omp for collapse(2)
	for (int k=0; k<nz; k++){ // left & right
		for (int j=0; j<ny; j++){
			a[k*nx*ny + j*nx] = 0.0; b[k*nx*ny + j*nx] = 0.0; c[k*nx*ny + j*nx] = 0.0;
			a[k*nx*ny + j*nx + nx-1] = 0.0; b[k*nx*ny + j*nx + nx-1] = 0.0; c[k*nx*ny + j*nx + nx-1] = 0.0;
		}}
}


// ********************************************************* \\

void prediction(double *__restrict u_star, double *__restrict v_star, double *__restrict w_star, const double *__restrict u, const double *__restrict v, const double *__restrict w, const double *__restrict p, const double *__restrict T, double Pr, int nx, int ny,  int nz,  double dx,  double dy,  double dz, double dt){
	int i, j, k, idx;
	double adv, diff, source;
	const double inv_dx = 1.0/dx, inv_dy = 1.0/dy, inv_dz = 1.0/dz;
	const double inv_dx2 = inv_dx*inv_dx, inv_dy2 = inv_dy*inv_dy, inv_dz2 = inv_dz*inv_dz;
	const double inv_2dx = 0.5*inv_dx, inv_2dy = 0.5*inv_dy, inv_2dz = 0.5*inv_dz;
	#pragma omp for collapse(2) private(i, j, k, idx, adv, diff, source)
	for (k=1; k<nz-1; k++){
		for (j=1; j<ny-1; j++){
			for (i=1; i<nx-1; i++){
				idx = k*ny*nx + j*nx + i;
				adv = advection(u[idx], v[idx], w[idx], u, nx, ny, idx, inv_dx, inv_dy, inv_dz);
				diff = Pr * diffusion(u, nx, ny, idx, inv_dx2, inv_dy2, inv_dz2) ;
				source = -(p[idx+1]-p[idx-1]) * inv_2dx;
				u_star[idx] = u[idx] + dt * (-adv + diff + source);

				adv = advection(u[idx], v[idx], w[idx], v, nx, ny, idx, inv_dx, inv_dy, inv_dz);
				diff = Pr * diffusion(v, nx, ny, idx, inv_dx2, inv_dy2, inv_dz2) ;
				source = -(p[idx+nx]-p[idx-nx]) * inv_2dy + Pr*T[idx];
				v_star[idx] = v[idx] + dt * (-adv + diff + source);

				adv = advection(u[idx], v[idx], w[idx], w, nx, ny, idx, inv_dx, inv_dy, inv_dz);
				diff = Pr * diffusion(w, nx, ny, idx, inv_dx2, inv_dy2, inv_dz2) ;
				source = -(p[idx+(nx*ny)]-p[idx-(nx*ny)]) * inv_2dz;
				w_star[idx] = w[idx] + dt * (-adv + diff + source);
			}}}
}

// ********************************************************* \\

void correction(const double *__restrict u_star, const double *__restrict v_star, const double *__restrict w_star, double *__restrict u, double *__restrict v, double *__restrict w, const double *__restrict p_prime, int nx, int ny,  int nz,  double dx,  double dy,  double dz, double dt){
	const double c_x = dt*0.5/dx, c_y = dt*0.5/dy, c_z = dt*0.5/dz;
	#pragma omp for collapse(2)
	for (int k=1; k<nz-1; k++){
		for (int j=1; j<ny-1; j++){
			for (int i=1; i<nx-1; i++){
				int idx = k*ny*nx + j*nx + i;
				u[idx] = u_star[idx] - c_x * (p_prime[idx+1]-p_prime[idx-1]);
				v[idx] = v_star[idx] - c_y * (p_prime[idx+nx]-p_prime[idx-nx]);
				w[idx] = w_star[idx] - c_z * (p_prime[idx+(nx*ny)]-p_prime[idx-(nx*ny)]);
			}}}
}


// ********************************************************* \\

void T_compute(const double *__restrict u, const double *__restrict v, const double *__restrict w, double *__restrict T, const double *__restrict I, double Ra, int nx, int ny,  int nz,  double dx,  double dy,  double dz, double dt, dummy_variables& dv){
	int i, j, k, idx;
	double adv, diff, source;
	const double inv_dx = 1.0/dx, inv_dy = 1.0/dy, inv_dz = 1.0/dz;
	const double inv_dx2 = inv_dx*inv_dx, inv_dy2 = inv_dy*inv_dy, inv_dz2 = inv_dz*inv_dz;
	double *__restrict T_new = dv.T_new;
	#pragma omp for collapse(2) private(i, j, k, idx, adv, diff, source)
	for (k=1; k<nz-1; k++){
		for (j=1; j<ny-1; j++){
			for (i=1; i<nx-1; i++){
				idx = k*ny*nx + j*nx + i;
				adv = advection(u[idx], v[idx], w[idx], T, nx, ny, idx, inv_dx, inv_dy, inv_dz);
				diff = diffusion(T, nx, ny, idx, inv_dx2, inv_dy2, inv_dz2);
				source = Ra * I[idx];
				T_new[idx] = T[idx] + dt * (-adv + diff + source);
			}}}
	#pragma omp for
	for (int n=0; n<nx*ny*nz; n++) T[n] = T_new[n];
}

// ********************************************************* \\

// The whole body runs inside a single thread team (SPMD-style): every thread
// executes the same call sequence, and each #pragma omp for / single inside
// it (here and in the kernels it calls) divides the actual work. This avoids
// spawning/tearing down a thread team per kernel call (~10+ times per
// timestep) -- Intel VTune showed that per-call team creation, not compute,
// was the dominant cost when each kernel opened its own "parallel for".
void ns_compute(double *__restrict u, double *__restrict v, double *__restrict w, double *__restrict p, double *__restrict T, const double *__restrict I, double Pr, double Ra, int nx, int ny,  int nz,  double dx,  double dy,  double dz, double dt, dummy_variables& dv){
	const int n = nx*ny*nz;
	// dv.p_prime is never read before poisson_solver_Neumann_fft below
	// overwrites it entirely (interior via inverse DCT, boundary via
	// Neumann_bc) -- no zero-init needed (that was only required by the old
	// Gauss-Seidel solver, which needed an initial guess to iterate from).
	#pragma omp parallel
	{
	prediction(dv.u_star, dv.v_star, dv.w_star, u, v, w, p, T, Pr, nx, ny, nz, dx, dy, dz, dt);
	Dirichlet_bc3(dv.u_star, dv.v_star, dv.w_star, nx, ny, nz);

	divergence(dv.div, dv.u_star, dv.v_star, dv.w_star, nx, ny, nz, dx, dy, dz);
	const double inv_dt = 1.0/dt;
	#pragma omp for
	for (int i=0; i<n; i++) {dv.div_dt[i] = dv.div[i] * inv_dt; }

	poisson_solver_Neumann_fft(dv.p_prime, dv.div_dt, nx, ny, nz, dv.dct_buf, dv.lam_x, dv.lam_y, dv.lam_z, dv.dct_fwd, dv.dct_bwd);
	correction(dv.u_star, dv.v_star, dv.w_star, u, v, w, dv.p_prime, nx, ny, nz, dx, dy, dz, dt);
	Dirichlet_bc3(u, v, w, nx, ny, nz);
	#pragma omp for
	for (int i=0; i<n; i++) {p[i] += dv.p_prime[i]; }
	Neumann_bc(p, nx, ny, nz);

	T_compute(u, v, w, T, I, Ra, nx, ny, nz, dx, dy, dz, dt, dv);
	Dirichlet_bc(T, nx, ny, nz);
	} // end #pragma omp parallel
}

// ********************************************************* \\

// Parallelized SPMD-style like ns_compute: the k-loop (propagation through z)
// has a genuine sequential dependency -- each plane's transform reads the
// previous plane's result -- so every thread walks the same k sequence in
// lockstep, and #pragma omp for/single inside each iteration divides the
// actual per-plane work. fftw_execute itself isn't safe to call concurrently
// on the same plan, hence #pragma omp single around it (same reasoning as
// poisson_solver_Neumann_fft). The normalize + write-back steps are fused
// into one pass (computing Ar/Ai directly from unnormalized `out`) to save
// a barrier relative to a separate normalize-in-place loop.
void A_compute_ss(double *__restrict Ar, double *__restrict Ai, const double *__restrict T, const double *__restrict kx, const double *__restrict ky, fftw_complex *__restrict in, fftw_complex *__restrict out, fftw_plan ff, fftw_plan fb, int nx, int ny, int nz, double beta, double wn, double C, double dz){
	int i, j, k; double ca;
	const double atten = exp(-beta*dz/2);
	const double inv_norm = 1.0 / ((nx-1)*(ny-1));
	const double inv_2wn = 1.0/(2*wn);
	const double c_phase = C*wn*dz/2.0;
	#pragma omp parallel private(i, j, k, ca)
	{
	for(k=1; k<nz; k++){
		#pragma omp for collapse(2)
		for(j=0; j<ny-1; j++){
			for(i=0; i<nx-1; i++){
			ca = c_phase*T[(k-1)*nx*ny + j*nx + i];
			in[j*(nx-1) + i][0] = atten*(Ar[(k-1)*nx*ny + j*nx + i]*cos(ca) - Ai[(k-1)*nx*ny + j*nx + i]*sin(ca));
			in[j*(nx-1) + i][1] = atten*(Ar[(k-1)*nx*ny + j*nx + i]*sin(ca) + Ai[(k-1)*nx*ny + j*nx + i]*cos(ca));
			}
		}
		#pragma omp single
		fftw_execute(ff);
		#pragma omp for collapse(2)
		for(j=0; j<ny-1; j++){
			for(i=0; i<nx-1; i++){
				ca = (kx[j*(nx-1) + i]*kx[j*(nx-1) + i] + ky[j*(nx-1) + i]*ky[j*(nx-1) + i])*dz*inv_2wn;
				in[j*(nx-1) + i][0] = out[j*(nx-1) + i][0]*cos(ca) - out[j*(nx-1) + i][1]*sin(ca);
				in[j*(nx-1) + i][1] = out[j*(nx-1) + i][0]*sin(ca) + out[j*(nx-1) + i][1]*cos(ca);
			}
		}
		#pragma omp single
		fftw_execute(fb);
		#pragma omp for collapse(2)
		for(j=0; j<ny-1; j++){
			for(i=0; i<nx-1; i++){
				Ar[k*nx*ny + j*nx + i] = out[j*(nx-1) + i][0] * inv_norm;
				Ai[k*nx*ny + j*nx + i] = out[j*(nx-1) + i][1] * inv_norm;
			}
		}
		#pragma omp for
		for(i=0; i<nx; i++){
			Ar[k*nx*ny + (ny-1)*nx + i] = Ar[k*nx*ny + 0*nx + i];
			Ai[k*nx*ny + (ny-1)*nx + i] = Ai[k*nx*ny + 0*nx + i];
		}
		#pragma omp for
		for(j=0; j<ny; j++){
			Ar[k*nx*ny + j*nx + nx-1] = Ar[k*nx*ny + j*nx + 0];
			Ai[k*nx*ny + j*nx + nx-1] = Ai[k*nx*ny + j*nx + 0];
		}

	}
	} // end #pragma omp parallel
}

// ********************************************************* \\

void data_write(const std::string& filename, const double *X, int nx, int ny, int nz){
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
