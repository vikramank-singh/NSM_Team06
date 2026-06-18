#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

__global__ void laplace (double*T, double *Tnew, int N){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int j = idx / N;
        int i = idx % N;     
        if (i > 0 && i < N-1 && j > 0 && j < N-1) Tnew[j*N + i] = 0.25 * (T[j*N + i+1] + T[j*N + i-1] + T[(j+1)*N + i] + T[(j-1)*N + i]);
        else if (i==0) Tnew[idx] = 2.0;
        else if (i==N-1) Tnew[idx] = 2.0;
        else if (j==0) Tnew[idx] = 0.0;
        else if (j==N-1) Tnew[idx] = 4.0;
        }

void data_write(const std::string& filename, double *X, int n);



int main()
{

int N = 200;

double *T = new double[N*N];

for (int i=0; i<N; i++) {
	T[i] = 0.0;
	T[i*N+N-1] = 2.0;
	T[(N-1)*N+i] = 4.0;
	T[i*N] = 2.0;
}


double *gT, *gTnew;

T[0] = 10.0;

cudaMalloc(&gT, N*N*sizeof(double));
cudaMalloc(&gTnew, N*N*sizeof(double));

cudaMemcpy(gT, T, N*N*sizeof(double), cudaMemcpyHostToDevice);

int threads_per_block = 1024;
int blocks_per_grid = (N*N + threads_per_block -1)/threads_per_block;

for (int itr=0; itr<40000; itr++){
        laplace<<<blocks_per_grid, threads_per_block>>>(gT, gTnew, N);
        cudaDeviceSynchronize();
        double *aa = gT;
        gT = gTnew;
        gTnew = aa;
}

cudaMemcpy(T, gT, N*N*sizeof(double), cudaMemcpyDeviceToHost);

data_write("T.txt", T, N);

cudaFree(gT);
cudaFree(gTnew);

return 0;
}

void data_write(const std::string& filename, double *X, int n){
	ofstream File(filename);
	for(int j=0; j<n; j++){
		for(int i=0; i<n; i++){
			File<<X[j*n + i]<<endl;
		}
	}
	File.close();
}
