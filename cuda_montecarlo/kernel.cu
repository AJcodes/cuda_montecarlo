#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <iomanip>
#include <iostream>

#define BLOCKSIZE 64

cudaError_t mcCuda(double *val, double *val1, const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T);

__global__ void init(unsigned int seed, curandState_t* states, double *normal, const int num_sims) {
	int id = blockIdx.x * BLOCKSIZE + threadIdx.x;
	if (id < num_sims) {
		curand_init(seed, id, 0, &states[id]);
		normal[id] = curand_normal_double(&states[id]);
	}
}

__global__ void mcKernel(double *normal, const int num_sims, const double S, const double K, const double r, const double v, const double T, double *val, double *val1) {
	__shared__ double c[BLOCKSIZE];
	__shared__ double p[BLOCKSIZE];
    double S_adjust = S * exp(T*(r-0.5*v*v));
	double S_cur = 0.0;
	double payoff_sum = 0.0;
	double payoff_sum1 = 0.0;
	double call_temp = 0.0;
	double put_temp = 0.0;
	int id = blockIdx.x * BLOCKSIZE + threadIdx.x;
	int bid = blockIdx.x;

	if (id < num_sims) {
			double gauss_bm = normal[id];
			S_cur = S_adjust * exp(sqrt(v*v*T)*gauss_bm);
			payoff_sum = max(S_cur - K, 0.0);
			payoff_sum1 = max(K - S_cur, 0.0);
			c[threadIdx.x] = (payoff_sum / num_sims) * exp(-r*T);
			p[threadIdx.x] = (payoff_sum1 / num_sims) * exp(-r*T);
			__syncthreads();

			for (int i = 0; i <BLOCKSIZE; ++i) {
				call_temp += c[i];
				put_temp += p[i];
			}
	}

	val[bid] = call_temp;
	val1[bid] = put_temp;
}

int main() {
	int num_sims = 100000;   // Number of simulated asset paths                                                       
	double S = 100.0;  // Option price                                                                                  
	double K = 100.0;  // Strike price                                                                                  
	double r = 0.05;   // Risk-free rate (5%)                                                                           
	double v = 0.2;    // Volatility of the underlying (20%)                                                            
	double T = 1.0;    // One year until expiry
	double *val = 0;
	double *val1 = 0;

	val = (double *) malloc(num_sims * sizeof(double));
	val1 = (double *) malloc(num_sims * sizeof(double));

    cudaError_t cudaStatus = mcCuda(val, val1, num_sims, S, K, r, v, T);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t mcCuda(double *val, double *val1, const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T) {
	double * dev_val = 0;
	double * dev_val1 = 0;
	double * dev_normal = 0;
    cudaError_t cudaStatus;
	curandState_t* states;
	dim3 dimBlock(BLOCKSIZE);
    dim3 dimGrid(ceil(float(num_sims) / float(BLOCKSIZE)));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	float milliseconds = 0.0f;
	float milliseconds1 = 0.0f;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**) &states, num_sims * sizeof(curandState_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_val, num_sims * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_val1, num_sims * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_normal, num_sims * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	init<<<dimGrid, dimBlock>>>(time(0), states, dev_normal, num_sims);

	cudaEventRecord(start);
    mcKernel<<<dimGrid, dimBlock>>>(dev_normal, num_sims, S, K, r, v, T, dev_val, dev_val1);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mcKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mcKernel!\n", cudaStatus);
        goto Error;
    }

	cudaStatus = cudaMemcpy(val, dev_val, num_sims * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(val1, dev_val1, num_sims * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaEventRecord(start1);
	double call = 0;
	for (int i = 0; i < ceil(float(num_sims) / float(BLOCKSIZE)); i++) {
		call += val[i];
	}
	double put = 0;
	for (int i = 0; i < ceil(float(num_sims) / float(BLOCKSIZE)); i++) {
		put += val1[i];
	}

	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);

	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventElapsedTime(&milliseconds1, start1, stop1);
	std::cout << "Number of Paths: " << num_sims << std::endl;
	std::cout << "Underlying:      " << S << std::endl;
	std::cout << "Strike:          " << K << std::endl;
	std::cout << "Risk-Free Rate:  " << r << std::endl;
	std::cout << "Volatility:      " << v << std::endl;
	std::cout << "Maturity:        " << T << std::endl;
	std::cout << "================================" << std::endl;
	std::cout << "Call Price:	   " << call << std::endl;
	std::cout << "Put Price:	   " << put << std::endl;
	std::cout << "================================" << std::endl;
	std::cout << "Execution Time : " << (milliseconds + milliseconds1) / 1000 << " seconds" << std::endl;
	std::cout << "Execution Time (GPU only): " << (milliseconds) / 1000 << " seconds" << std::endl;
	std::cout << "Execution Time (Summation only): " << (milliseconds1) / 1000 << " seconds" << std::endl;
	std::cout << "Effective Bandwidth : " << (num_sims*sizeof(double)*2) / ((milliseconds + milliseconds1) / 1000) << " GB/s" << std::endl;

Error:
    cudaFree(dev_val);
	cudaFree(dev_val1);
	cudaFree(dev_normal);
	cudaFree(states);
    
    return cudaStatus;
}
