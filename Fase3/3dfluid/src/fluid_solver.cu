#include "fluid_solver.h"
#include <cmath>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdio.h>
#include <iostream>
//#define IX(i, j, k, M, N) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
/*
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
*/
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

// Kernel para adicionar fontes
__global__ void add_source_kernel(int size, float *x, const float *s, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}

void add_source_cuda(int M, int N, int O, float *x, const float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);

    // Configuração de threads e blocos
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Executar o kernel
    add_source_kernel<<<blocksPerGrid, threadsPerBlock>>>(size, x, s, dt);
    //cudaDeviceSynchronize();
}


// Kernel para ajustar as condições de contorno
__global__ void set_bnd_kernel(int M, int N, int O, int b, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // Ajustar as faces
    if (i <= M && j <= N) {
        x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
        x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];
        x[IX(0, i, j)] = (b == 1) ? -x[IX(1, i, j)] : x[IX(1, i, j)];
        x[IX(M + 1, i, j)] = (b == 1) ? -x[IX(M, i, j)] : x[IX(M, i, j)];
        x[IX(i, 0, j)] = (b == 2) ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
        x[IX(i, N + 1, j)] = (b == 2) ? -x[IX(i, N, j)] : x[IX(i, N, j)];
    }

    // Configurar os cantos (somente thread 0 para evitar redundância)
    if (i == 1 && j == 1) {
        x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
        x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
        x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
        x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
    }
}


void set_bnd_cuda(int M, int N, int O, int b, float *x) {
    // Configuração de threads e blocos
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Executar o kernel
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, x);

    // Sincronizar para garantir a execução
    //cudaDeviceSynchronize();
}


//__device__ float atomicMaxFloat(float *address, float val) {
//    int *address_as_int = (int *)address;
//    int old = *address_as_int, assumed;
//
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_int, assumed,
//                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
//    } while (assumed != old);
//
//    return __int_as_float(old);
//}

__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]);
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 8]);
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 4]);
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 2]);
    sdata[tid] = fmaxf(sdata[tid], sdata[tid + 1]);
}


// Kernel para a fase vermelha
__global__ void lin_solve_red_kernel(int M, int N, int O, float *__restrict__ x, const float *__restrict__ x0, float a, float c, float *__restrict__ max_c) {
    //extern __shared__ float local_max[]; // Memória compartilhada para o bloco

    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    //if (i > M || j > N || k > O) return;

    //if ((i + k + j) % 2 != 0) return;

    //int local_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    //local_max[local_idx] = 0.0f; // Inicializar o máximo local

    

    if ((i + k + j) % 2 == 0 && i <= M && j <= N && k <= O) {
        //int idx = IX(i, j, k);
        int idx = k * (M + 2) * (N + 2) + j * (M + 2) + i;
        float old_x = __ldg(&x[idx]);
        //x[idx] = (x0[idx] +
        //          a * (__ldg(&x[IX(i - 1, j, k)]) + __ldg(&x[IX(i + 1, j, k)]) +
        //               __ldg(&x[IX(i, j - 1, k)]) + __ldg(&x[IX(i, j + 1, k)]) +
        //               __ldg(&x[IX(i, j, k - 1)]) + __ldg(&x[IX(i, j, k + 1)]))) / c;

        x[idx] = __fmaf_rn(a, (__ldg(&x[IX(i - 1, j, k)]) + __ldg(&x[IX(i + 1, j, k)]) +
                       __ldg(&x[IX(i, j - 1, k)]) + __ldg(&x[IX(i, j + 1, k)]) +
                       __ldg(&x[IX(i, j, k - 1)]) + __ldg(&x[IX(i, j, k + 1)])), x0[idx]) / c;


        float change = fabsf(x[idx] - old_x);
        //local_max[local_idx] = change; // Armazenar mudança local

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            change = fmaxf(change, __shfl_down_sync(0xFFFFFFFF, change, offset));
        }

        if ((threadIdx.x & 31) == 0) {
            //atomicMax((unsigned int *)max_c, __float_as_int(local_max[0])); // Atualização atômica
            atomicMax(reinterpret_cast<unsigned int *>(max_c), __float_as_int(change));
            //atomicMaxDouble((double*)max_c, (double)local_max[0]);
        }
    }  
}


// Kernel para a fase preta
__global__ void lin_solve_black_kernel(int M, int N, int O, float *__restrict__ x, const float *__restrict__ x0, float a, float c, float *__restrict__ max_c) {
    //extern __shared__ float local_max[];
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    //if (i > M || j > N || k > O) return;

    //if ((i + k + j) % 2 != 1) return;

    //int local_idx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    //local_max[local_idx] = 0.0f;

    if ((i + k + j) % 2 != 0 && i <= M && j <= N && k <= O) {
        //int idx = IX(i, j, k);
        int idx = k * (M + 2) * (N + 2) + j * (M + 2) + i;
        float old_x = __ldg(&x[idx]);
        //x[idx] = (x0[idx] +
        //          a * (__ldg(&x[IX(i - 1, j, k)]) + __ldg(&x[IX(i + 1, j, k)]) +
        //               __ldg(&x[IX(i, j - 1, k)]) + __ldg(&x[IX(i, j + 1, k)]) +
        //               __ldg(&x[IX(i, j, k - 1)]) + __ldg(&x[IX(i, j, k + 1)]))) / c;

        x[idx] = __fmaf_rn(a, (__ldg(&x[IX(i - 1, j, k)]) + __ldg(&x[IX(i + 1, j, k)]) +
                       __ldg(&x[IX(i, j - 1, k)]) + __ldg(&x[IX(i, j + 1, k)]) +
                       __ldg(&x[IX(i, j, k - 1)]) + __ldg(&x[IX(i, j, k + 1)])), x0[idx]) / c;

        float change = fabsf(x[idx] - old_x);
        //local_max[local_idx] = change;

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            change = fmaxf(change, __shfl_down_sync(0xFFFFFFFF, change, offset));
        }

        if ((threadIdx.x & 31) == 0) {
            atomicMax(reinterpret_cast<unsigned int *>(max_c), __float_as_int(change));
        }
    }

}



__global__ void lin_solve_combined_kernel(int M, int N, int O, float *x, const float *x0, float a, float c, float *max_c) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    sdata[tid] = 0.0f;
    //if (i <= M && j <= N && k <= O) {
    //    int idx = IX(i, j, k);
    //    float old_x = x[idx];
    //    x[idx] = (x0[idx] + a * (x[IX(i-1,j,k)] + x[IX(i+1,j,k)] + x[IX(i,j-1,k)] + x[IX(i,j+1,k)] + x[IX(i,j,k-1)] + x[IX(i,j,k+1)])) / c;
    //    float change = fabsf(x[idx] - old_x);
    //    sdata[tid] = change;
    //}
    //__syncthreads();

    // Fase vermelha: células pares
    if ((i + j + k) % 2 == 0 && i <= M && j <= N && k <= O) {
        int idx = IX(i, j, k);
        float old_x = x[idx];
        x[idx] = (x0[idx] + a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                 x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                 x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
        sdata[tid] = fabsf(x[idx] - old_x);
    }
    __syncthreads();

    // Fase preta: células ímpares
    if ((i + j + k) % 2 != 0 && i <= M && j <= N && k <= O) {
        int idx = IX(i, j, k);
        float old_x = x[idx];
        x[idx] = (x0[idx] + a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                 x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                 x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
        sdata[tid] = fmaxf(sdata[tid], fabsf(x[idx] - old_x));
    }
    __syncthreads();

    for (int s = blockDim.x * blockDim.y * blockDim.z / 2; s > 32; s /= 2) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) atomicMax((unsigned int *)max_c, __float_as_int(sdata[0]));
}


void lin_solve_cuda(int M, int N, int O, int b, float *x, float *x0, float a, float c, float *d_max_c_global, float *d_max_c2_global) {
    // Configuração de blocos e grids
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    float tol = 1e-7;
    int iterations = 0, max_iter = 20;
    float max_val, max_val2;

    do {
        cudaMemset(d_max_c_global, 0, sizeof(float));
        cudaMemset(d_max_c2_global, 0, sizeof(float));

        // Fase vermelha
        lin_solve_red_kernel<<<numBlocks, threadsPerBlock, threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * sizeof(float)>>>(M, N, O, x, x0, a, c, d_max_c_global);
        cudaDeviceSynchronize();

        // Fase preta
        lin_solve_black_kernel<<<numBlocks, threadsPerBlock, threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * sizeof(float)>>>(M, N, O, x, x0, a, c, d_max_c2_global);
        cudaDeviceSynchronize();


        //lin_solve_combined_kernel<<<numBlocks, threadsPerBlock, threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * sizeof(float)>>>(M, N, O, x, x0, a, c, d_max_c_global);
        
        cudaMemcpy(&max_val, d_max_c_global, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&max_val2, d_max_c2_global, sizeof(float), cudaMemcpyDeviceToHost);


        // Copiar max_c de volta para o host
        //cudaMemcpy(&max_c_host, d_max_c_global, sizeof(float), cudaMemcpyDeviceToHost);
        //cudaMemcpy(&max_c2_host, d_max_c2_global, sizeof(float), cudaMemcpyDeviceToHost);


        //if ((*d_max_c_global <= tol) && (*d_max_c2_global <= tol)){
        //    break;
        //}

        set_bnd_cuda(M, N, O, b, x);
        
        iterations++;
    //} while ((max_c_host > tol || max_c2_host > tol) && iterations < max_iter);
    } while ((max_val > tol || max_val2 > tol) && iterations < max_iter);
}


void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt, float *d_max_c_global, float *d_max_c2_global) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve_cuda(M, N, O, b, x, x0, a, 1 + 6 * a, d_max_c_global, d_max_c2_global);
}


// Kernel para realizar a advecção
__global__ void advect_kernel(int M, int N, int O, int b, float *d, const float *d0, const float *u, const float *v, const float *w, float dt) {
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i <= M && j <= N && k <= O) {
        int curr = IX(i, j, k);

        float x = i - dt * M * u[curr];
        float y = j - dt * N * v[curr];
        float z = k - dt * O * w[curr];

        // Clamp os valores para dentro do domínio
        x = fminf(fmaxf(x, 0.5f), M + 0.5f);
        y = fminf(fmaxf(y, 0.5f), N + 0.5f);
        z = fminf(fmaxf(z, 0.5f), O + 0.5f);

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1;
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        d[curr] =
            s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                  t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
            s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                  t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
    }
}

void advect_cuda(int M, int N, int O, int b, float *d, const float *d0, const float *u, const float *v, const float *w, float dt) {
    // Configuração de blocos e threads
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Executar o kernel
    advect_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d, d0, u, v, w, dt);
    //cudaDeviceSynchronize();

    set_bnd_cuda(M, N, O, b, d);
}


// Kernel para calcular a divergência e inicializar p
__global__ void calculate_divergence_and_initialize_p(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i <= M && j <= N && k <= O) {
        int idx = IX(i, j, k);
        div[idx] = -0.5f * ((u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)]) +
                            (v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)]) +
                            (w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)])) / M;
        p[idx] = 0;
    }
}

// Kernel para corrigir u, v, w usando p
__global__ void correct_velocity(int M, int N, int O, float *u, float *v, float *w, float *p) {
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i <= M && j <= N && k <= O) {
        int idx = IX(i, j, k);
        u[idx] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[idx] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[idx] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    }
}

// Função principal para realizar a projeção
void project_cuda(int M, int N, int O, float *u, float *v, float *w, float *d_u, float *d_v, float *d_max_c_global, float *d_max_c2_global) {
    // Configuração de blocos e threads
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Calcular divergência e inicializar p
    calculate_divergence_and_initialize_p<<<numBlocks, threadsPerBlock>>>(M, N, O, u, v, w, d_u, d_v);
    //cudaDeviceSynchronize();

    // Aplicar condições de contorno para div e p
    set_bnd_cuda(M, N, O, 0, d_v);
    set_bnd_cuda(M, N, O, 0, d_u);

    // Resolver o sistema linear para p
    lin_solve_cuda(M, N, O, 0, d_u, d_v, 1, 6, d_max_c_global, d_max_c2_global);

    // Corrigir u, v, w
    correct_velocity<<<numBlocks, threadsPerBlock>>>(M, N, O, u, v, w, d_u);
    //cudaDeviceSynchronize();

    // Aplicar condições de contorno para u, v, w
    set_bnd_cuda(M, N, O, 1, u);
    set_bnd_cuda(M, N, O, 2, v);
    set_bnd_cuda(M, N, O, 3, w);
}


void diffuse_3(int M, int N, int O, int b1, int b2, int b3, float *x, float *x0, float *y, float *y0, float *z, float *z0, float diff, float dt, float *d_max_c_global, float *d_max_c2_global) {
    float a = dt * diff * O * O;

    lin_solve_cuda(M, N, O, b1, x, x0, a, 1 + 6 * a, d_max_c_global, d_max_c2_global);
           
    lin_solve_cuda(M, N, O, b2, y, y0, a, 1 + 6 * a, d_max_c_global, d_max_c2_global);
 
    lin_solve_cuda(M, N, O, b3, z, z0, a, 1 + 6 * a, d_max_c_global, d_max_c2_global);
}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt, float *d_max_c_global, float *d_max_c2_global) {
  add_source_cuda(M, N, O, x, x0, dt);

  //SWAP(x0, x);
  diffuse(M, N, O, 0, x0, x, diff, dt, d_max_c_global, d_max_c2_global);

  //SWAP(x0, x);
  advect_cuda(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt, float *d_max_c_global, float *d_max_c2_global) {
  add_source_cuda(M,N,O,u,u0,dt);
  add_source_cuda(M,N,O,v,v0,dt);
  add_source_cuda(M,N,O,w,w0,dt);

  diffuse_3(M, N, O, 1, 2, 3, u0, u, v0, v, w0, w, visc, dt, d_max_c_global, d_max_c2_global);

  project_cuda(M, N, O, u0, v0, w0, u, v, d_max_c_global, d_max_c2_global);

  advect_cuda(M,N,O,1,u,u0,u0,v0,w0,dt);
  advect_cuda(M,N,O,2,v,v0,u0,v0,w0,dt);
  advect_cuda(M,N,O,3,w,w0,u0,v0,w0,dt);

  project_cuda(M, N, O, u, v, w, u0, v0, d_max_c_global, d_max_c2_global);
}