#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt, float *d_max_c_global, float *d_max_c2_global);
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt, float *d_max_c_global, float *d_max_c2_global);
//void allocate_device_memory(int size);
//void free_device_memory();
//void copy_data_to_host(int size, float *h_x, float *u, float *v, float *w);
//void copy_data_to_device(int size, float *h_u, float *h_v, float *h_w, float *u0, float *v0, float *w0, float *h_x, float *h_x0);
#endif // FLUID_SOLVER_H
