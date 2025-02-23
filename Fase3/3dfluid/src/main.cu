#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>

#define SIZE 168

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;

float *d_x_global, *d_x0_global, *d_u, *d_v, *d_w, *d_u0, *d_v0, *d_w0, *d_max_c_global, *d_max_c2_global;


void copy_data_to_device(int size, float *u, float *v, float *w, float *u0, float *v0, float *w0, float *h_x, float *h_x0) {
  cudaMemcpy(d_x_global, h_x, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x0_global, h_x0, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u, u, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u0, u0, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v0, v0, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w0, w0, size * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_u, h_u, size * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_v, h_v, size * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_w, h_w, size * sizeof(float), cudaMemcpyHostToDevice);
}


void copy_data_to_host(int size, float *h_x, float *u, float *v, float *w) {
  cudaMemcpy(h_x, d_x_global, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(u, d_u, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(v, d_v, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(w, d_w, size * sizeof(float), cudaMemcpyDeviceToHost);
}

 void allocate_device_memory(int size) {
    cudaMalloc(&d_x_global, size * sizeof(float));
    cudaMalloc(&d_x0_global, size * sizeof(float));
    cudaMalloc(&d_max_c_global, sizeof(float));
    cudaMalloc(&d_max_c2_global, sizeof(float));
    //cudaMallocManaged(&d_max_c_global, sizeof(float));
    //cudaMallocManaged(&d_max_c2_global, sizeof(float));
    cudaMalloc(&d_u, size * sizeof(float));
    cudaMalloc(&d_v, size * sizeof(float));
    cudaMalloc(&d_w, size * sizeof(float));
    cudaMalloc(&d_u0, size * sizeof(float));
    cudaMalloc(&d_v0, size * sizeof(float));
    cudaMalloc(&d_w0, size * sizeof(float));
 }
 
 void free_device_memory() {
    cudaFree(d_x_global);
    cudaFree(d_x0_global);
    cudaFree(d_max_c_global);
    cudaFree(d_max_c2_global);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_u0);
    cudaFree(d_v0);
    cudaFree(d_w0);
 }


// Function to allocate simulation data
int allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  u = new float[size];
  v = new float[size];
  w = new float[size];
  u_prev = new float[size];
  v_prev = new float[size];
  w_prev = new float[size];
  dens = new float[size];
  dens_prev = new float[size];
  allocate_device_memory(size);

  if (!u || !v || !w || !u_prev || !v_prev || !w_prev || !dens || !dens_prev) {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }
  return 1;
}

// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    u[i] = v[i] = w[i] = u_prev[i] = v_prev[i] = w_prev[i] = dens[i] =
        dens_prev[i] = 0.0f;
  }
}

// Free allocated memory
void free_data() {
  delete[] u;
  delete[] v;
  delete[] w;
  delete[] u_prev;
  delete[] v_prev;
  delete[] w_prev;
  delete[] dens;
  delete[] dens_prev;
  free_device_memory();
}


// Define the kernel function
__global__ void apply_events_kernel(Event* events, int num_events, int M, int N, int O, float* d_x_global, float* d_u, float* d_v, float* d_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_events) {
    Event event = events[idx];
    int i = M / 2, j = N / 2, k = O / 2;
    if (event.type == ADD_SOURCE) {
      d_x_global[IX(i, j, k)] = event.density;
    } else if (event.type == APPLY_FORCE) {
      d_u[IX(i, j, k)] = event.force.x;
      d_v[IX(i, j, k)] = event.force.y;
      d_w[IX(i, j, k)] = event.force.z;
    }
  }
}

// Apply events (source or force) for the current timestep
void apply_events(const std::vector<Event> &events) {

  int size = events.size();
  Event *d_events;

  if(size > 0){
    cudaMalloc((void **)&d_events, size * sizeof(Event));
    cudaMemcpy(d_events, events.data(), size * sizeof(Event), cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocks_per_grid = (size + threadsPerBlock - 1) / threadsPerBlock;

    apply_events_kernel<<<blocks_per_grid, threadsPerBlock>>>(d_events, size, M, N, O, d_x_global, d_u, d_v, d_w);
    //cudaDeviceSynchronize();

    cudaFree(d_events);
  }
}


// Function to sum the total density
float sum_density() {
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    //std::cerr << "dens " << (i) << ": " << (dens[i]) << std::endl;
    total_density += dens[i];
  }
  return total_density;
}

// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
  for (int t = 0; t < timesteps; t++) {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    // Apply events to the simulation
    apply_events(events);

    // Perform the simulation steps
    vel_step(M, N, O, d_u, d_v, d_w, d_u0, d_v0, d_w0, visc, dt, d_max_c_global, d_max_c2_global);
    dens_step(M, N, O, d_x_global, d_x0_global, d_u, d_v, d_w, diff, dt, d_max_c_global, d_max_c2_global);
  }
}

int main() {
  // Initialize EventManager
  EventManager eventManager;
  eventManager.read_events("events.txt");

  // Get the total number of timesteps from the event file
  int timesteps = eventManager.get_total_timesteps();

  // Allocate and clear data
  if (!allocate_data())
    return -1;
  clear_data();

  int size = (M + 2) * (N + 2) * (O + 2);

  copy_data_to_device(size,u,v,w,u_prev,v_prev,w_prev,dens,dens_prev);

  // Run simulation with events
  simulate(eventManager, timesteps);

  copy_data_to_host(size,dens,u,v,w);

  // Print total density at the end of simulation
  float total_density = sum_density();
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  free_data();

  return 0;
}
