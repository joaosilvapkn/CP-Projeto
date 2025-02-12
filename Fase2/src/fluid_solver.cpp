#include "fluid_solver.h"
#include <cmath>
#include <omp.h>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}


// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
    int i, j;

    #pragma omp parallel for
    for (j = 1; j <= M; j++) {
        for (i = 1; i <= N; i++) {
            x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
            x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];

            x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
            x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];

            x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
            x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
        }
    }

    // Configuração dos cantos, onde a paralelização não é necessária, pois é uma operação pequena
    x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
    x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
    x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
}




void __attribute__((always_inline)) inline lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c, max_c2;
    float old_x=0;
    float change=0;
    int l = 0;

    do {

        max_c = 0.0f;
        max_c2= 0.0f; 

        //red phase
        #pragma omp parallel for collapse(2) private(old_x,change) reduction(max:max_c)
        for (int kblock = 1; kblock <= 79; kblock += 6) {
            for (int jblock = 1; jblock <= 81; jblock += 4) {
                for (int k = kblock; k <= kblock+5; k++) {
                    for (int j = jblock; j <= jblock+3; j++) {
                        for (int i =1+ (k+j)%2; i <= 84; i+=2){

                            old_x = x[IX(i, j, k)];
                            x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                                              a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                                   x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                                   x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
                            change = fabs(x[IX(i, j, k)] - old_x);
                            max_c=std::max(max_c,change);

                        }
                    }
                }

            }

        }

        //black phase
        #pragma omp parallel for collapse(2) private(old_x,change) reduction(max:max_c2)
        for (int kblock = 1; kblock <= 79; kblock += 6) {
            for (int jblock = 1; jblock <= 81; jblock += 4) {
                for (int k = kblock; k <= kblock+5; k++) {
                    for (int j = jblock; j <= jblock+3; j++) {
                        for (int i = 1+ (k+j+1)%2; i <= 84; i+=2){

                            old_x = x[IX(i, j, k)];
                            x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                                              a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                                   x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                                   x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
                            change = fabs(x[IX(i, j, k)] - old_x);
                            max_c2=std::max(max_c2,change);
                        }

                    }

                }

            }

        }

        set_bnd(M, N, O, b, x);
    } while ((max_c2 > tol || max_c>tol) && ++l < 20);
}



void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}


// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    #pragma omp parallel for collapse(2) 
    for (int k = 1; k <= M; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= O ; i ++) {
		int curr = IX(i, j, k);
		    float x = i - dtX * u[curr];
		    float y = j - dtY * v[curr];
		    float z = k - dtZ * w[curr];

		    // Clamping de variáveis
		    x = fminf(fmaxf(x, 0.5f), M + 0.5f);
		    y = fminf(fmaxf(y, 0.5f), N + 0.5f);
		    z = fminf(fmaxf(z, 0.5f), O + 0.5f);

		    int i0 = (int)x, i1 = i0 + 1;
		    int j0 = (int)y, j1 = j0 + 1;
		    int k0 = (int)z, k1 = k0 + 1;

		    float s1 = x - i0, s0 = 1 - s1;
		    float t1 = y - j0, t0 = 1 - t1;
		    float u1 = z - k0, u0 = 1 - u1;
		    // Cálculo das interpolações trilineares
		    d[curr] =
			s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
			      t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
			s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
			      t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));

            }
        }
    }
    set_bnd(M, N, O, b, d);
}
                         


// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    int maximo = MAX(MAX(M, N), O);

    #pragma omp parallel for collapse(3)
    for (int k = 1; k <= M; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= O; i++) {  // `i` é declarado aqui localmente
                div[IX(i, j, k)] =
                    -0.5f *
                    (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
                     v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
                     w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) /
                    maximo;
                p[IX(i, j, k)] = 0;
            }
        }
    }



    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);
    lin_solve(M, N, O, 0, p, div, 1, 6);

    #pragma omp parallel for collapse(3)
    for (int k = 1; k <= M; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= O; i++) {  // `i` é novamente local aqui
                int curr = IX(i, j, k);
                u[curr] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
                v[curr] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
                w[curr] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
            }
        }
    }

    set_bnd(M, N, O, 1, u);

    set_bnd(M, N, O, 2, v);

    set_bnd(M, N, O, 3, w);
}


// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  //SWAP(x0, x);
  diffuse(M, N, O, 0, x0, x, diff, dt);
  //SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}



// void diffuse_3(int M, int N, int O, int b1, int b2, int b3, float *x, float *x0, float *y,float *y0, float *z, float *z0, float diff, float dt) {
//   int max = MAX(MAX(M, N), O);
//   float a = dt * diff * max * max;


//     #pragma omp parallel
//     {
//         #pragma omp single
//         {

//             #pragma omp task 
//             {
//                 lin_solve(M,N,O,b1,x,x0,a,1 + 6 * a);

//             }


//             #pragma omp task 
//             {
//                 lin_solve(M,N,O,b2,y,y0,a, 1 + 6 * a);

//             }


//              #pragma omp task 
//             {
//                 lin_solve(M,N,O,b3,z,z0,a,1 + 6 * a);

//             }

//         }
//     }

// }

void diffuse_3(int M, int N, int O, int b1, int b2, int b3, float *x, float *x0, float *y, float *y0, float *z, float *z0, float diff, float dt) {
   
    float a = dt * diff * O * O;

   
 
       
        
          
    lin_solve(M, N, O, b1, x, x0, a, 1 + 6 * a);
           
    lin_solve(M, N, O, b2, y, y0, a, 1 + 6 * a);
 
    lin_solve(M, N, O, b3, z, z0, a, 1 + 6 * a);
            
        
    
}


// Add sources (density or velocity)
void add_source_3(int M, int N, int O, float *x, float *s1, float *y, float *s2, float *z, float *s3, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  int i=0;
  #pragma omp parallel for
  for (i = 0; i <= size; i++) {

    x[i] += dt * s1[i];
    y[i] += dt * s2[i];
    z[i] += dt * s3[i];

  }
}




void advect_3(int M, int N, int O, int b1,int b2, int b3, float *d1, float *d2, float *d3, float *u, float *v, float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  #pragma omp parallel for 
  for (int k = 1; k <= M; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= O; i++) {



        int curr = IX(i,j,k);
        float x = i - dtX * u[curr];
        float y = j - dtY * v[curr];  // i
        float z = k - dtZ * w[curr];

        // Clamp to grid boundaries
        if (x < 0.5f)
          x = 0.5f;
        if (x > M + 0.5f)
          x = M + 0.5f;
        if (y < 0.5f)
          y = 0.5f;
        if (y > N + 0.5f)
          y = N + 0.5f;
        if (z < 0.5f)
          z = 0.5f;
        if (z > O + 0.5f)
          z = O + 0.5f;

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1; // i
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1; // i
        float u1 = z - k0, u0 = 1 - u1;

        d1[curr] =
            s0 * (t0 * (u0 * u[IX(i0, j0, k0)] + u1 * u[IX(i0, j0, k1)]) +
                  t1 * (u0 * u[IX(i0, j1, k0)] + u1 * u[IX(i0, j1, k1)])) +  // i
            s1 * (t0 * (u0 * u[IX(i1 ,j0, k0)] + u1 * u[IX(i1, j0, k1)]) +
                  t1 * (u0 * u[IX(i1, j1, k0)] + u1 * u[IX(i1, j1, k1)]));




        d2[curr] =
            s0 * (t0 * (u0 * v[IX(i0, j0, k0)] + u1 * v[IX(i0, j0, k1)]) +
                  t1 * (u0 * v[IX(i0, j1, k0)] + u1 * v[IX(i0, j1, k1)])) +  // i
            s1 * (t0 * (u0 * v[IX(i1 ,j0, k0)] + u1 * v[IX(i1, j0, k1)]) +
                  t1 * (u0 * v[IX(i1, j1, k0)] + u1 * v[IX(i1, j1, k1)]));

        d3[curr] =
            s0 * (t0 * (u0 * w[IX(i0, j0, k0)] + u1 * w[IX(i0, j0, k1)]) +
                  t1 * (u0 * w[IX(i0, j1, k0)] + u1 * w[IX(i0, j1, k1)])) + // i
            s1 * (t0 * (u0 * w[IX(i1 ,j0, k0)] + u1 * w[IX(i1, j0, k1)]) +
                  t1 * (u0 * w[IX(i1, j1, k0)] + u1 * w[IX(i1, j1, k1)]));

      }
    }

    set_bnd(M, N, O, b1, d1);

    set_bnd(M, N, O, b2, d3);

    set_bnd(M, N, O, b3, d3);
  }

}


// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt) {

 // add_source_3(M, N, O, u, u0, v, v0, w, w0, dt);
  add_source(M,N,O,u,u0,dt);
  add_source(M,N,O,v,v0,dt);
  add_source(M,N,O,w,w0,dt);

  diffuse_3(M, N, O, 1, 2, 3, u0, u, v0, v, w0, w, visc, dt);

  project(M, N, O, u0, v0, w0, u, v);

  advect(M,N,O,1,u,u0,u0,v0,w0,dt);
  advect(M,N,O,2,v,v0,u0,v0,w0,dt);
  advect(M,N,O,3,w,w0,u0,v0,w0,dt);


 // advect_3(M,N,O,1,2,3,u,v,w,u0,v0,w0,dt);

  project(M, N, O, u, v, w, u0, v0);
}



