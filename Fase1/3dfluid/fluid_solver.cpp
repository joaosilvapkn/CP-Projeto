#include "fluid_solver.h"
#include <cmath>

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
  for (int i = 0; i < size-15; i+=16) {
    x[i] += dt * s[i]; x[i+1] += dt * s[i+1];
    x[i+2] += dt * s[i+2]; x[i+3] += dt * s[i+3];
    x[i+4] += dt * s[i+4]; x[i+5] += dt * s[i+5];
    x[i+6] += dt * s[i+6]; x[i+7] += dt * s[i+7];
    x[i+8] += dt * s[i+8]; x[i+9] += dt * s[i+9];
    x[i+10] += dt * s[i+10]; x[i+11] += dt * s[i+11];
    x[i+12] += dt * s[i+12]; x[i+13] += dt * s[i+13];
    x[i+14] += dt * s[i+14]; x[i+15] += dt * s[i+15];
  }
}

// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;

  // Set boundary on faces
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

  // Set corners
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);

  x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);

  x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);

  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
}

// Linear solve for implicit methods (diffusion)
__attribute__((always_inline)) inline void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {

    for (int l = 0; l < LINEARSOLVERTIMES; l++) {
        for (int k = 1; k <= O; k++) {
            for (int j = 1; j <= N; j++) {
                for (int i = 1; i <= M-6; i+=7) {

                  int curr = IX(i,j,k);
            
                  x[curr] = (x0[curr] +
                                    a * (x[curr-1] + x[curr+1] +
                                         x[curr-44] + x[curr+44] +
                                         x[curr-1936] + x[curr+1936])) / c;
          
                  x[curr+1] = (x0[curr+1] +
                                    a * (x[curr] + x[curr+2] +
                                         x[curr+1-44] + x[curr+1+44] +
                                         x[curr+1-1936] + x[curr+1+1936])) / c;
          
                  x[curr+2] = (x0[curr+2] +
                                    a * (x[curr+1] + x[curr+3] +
                                         x[curr+2-44] + x[curr+2+44] +
                                         x[curr+2-1936] + x[curr+2+1936])) / c;
          
                  x[curr+3] = (x0[curr+3] +
                                    a * (x[curr+2] + x[curr+4] +
                                         x[curr+3-44] + x[curr+3+44] +
                                         x[curr+3-1936] + x[curr+3+1936])) / c;
        
                  x[curr+4] = (x0[curr+4] +
                                      a * (x[curr+3] + x[curr+5] +
                                           x[curr+4-44] + x[curr+4+44] +
                                         x[curr+4-1936] + x[curr+4+1936])) / c;

                  x[curr+5] = (x0[curr+5] +
                                      a * (x[curr+4] + x[curr+6] +
                                           x[curr+5-44] + x[curr+5+44] +
                                         x[curr+5-1936] + x[curr+5+1936])) / c;
            
                  x[curr+6] = (x0[curr+6] +
                                      a * (x[curr+5] + x[curr+7] +
                                           x[curr+6-44] + x[curr+6+44] +
                                         x[curr+6-1936] + x[curr+6+1936])) / c;                                                                         
                }
            }
        }
        set_bnd(M, N, O, b, x);
    }
}

// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  for (int k = 1; k <= M; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= O-2; i+=3) {

        int curr = IX(i,j,k);

        float x = i - dtX * u[curr];
        float y = j - dtY * v[curr];
        float z = k - dtZ * w[curr];

        float a = i+1 - dtX * u[curr+1];
        float b = j - dtY * v[curr+1];
        float c = k - dtZ * w[curr+1];

        float a2 = i+2 - dtX * u[curr+2];
        float b2 = j - dtY * v[curr+2];
        float c2 = k - dtZ * w[curr+2];

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

        if (a < 0.5f)
          a = 0.5f;
        if (a > M + 0.5f)
          a = M + 0.5f;
        if (b < 0.5f)
          b = 0.5f;
        if (b > N + 0.5f)
          b = N + 0.5f;
        if (c < 0.5f)
          c = 0.5f;
        if (c > O + 0.5f)
          c = O + 0.5f;

        if (a2 < 0.5f)
          a2 = 0.5f;
        if (a2 > M + 0.5f)
          a2 = M + 0.5f;
        if (b2 < 0.5f)
          b2 = 0.5f;
        if (b2 > N + 0.5f)
          b2 = N + 0.5f;
        if (c2 < 0.5f)
          c2 = 0.5f;
        if (c2 > O + 0.5f)
          c2 = O + 0.5f;

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1; // i
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1; // i
        float u1 = z - k0, u0 = 1 - u1;

        int i0_a = (int)a, i1_a = i0_a + 1;
        int j0_b = (int)b, j1_b = j0_b + 1; // i + 1
        int k0_c = (int)c, k1_c = k0_c + 1;

        float s1_a = a - i0_a, s0_a = 1 - s1_a;
        float t1_b = b - j0_b, t0_b = 1 - t1_b; // i + 1
        float u1_c = c - k0_c, u0_c = 1 - u1_c;

        int i0_a2 = (int)a2, i1_a2 = i0_a2 + 1;
        int j0_b2 = (int)b2, j1_b2 = j0_b2 + 1; // i + 2
        int k0_c2 = (int)c2, k1_c2 = k0_c2 + 1;

        float s1_a2 = a2 - i0_a2, s0_a2 = 1 - s1_a2;
        float t1_b2 = b2 - j0_b2, t0_b2 = 1 - t1_b2; // i + 2
        float u1_c2 = c2 - k0_c2, u0_c2 = 1 - u1_c2;

        d[curr] =
            s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                  t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) + // i
            s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                  t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));

        d[curr+1] =
            s0_a * (t0_b * (u0_c * d0[IX(i0_a, j0_b, k0_c)] + u1_c * d0[IX(i0_a, j0_b, k1_c)]) +
                  t1_b * (u0_c * d0[IX(i0_a, j1_b, k0_c)] + u1_c * d0[IX(i0_a, j1_b, k1_c)])) +  // i+1
            s1_a * (t0_b * (u0_c * d0[IX(i1_a, j0_b, k0_c)] + u1_c * d0[IX(i1_a, j0_b, k1_c)]) +
                  t1_b * (u0_c * d0[IX(i1_a, j1_b, k0_c)] + u1_c * d0[IX(i1_a, j1_b, k1_c)]));

        d[curr+2] =
            s0_a2 * (t0_b2 * (u0_c2 * d0[IX(i0_a2, j0_b2, k0_c2)] + u1_c2 * d0[IX(i0_a2, j0_b2, k1_c2)]) +
                  t1_b2 * (u0_c2 * d0[IX(i0_a2, j1_b2, k0_c2)] + u1_c2 * d0[IX(i0_a2, j1_b2, k1_c2)])) +  //i+2
            s1_a2 * (t0_b2 * (u0_c2 * d0[IX(i1_a2, j0_b2, k0_c2)] + u1_c2 * d0[IX(i1_a2, j0_b2, k1_c2)]) +
                  t1_b2 * (u0_c2 * d0[IX(i1_a2, j1_b2, k0_c2)] + u1_c2 * d0[IX(i1_a2, j1_b2, k1_c2)]));

      }
    }
  }
  set_bnd(M, N, O, b, d);
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
  int maximo = MAX(M, MAX(N, O));

  for (int k = 1; k <= M; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= O; i++) {
        div[IX(i, j, k)] =
            -0.5f *
            (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
             v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) /
            maximo;
        p[IX(i, j, k)] = 0;
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  for (int k = 1; k <= M; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= O; i++) {
        int curr = IX(i,j,k);
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

void lin_solve_3(int M, int N, int O, int b1, int b2, int b3, float *x, float *x0, float *y,float *y0, float *z, float *z0, float a, float c) {
  for (int l = 0; l < LINEARSOLVERTIMES; l++) {
    for (int k = 1; k <= M; k++) {
      for (int j = 1; j <= N; j++) {
        for (int i = 1; i <= O-6; i+=7) {

          int curr = IX(i,j,k);

          x[curr] = (x0[curr] +
                            a * (x[curr-1] + x[curr+1] +
                                 x[curr-44] + x[curr+44] +
                                 x[curr-1936] + x[curr+1936])) / c;
                           
          x[curr+1] = (x0[curr+1] +
                            a * (x[curr] + x[curr+2] +
                                 x[curr+1-44] + x[curr+1+44] +
                                 x[curr+1-1936] + x[curr+1+1936])) / c;
                           
          x[curr+2] = (x0[curr+2] +
                            a * (x[curr+1] + x[curr+3] +
                                 x[curr+2-44] + x[curr+2+44] +
                                 x[curr+2-1936] + x[curr+2+1936])) / c;
                           
          x[curr+3] = (x0[curr+3] +
                            a * (x[curr+2] + x[curr+4] +
                                 x[curr+3-44] + x[curr+3+44] +
                                 x[curr+3-1936] + x[curr+3+1936])) / c;
                           
          x[curr+4] = (x0[curr+4] +
                              a * (x[curr+3] + x[curr+5] +
                                   x[curr+4-44] + x[curr+4+44] +
                                 x[curr+4-1936] + x[curr+4+1936])) / c;
                             
          x[curr+5] = (x0[curr+5] +
                              a * (x[curr+4] + x[curr+6] +
                                   x[curr+5-44] + x[curr+5+44] +
                                 x[curr+5-1936] + x[curr+5+1936])) / c;
                             
          x[curr+6] = (x0[curr+6] +
                              a * (x[curr+5] + x[curr+7] +
                                   x[curr+6-44] + x[curr+6+44] +
                                 x[curr+6-1936] + x[curr+6+1936])) / c;
                             
          y[curr] = (y0[curr] +
                            a * (y[curr-1] + y[curr+1] +
                                 y[curr-44] + y[curr+44] +
                                 y[curr-1936] + y[curr+1936])) / c;
                           
          y[curr+1] = (y0[curr+1] +
                            a * (y[curr] + y[curr+2] +
                                 y[curr+1-44] + y[curr+1+44] +
                                 y[curr+1-1936] + y[curr+1+1936])) / c;
                           
          y[curr+2] = (y0[curr+2] +
                            a * (y[curr+1] + y[curr+3] +
                                 y[curr+2-44] + y[curr+2+44] +
                                 y[curr+2-1936] + y[curr+2+1936])) / c;
                           
          y[curr+3] = (y0[curr+3] +
                            a * (y[curr+2] + y[curr+4] +
                                 y[curr+3-44] + y[curr+3+44] +
                                 y[curr+3-1936] + y[curr+3+1936])) / c;
                           
          y[curr+4] = (y0[curr+4] +
                              a * (y[curr+3] + y[curr+5] +
                                   y[curr+4-44] + y[curr+4+44] +
                                 y[curr+4-1936] + y[curr+4+1936])) / c;
                             
          y[curr+5] = (y0[curr+5] +
                              a * (y[curr+4] + y[curr+6] +
                                   y[curr+5-44] + y[curr+5+44] +
                                 y[curr+5-1936] + y[curr+5+1936])) / c;
                             
          y[curr+6] = (y0[curr+6] +
                              a * (y[curr+5] + y[curr+7] +
                                   y[curr+6-44] + y[curr+6+44] +
                                 y[curr+6-1936] + y[curr+6+1936])) / c;
                             
          z[curr] = (z0[curr] +
                            a * (z[curr-1] + z[curr+1] +
                                 z[curr-44] + z[curr+44] +
                                 z[curr-1936] + z[curr+1936])) / c;
                           
          z[curr+1] = (z0[curr+1] +
                            a * (z[curr] + z[curr+2] +
                                 z[curr+1-44] + z[curr+1+44] +
                                 z[curr+1-1936] + z[curr+1+1936])) / c;
                           
          z[curr+2] = (z0[curr+2] +
                            a * (z[curr+1] + z[curr+3] +
                                 z[curr+2-44] + z[curr+2+44] +
                                 z[curr+2-1936] + z[curr+2+1936])) / c;
                           
          z[curr+3] = (z0[curr+3] +
                            a * (z[curr+2] + z[curr+4] +
                                 z[curr+3-44] + z[curr+3+44] +
                                 z[curr+3-1936] + z[curr+3+1936])) / c;

          z[curr+4] = (z0[curr+4] +
                              a * (z[curr+3] + z[curr+5] +
                                   z[curr+4-44] + z[curr+4+44] +
                                 z[curr+4-1936] + z[curr+4+1936])) / c;
                             
          z[curr+5] = (z0[curr+5] +
                              a * (z[curr+4] + z[curr+6] +
                                   z[curr+5-44] + z[curr+5+44] +
                                 z[curr+5-1936] + z[curr+5+1936])) / c;
                             
          z[curr+6] = (z0[curr+6] +
                              a * (z[curr+5] + z[curr+7] +
                                   z[curr+6-44] + z[curr+6+44] +
                                 z[curr+6-1936] + z[curr+6+1936])) / c;                           
        }
      }
    }
    set_bnd(M, N, O, b1, x);
    set_bnd(M, N, O, b2, y);
    set_bnd(M, N, O, b3, z);
  }
}

void diffuse_3(int M, int N, int O, int b1, int b2, int b3, float *x, float *x0, float *y,float *y0, float *z, float *z0, float diff, float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve_3(M, N, O, b1, b2, b3, x, x0, y, y0, z, z0,a, 1 + 6 * a);
}

// Add sources (density or velocity)
void add_source_3(int M, int N, int O, float *x, float *s1, float *y, float *s2, float *z, float *s3, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i <= size-7; i+=8) {

    x[i] += dt * s1[i]; x[i+1] += dt * s1[i+1];
    x[i+2] += dt * s1[i+2]; x[i+3] += dt * s1[i+3];
    x[i+4] += dt * s1[i+4]; x[i+5] += dt * s1[i+5];
    x[i+6] += dt * s1[i+6]; x[i+7] += dt * s1[i+7];

    y[i] += dt * s2[i]; y[i+1] += dt * s2[i+1];
    y[i+2] += dt * s2[i+2]; y[i+3] += dt * s2[i+3];
    y[i+4] += dt * s2[i+4]; y[i+5] += dt * s2[i+5];
    y[i+6] += dt * s2[i+6]; y[i+7] += dt * s2[i+7];

    z[i] += dt * s3[i]; z[i+1] += dt * s3[i+1];
    z[i+2] += dt * s3[i+2]; z[i+3] += dt * s3[i+3];
    z[i+4] += dt * s3[i+4]; z[i+5] += dt * s3[i+5];
    z[i+6] += dt * s3[i+6]; z[i+7] += dt * s3[i+7];
  }
}
                                                                                 
void advect_3(int M, int N, int O, int b1,int b2, int b3, float *d1, float *d2, float *d3, float *u, float *v, float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  for (int k = 1; k <= M; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= O-1; i+=2) {

        int curr = IX(i,j,k);
        float x = i - dtX * u[curr];   
        float y = j - dtY * v[curr];  // i
        float z = k - dtZ * w[curr];

        int curr2 = IX(i+1,j,k);
        float x2 = i+1 - dtX * u[curr2];   
        float y2 = j - dtY * v[curr2]; // i+1
        float z2 = k - dtZ * w[curr2];

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

        if (x2 < 0.5f)
          x2 = 0.5f;
        if (x2 > M + 0.5f)
          x2 = M + 0.5f;
        if (y2 < 0.5f)
          y2 = 0.5f;
        if (y2 > N + 0.5f)
          y2 = N + 0.5f;
        if (z2 < 0.5f)
          z2 = 0.5f;
        if (z2 > O + 0.5f)
          z2 = O + 0.5f;

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1; // i
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1; // i
        float u1 = z - k0, u0 = 1 - u1;

        int i02 = (int)x2, i12 = i02 + 1;
        int j02 = (int)y2, j12 = j02 + 1; // i+1
        int k02 = (int)z2, k12 = k02 + 1;

        float s12 = x2 - i02, s02 = 1 - s12;
        float t12 = y2 - j02, t02 = 1 - t12; // i+1
        float u12 = z2 - k02, u02 = 1 - u12;

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

        d1[curr2] =
            s02 * (t02 * (u02 * u[IX(i02, j02, k02)] + u12 * u[IX(i02, j02, k12)]) +
                  t12 * (u02 * u[IX(i02, j12, k02)] + u12 * u[IX(i02, j12, k12)])) + // i+1
            s12 * (t02 * (u02 * u[IX(i12 ,j02, k02)] + u12 * u[IX(i12, j02, k12)]) +
                  t12 * (u02 * u[IX(i12, j12, k02)] + u12 * u[IX(i12, j12, k12)]));

        d2[curr2] =
            s02 * (t02 * (u02 * v[IX(i02, j02, k02)] + u12 * v[IX(i02, j02, k12)]) +
                  t12 * (u02 * v[IX(i02, j12, k02)] + u12 * v[IX(i02, j12, k12)])) + // i+1
            s12 * (t02 * (u02 * v[IX(i12 ,j02, k02)] + u12 * v[IX(i12, j02, k12)]) +
                  t12 * (u02 * v[IX(i12, j12, k02)] + u12 * v[IX(i12, j12, k12)]));

        d3[curr2] =
            s02 * (t02 * (u02 * w[IX(i02, j02, k02)] + u12 * w[IX(i02, j02, k12)]) +
                  t12 * (u02 * w[IX(i02, j12, k02)] + u12 * w[IX(i02, j12, k12)])) + // i+1
            s12 * (t02 * (u02 * w[IX(i12 ,j02, k02)] + u12 * w[IX(i12, j02, k12)]) +
                  t12 * (u02 * w[IX(i12, j12, k02)] + u12 * w[IX(i12, j12, k12)]));
      }
    }
  }
  set_bnd(M, N, O, b1, d1);
  set_bnd(M, N, O, b2, d3);
  set_bnd(M, N, O, b3, d3);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt) {

  add_source_3(M, N, O, u, u0, v, v0, w, w0, dt);

  diffuse_3(M, N, O, 1, 2, 3, u0, u, v0, v, w0, w, visc, dt);

  project(M, N, O, u0, v0, w0, u, v);

  advect_3(M,N,O,1,2,3,u,v,w,u0,v0,w0,dt);

  project(M, N, O, u, v, w, u0, v0);
}
