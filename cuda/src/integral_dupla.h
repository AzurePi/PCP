#ifndef INTEGRAL_DUPLA_H
#define INTEGRAL_DUPLA_H

#include <cuda_runtime.h>
#include <cstdio>
#include <unistd.h>
#include <math.h>

#define F(x, y) sin(x*x + y*y)

__global__ void integral_kernel(int x_intervalos, int y_intervalos, double h_x, double h_y, double *d_resultado);
double integral_dupla_cuda(int n_blocos, int x_intervalos, int y_intervalos);
void salvar_tempos(int cores, int x_intervalos, int y_intervalos, double tempo_medio);

#endif //INTEGRAL_DUPLA_H
