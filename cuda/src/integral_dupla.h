#ifndef INTEGRAL_DUPLA_H
#define INTEGRAL_DUPLA_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#define F(x, y) sin(x*x + y*y)
#define THREADS_POR_BLOCO 512

float cpuSecond();
__global__ void integral_dupla_cuda(float *resultado, float h_x, float h_d, int x_intervalos, int y_intervalos,
                                    float limite_inf);
void salvar_tempos(int blocos, int x_intervalos, int y_intervalos, double tempo_medio);

#endif //INTEGRAL_DUPLA_H
