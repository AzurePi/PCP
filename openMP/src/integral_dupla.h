#ifndef INTEGRAL_DUPLA_H
#define INTEGRAL_DUPLA_H

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define F(x, y) sin(x*x + y*y)

double integral_dupla_omp(int n_threads, int x_intervalos, int y_intervalos);
void salvar_tempos(double (*tempos)[3][3], int threads_count, int intervalos_count);

#endif //INTEGRAL_DUPLA_H
