#ifndef INTEGRAL_DUPLA_H
#define INTEGRAL_DUPLA_H

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define F(x, y) sin(x*x + y*y)

double integral_dupla_omp(int x_intervalos, int y_intervalos);
void salvar_tempos(double (*tempos)[3][3], int cores_count, int intervalos_count);

#endif //INTEGRAL_DUPLA_H
