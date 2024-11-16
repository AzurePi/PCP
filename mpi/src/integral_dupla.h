#ifndef INTEGRAL_DUPLA_H
#define INTEGRAL_DUPLA_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#define F(x, y) sin(x*x + y*y)

double integral_dupla_mpi(int x_intervalos, int y_intervalos);
void salvar_tempos(int cores, int x_intervalos, int y_intervalos, double tempo_medio);

#endif //INTEGRAL_DUPLA_H
