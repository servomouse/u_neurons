#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>


// Define whether to use float or double
#define FLOAT float // double

FLOAT get_random(void);
void create_network(int n_layers, ...);
