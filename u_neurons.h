#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// Define whether to use float or double
#define FLOAT float
// #define FLOAT double

// returns random value in range [-0.99; 0.99]
FLOAT get_random(void);

void create_network(int n_layers, ...);


void train_network(FLOAT * inputs, FLOAT * outputs, FLOAT coefficient);

// calculate the outputs array for the provied inputs array
void get_outputs(FLOAT * inputs, FLOAT * outputs);

 // returns either -1 in case of error or 0 if the network stored successfully
int store_network(char * filename);

 // returns either -1 in case of error or 0 if the network restored successfully
int restore_network(char * filename);


