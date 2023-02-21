#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// Define whether to use float or double
#define FLOAT float
// #define FLOAT double

// returns random value in range [-0.99; 0.99]
FLOAT get_random(void);

void * create_network(int n_layers, ...);
void delete_network(void * net);


void clear_network(void *network);
void train_network(void *network, FLOAT * inputs, FLOAT * outputs);
void update_network(void *network, FLOAT coefficient);

// calculate the outputs array for the provied inputs array
void get_outputs(FLOAT * inputs, FLOAT * outputs, void *network);

void store_network(char * filename, void * n);

void * restore_network(char * filename);


