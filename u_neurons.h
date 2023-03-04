#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

// Define whether to use float or double
#define FLOAT float
// #define FLOAT double

#define WEIGHT  1
#define BIAS    2

typedef struct
{
    int type;
    int neuron;
    int weight;
    FLOAT prev_value;
}pre_val_t;

// returns random value in range [-0.99; 0.99]
FLOAT get_random(void);

void * create_network(int n_layers, ...);
void * copy_network(void * network);
void delete_network(void * net);


void clear_network(void *network);
void train_network(void *network, FLOAT * inputs, FLOAT * outputs);
void update_network(void *network, FLOAT coefficient);
void print_network(void *network);

// evolution funcs:
void change_random_weight(void *network, pre_val_t * prev_value);
void undo_changes(void * network, pre_val_t * prev_value);

// calculate the outputs array for the provied inputs array
void get_outputs(FLOAT * inputs, FLOAT * outputs, void *network);

void store_network(char * filename, void * network);

void * restore_network(char * filename);


