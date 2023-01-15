#include "u_neurons.h"
#include <math.h>
#include <stdarg.h>

FLOAT get_random(void)  // returns random value in range [-0.99; 0.99]
{
    return 0.99 * sin(rand());
}

typedef struct _neuron_t
{
    int num_of_inputs;
    int * inputs;
    FLOAT * i_values;   // input values
    FLOAT * weights;
    FLOAT * w_deltas;
    FLOAT bias;
    FLOAT b_delta;
    FLOAT sum;
    FLOAT output;
    FLOAT error;
}neuron_t;

typedef struct _network_t
{
    int n_inputs;
    int n_outputs;
    int n_neurons;
    neuron_t * network;
    FLOAT * outputs;
}network_t;

network_t * net = NULL;

static void create_neuron(neuron_t *n, int num_of_inputs, int first_input_index)
{
    n = (neuron_t *)malloc(sizeof(neuron_t));
    n->num_of_inputs = num_of_inputs;
    n->inputs = (int*)malloc(sizeof(int)*num_of_inputs);
    n->i_values = (FLOAT*)malloc(sizeof(FLOAT)*num_of_inputs);
    n->weights = (FLOAT*)malloc(sizeof(FLOAT)*num_of_inputs);
    n->w_deltas = (FLOAT*)malloc(sizeof(FLOAT)*num_of_inputs);

    for(int i=0; i<num_of_inputs; i++)
    {
        n->weights[i] = get_random();    // Init weights with random values
        n->inputs[i] = first_input_index + i;   // set inputs indexes
    }
    n->bias = get_random();
}

static void delete_neuron(neuron_t * n)
{
    free(n->inputs);
    free(n->i_values);
    free(n->weights);
    free(n->w_deltas);
}

static FLOAT get_output(neuron_t *n, neuron_t **net)
{
    n->b_delta = 0;
    n->sum = 0;
    n->output = 0;
    n->error = 0;
    for(int i=0; i<n->num_of_inputs; i++)
    {
        n->i_values[i] = net[n->inputs[i]]->output;
        n->sum += net[n->inputs[i]]->output * n->weights[i];
    }
    n->sum += n->bias;
    n->output = tanh(n->sum);
    return n->output;
}

static void set_output(neuron_t *n, FLOAT value)
{
    n->output = value;
}

static void add_error(neuron_t *n, FLOAT error)
{
    n->error += error;
}

static void teach_neuron(neuron_t *n, neuron_t **net, FLOAT coefficient)
{
    n->b_delta = n->error * (1 - pow(tanh(n->sum), 2));    // get bias delta
    
    for(int i=0; i<n->num_of_inputs; i++)
    {
        // n->w_deltas[i] = n->b_delta * n->i_values[i];
        add_error(net[n->inputs[i]], n->b_delta * n->weights[i]);   // set error for each connected neuron
        n->weights[i] += coefficient * (n->b_delta * n->i_values[i]);   // get delta for each weight
    }
    n->bias += n->b_delta;
}

void create_network(int n_layers, ...)
{
    if(net != NULL)
        return;
    net = (network_t*)calloc(sizeof(network_t), 1);

    int layers[128] = {[0 ... 63] = -1, [64 ... 127] = 8};

    va_list ptr;
    va_start(ptr, n_layers);
    for (int i = 0; i < n_layers; i++)
    {
        layers[i] = va_arg(ptr, int);
        net->n_neurons += layers[i];
    }
    va_end(ptr);


}