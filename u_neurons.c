#include "u_neurons.h"
#include <math.h>
#include <stdarg.h>

#define ERROR ((int)-1)

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
    neuron_t ** network;
    FLOAT * outputs;
}network_t;

network_t * net = NULL;

typedef struct _packed_weigth_t
{
    int index; 
    FLOAT weight;
}packed_weigth_t;

typedef struct _packed_neuron_t
{
    int size;
    int num_of_inputs;
    FLOAT bias;
    packed_weigth_t inputs[0];    // the length of the array is num_of_inputs
}packed_neuron_t;

typedef struct _packed_net_t
{
    int total_size;
    int num_of_inputs;          // number of the dummy input neurons
    int num_of_neurons;         // total number of the useful neurons in the network
    int num_of_outputs;         // number of the neurons in the output layer
    packed_neuron_t net[0];     // the length of the array is num_of_neurons
}packed_net_t;

static neuron_t * create_neuron(int num_of_inputs, int first_input_index)
{
    neuron_t *n = (neuron_t *)malloc(sizeof(neuron_t));
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
    return n;
}

static neuron_t * restore_neuron(FLOAT bias, int num_of_inputs, packed_weigth_t *inputs)
{
    neuron_t *n = (neuron_t *)malloc(sizeof(neuron_t));
    n->num_of_inputs = num_of_inputs;
    n->inputs = (int*)malloc(sizeof(int)*num_of_inputs);
    n->i_values = (FLOAT*)malloc(sizeof(FLOAT)*num_of_inputs);
    n->weights = (FLOAT*)malloc(sizeof(FLOAT)*num_of_inputs);
    n->w_deltas = (FLOAT*)malloc(sizeof(FLOAT)*num_of_inputs);

    for(int i=0; i<num_of_inputs; i++)
    {
        n->weights[i] = inputs[i].weight;
        n->inputs[i] = inputs[i].index;
    }
    n->bias = bias;
    return n;
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

    int layers[128] = {[0 ... 127] = -1};

    va_list ptr;
    va_start(ptr, n_layers);
    for (int i = 0; i < n_layers; i++)
    {
        layers[i] = va_arg(ptr, int);
        net->n_neurons += layers[i];
        net->n_outputs = layers[i];
    }
    net->n_inputs = layers[0];
    va_end(ptr);
    printf("%d, %d, %d\n", layers[0], layers[1], layers[2]);
    net->network = (neuron_t**)malloc(sizeof(neuron_t*) * net->n_neurons);
    net->outputs = (FLOAT*)malloc(sizeof(FLOAT) * net->n_outputs);
    int index = 0, pointer = 0;
    for(int i=0; i<layers[0]; i++)
        net->network[pointer++] = create_neuron(0, 0);
    for(int i=1; i<n_layers; i++)
    {
        for(int j=0; j<layers[i]; j++)
            net->network[pointer++] = create_neuron(layers[i-1], index);
        index += layers[i-1];
    }
    printf("%d, %d, %d\n", net->n_inputs, net->n_outputs, net->n_neurons);
}

// params: inputs - input data;
// outputs - expected output data
void train_network(FLOAT * inputs, FLOAT * outputs, FLOAT coefficient)
{
    for(int i=0; i<net->n_inputs; i++)  // set inputs values
        net->network[i]->output = inputs[i];
    for(int i=net->n_inputs; i<net->n_neurons; i++)   // calculate output value for each neuron
        get_output(net->network[i], net->network);
    int offset = net->n_neurons-net->n_outputs;
    for(int i=0; i<net->n_outputs; i++) // set errors for output neurons
        net->network[offset+i]->error = outputs[i] - net->network[offset+i]->output;
    for(int i=net->n_neurons-1; i>=net->n_inputs; i--)
        teach_neuron(net->network[i], net->network, coefficient);
}

// outputs - an array for the output data
void get_outputs(FLOAT * inputs, FLOAT * outputs)
{
    for(int i=0; i<net->n_inputs; i++)  // set inputs values
        net->network[i]->output = inputs[i];
    for(int i=net->n_inputs; i<net->n_neurons; i++)   // calculate output value for each neuron
        get_output(net->network[i], net->network);
    int offset = net->n_neurons-net->n_outputs;
    for(int i=0; i<net->n_outputs; i++) // set errors for output neurons
        outputs[i] = net->network[offset+i]->output;
}

// p = (int*)((size_t)p + 4);
int store_network(char * filename)
{
    // calculate the required size
    int size = sizeof(packed_net_t);
    size += sizeof(packed_neuron_t) * (net->n_neurons - net->n_inputs);
    for(int i=net->n_inputs; i<net->n_neurons; i++)
        size += net->network[i]->num_of_inputs * (sizeof(int) + sizeof(FLOAT));
    
    // pack the network:
    packed_net_t *zip = (packed_net_t *)malloc(size);
    zip->total_size = size;
    zip->num_of_neurons = net->n_neurons - net->n_inputs;
    zip->num_of_inputs = net->n_inputs;
    zip->num_of_outputs = net->n_outputs;
    packed_neuron_t *current = &zip->net[0];
    for(int i=0; i<zip->num_of_neurons; i++)
    {
        int n_inputs = net->network[net->n_inputs+i]->num_of_inputs;
        int temp_size = sizeof(int) + sizeof(FLOAT);
        temp_size *= n_inputs;
        temp_size += sizeof(packed_neuron_t);
        current->size =  temp_size;
        current->num_of_inputs = n_inputs;
        current->bias = net->network[net->n_inputs+i]->bias;
        for(int j=0; j<n_inputs; j++)
        {
            current->inputs->index = net->network[net->n_inputs+i]->inputs[j];
            current->inputs->weight = net->network[net->n_inputs+i]->weights[j];
        }
        current = (packed_neuron_t*)((size_t)current + temp_size);
    }

    FILE *file = fopen(filename, "wb");
    fwrite(zip, size, 1, file);
    return 0;
}

int restore_network(char * filename)
{
    packed_net_t meta;    // network metadata
    FILE *file = fopen(filename, "rb");
    fread(&meta, sizeof(meta), 1, file);
    // allocate space for th network
    packed_neuron_t * arr = (packed_neuron_t *)malloc(meta.total_size);
    if(NULL == arr)
        return ERROR;

    fread(arr, meta.total_size, 1, file);    // read the compressed network

    if(net != NULL) // check if the network is already allocated
        return ERROR;
    net = (network_t*)calloc(sizeof(network_t), 1);
    net->n_inputs = meta.num_of_inputs;
    net->n_outputs = meta.num_of_outputs;
    net->n_neurons = meta.num_of_inputs + meta.num_of_neurons;
    printf("%d, %d, %d\n", net->n_inputs, net->n_outputs, net->n_neurons);
    net->network = (neuron_t**)malloc(sizeof(neuron_t*) * net->n_neurons);
    net->outputs = (FLOAT*)malloc(sizeof(FLOAT) * net->n_outputs);
    int i = 0;
    for(i; i<net->n_inputs; i++)    // create input neurons
        net->network[i] = create_neuron(0, 0);

    packed_neuron_t *current = &meta.net[0];
    for(i; i<net->n_neurons; i++)   // restore useful neurons
    {
        net->network[i] = restore_neuron(current->bias, current->num_of_inputs, current->inputs);
        current = (packed_neuron_t*)((size_t)current + current->size);
    }

    return 0;
}