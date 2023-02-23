#include "u_neurons.h"
#include <math.h>
#include <stdarg.h>

#define ERROR ((int)-1)

FLOAT get_random(void)  // returns random value in range [-0.99; 0.99]
{
    return 0.99 * sin(rand());
}

#ifdef NEW

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
int train_counter = 0;

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

static void train_neuron(neuron_t *n, neuron_t **net)
{
    n->b_delta += n->error * (1 - pow(tanh(n->sum), 2));    // get bias delta
    
    for(int i=0; i<n->num_of_inputs; i++)
    {
        n->w_deltas[i] += n->b_delta * n->i_values[i];
        add_error(net[n->inputs[i]], n->b_delta * n->weights[i]);   // set error for each connected neuron
        // n->weights[i] += coefficient * (n->b_delta * n->i_values[i]);   // get delta for each weight
    }
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

static void update_neuron(neuron_t *n, FLOAT coefficient)
{
    n->bias += coefficient * (n->b_delta / train_counter);
    
    for(int i=0; i<n->num_of_inputs; i++)
    {
        n->weights[i] += coefficient * (n->w_deltas[i] / train_counter);   // get delta for each weight
    }
}

void update_network(FLOAT coefficient)
{
    for(int i=net->n_neurons-1; i>=net->n_inputs; i--)
        update_neuron(net->network[i], coefficient);
}

static void clear_delta(neuron_t *n, neuron_t **net)
{
    n->b_delta = 0;    // get bias delta
    
    for(int i=0; i<n->num_of_inputs; i++)
        n->w_deltas[i] = 0;
}

void clear_deltas(void)
{
    train_counter = 0;
    for(int i=net->n_neurons-1; i>=net->n_inputs; i--)
        clear_delta(net->network[i], net->network);
}

// params: inputs - input data;
// outputs - expected output data
void train_network(FLOAT * inputs, FLOAT * outputs)
{
    for(int i=0; i<net->n_inputs; i++)  // set inputs values
        net->network[i]->output = inputs[i];
    for(int i=net->n_inputs; i<net->n_neurons; i++)   // calculate output value for each neuron
        get_output(net->network[i], net->network);
    int offset = net->n_neurons-net->n_outputs;
    for(int i=0; i<net->n_outputs; i++) // set errors for output neurons
        net->network[offset+i]->error = outputs[i] - net->network[offset+i]->output;
    for(int i=net->n_neurons-1; i>=net->n_inputs; i--)
        train_neuron(net->network[i], net->network);
    train_counter++;
}

// outputs - an array for the output data
void get_outputs(FLOAT * inputs, FLOAT * outputs)
{
    for(int i=0; i<net->n_inputs; i++)  // set inputs values
        net->network[i]->output = inputs[i];
    for(int i=net->n_inputs; i<net->n_neurons; i++)   // calculate output value for each neuron
        get_output(net->network[i], net->network);
    int offset = net->n_neurons - net->n_outputs;
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

#else

#define MAX_NEURON_INPUTS_NUMBER    100 // how many inputs each neuron can has

typedef struct __attribute__((packed))
{
    int index;      // input index in the neurons table
    FLOAT value; 
    FLOAT weight; 
    FLOAT w_error;
}input_t;

typedef struct __attribute__((packed))
{
    FLOAT bias;     // the bias itself
    FLOAT b_error;  // bias error
    FLOAT sum;      // bias + sum(inpit[i] * weight[i] for i in [offset ... offset+n_inputs])
    FLOAT error;
    FLOAT output;
    input_t inputs[MAX_NEURON_INPUTS_NUMBER];
}neuron_t;

typedef struct __attribute__((packed))
{
    int size;           // total size in bytes, need to correctly store and restore network
    int n_inputs;       // number of input neurons
    int n_outputs;      // size of the outputs array
    int n_neurons;      // total amount of neurons in the network
    int train_counter;
    neuron_t neuron[0];
}network_t;

void create_weight(int source_neuron, int dest_neuron, void * network)
{
    network_t *net = (network_t*)network;
    int index;
    for(int i=0; i<MAX_NEURON_INPUTS_NUMBER; i++)
    {
        if(-1 == net->neuron[dest_neuron].inputs[source_neuron].index)
        {
            index = i;
            break;
        }
    }
    if(index < MAX_NEURON_INPUTS_NUMBER)
    {
        net->neuron[dest_neuron].inputs[index].weight = 0.1 * get_random();
        net->neuron[dest_neuron].inputs[index].index = dest_neuron;
    }
}

void *create_network(int n_layers, ...)
{
    // get the structure of the network
    int layers[128] = {[0 ... 127] = -1};
    int n_total = 0, inputs_arr_size = 0;

    va_list ptr;
    va_start(ptr, n_layers);
    for (int i = 0; i < n_layers; i++)
    {
        layers[i] = va_arg(ptr, int);
        n_total += layers[i];
        if(i>0)
            inputs_arr_size += layers[i] * layers[i-1];
    }
    va_end(ptr);

    // print network info
    printf("network structure:\n");
    printf("layer 0: \t %d inputs\n", layers[0]);
    for(int i=1; i<n_layers-1; i++)
        printf("layer %d: \t %d neurons\n", i, layers[i]);
    printf("output layer: \t %d outputs\n", layers[n_layers-1]);

    // create network
    int net_size = sizeof(network_t) + n_total * sizeof(neuron_t);
    network_t *net = (network_t*)calloc(net_size, 1);

    net->size = net_size;
    net->n_inputs = layers[0];
    net->n_neurons = n_total;
    net->n_outputs = layers[n_layers-1];
    for(int i=0; i<n_total; i++)    // mark all the indexes as unused
    {
        for(int j=0; j<MAX_NEURON_INPUTS_NUMBER; j++)
            net->neuron[i].inputs[j].index = -1;
    }
    
    // create the rest of the network
    int neuron = layers[0];     // the first neuron in the current layer
    int first_input = 0;        // index of the first source neuron
    for(int i=1; i<n_layers; i++)
    {
        for(int dest=0; dest<layers[i]; dest++)
        {
            for(int src=0; src<layers[i-1]; src++)
            {
                create_weight(neuron+dest, first_input+src, net);
            }
        }
        first_input = neuron;
        neuron = layers[i];
    }
    printf("network created\n");
    return net;
}

void delete_network(void * net)
{
    free(net);
}

void store_network(char * filename, void * n)
{
    network_t *net = (network_t*)n;
    FILE *file = fopen(filename, "wb");
    fwrite(net, net->size, 1, file);
    fclose(file);
}

void * restore_network(char * filename)
{
    int net_size;
    FILE *file = fopen(filename, "rb");
    fread(&net_size, sizeof(net_size), 1, file);
    // allocate space for the network
    network_t *net = (network_t*)malloc(net_size);
    rewind(file);
    fread(net, net_size, 1, file);
    fclose(file);

    return net;
}

static FLOAT get_output(neuron_t *n, network_t *net)
{
    n->sum = 0;
    n->output = 0;
    n->error = 0;
    input_t *inputs = &net->inputs[n->inputs_off];
    for(int i=0; i<n->n_inputs; i++)
    {
        inputs[i].value = net->neuron[inputs[i].index].output;
        n->sum += inputs[i].value * inputs[i].weight;
    }
    n->sum += n->bias;
    n->output = tanh(n->sum);
    return n->output;
}

// outputs - an array for the output data
void get_outputs(FLOAT * inputs, FLOAT * outputs, void *network)
{
    network_t *n = (network_t*)network;
    for(int i=0; i<n->n_inputs; i++)  // set inputs values
        n->neuron[i].output = inputs[i];
    for(int i=n->n_inputs; i<n->n_neurons; i++)   // calculate output value for each neuron
        get_output(&n->neuron[i], n);
    int offset = n->n_neurons - n->n_outputs;
    for(int i=0; i<n->n_outputs; i++) // set errors for output neurons
        outputs[i] = n->neuron[offset+i].output;
}

void update_network(void *network, FLOAT coefficient)
{
    network_t *net = (network_t*)network;
    neuron_t * neuron;
    input_t * inputs;
    for(int i=net->n_neurons-1; i>=net->n_inputs; i--)
    {
        neuron = &net->neuron[i];
        inputs = &net->inputs[net->neuron[i].inputs_off];

        neuron->bias -= coefficient * (neuron->b_error / net->train_counter);
        printf("neuron %d weights: ", i);
        for(int j=0; j<neuron->n_inputs; j++)
        {
            inputs[j].weight -= coefficient * (inputs[j].w_error / net->train_counter);
            printf("%f, ", inputs[j].weight);
        }
        printf("\n");
    }
}

static void train_neuron(neuron_t *n, network_t *net)
{
    n->error *= (1 - pow(tanh(n->sum), 2));    // get bias error
    n->b_error += n->error;
    input_t * inputs = &net->inputs[n->inputs_off];
    
    for(int i=0; i<n->n_inputs; i++)
    {
        inputs[i].w_error += n->error * inputs[i].value;
        net->neuron[inputs[i].index].error += n->error * inputs[i].weight;   // set error for each connected neuron
    }
}

// params: inputs - input data;
// outputs - expected output data
void train_network(void *network, FLOAT * inputs, FLOAT * outputs)
{
    network_t *net = (network_t*)network;
    // set inputs values
    for(int i=0; i<net->n_inputs; i++)
        net->neuron[i].output = inputs[i];
    // calculate output value for each neuron
    for(int i=net->n_inputs; i<net->n_neurons; i++)
        get_output(&net->neuron[i], net);
    
    for(int i=0; i<net->n_neurons; i++)
        net->neuron[i].error = 0;   // set errors to 0

    // set errors for output neurons
    int offset = net->n_neurons - net->n_outputs;
    for(int i=0; i<net->n_outputs; i++)
        net->neuron[offset+i].error = 2 * (net->neuron[offset+i].output - outputs[i]);
    for(int i=net->n_neurons-1; i>=net->n_inputs; i--)
        train_neuron(&net->neuron[i], net);
    net->train_counter++;
}

// prepare network to training
void clear_network(void *network)
{
    network_t *net = (network_t*)network;
    for(int i=net->n_neurons-1; i>=net->n_inputs; i--)
    {
        net->neuron[i].b_error = 0;
        net->neuron[i].sum = 0;
        net->neuron[i].output = 0;
        input_t * inputs = &net->inputs[net->neuron[i].inputs_off];
        
        for(int i=0; i<net->neuron[i].n_inputs; i++)
            inputs[i].w_error = 0;  // weight error = 0
    }
    net->train_counter = 0;
}

#endif