#include "u_neurons.h"
#include <math.h>
#include <stdarg.h>

#define ERROR ((int)-1)

FLOAT get_random(void)  // returns random value in range [-0.99; 0.99]
{
    return 0.99 * sin(rand());
}

int get_random_from_interval(int a1, int a2)
{
    int temp = a2 - a1;
    return a1 + (rand()%temp);
}

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
    int inputs_array_size;
    int train_counter;
    neuron_t neuron[0];
}network_t;

int neuron = 0;
int weight = 0;
FLOAT prev_value = 0;

int pick_neuron(network_t *net)
{
    return get_random_from_interval(net->n_inputs, net->n_neurons);
}

int pick_weight(network_t *net, neuron_t *n)
{
    int num_of_weights = 0;
    for(int i=0; i<net->inputs_array_size; i++)
    {
        if(n->inputs[i].index != -1)
            num_of_weights++;
        else
            break;
    }
    return get_random_from_interval(0, num_of_weights);
}

void change_random_weight(void *network)
{
    network_t *net = (network_t*)network;
    neuron = pick_neuron(net);
    weight = pick_weight(net, &net->neuron[neuron]);
    prev_value = net->neuron[neuron].inputs[weight].weight;
    net->neuron[neuron].inputs[weight].weight += tanh(prev_value + 0.1 * get_random());
}

void undo_changes(void * network)
{
    network_t *net = (network_t*)network;
    net->neuron[neuron].inputs[weight].weight = prev_value;
}

void create_weight(int source_neuron, int dest_neuron, void * network)
{
    network_t *net = (network_t*)network;
    int index;
    for(int i=0; i<MAX_NEURON_INPUTS_NUMBER; i++)
    {
        if(-1 == net->neuron[dest_neuron].inputs[i].index)
        {
            index = i;
            break;
        }
    }
    if(index < MAX_NEURON_INPUTS_NUMBER)
    {
        net->neuron[dest_neuron].inputs[index].weight = 0.1 * get_random();
        net->neuron[dest_neuron].inputs[index].index = source_neuron;
    }
    printf("created_weight src: %d, dst: %d, index:%d, weight: %f\n", source_neuron, dest_neuron, index, net->neuron[dest_neuron].inputs[index].weight);
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
    net->inputs_array_size = MAX_NEURON_INPUTS_NUMBER;
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
                create_weight(first_input+src, neuron+dest, net);
            }
            net->neuron[i].bias = 0.1 * get_random();
            printf("bias: %f\n", net->neuron[i].bias);
        }
        first_input = neuron;
        neuron += layers[i];
    }
    printf("network created\n");
    return net;
}

// free memory previously allocated for network
void delete_network(void * net)
{
    free(net);
}

// store network to file
void store_network(char * filename, void * n)
{
    network_t *net = (network_t*)n;
    FILE *file = fopen(filename, "wb");
    fwrite(net, net->size, 1, file);
    fclose(file);
}

// restore network from file
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

// calculate the output for the specific neuron
static FLOAT get_output(neuron_t *n, network_t *net)
{
    n->sum = 0;
    n->output = 0;
    for(int i=0; i<net->inputs_array_size; i++)
    {
        if(-1 == n->inputs[i].index)
            break;
        n->inputs[i].value = net->neuron[n->inputs[i].index].output;
        n->sum += n->inputs[i].value * n->inputs[i].weight;
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

void print_network(void *network)
{
    network_t *net = (network_t*)network;
    for(int i=net->n_inputs; i<net->n_neurons; i++)
    {
        printf("neuron %d:\tweights: ", i);
        for(int j=0; j<net->inputs_array_size; j++)
        {
            if(-1 == net->neuron[i].inputs[j].index)
                break;
            printf("\t%d: %f ", j, net->neuron[i].inputs[j].weight);
        }
        printf("bias: %f\n", net->neuron[i].bias);
    }
}

void print_weights(network_t * net)
{
    for(int i=0; i<net->n_neurons; i++)
    {
        printf("neuron %d weights: ", i);
        for(int j=0; j<net->inputs_array_size; j++)
        {
            if(-1 == net->neuron[i].inputs[j].index)
                break;
            printf("\t%d: %f ", j, net->neuron[i].inputs[j].weight);
        }
        printf("\n");
    }
}

void update_network(void *network, FLOAT coefficient)
{
    network_t *net = (network_t*)network;
    neuron_t * neuron;
    input_t * inputs;
    for(int i=net->n_inputs; i<net->n_neurons; i++)
    {
        neuron = &net->neuron[i];

        neuron->bias += coefficient * (neuron->b_error / net->train_counter);
        neuron->bias = tanh(neuron->bias);
        inputs = net->neuron[i].inputs;
        for(int j=0; j<net->inputs_array_size; j++)
        {
            if(-1 == inputs[j].index)
                break;
            inputs[j].weight += coefficient * (inputs[j].w_error / net->train_counter);
            inputs[j].weight = tanh(inputs[j].weight);
        }
    }
    // print_weights(net);
}

static void train_neuron(neuron_t *n, network_t *net)
{
    n->error *= (1 - pow(tanh(n->sum), 2));    // get bias error
    n->b_error += n->error;
    for(int i=0; i<net->inputs_array_size; i++)
    {
        if(-1 == n->inputs[i].index)
            break;
        n->inputs[i].w_error += n->error * n->inputs[i].value;

        // set error for each connected neuron
        net->neuron[n->inputs[i].index].error += n->error * n->inputs[i].weight;
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
        net->neuron[offset+i].error = 2 * (outputs[i] - net->neuron[offset+i].output);
    for(int i=net->n_neurons-1; i>=net->n_inputs; i--)
        train_neuron(&net->neuron[i], net);
    net->train_counter++;
}

// prepare network to training
void clear_network(void *network)
{
    network_t *net = (network_t*)network;
    for(int i=0; i<net->n_neurons; i++)
    {
        net->neuron[i].b_error = 0;
        net->neuron[i].sum = 0;
        net->neuron[i].output = 0;
        net->neuron[i].error = 0;
        
        for(int j=0; j<net->inputs_array_size; j++)
        {
            if(-1 == net->neuron[i].inputs[i].index)
                break;
            net->neuron[i].inputs[i].w_error = 0;  // weight error = 0
        }
    }
    net->train_counter = 0;
}