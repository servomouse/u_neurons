#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "u_neurons.h"

#define CREATE

void get_values(float *ins, float *expects)
{
    // ins[0] = get_random();
    // ins[1] = get_random();

    // if(ins[0] > ins[1])
    //     expects[0] = 1;
    // else
    //     expects[0] = -1;
    static int temp = 0;
    if(temp == 0)
    {
        float base = rand()%10000;
        ins[0] = sin(base);
        printf("values: %f, ", base);
        base += 0.1;
        ins[1] = sin(base);
        printf("%f, ", base);
        base += 0.1;
        expects[0] = sin(base);
        printf("%f\n", base);
        temp = 1;
    }
    else
    {
        float base = rand()%10000;
        ins[0] = sin(base);
        base += 0.1;
        ins[1] = sin(base);
        base += 0.1;
        expects[0] = sin(base);
    }
}

float get_average_error(void *net, float *outs)
{
    float ins[2];
    float expects[1];
    float error = 0;
    int error_count = 0;
    for(int j=0; j<100000; j++)
    {
        get_values(ins, expects);

        get_outputs(ins, outs, net);
        error += pow(expects[0] - outs[0], 2);
        error_count ++;
    }
    return error / error_count;
}

void get_outputs_csv(void *net, float *outs)
{
    float ins[2];
    float expects[1];
    float error = 0;
    int error_count = 0;
    FILE *file = fopen("outputs.csv", "wb");
    for(float j=-3; j<3; j+=0.1)
    {
        ins[0] = sin(j);
        ins[1] = sin(j+0.1);
        expects[0] = sin(j+0.2);

        get_outputs(ins, outs, net);
        char str[128] = {0};
        int len = sprintf(str, "%f; %f; %f;\n", j, expects[0], outs[0]);
        fwrite(str, len, 1, file);
        // printf("%f; %f;\n", expects[0], outs[0]);
    }
    fclose(file);
}

void evolution(void * network)
{
    float outs[1];
    float error, prev_error;
    float min_error = get_average_error(network, outs);
    for(int i=0; i<20000001; i++)
    {
        prev_error = get_average_error(network, outs);
        change_random_weight(network);
        error = get_average_error(network, outs);
        if(error > prev_error)
        {
            undo_changes(network);
            continue;
        }
        printf("average error: %f\n", error);
        if(error < min_error)
        {
            store_network("network.dat", network);
            min_error = error;
        }

        if(error < 0.0001)
        {
            printf("train complete!\n");
            break;
        }
    }
    get_outputs_csv(network, outs);
}

#ifdef CREATE
int main(void)  // create, train and store
{
    srand((unsigned int)time(NULL));
    // void * net = create_network(5, 2, 32, 32, 16, 1);
    void * net = restore_network("network.dat");
    // evolution(net);
    // return 0;
    float ins[2];
    float outs[1];
    float expects[1];
    float error = 0;
    clear_network(net);
    // print_network(net);
    float min_error = get_average_error(net, outs);
    float train_coeff = 1;
    
    for(int i=0; i<20000001; i++)
    {
        get_values(ins, expects);

        train_network(net, ins, expects);

        if(i%1000 == 0)
        {
            if(i > 0)
            {
                update_network(net, 0.01);
                clear_network(net);
            }

            error = get_average_error(net, outs);
            printf("inputs: %f, %f,", ins[0], ins[1]);
            printf("\texpects: %f", expects[0]);
            printf("\toutput: %f,", outs[0]);
            printf("\taverage error: %f\n", error);
            if(error < min_error)
            {
                store_network("network.dat", net);
                // print_network(net);
                min_error = error;
            }
            // printf("output: %f,\taverage error: %f\n", outs[0], error/error_count);

            if(error < 0.0001)
            {
                printf("train complete!\n");
                break;
            }
        }
    }
    get_outputs_csv(net, outs);

    // store_network("network.dat", net);
    // delete_network(net);
    // net = restore_network("network.dat");

    // get_outputs(ins, outs, net);
    // error = pow(expects[0] - outs[0], 2);
    // printf("inputs: %f,  %f,\toutput = %f,\t error: %f\n", ins[0], ins[1], outs[0], error);
    
    return 0;
}

#else
int main(void)  // restore and test
{
    srand((unsigned int)time(NULL));
    void * net = restore_network("network.dat");
    float ins[2];
    float outs[1];
    float expects[1];
    float error = 0;
    int error_count = 0;
    for(int j=0; j<1000; j++)
    {
        get_values(ins, expects);

        get_outputs(ins, outs, net);
        error += pow(expects[0] - outs[0], 2);
        error_count ++;
    }
    printf("inputs: %f, %f,", ins[0], ins[1]);
    printf("\texpects: %f", expects[0]);
    printf("\toutput: %f,", outs[0]);
    printf("\taverage error: %f\n", error/error_count);
    delete_network(net);
    
    return 0;
}
#endif