#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "u_neurons.h"

int main(void)
{
    srand((unsigned int)time(NULL));
    create_network(2, 2, 2, 1);    // 2 layers: 2 inputs and 1 output neuron
    float ins[2];
    float outs[1];
    float expects[1];
    float error = 0;
    for(int i=1; i<10000; i++)
    {
        ins[0] = get_random();
        ins[1] = get_random();
        if(ins[0] > ins[1])
            expects[0] = 1;
        else
            expects[0] = 0;
        get_outputs(ins, outs);
        error += pow(expects[0] - outs[0], 2);
        if(i%1000 == 0)
        {
            printf("error = %f\n", error/1000);
            error = 0;
        }
        train_network(ins, expects, 0.01);
    }
    // for(int i=0; i<10; i++)
    // {
    //     ins[0] = get_random();
    //     ins[1] = get_random();
    //     get_outputs(ins, outs);
    //     printf("inputs: %f, %f;\t output: %f\n", ins[0], ins[1], outs[0]);
    // }
    
    return 0;
}