#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "u_neurons.h"

#define CREATE

#ifdef CREATE
int main(void)  // create, train and store
{
    srand((unsigned int)time(NULL));
    // void * net = create_network(3, 2, 2, 1);
    void * net = restore_network("network.dat");
    float ins[2];
    float outs[1];
    float expects[1];
    float error = 0;
    clear_network(net);
    
    for(int i=0; i<200001; i++)
    {
        ins[0] = get_random();
        ins[1] = get_random();

        if(ins[0] > ins[1])
            expects[0] = 1;
        else
            expects[0] = -1;

        train_network(net, ins, expects);

        if(i%1000 == 0)
        {
            if(i> 0)
            {
                update_network(net, 0.1);
                clear_network(net);
            }

            error = 0;
            int error_count = 0;
            for(int j=0; j<1000; j++)
            {
                ins[0] = get_random();
                ins[1] = get_random();

                if(ins[0] > ins[1])
                    expects[0] = 1;
                else
                    expects[0] = -1;

                get_outputs(ins, outs, net);
                error += pow(expects[0] - outs[0], 2);
                error_count ++;
            }
            printf("output: %f,\taverage error: %f\n", outs[0], error/error_count);
        }
    }

    store_network("network.dat", net);
    delete_network(net);
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
    restore_network("network.dat");
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
        // train_network(ins, expects);
    }
    
    return 0;
}
#endif