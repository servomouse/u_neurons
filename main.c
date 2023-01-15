#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "u_neurons.h"

int main(void)
{
    srand((unsigned int)time(NULL));
    create_network(2, 1, 2);
    for(int i=0; i<10; i++)
        printf("%f\n", get_random());
    
    return 0;
}