#include "debug_print.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define OUTPUT 0
#define DEBUG 0

layer make_debug_print_layer(int batch, int dim1, int dim2, int dim3)
{
    fprintf(stderr, "print: batch size %d, dim1 %d,  dim2 %d, dim3 %d\n", batch, dim1, dim2, dim3);
    layer layer = { (LAYER_TYPE)0 };
    layer.batch = batch;
    layer.type = DEBUG_PRINT;
    layer.c = dim1;
    layer.h = dim2;
    layer.w = dim3;

    layer.forward = forward_debug_print_layer;
    return layer;
}


void forward_debug_print_layer(layer l, network_state state)
{
    printf("The shape is %d x %d x %d X %d \n", l.batch, l.c, l.h, l.w);
    for(int bi = 0; bi < l.batch; bi++) {
        for(int d1 = 0; d1 < l.c; d1++) {
            for(int d2 = 0; d2 < l.h; d2++) {
                for(int d3 = 0; d3 < l.w; d3++) {
                    printf("layer input[%d][%d][%d][%d] = %f\n", bi, d1, d2, d3, state.input[bi*l.c*l.h*l.w + d1*l.h*l.w + d2*l.w + d3]);
                }
            }
        }
    }
printf("Finish.\n\n");
    
}

