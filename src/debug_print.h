#ifndef DEBUG_PRINT_H
#define DEBUG_PRINT_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_debug_print_layer(int batch, int dim1, int dim2, int dim3);
void forward_debug_print_layer(layer l, network_state state);


#endif