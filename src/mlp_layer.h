#ifndef MLP_LAYER_H
#define MLP_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_mlp_layer(int batch, int input_size, int model_dim, int hidden_size, float dropout, ACTIVATION activation);
void forward_mlp_layer(layer l, network_state state);


#endif