#ifndef FLATTEN_H
#define FLATTEN_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_flatten_layer(int batch, int model_dim, int gh, int gw);
void forward_flatten_layer(layer l, network_state state);


// void test_positional_embedding_layer();

// void backward_positional_embedding_layer(const layer l, network_state state);

// void update_positional_embedding_layer(layer l, int batch, float learning_rate, float momentum, float decay);

#endif