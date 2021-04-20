#ifndef POSITIONAL_EMBEDDING_LAYER_H
#define POSITIONAL_EMBEDDING_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_positional_embedding_layer(int batch, int input_size, int model_dim);
void forward_positional_embedding_layer(layer l, network_state state);


// void test_positional_embedding_layer();

// void backward_positional_embedding_layer(const layer l, network_state state);

// void update_positional_embedding_layer(layer l, int batch, float learning_rate, float momentum, float decay);

#endif