#include "positional_embedding.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define OUTPUT 0
#define DEBUG 0

layer make_positional_embedding_layer(int batch, int input_size, int model_dim)
{
    // input dim [batch, input_size (sequence_length), model_dim]
    fprintf(stderr, "positional embedding layer: batch size %d, input size %d,  model dim %d\n", batch, input_size, model_dim);
    layer layer = { (LAYER_TYPE)0 };
    layer.type =  POSITIONAL_EMBEDDING;
    layer.batch = batch;
    layer.input_size = input_size;
    layer.model_dim = model_dim;


    layer.output = (float*)xcalloc(batch*input_size * model_dim , sizeof(float));
    layer.weights = (float*)xcalloc(batch*input_size * model_dim , sizeof(float));
    memset(layer.weights, 0, batch*input_size*model_dim*sizeof(float));
    layer.inputs = input_size;
    layer.outputs = layer.inputs;
    layer.out_c = layer.c = 576;
    layer.out_h = layer.h = 12;
    layer.out_w = layer.w = 64;


    layer.forward = forward_positional_embedding_layer;
    // layer.backward = backward_layernorm_layer;
    // layer.update = update_layernorm_layer;

    return layer;
}


void forward_positional_embedding_layer(layer l, network_state state)
{
    int model_dim = state.net.model_dim;
    int input_size = l.input_size;
    int batch = l.batch;
   
    // l.out_c = l.c = 576;
    // l.out_h = l.h = 12;
    // l.out_w = l.w = 64;
    printf("positional_embedding, l.out_c=%d, l.c=%d, state.net.c=%d\n",  l.out_c, l.c, state.net.c);
    // printf("positional_embedding, state.input[0]= %f, state.input[12]=%f\n", state.input[0], state.input[12]);
    
    for(int bi = 0; bi < batch; bi++) {
        for(int ii = 0; ii < input_size; ii++) {
            for(int di = 0; di < model_dim; di++) {
                l.output[bi*input_size*model_dim + ii*model_dim + di] = 
                state.input[bi*input_size*model_dim + ii*model_dim + di] + l.weights[bi*input_size*model_dim + ii*model_dim + di];
            }
        }
    }
    
}

