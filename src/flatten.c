#include "flatten.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define OUTPUT 0
#define DEBUG 0

layer make_flatten_layer(int batch, int model_dim, int gh, int gw)
{
    // input dim [batch, input_size (sequence_length), model_dim]
    fprintf(stderr, "flatten layer: batch size %d, model_dim%d, gh %d, gw %d\n", batch, model_dim, gh, gw);
    layer layer = { (LAYER_TYPE)0 };
    layer.type =  FLATTEN;
    layer.batch = batch;
    layer.model_dim = model_dim;
    layer.h = gh;
    layer.w = gw;


    layer.output = (float*)xcalloc(batch* (gh * gw) * model_dim , sizeof(float));
    // layer.inputs = input_size;
    // layer.outputs = layer.inputs;


    layer.forward = forward_flatten_layer;
    // layer.backward = backward_layernorm_layer;
    // layer.update = update_layernorm_layer;

    return layer;
}


void forward_flatten_layer(layer l, network_state state)
{
    int model_dim = state.net.model_dim;
    int gw = l.w;
    int gh = l.h;
    int batch = l.batch;
    printf("flatten: gw=%d, gh=%d", l.w, l.h);

    
    for(int bi = 0; bi < batch; bi++) {
        for(int mi = 0; mi < model_dim; mi++) {
            for(int hi = 0; hi < gh; hi++) {
                for(int wi = 0; wi < gw; wi++)

                l.output[bi*gh*gw*model_dim + hi*gw*model_dim + wi*model_dim +mi] = 
                state.input[bi*model_dim*gw*gh + mi*gh*gw + hi*gw + wi];
            }
        }
    }
    
}

