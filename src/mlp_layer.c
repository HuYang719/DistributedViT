#include "mlp_layer.h"

#include "utils.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



layer  make_mlp_layer(int batch, int input_size, int model_dim, int hidden_size, float dropout, ACTIVATION activation)
{
    int i;
    layer l = { (LAYER_TYPE)0 };
    l.type = MLP;
    l.activation = activation;

    l.model_dim = model_dim;
    l.input_size = input_size;
    l.hidden = hidden_size;
    l.inputs = input_size*model_dim;
    l.outputs = input_size*hidden_size;
    l.inputs = input_size*model_dim;
    l.batch= batch;
    l.scale = dropout;


    l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));

    l.weights = (float*)xcalloc(model_dim*hidden_size, sizeof(float));
    l.biases = (float*)xcalloc(hidden_size, sizeof(float));

    l.forward = forward_mlp_layer;





    // float scale = sqrt(2.f/(input_size*model_dim));
    // for(i = 0; i < model_dim*hidden_size; ++i){
    //     l.weights[i] = scale*rand_uniform(-1, 1);
    // }

    fprintf(stderr, "mlp 1 layer                            %4d  x %4d\n", model_dim, hidden_size);
    return l;
}



void forward_mlp_layer(layer l, network_state state)
{
    l.out_c = l.c = state.net.c;
    l.out_h = l.h = state.net.h;
    l.out_w = l.w = state.net.w;
    int i;
    
    int m = l.input_size;
    int k = l.model_dim;
    int n = l.hidden;
    float *a = state.input;
    float *b = l.weights;
    float *c = l.output;
    for(i = 0; i < l.batch; i++) {
        gemm(0,0,m,n,k,1,a,k,b,n,1,c + i*l.outputs,n);

    }
    // printf("mlp output[%d] = %f\n", 2, l.output[2]);
    
    for(i = 0; i < l.batch*m; ++i){
        axpy_cpu(n, 1, l.biases, 1, l.output + i*n, 1);
    }
    if(l.activation == SOFTMAXAC) {
        softmax_cpu(l.output, l.hidden, l.batch,  l.input_size*l.hidden, l.input_size, l.hidden, 1, 1, l.output);

    } else {
        activate_array(l.output, l.outputs*l.batch, l.activation);
    }
    if(l.scale > 0 && l.scale < 1) {
        float p_keep = 1-l.scale;
        scal_cpu(l.batch*l.outputs, p_keep, l.output, 1);
    }
    free(l.weights);
    free(l.biases);
}





