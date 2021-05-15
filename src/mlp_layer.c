#include "mlp_layer.h"

#include "utils.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
// #define MPI 1

int world_size, world_rank;

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

    l.weights = (float*)xcalloc(hidden_size*model_dim, sizeof(float));
    l.biases = (float*)xcalloc(hidden_size, sizeof(float));

    l.forward = forward_mlp_layer;





    // float scale = sqrt(2.f/(input_size*model_dim));
    // for(i = 0; i < model_dim*hidden_size; ++i){
    //     l.weights[i] = scale*rand_uniform(-1, 1);
    // }
    // MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // // Get the rank of the process
   
    // MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    fprintf(stderr, "mlp 1 layer                            %4d  x %4d\n", model_dim, hidden_size);
    return l;
}



void forward_mlp_layer(layer l, network_state state)
{
    l.out_c = l.c = state.net.c;
    l.out_h = l.h = state.net.h;
    l.out_w = l.w = state.net.w;
    int i;
    int batch = l.batch;
#ifdef MPI
int displs[world_size], recvcounts[world_size];
int displs_o[world_size], recvcounts_o[world_size];
int input_size_avg =  l.input_size/world_size;
int input_size_rst = l.input_size - l.input_size/world_size*(world_size-1);

float *input_temp, *output_temp;

input_temp = (float *)malloc(batch*input_size_rst*l.model_dim*sizeof(float));
output_temp = (float *)malloc(batch*input_size_rst*l.hidden*sizeof(float));

memset(input_temp, 0, batch*input_size_rst*l.model_dim*sizeof(float));
memset(output_temp, 0, batch*input_size_rst*l.hidden*sizeof(float));


for(int i = 0; i<world_size; i++){
    if(i==0){
        displs[i] = 0;
        displs_o[i] = 0;
        recvcounts[i] = l.batch*l.model_dim*input_size_rst;
        recvcounts_o[i] = l.batch*l.hidden*input_size_rst;
    }else{
        displs[i] = displs[i-1]+recvcounts[i-1];
        displs_o[i] = displs_o[i-1] + recvcounts_o[i-1];
        recvcounts[i] = batch*l.model_dim*input_size_avg;
        recvcounts_o[i] = batch*l.hidden*input_size_avg;
    }
}

int m;
if(world_rank == 0){
    m = input_size_rst;
}else{ 
    m = input_size_avg;
}
int k = l.model_dim;
int n = l.hidden;
float *a = input_temp;
float *b = l.weights;
float *c = output_temp;

if(world_rank == 0){
MPI_Scatterv(state.input, recvcounts, displs, MPI_FLOAT, input_temp, recvcounts[world_rank], MPI_FLOAT,0,  MPI_COMM_WORLD); 


for(i = 0; i < l.batch; i++) {
    gemm(0,0,m,n,k,1,input_temp,k,b,n,1,c + i*l.outputs,n);
}
MPI_Allgatherv(output_temp, recvcounts_o[world_rank], MPI_FLOAT,l.output, recvcounts_o,displs_o, MPI_FLOAT, MPI_COMM_WORLD);   
 
}else{
MPI_Scatterv(NULL, recvcounts, displs, MPI_FLOAT, input_temp, recvcounts[world_rank], MPI_FLOAT,0,  MPI_COMM_WORLD); 
for(i = 0; i < l.batch; i++) {
    gemm(0,0,m,n,k,1,a,k,b,n,1,c + i*l.outputs,n);
} 
MPI_Allgatherv(output_temp, recvcounts_o[world_rank], MPI_FLOAT,l.output, recvcounts_o,displs_o, MPI_FLOAT, MPI_COMM_WORLD);   

}

#else
//[m, k] *[k, n] -> [m, n]
    int m = l.input_size;
    int k = l.model_dim;
    int n = l.hidden;
    float *a = state.input;
    float *b = l.weights;
    float *c = l.output;

    for(i = 0; i < l.batch; i++) {
        gemm(0,1,m,n,k,1,a,k,b,k,1,c + i*l.outputs,n);
    }
 

    // printf("mlp output[%d] = %f\n", 2, l.output[2]);
    for(i = 0; i < l.batch*m; ++i){
        axpy_cpu(n, 1, l.biases, 1, l.output + i*n, 1);
    }


    // if(l.scale > 0 && l.scale < 1) {
    //     float p_keep = 1-l.scale;
    //     scal_cpu(l.batch*l.outputs, p_keep, l.output, 1);
    // }

    if(l.activation == SOFTMAXAC) {
        softmax_cpu(l.output, l.hidden, l.batch,  l.input_size*l.hidden, l.input_size, l.hidden, 1, 1, l.output);

    } else {
        activate_array(l.output, l.outputs*l.batch, l.activation);
    }

    // printf(" output shaoe is %d x %d\n", l.input_size, l.hidden);
    //     int s = 0;
    //     if(state.index == 88){
    //         for(int i = 0; i < l.input_size; i++){
    //         printf("[");
    //         for(int j = 0; j < l.hidden; j++){
    //                 printf("%f ", l.output[i*l.hidden+j]);
    //                 s++;
    //                 if(s == 6){
    //                     printf("\n");
    //                     s=0;
    //                 }    
    //         }
    //         printf("]\n");
    //         s=0;
    //     }

    //     }
 #endif  
    free(l.weights);
    free(l.biases);
}





