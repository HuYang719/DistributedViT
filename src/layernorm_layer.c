#include "layernorm_layer.h"
#include "blas.h"
#include "utils.h"
#include "configure.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define OUTPUT 0
#define DEBUG 0

layer make_layernorm_layer(int batch, int input_size, int model_dim, int train)
{
    fprintf(stderr, "Layer Normalization Layer: batch size %d, input size %d,  model dim %d\n", batch, input_size, model_dim);
    layer layer = { (LAYER_TYPE)0 };
    layer.type =  LAYERNORM_LAYER;
    layer.batch = batch;
    layer.train = train;
    layer.input_size = input_size;
    layer.model_dim = model_dim;


    layer.output = (float*)xcalloc(input_size * model_dim * batch, sizeof(float));
    layer.delta = (float*)xcalloc(input_size * model_dim * batch, sizeof(float));
    layer.inputs = input_size;
    layer.outputs = layer.inputs;

    layer.biases = (float*)xcalloc(batch*model_dim, sizeof(float));
    layer.scales = (float*)xcalloc(batch*model_dim, sizeof(float));


#if OMP
omp_set_num_threads(THRD_NUM);
#pragma omp parallel for 
#endif  
    for(int i = 0; i < batch; ++i){
        for(int j = 0; j < model_dim; j++)
        layer.scales[i*model_dim + j] = 1;
    }


    
    layer.mean = (float*)xcalloc(batch*input_size, sizeof(float));
    layer.variance = (float*)xcalloc(batch*input_size, sizeof(float));



    layer.forward = forward_layernorm_layer;
    layer.backward = backward_layernorm_layer;
    layer.update = update_layernorm_layer;

    return layer;
}


void forward_layernorm_layer(layer l, network_state state)
{
    int model_dim = state.net.model_dim;
   
        l.out_c = l.c = 576;
        l.out_h = l.h = 12;
        l.out_w = l.w = 64;
  
    if(model_dim != l.model_dim) {
        fprintf(stderr, "Layer Normlization: input model dim %d is different from required dim %d\n",
        model_dim, l.model_dim);
        return;
    }
    // printf("layer norm input is state.input[0]=%f, state.input[1]=%f\n", state.input[0], state.input[1]);
    calculate_mean(state.input, l.batch, l.input_size, model_dim, l.mean);
    calculate_variance(state.input, l.batch, l.input_size, model_dim, l.mean, l.variance);
    layer_normalization(state.input, l.output, l.mean, l.variance, l.batch, l.input_size, model_dim);
    multiply_scales(l.output, l.scales, l.batch, l.input_size, model_dim);
    add_biases(l.output, l.biases, l.batch, l.input_size, model_dim);  
    // printf("layer norm output is l.input[0]=%f, l.input[1]=%f\n", l.output[0], l.output[1]);

#if OUTPUT
    char name[30] = "layernorm";
    output_printf_layer(1, 1, l.input_size, model_dim, l.output, name);
#endif
}

void backward_layernorm_layer(const layer l, network_state state){}

void update_layernorm_layer(layer l, int batch, float learning_rate, float momentum, float decay){}

void calculate_mean(float *input, int batch, int input_size, int model_dim, float *mean){

#if OMP
    omp_set_num_threads(THRD_NUM);
    #pragma omp parallel for 
#endif
    for(int bi = 0; bi < batch; bi++) {
        for(int ii = 0; ii < input_size; ii++) {
            float tmp = 0;
            for(int mi = 0; mi < model_dim; mi++) {
                tmp += input[bi*input_size*model_dim+ii*model_dim+mi];
            }
                mean[bi*input_size + ii] = tmp/model_dim;
        }
    }
}

void calculate_variance(float *input, int batch, int input_size, int model_dim, const float *mean, float *variance){
#if OMP
    omp_set_num_threads(THRD_NUM);
    #pragma omp parallel for 
#endif
    for(int bi = 0; bi < batch; bi++) {
        for(int ii = 0; ii < input_size; ii++) {
            for(int mi = 0; mi < model_dim; mi++) {
                variance[bi*input_size + ii] += 
                pow((input[bi*input_size*model_dim+ii*model_dim+mi] - mean[bi*input_size+ii]),2);
            }
            variance[bi*input_size + ii] /= model_dim;  
        }
    }
}

void layer_normalization(float *input, float *output, const float *mean, const float *variance, int batch, int input_size, int model_dim){
#if OMP
    omp_set_num_threads(THRD_NUM);
    #pragma omp parallel for 
#endif  
    for(int bi = 0; bi < batch; bi++) {
        for(int ii = 0; ii < input_size; ii++) {
            for(int mi = 0; mi < model_dim; mi++) {
                output[bi*input_size*model_dim + ii*model_dim + mi] = 
                (input[bi*input_size*model_dim + ii*model_dim + mi] - mean[bi*input_size+ii])/sqrt(variance[bi*input_size+ii]+0.001);
// #ifdef DEBUG
//             printf("output[%d][%d][%d](%f) = input[%d][%d][%d](%f) - mean[]%d[%d](%f) / sqrt var[%d][%d](%f)\n",
//             bi, ii, mi, output[bi*input_size*model_dim + ii*model_dim + mi],
//             bi, ii, mi, input[bi*input_size*model_dim + ii*model_dim + mi],
//             bi, ii, mean[bi*input_size+ii],
//             bi, ii, variance[bi*input_size+ii]);
// #endif
            }
              
        }
    }

}

void  multiply_scales(float *output, float *scales, int batch, int input_size, int model_dim){
#if OMP
    omp_set_num_threads(THRD_NUM);
    #pragma omp parallel for 
#endif
    for(int bi = 0; bi < batch; bi++) {
        for(int ii = 0; ii < input_size; ii++) {
            for(int mi = 0; mi < model_dim; mi++) {
                output[bi*input_size*model_dim + ii*model_dim + mi] = 
                output[bi*input_size*model_dim + ii*model_dim + mi]*scales[bi*model_dim+mi];
            }
              
        }
    }


}

void  add_biases(float *output, float *biases, int batch, int input_size, int model_dim){
#if OMP
    omp_set_num_threads(THRD_NUM);
    #pragma omp parallel for 
#endif
    for(int bi = 0; bi < batch; bi++) {
        for(int ii = 0; ii < input_size; ii++) {
            for(int mi = 0; mi < model_dim; mi++) {
                output[bi*input_size*model_dim + ii*model_dim + mi] = 
                output[bi*input_size*model_dim + ii*model_dim + mi]+biases[bi*model_dim+mi];
            }
              
        }
    }


}


void output_printf_layer(int b, int h, int q, int k, float *score, char *name){
        printf("%s result is \n", name);
     for(int bi = 0; bi < b; bi++) {
        for(int hi = 0; hi < h; hi++) {
            for(int qni = 0; qni < q; qni++) {
                for(int kni = 0; kni < k; kni++) {
#if OUTPUT                   
                    printf("%f ", score[bi*h*q*k + hi*q*k + qni*k + kni]);
#endif
                }
#if OUTPUT
                printf("\n");
#endif
            }
#if OUTPUT
             printf("\n\n");
#endif
        }
#if OUTPUT
        printf("\n\n\n");
#endif

    }
    printf("shape is [%d %d %d %d]\n", b, h, q, k);
}


void test_initial_layer_kernel(int k, int l, int m, int n, float *v, const float value, int rand, float incre) {
    float value_tmp = value;
    for (int ki = 0; ki < k; ki++) {
        for (int li = 0; li < l; li++) {
            for (int mi = 0; mi < m; mi ++) {
                for (int ni = 0; ni < n; ni++) {
                    if(rand == 0){
                        v[ki*l*m*n + li*m*n + mi*n + ni] = value;
                    } else {
                        v[ki*l*m*n + li*m*n + mi*n + ni] = value_tmp;
                        value_tmp += incre;
                    }
                    
                }
            }
        }
    }
}
void test_initial_layer_value(int batch, int input_size, int model_dim, float *input, float  *mean, 
float *variance, float *scales, float *biases, float *output) {
    test_initial_kernel(1, batch, input_size, model_dim, input, 1.0, 1, -0.5);
    test_initial_kernel(1, batch, 1, model_dim, scales, 1.0, 0, 1);  
}

void test_layer_norm(){
    int batch = 2;
    int input_size = 2;
    int model_dim = 4;

    float *input, *mean, *variance, *scales, *biases, *output;

    // malloc data
    input = (float*) malloc(batch*input_size*model_dim*sizeof(float));
    mean =  (float*) malloc(batch*input_size*sizeof(float));
    variance =  (float*) malloc(batch*input_size*sizeof(float));
    scales = (float*) malloc(batch*model_dim*sizeof(float));
    biases = (float*) malloc(batch*model_dim*sizeof(float));
    output = (float*) malloc(batch*input_size*model_dim*sizeof(float));

     
    // initial data
    test_initial_layer_value(batch, input_size,model_dim, input, mean, variance, scales, biases, output);
    calculate_mean(input, batch, input_size, model_dim, mean);
    calculate_variance(input, batch, input_size, model_dim, mean, variance);
    layer_normalization(input, output, mean, variance, batch, input_size, model_dim);
    multiply_scales(output, scales, batch, input_size, model_dim);
    add_biases(output, biases, batch, input_size, model_dim);  
    // process 
    

    
    // char name[30] = "output_result";
    // output_printf_layer(batch, input_size, 1, model_dim, output, name);
    // char name2[30] = "mean";
    // output_printf_layer(batch, input_size, 1, 1, mean, name2);
    //  char name3[30] = "var";
    // output_printf_layer(batch, input_size, 1, 1, variance, name3);
    // char name4[30] = "input";
    // output_printf(batch, input_size, 1, model_dim, input, name4);
}