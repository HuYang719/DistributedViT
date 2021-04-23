#ifndef LAYERNORM_LAYER_H
#define LAYERNORM_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

layer make_layernorm_layer(int batch, int input_size, int model_dim, int train, int cut);
void forward_layernorm_layer(layer l, network_state state);

void calculate_mean(float *input, int batch, int input_size, int model_dim, float *mean);

void calculate_variance(float *input, int batch, int input_size, int model_dim, const float *mean, float *variance);
void layer_normalization(float *input, float *output, const float *mean, const float *variance, int batch, int input_size, int model_dim);
void  multiply_scales(float *output, float *scales, int batch, int input_size, int model_dim);
void  add_biases(float *output, float *biases, int batch, int input_size, int model_dim);
void output_printf_layer(int b, int h, int q, int k, float *score, char *name);
void test_initial__layer_kernel(int k, int l, int m, int n, float *v, const float value, int rand, float incre);
void test_initial_layer_value(int batch, int input_size, int model_dim, float *input, float  *mean, 
float *variance, float *scales, float *biases, float *output);
void test_layer_norm();

void backward_layernorm_layer(const layer l, network_state state);

void update_layernorm_layer(layer l, int batch, float learning_rate, float momentum, float decay);

#endif