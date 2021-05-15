#ifndef MULTI_HEAD_ATTENTION_LAYER_H
#define MULTI_HEAD_ATTENTION_LAYER_H

#include "dark_cuda.h"
#include "image.h"
#include "layer.h"
#include "network.h"

layer make_multi_head_attention_layer(int batch, int input_size, int head_num, int key_dim);
void forward_multi_head_attention_layer(const layer layer, network_state state);
void weight_multiply(int batch, int input_size, int head_num, int key_dim, int model_dim, float *qweight,float *kweight, float *vweight, float *input, float *q, float *k, float*v,float *Bq, float *Bv, float *Bk);
void transpose_qkv(int batch, int input_size, int head_num, int key_dim, float *q, float *qt, float *k, float *kt, float *v, float *vt);
void attention_score(int batch, int input_size, int head_num, int key_dim, float *q, float *k, float *score);
void matmul_v(int batch, int input_size, int head_num, int key_dim, float *output, float *score, float *v);
void transpose_output(int batch, int input_size, int head_num, int key_dim, float* input, float* output);
void multi_head_output(int batch, int input_size, int head_num, int key_dim, float *output, float *concat_head, float *output_weights);
void test_initial_value(int batch, int input_size, int head_num, int key_dim, int model_dim, float *input_data, float *Wq, float *Wk, float *Wv,
float *q, float *k, float *v, float *score, float *output, float *output_weights);
void test_initial_kernel(int k, int l, int m, int n, float *v, const float value, int rand, float incre);
void test_multi_head_attention();
void output_printf(int b, int h, int q, int k, float *score, char *name);
#endif