#include "multi_head_attention_layer.h"
#include "configure.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gemm.h"
#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
#endif
// #include <omp.h>
#define DEBUG 0
#define OUTPUT 0
#define OUTPUT_SHAPE 0
#define TIME 1
#define TMP 1
layer make_multi_head_attention_layer(int batch, int input_size, int head_num, int key_dim)
{
    // projection_dim for q,v,k
    // input data dimension is : [batch, input_size, model_dim]
    fprintf(stderr, "Multi_Head_Attention_Layer: batch_size %d, head_num: %d, qkv_dim %d\n", batch, head_num, key_dim);
    layer l = { (LAYER_TYPE)0 };
    l.type = MULTI_HEAD_ATTENTION;
    l.batch = batch;
    l.head_num = head_num;
    l.key_dim = key_dim;

    l.inputs = input_size;
    l.outputs = input_size;
    l.c = input_size;
    l.h = head_num;
    l.w = key_dim;
    l.out_c = input_size;
    l.out_h = head_num;
    l.out_w = key_dim;
    int model_dim = head_num*key_dim;
    l.model_dim = model_dim;

    printf("in multi_head_attention, input size is %d\n", input_size);
    l.Wq = (float*)xcalloc(model_dim*head_num*key_dim, sizeof(float));
    l.Wk = (float*)xcalloc(model_dim*head_num*key_dim, sizeof(float));
    l.Wv = (float*)xcalloc(model_dim*head_num*key_dim, sizeof(float));
    l.Bq = (float*)xcalloc(head_num*key_dim, sizeof(float));
    l.Bk = (float*)xcalloc(head_num*key_dim, sizeof(float));
    l.Bv = (float*)xcalloc(head_num*key_dim, sizeof(float));
    l.q = (float*)xcalloc(batch * input_size * head_num * key_dim, sizeof(float));
    l.v = (float*)xcalloc(batch * input_size * head_num * key_dim, sizeof(float));
    l.qt = (float*)xcalloc(batch *  head_num * input_size * key_dim, sizeof(float));
    l.kt = (float*)xcalloc(batch  * head_num * key_dim* input_size, sizeof(float));
    l.vt = (float*)xcalloc(batch  * head_num *input_size * key_dim, sizeof(float));
    l.k = (float*)xcalloc(batch * input_size * head_num * key_dim, sizeof(float));
    l.score = (float*)xcalloc(batch * head_num * input_size *input_size, sizeof(float));
    // Wq, Wk, Wv?
    l.output_weight = (float*)xcalloc(head_num * key_dim * (head_num*key_dim), sizeof(float));
    l.biases = (float*)xcalloc(head_num * key_dim, sizeof(float));
    l.concat_head = (float*)xcalloc(batch * input_size * head_num * key_dim, sizeof(float));
    l.output = (float*)xcalloc(batch * input_size * head_num * key_dim, sizeof(float));
    // l.finaloutput = (float*)xcalloc(batch * input_size * head_num * key_dim, sizeof(float))

    l.forward = forward_multi_head_attention_layer;
    // l.backward = backward_multi_head_attention_layer;
    // l.update = update_multi_head_attention_layer;
    l.bflops += (6*input_size*model_dim*key_dim*head_num + 2*key_dim*input_size*input_size*head_num 
    + 2*input_size*input_size*head_num + input_size*input_size*key_dim*head_num + input_size*model_dim*model_dim)/1000000000;

    return l;
}



void forward_multi_head_attention_layer(const layer layer, network_state state)
{
    
    float *q = layer.q;
    float *k = layer.k;
    float *v = layer.v;
    float *qt = layer.qt;
    float *kt = layer.kt;
    float *vt = layer.vt;
    float *Wq = layer.Wq; // should be initial with weight result
    float *Wk = layer.Wk;
    float *Wv = layer.Wv;
    float *Bq = layer.Bq;
    float *Bv = layer.Bv;
    float *Bk = layer.Bk;
    float *score = layer.score;
    float *output = layer.output;
    float *concat_head = layer.concat_head;
    float *output_weights = layer.output_weight;
    float *input_data = state.input;
    int batch = layer.batch;
    int dim = layer.key_dim;
    int head_num = layer.head_num;
    int input_size = layer.inputs;
    int model_dim = state.net.model_dim;
#if OUTPUT
    printf("input_size is %d, model_dim is %d, key_dim is %d, head_num is %d\n", input_size, model_dim, dim, head_num);
#endif

    //assert model_dim = key_dim*head_num
    if(model_dim != dim*head_num) {
        fprintf(stderr, "Multi_Head_Attention_Layer: model_dim %d need to equal key_dim %d * head_num %d\n", model_dim, dim, head_num);
        return;

    }

       
// #if TMP
//     test_initial_kernel(1, model_dim, head_num, dim, Wq,0, 2,0);
//     test_initial_kernel(1, model_dim, head_num, dim, Wk, 0, 2, 0);
//     test_initial_kernel(1, model_dim, head_num, dim, Wv, 0, 2, 0);
//     test_initial_kernel(1, head_num, dim, head_num*head_num, output_weights,  0, 2, 0);

// #endif
     
       

    // generate q, k, v
    weight_multiply(batch, input_size, head_num, dim, model_dim,  Wq, Wk, Wv, input_data, q, k, v, Bq, Bv, Bk);


    // transpose q, ks
    transpose_qkv(batch, input_size, head_num, dim, q, qt, k, kt, v, vt);

    // match score function
    attention_score(batch, input_size, head_num, dim, qt, kt, v, score);

    // char name2[30] = "before softmax";
    // output_printf(batch,  head_num, input_size, input_size, score, name2);

    softmax_cpu(score, input_size, batch, head_num*input_size*input_size, head_num*input_size, input_size, 1, 1, score);

    char name3[30] = "after softmax";
    output_printf(batch,  head_num, input_size, input_size, score, name3);
    // matmul with v
    matmul_v(batch, input_size, head_num, dim, concat_head, score, v);

#if OUTPUT
    char name4[30] = "concat_head";
    output_printf(batch,  head_num, input_size, dim, concat_head, name4);
#endif
    
    // MultiHead output for next sub layer
    multi_head_output(batch, input_size, head_num, dim, output, concat_head, output_weights);

    // add bias
    for(int i = 0; i < batch*input_size; ++i){
        axpy_cpu(model_dim, 1, layer.biases, 1, layer.output + i*model_dim, 1);
    }

#if DEBUG
    char name5[30] = "output_weights";
    output_printf(1,  head_num, dim, head_num*dim, output_weights, name5);
#endif
    
#if OUTPUT
    // char namei[30] = "input_data";
    // output_printf(batch, input_size, 1, model_dim, input_data, namei);
    // char nameBq[30] = "Biase_q";
    // output_printf(batch, 1, 1, model_dim, Bq, nameBq);
    // char nameWq[30] = "weight_q";
    // output_printf(batch, 1, model_dim, model_dim, Wq, nameWq);

    // char nameBk[30] = "Biase_k";
    // output_printf(batch, 1, 1, model_dim, Bk, nameBk);
    // char nameWk[30] = "weight_k";
    // output_printf(batch, 1, model_dim, model_dim, Wk, nameWk);

    // char nameBv[30] = "Biase_v";
    // output_printf(batch, 1, 1, model_dim, Bv, nameBv);
    // char nameWv[30] = "weight_v";
    // output_printf(batch, 1, model_dim, model_dim, Wv, nameWv);

    // char nameBo[30] = "Biase_output";
    // output_printf(batch, 1, 1, model_dim, layer.biases, nameBo);
    // char nameWo[30] = "weight_output";
    // output_printf(batch, 1, model_dim, model_dim, output_weights, nameWo);
   char name2[30] = "score";
    output_printf(batch,  head_num, input_size, input_size, score, name2);
    char name[30] = "attention output";
    output_printf(batch, input_size, 1, head_num*dim, output, name);
#endif
     
    

    // free(layer.Wq);
    // free(layer.Wk);
    // free(layer.Wv);
    // free(layer.output_weight);

}


void weight_multiply(int batch, int input_size, int head_num, int key_dim, int model_dim, float *qweight,float *kweight, float *vweight, float *input, float *q, float *k, float*v, float *Bq, float *Bv, float *Bk){
// output [b, input_size, head_num*key_dim]
//output = np.einsum("bim, imt -> bit", input, weight)


// omp_set_num_threads(THRD_NUM);
// #pragma omp parallel for
int M = batch*input_size;
int N = head_num*key_dim;
int K = model_dim;

gemm(0, 0,M, N, K, 1, input,K, qweight,N,1,q,N);
gemm(0, 0,M, N, K, 1, input,K, vweight,N,1,v,N);
gemm(0, 0,M, N, K, 1, input,K, kweight,N,1,k,N);
axpy_cpu(N, 1, Bq, 1, q, 1);
axpy_cpu(N, 1, Bv, 1, v, 1);
axpy_cpu(N, 1, Bk, 1, k, 1);
//  for(int bi = 0; bi < batch; bi++) {
//         // #pragma omp parallel for
//         for(int ii = 0; ii < input_size; ii++) {
//             for(int mi = 0; mi < model_dim; mi++){
//                      float A_PART = input[bi*input_size*model_dim + ii*model_dim + mi];
//                      for(int ki = 0; ki < head_num*key_dim; ki++){
//                         int tmp1 =  bi*input_size*head_num*key_dim + ii*head_num*key_dim +ki;
//                         int tmp2 =  mi*head_num*key_dim + ki;
//                         q[tmp1] += A_PART *qweight[tmp2];

//                     }       
//             }
//         }
//     }

//  for(int bi = 0; bi < batch; bi++) {
//         // #pragma omp parallel for
//         for(int ii = 0; ii < input_size; ii++) {
//             for(int mi = 0; mi < model_dim; mi++){
//                      float A_PART = input[bi*input_size*model_dim + ii*model_dim + mi];
//                      for(int ki = 0; ki < head_num*key_dim; ki++){
//                         int tmp1 =  bi*input_size*head_num*key_dim + ii*head_num*key_dim +ki;
//                         int tmp2 =  mi*head_num*key_dim + ki;
//                         k[tmp1] += A_PART *kweight[tmp2];
//                     }       
//             }
//         }
//     }

//  for(int bi = 0; bi < batch; bi++) {
//         // #pragma omp parallel for
//         for(int ii = 0; ii < input_size; ii++) {
//             for(int mi = 0; mi < model_dim; mi++){
//                      float A_PART = input[bi*input_size*model_dim + ii*model_dim + mi];
//                      for(int ki = 0; ki < head_num*key_dim; ki++){
//                         int tmp1 =  bi*input_size*head_num*key_dim + ii*head_num*key_dim +ki;
//                         int tmp2 =  mi*head_num*key_dim + ki;
//                         v[tmp1] += A_PART *vweight[tmp2];
//                     }       
//             }
//         }
//     }
}

void transpose_qkv(int batch, int input_size, int head_num, int key_dim, float *q, float *qt, float *k, float *kt, float *v, float *vt) {
int bi, ii, hi, di;
    for(bi = 0; bi < batch; bi++) {
            // #pragma omp parallel for
            for(ii = 0; ii < input_size; ii++) {
                for(hi = 0; hi < head_num; hi++) {
                    for(di = 0; di < key_dim; di++) {
                        qt[bi*head_num*input_size*key_dim + hi*input_size*key_dim + ii*key_dim +di] =
                        q[bi*input_size*head_num*key_dim + ii*head_num*key_dim + hi*key_dim + di];
                        kt[bi*head_num*key_dim*input_size + hi*key_dim*input_size + di*input_size +ii] = 
                        k[bi*input_size*head_num*key_dim + ii*head_num*key_dim + hi*key_dim + di];
                        vt[bi*head_num*input_size*key_dim + hi*input_size*key_dim + ii*key_dim + di] = 
                        v[bi*input_size*head_num*key_dim + ii*head_num*key_dim + hi*key_dim + di];
                    }
                }
            }
     
    }

}

void attention_score(int batch, int input_size, int head_num, int key_dim, float *qt, float *kt, float *v, float *score){

    // [batch, head_num, input_size, key_dim] X [b,H, keydim, input_size]
    // -> [batch, H, input_size, input_size]

int i;
int M = input_size;
int K = key_dim;
int N = input_size;
int stride1 = input_size*key_dim;
int stride2 = key_dim*input_size;
int stride3 = input_size*input_size;
float div_number = 1/pow(key_dim, 0.5);
// gemm(0, 0,M, N, K, 1, input,K, qweight,N,1,q,K);
for(i = 0; i < batch*head_num; i++) {
    gemm(0, 0, M, N, K, 1, qt+i*stride1, K, kt + i*stride2, N, 1, score + i*stride3, N);
}

int iter = batch*head_num*input_size*input_size;
for(i = 0; i < iter; i++) {
    score[i] *= div_number;
}
// #if DEBUG
//     printf("intermid result:\n");
// #endif
// int bi, qni, kni, hi, di;
// float div_number = pow(key_dim, 0.5);
// #if OMP
// omp_set_num_threads(THRD_NUM);
// // #pragma omp parallel for
// #endif
//     for(bi = 0; bi < batch; bi++) {
//         // #pragma omp parallel for
//         for(hi = 0; hi < head_num; hi++) {
//             for(qni = 0; qni < input_size; qni++) {
//                 for(di = 0; di < key_dim; di++) {
//                     PUT_IN_REGISTER float A_PART = qt[bi*head_num*input_size*key_dim + hi*input_size*key_dim + qni*key_dim  + di];
//                     for(kni = 0; kni < input_size; kni++) {
//                         score[bi*head_num*input_size*input_size + hi*input_size*input_size + qni*input_size + kni] += 
//                         A_PART*kt[bi*head_num*key_dim*input_size + hi*key_dim*input_size + di*input_size  + kni];
//                         score[bi*head_num*input_size*input_size + hi*input_size*input_size + qni*input_size + kni] /= div_number;
//                     }
// #if DEBUG
//                         printf("\n\n");
// #endif
//                 }
//             }
//         }
//     }
}

void matmul_v(int batch, int input_size, int head_num, int key_dim, float *output, float *score, float *v) {

// [B, H, S, S] X [B,H,S,W] -> [B,H,S,w]
int i;
int M = input_size;
int K = input_size;
int N = key_dim;
int stride1 = input_size*input_size;
int stride2 = input_size*key_dim;
float *output_temp;
output_temp = (float*) malloc(batch*input_size*head_num*key_dim*sizeof(float));
memset(output_temp, 0, batch*input_size*head_num*key_dim*sizeof(float));

// gemm(0, 0,M, N, K, 1, input,K, qweight,N,1,q,K);
for(i = 0; i < batch*head_num; i++) {
    gemm(0, 0, M, N, K, 1, score+i*stride1, K, v + i*stride2, N, 1, output_temp+ i*stride2, N);
}

int bi, ii, hi, di;
for(bi = 0; bi < batch; bi++) {
        // #pragma omp parallel for
        for(ii = 0; ii < input_size; ii++) {
            for(hi = 0; hi < head_num; hi++) {
                for(di = 0; di < key_dim; di++) {
                    output[bi*input_size*head_num*key_dim + ii*head_num*key_dim + hi*key_dim +di] =
                    output_temp[bi*head_num*input_size*key_dim + hi*input_size*key_dim + ii*key_dim + di];
                }
            }
        }
    
}

// #if OMP
// omp_set_num_threads(THRD_NUM);
// #pragma omp parallel for
// #endif      
//     for(int bi = 0; bi < batch; bi++) {
//         for(int hi = 0; hi < head_num; hi++) {
//                 for(int sni = 0; sni < input_size; sni++) {
//                     for(int vni = 0; vni < input_size; vni++) {
//                         PUT_IN_REGISTER float A_PART = score[bi*input_size*head_num*input_size + hi*input_size*input_size + sni*input_size  + vni];
//                         for (int di = 0; di < key_dim; di++) {
//                         output[bi*head_num*input_size*key_dim + sni*head_num*key_dim + hi*key_dim + di] += 
//                         A_PART*v[bi*input_size*head_num*key_dim + vni*head_num*key_dim + hi*key_dim  + di];
// #if DEBUG
//                         printf("output[%d][%d][%d][%d] (%f) += score[%d][%d][%d][%d](%f) * v[%d][%d][%d][%d](%f)\n", bi, sni,hi, di,
//                         output[bi*head_num*input_size*key_dim + sni*head_num*key_dim + hi*key_dim + di],bi,hi, sni, vni, score[bi*input_size*head_num*input_size + hi*input_size*input_size + sni*input_size  + vni],bi, vni, hi, di,v[bi*input_size*head_num*key_dim + vni*head_num*key_dim + hi*key_dim  + di]);
// #endif
//                         }

//                     }
// #if DEBUG
//                         printf("\n\n");
// #endif               
//                 }
            
//         }

//     }

}

void multi_head_output(int batch, int input_size, int head_num, int key_dim, float *output, float *concat_head, float *output_weights) {
    // output = concat_head * W^Q    // ([b, n, h x d] X [h x d, d_{model}] = [b, n, d_{model}])
int M = batch*input_size;
int K = head_num*key_dim;
int N = head_num*key_dim;

gemm(0, 0, M, N, K, 1, concat_head, K, output_weights , N, 1, output , N);


// int bi, ni, mi, hi;
// #if OMP
// omp_set_num_threads(THRD_NUM);
// #pragma omp parallel for
// #endif
//     for( bi = 0; bi < batch; bi++) {
//         for( ni = 0; ni < input_size; ni++) {
//             for( mi = 0; mi < (head_num*key_dim); mi++) {
//                     PUT_IN_REGISTER float A_PART = concat_head[bi*input_size*head_num*key_dim + ni*head_num*key_dim + mi];
//                     for (hi = 0; hi < head_num*key_dim; hi++){
//                         output[bi*input_size*head_num*key_dim + ni*head_num*key_dim + hi] += 
//                         A_PART*output_weights[mi*head_num*key_dim  + hi];
// #if DEBUG
//                         printf("output[%d][%d][%d](%f) += concat_head[%d][%d][%d](%f) * output_weights[%d][%d](%f)\n", 
//                         bi, ni,hi,output[bi*input_size*head_num*key_dim + ni*head_num*key_dim + hi],
//                         bi,ni,mi,concat_head[bi*input_size*head_num*key_dim + ni*head_num*key_dim + mi], 
//                         mi, hi,  output_weights[mi*head_num*key_dim  + hi]);
// #endif
//                     }
// #if DEBUG
//                         printf("\n\n");
// #endif               
//             }
//         }
//     }

}


void backward_multi_head_attention_layer() {}

void update_multi_head_attention_layer() {}


void test_multi_head_attention(){

    int batch = 1;
    int input_size = 4;
    int head_num = 5;
    int key_dim = 6;
    int model_dim = 30;
#if TIME
    struct timespec start, end, m1, m2, m3, m4, m5,m6, mt;
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) {perror("clock unable gettune");}
#endif

    float *input_data, *Wq,*Wk, *Wv, *q, *qt, *k, *kt, *v, *vt, *Bv, *Bq, *Bk, *score, *output, *output_weights, *final_result;
    input_data = (float*) malloc(batch*input_size*model_dim*sizeof(float));
    Wq =  (float*) malloc(model_dim*head_num*key_dim*sizeof(float));
    Wk =  (float*) malloc(model_dim*head_num*key_dim*sizeof(float));
    Wv =  (float*) malloc(model_dim*head_num*key_dim*sizeof(float));
    q = (float*) malloc(batch*input_size*head_num*key_dim*sizeof(float));
    qt = (float*) malloc(batch*head_num*input_size*key_dim*sizeof(float)); //qt [B, H, input_size, key_dim]
    k = (float*) malloc(batch*input_size*head_num*key_dim*sizeof(float));
    kt = (float*) malloc(batch*head_num*key_dim*input_size*sizeof(float)); //kt [B,H, key_dim, input_size]
    v = (float*) malloc(batch*input_size*head_num*key_dim*sizeof(float));
    vt = (float*) malloc(batch*head_num*input_size*key_dim*sizeof(float)); //kt [B,H, input_size, key_dim]
    Bq = (float*) malloc(batch*head_num*input_size*key_dim*sizeof(float));
    Bv = (float*) malloc(batch*head_num*input_size*key_dim*sizeof(float));
    Bk = (float*) malloc(batch*head_num*input_size*key_dim*sizeof(float));
    score = (float*) malloc(batch*head_num*input_size*input_size*sizeof(float));
    output = (float*) malloc(batch*input_size*head_num*key_dim*sizeof(float));
    output_weights = (float*) malloc(head_num*key_dim*head_num*key_dim*sizeof(float));
    final_result = (float*) malloc(batch*input_size*head_num*key_dim*sizeof(float));
     
    // initial value for input_data, Weights(Wq, Wk, Wv, output_weight);
    test_initial_value(batch, input_size, head_num, key_dim,model_dim, input_data, Wq, Wk, Wv, q,k, v, score, output, output_weights);
    // printf("after initial, Wq[1][0][0]=%f\n",Wq[1*key_dim*head_num]);
    // printf("The size is : Wq = %ld, q = %ld, score = %ld, output = %ld, output_weights = %ld, final_result = %ld\n", 
    // (int)sizeof(Wq)/sizeof(Wq[12]), sizeof(*q), sizeof(*score), sizeof(*output), sizeof(*output_weights), sizeof(*final_result));
#if TIME
    if( clock_gettime(CLOCK_REALTIME, &m1) == -1) {perror("clock unable gettune");}
    double time1 = (m1.tv_sec - start.tv_sec) + (double)(m1.tv_nsec - start.tv_nsec)/1e9;
    printf("\n Allocation Time is %f sec \n", time1);
#endif
    weight_multiply(batch, input_size, head_num, key_dim, model_dim,  Wq, Wk, Wv, input_data, q, k, v, Bq, Bv, Bk);

#if TIME
    if( clock_gettime(CLOCK_REALTIME, &m2) == -1) {perror("clock unable gettune");}
    double time2 = (m2.tv_sec - m1.tv_sec) + (double)(m2.tv_nsec - m1.tv_nsec)/1e9;
    printf("\n Weight Multiply Time is %f sec \n", time2);
#endif

transpose_qkv(batch, input_size, head_num, key_dim, q, qt, k, kt, v, vt);
#if TIME
    if( clock_gettime(CLOCK_REALTIME, &mt) == -1) {perror("clock unable gettune");}
    double timet = (mt.tv_sec - m2.tv_sec) + (double)(mt.tv_nsec - m2.tv_nsec)/1e9;
    printf("\n Transpose Time is %f sec \n", timet);
#endif
    char nameq[30] = "result q";
    output_printf(batch, input_size, head_num, key_dim, q, nameq);
    char namek[30] = "result k";
    output_printf(batch, input_size, head_num, key_dim, k, namek);
    char namev[30] = "result v";
    output_printf(batch, input_size, head_num, key_dim, v, namev);
    attention_score(batch, input_size, head_num, key_dim, qt, kt, NULL, score);
#if TIME
    if( clock_gettime(CLOCK_REALTIME, &m3) == -1) {perror("clock unable gettune");}
    double time3 = (m3.tv_sec - m2.tv_sec) + (double)(m3.tv_nsec - m2.tv_nsec)/1e9;
    printf("\n Attention Score Time is %f sec \n", time3);
#endif
    // char name2[30] = "score";
    // output_printf(batch,  head_num, input_size, input_size, score, name2);
    softmax_cpu(score, input_size, batch, head_num*input_size*input_size, head_num*input_size, input_size, 1, 1, score);
#if TIME
    if( clock_gettime(CLOCK_REALTIME, &m4) == -1) {perror("clock unable gettune");}
    double time4 = (m4.tv_sec - m3.tv_sec) + (double)(m4.tv_nsec - m3.tv_nsec)/1e9;
    printf("\n Softmax Score Time is %f sec \n", time4);
#endif
    

    matmul_v(batch, input_size, head_num, key_dim, output, score, v);
#if TIME
    if( clock_gettime(CLOCK_REALTIME, &m5) == -1) {perror("clock unable gettune");}
    double time5 = (m5.tv_sec - m4.tv_sec) + (double)(m5.tv_nsec - m4.tv_nsec)/1e9;
    printf("\n Matmul_v Time is %f sec \n", time5);
#endif
    multi_head_output(batch, input_size, head_num, key_dim, final_result, output, output_weights);
#if TIME
    if( clock_gettime(CLOCK_REALTIME, &m6) == -1) {perror("clock unable gettune");}
    double time6 = (m6.tv_sec - m5.tv_sec) + (double)(m6.tv_nsec - m5.tv_nsec)/1e9;
    printf("\n Multi_head_output Time is %f sec \n", time6);
#endif
#if TIME
    if( clock_gettime(CLOCK_REALTIME, &end) == -1) {perror("clock unable gettune");}
    double time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/1e9;
    printf("\n Attention Time is %f sec \n", time);
#endif
    char name[30] = "final_result";
    output_printf(batch, input_size, 1, head_num*key_dim, final_result, name);
}

void test_initial_kernel(int k, int l, int m, int n, float *v, const float value, int rand, float incre) {
    float value_tmp = value;
    for (int ki = 0; ki < k; ki++) {
        for (int li = 0; li < l; li++) {
            for (int mi = 0; mi < m; mi ++) {
                for (int ni = 0; ni < n; ni++) {
                    if(rand == 0){
                        v[ki*l*m*n + li*m*n + mi*n + ni] = value;
                    } else if (rand == 1) {
                        v[ki*l*m*n + li*m*n + mi*n + ni] = value_tmp;
                        value_tmp += incre;
                    }else if (rand == 2) {
                         v[ki*l*m*n + li*m*n + mi*n + ni] = 0.0001*rand_uniform(-1, 1);
                    }
                    
                }
            }
        }
    }
}
void test_initial_value(int batch, int input_size, int head_num, int key_dim, int model_dim, float *input_data, float *Wq, float *Wk, float *Wv,
float *q, float *k, float *v, float *score, float *output, float *output_weights) {
    test_initial_kernel(1, batch, input_size, model_dim, input_data, 1, 0, 0.05 );
    test_initial_kernel(1, model_dim, head_num, key_dim, Wq, 1.0, 0, 0.005 );
    test_initial_kernel(1, model_dim, head_num, key_dim, Wk, 0.2, 0, -0.001 );
    test_initial_kernel(1, model_dim, head_num, key_dim, Wv, 1.2, 0, 0.005);
    // test_initial_kernel(batch, head_num, input_size, input_size, score, 1, 0, 0);
    // test_initial_kernel(batch, input_size, head_num, key_dim, output, 1, 0, 0);
    test_initial_kernel(1, head_num, key_dim, head_num*key_dim, output_weights,  1, 1, 0.002);
}

void output_printf(int b, int h, int q, int k, float *score, char *name){
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
#if OUTPUT_SHAPE
    printf("shape is [%d %d %d %d]\n", b, h, q, k);
#endif
}





