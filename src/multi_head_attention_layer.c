#include "multi_head_attention_layer.h"
#include "configure.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

    l.Wq = (float*)xcalloc(model_dim*head_num*key_dim, sizeof(float));
    l.Wk = (float*)xcalloc(model_dim*head_num*key_dim, sizeof(float));
    l.Wv = (float*)xcalloc(model_dim*head_num*key_dim, sizeof(float));
    l.q = (float*)xcalloc(batch * input_size * head_num * key_dim, sizeof(float));
    l.v = (float*)xcalloc(batch * input_size * head_num * key_dim, sizeof(float));
    l.k = (float*)xcalloc(batch * input_size * head_num * key_dim, sizeof(float));
    l.score = (float*)xcalloc(batch * head_num * input_size *input_size, sizeof(float));
    // Wq, Wk, Wv?
    l.output_weight = (float*)xcalloc(head_num * key_dim * (head_num*key_dim), sizeof(float));
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
    float *Wq = layer.Wq; // should be initial with weight result
    float *Wk = layer.Wk;
    float *Wv = layer.Wv;
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

       
#if TMP
    test_initial_kernel(1, model_dim, head_num, dim, Wq,0, 2,0);
    test_initial_kernel(1, model_dim, head_num, dim, Wk, 0, 2, 0);
    test_initial_kernel(1, model_dim, head_num, dim, Wv, 0, 2, 0);
    test_initial_kernel(1, head_num, dim, head_num*head_num, output_weights,  0, 2, 0);

#endif
     
       

    // generate q, k, v
    weight_multiply(batch, input_size, head_num, dim, model_dim, Wq, input_data, q);
    weight_multiply(batch, input_size, head_num, dim, model_dim, Wv, input_data, v);
    weight_multiply(batch, input_size, head_num, dim, model_dim, Wk, input_data, k);

    // match score function
    attention_score(batch, input_size, head_num, dim, q, k, v, score);

    // char name2[30] = "before softmax";
    // output_printf(batch,  head_num, input_size, input_size, score, name2);

    softmax_cpu(score, input_size, batch, head_num*input_size*input_size, head_num*input_size, input_size, 1, 1, score);

    // char name3[30] = "after softmax";
    // output_printf(batch,  head_num, input_size, input_size, score, name3);
    // matmul with v
    matmul_v(batch, input_size, head_num, dim, concat_head, score, v);

#if DEBUG
    char name4[30] = "concat_head";
    output_printf(batch,  head_num, input_size, dim, concat_head, name4);
#endif
    
    // MultiHead output for next sub layer
    multi_head_output(batch, input_size, head_num, dim, output, concat_head, output_weights);

#if DEBUG
    char name5[30] = "output_weights";
    output_printf(1,  head_num, dim, head_num*dim, output_weights, name5);
#endif
    
#if OUTPUT
    char namei[30] = "input_data";
    output_printf(batch, input_size, 1, model_dim, input_data, namei);
   // char name2[30] = "score";
    // output_printf(batch,  head_num, input_size, input_size, score, name2);
    char name[30] = "attention output";
    output_printf(batch, input_size, 1, head_num*dim, output, name);
#endif
     
    
    // free(layer.score);
    free(layer.Wq);
    free(layer.Wk);
    free(layer.Wv);
    // free(layer.q);
    // free(layer.k);
    // free(layer.v);
    // free(layer.concat_head);
    free(layer.output_weight);

}

void weight_multiply(int batch, int input_size, int head_num, int key_dim, int model_dim, float *weight, float*input, float *output){
// output [b, input_size, head_num*key_dim]
//output = np.einsum("bim, imt -> bit", input, weight)

#if OMP
omp_set_num_threads(THRD_NUM);
#pragma omp parallel for
#endif
 for(int bi = 0; bi < batch; bi++) {
        for(int ii = 0; ii < input_size; ii++) {
            for(int mi = 0; mi < model_dim; mi++) {
                for(int hi = 0; hi < head_num; hi++) {
                    for(int ki = 0; ki < key_dim; ki++){
                        output[bi*input_size*head_num*key_dim + ii*head_num*key_dim + hi*key_dim+ki] += 
                        input[bi*input_size*model_dim + ii*model_dim + mi] * 
                        weight[mi*head_num*key_dim + hi*key_dim  + ki];

                    }       
                }
            }
        }

    }

}

void attention_score(int batch, int input_size, int head_num, int key_dim, float *q, float *k, float *v, float *score){

    // score[bi][][]
#if DEBUG
    printf("intermid result:\n");
#endif

#if OMP
omp_set_num_threads(THRD_NUM);
#pragma omp parallel for
#endif
    for(int bi = 0; bi < batch; bi++) {
        for(int hi = 0; hi < head_num; hi++) {
            for(int qni = 0; qni < input_size; qni++) {
                for(int kni = 0; kni < input_size; kni++) {
                    for(int di = 0; di < key_dim; di++) {
                        score[bi*head_num*input_size*input_size + hi*input_size*input_size + qni*input_size + kni] += 
                        q[bi*input_size*head_num*key_dim + qni*head_num*key_dim + hi*key_dim  + di]*k[bi*input_size*head_num*key_dim + kni*head_num*key_dim + hi*key_dim  + di];
#if DEBUG
                        printf("score[%d][%d][%d][%d] (%f) += q[%d][%d][%d][%d](%f) * k[%d][%d][%d][%d](%f)\n", bi, hi,qni, kni,
                        score[bi*head_num*input_size*input_size + hi*input_size*input_size + qni*input_size + kni],bi,qni, hi, di, q[bi*input_size*head_num*key_dim + qni*head_num*key_dim + hi*key_dim  + di],bi, kni, hi, di,k[bi*input_size*head_num*key_dim + kni*head_num*key_dim + hi*key_dim  + di]);
#endif
                    }
#if DEBUG
                        printf("\n\n");
#endif
                score[bi*head_num*input_size*input_size + hi*input_size*input_size + qni*input_size + kni] /= pow(key_dim, 0.5);
                }
            }
        }

    }

}

void matmul_v(int batch, int input_size, int head_num, int key_dim, float *output, float *score, float *v) {

#if OMP
omp_set_num_threads(THRD_NUM);
#pragma omp parallel for
#endif      
    for(int bi = 0; bi < batch; bi++) {
        for(int hi = 0; hi < head_num; hi++) {
            for (int di = 0; di < key_dim; di++) {
                for(int sni = 0; sni < input_size; sni++) {
                    for(int vni = 0; vni < input_size; vni++) {
                        output[bi*head_num*input_size*key_dim + sni*head_num*key_dim + hi*key_dim + di] += 
                        score[bi*input_size*head_num*input_size + hi*input_size*input_size + sni*input_size  + vni]*v[bi*input_size*head_num*key_dim + vni*head_num*key_dim + hi*key_dim  + di];
#if DEBUG
                        printf("output[%d][%d][%d][%d] (%f) += score[%d][%d][%d][%d](%f) * v[%d][%d][%d][%d](%f)\n", bi, sni,hi, di,
                        output[bi*head_num*input_size*key_dim + sni*head_num*key_dim + hi*key_dim + di],bi,hi, sni, vni, score[bi*input_size*head_num*input_size + hi*input_size*input_size + sni*input_size  + vni],bi, vni, hi, di,v[bi*input_size*head_num*key_dim + vni*head_num*key_dim + hi*key_dim  + di]);
#endif
                    }
#if DEBUG
                        printf("\n\n");
#endif               
                }
            }
        }

    }

}

void multi_head_output(int batch, int input_size, int head_num, int key_dim, float *output, float *concat_head, float *output_weights) {
    // output = concat_head * W^Q    // ([b, n, h x d] X [h x d, d_{model}] = [b, n, d_{model}])
#if OMP
omp_set_num_threads(THRD_NUM);
#pragma omp parallel for
#endif
    for(int bi = 0; bi < batch; bi++) {
        for(int ni = 0; ni < input_size; ni++) {
            for (int hi = 0; hi < head_num*key_dim; hi++) {
                    for(int mi = 0; mi < (head_num*key_dim); mi++) {
                        output[bi*input_size*head_num*key_dim + ni*head_num*key_dim + hi] += 
                        concat_head[bi*input_size*head_num*key_dim + ni*head_num*key_dim + mi]*output_weights[mi*head_num*key_dim  + hi];
#if DEBUG
                        printf("output[%d][%d][%d](%f) += concat_head[%d][%d][%d](%f) * output_weights[%d][%d](%f)\n", 
                        bi, ni,hi,output[bi*input_size*head_num*key_dim + ni*head_num*key_dim + hi],
                        bi,ni,mi,concat_head[bi*input_size*head_num*key_dim + ni*head_num*key_dim + mi], 
                        mi, hi,  output_weights[mi*head_num*key_dim  + hi]);
#endif
                    }
#if DEBUG
                        printf("\n\n");
#endif               
                 }
            }
        }

    }


void backward_multi_head_attention_layer() {}

void update_multi_head_attention_layer() {}


void test_multi_head_attention(){

    int batch = 16;
    int input_size = 196;
    int head_num = 16;
    int key_dim = 80;
    int model_dim = 1280;
#if TIME
    struct timespec start, end;
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) {perror("clock unable gettune");}
#endif

    float *input_data, *Wq,*Wk, *Wv, *q, *k, *v, *score, *output, *output_weights, *final_result;
    input_data = (float*) malloc(batch*input_size*model_dim*sizeof(float));
    Wq =  (float*) malloc(model_dim*head_num*key_dim*sizeof(float));
    Wk =  (float*) malloc(model_dim*head_num*key_dim*sizeof(float));
    Wv =  (float*) malloc(model_dim*head_num*key_dim*sizeof(float));
    q = (float*) malloc(batch*input_size*head_num*key_dim*sizeof(float));
    k = (float*) malloc(batch*input_size*head_num*key_dim*sizeof(float));
    v = (float*) malloc(batch*input_size*head_num*key_dim*sizeof(float));
    score = (float*) malloc(batch*head_num*input_size*input_size*sizeof(float));
    output = (float*) malloc(batch*input_size*head_num*key_dim*sizeof(float));
    output_weights = (float*) malloc(head_num*key_dim*head_num*key_dim*sizeof(float));
    final_result = (float*) malloc(batch*input_size*head_num*key_dim*sizeof(float));
     
    // initial value for input_data, Weights(Wq, Wk, Wv, output_weight);
    test_initial_value(batch, input_size, head_num, key_dim,model_dim, input_data, Wq, Wk, Wv, q,k, v, score, output, output_weights);
    // printf("after initial, Wq[1][0][0]=%f\n",Wq[1*key_dim*head_num]);
    // printf("The size is : Wq = %ld, q = %ld, score = %ld, output = %ld, output_weights = %ld, final_result = %ld\n", 
    // (int)sizeof(Wq)/sizeof(Wq[12]), sizeof(*q), sizeof(*score), sizeof(*output), sizeof(*output_weights), sizeof(*final_result));
  
    weight_multiply(batch, input_size, head_num, key_dim, model_dim, Wq, input_data, q);
    weight_multiply(batch, input_size, head_num, key_dim, model_dim, Wk, input_data, k);
    weight_multiply(batch, input_size, head_num, key_dim, model_dim, Wv, input_data, v);

    char nameq[30] = "result q";
    output_printf(batch, input_size, head_num, key_dim, q, nameq);
    char namek[30] = "result k";
    output_printf(batch, input_size, head_num, key_dim, k, namek);
    char namev[30] = "result v";
    output_printf(batch, input_size, head_num, key_dim, v, namev);
    attention_score(batch, input_size, head_num, key_dim, q, k, NULL, score);
    char name2[30] = "score";
    output_printf(batch,  head_num, input_size, input_size, score, name2);
    softmax_cpu(score, input_size, batch, head_num*input_size*input_size, head_num*input_size, input_size, 1, 1, score);
    

    matmul_v(batch, input_size, head_num, key_dim, output, score, v);
    multi_head_output(batch, input_size, head_num, key_dim, final_result, output, output_weights);
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
    test_initial_kernel(1, batch, input_size, model_dim, input_data, 1, 1, 0.05 );
    test_initial_kernel(1, model_dim, head_num, key_dim, Wq, 1.0, 1, 0.005 );
    test_initial_kernel(1, model_dim, head_num, key_dim, Wk, 0.2, 1, -0.001 );
    test_initial_kernel(1, model_dim, head_num, key_dim, Wv, 1.2, 1, 0.005);
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





