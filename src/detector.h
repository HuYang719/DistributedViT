#include <stdlib.h>
#include "darknet.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void*, const void*);
#ifdef __USE_GNU
typedef __compar_fn_t comparison_fn_t;
#endif
#endif

#include "http_stream.h"

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, int mjpeg_port, int show_imgs, int benchmark_layers, char* chart_path);
static int get_coco_image_id(char *filename);
static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h);
void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h);
void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h);
static void print_kitti_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h, char *outfile, char *prefix);
static void eliminate_bdd(char *buf, char *a);
static void get_bdd_image_id(char *filename);
static void print_bdd_detections(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h);
void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile);
void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile);
int detections_comparator(const void *pa, const void *pb);
float validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, network *existing_net);
void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);


