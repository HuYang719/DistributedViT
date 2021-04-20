#include "network.h"
#include "utils.h"
#include "parser.h"

void test_transformer(char *cfgfile, char *weightfile, char *filename)
{
    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    char buff[256];
    char *input = buff;

    if(weightfile){
        load_weights(&net, weightfile);
    }

    filename = "/home/lucyyang/Documents/02-Darknet/darknet/data/horses.jpg";
    printf("filename is %s\n", filename);
    
    while(1){
        if(filename){
            strncpy(input, filename, 256);
            printf("input is %s\n", input);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.h, net.w);
        float *X = sized.data;
        printf("the size of data image is %d\n", sizeof(X));
        clock_t time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));




        free_image(im);
        free_image(sized);



        if (filename) break;
    }
  
}

void run_transformer(int argc, char **argv)
{
// 	int dont_show = find_arg(argc, argv, "-dont_show");
// 	int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
//     int json_port = find_int_arg(argc, argv, "-json_port", -1);
// 	char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
//     char *prefix = find_char_arg(argc, argv, "-prefix", 0);
//     float thresh = find_float_arg(argc, argv, "-thresh", .2);
// 	float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
//     int cam_index = find_int_arg(argc, argv, "-c", 0);
//     int frame_skip = find_int_arg(argc, argv, "-s", 0);
// 	int ext_output = find_arg(argc, argv, "-ext_output");
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;

    if(0==strcmp(argv[2], "test")) test_transformer(cfg, weights, filename);
}