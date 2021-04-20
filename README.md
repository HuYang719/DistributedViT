## Distributed ViT

#### Build
```sh
cd ./distributedViT
make
```
#### Test

1. First change the image filepath in test_transformer() in transformer.c
2. Run with command:
```sh
./darknet transformer test ./cfg/vit-test.cfg 
```
or test the inference time

```sh
time ./darknet transformer test ./cfg/vit-test.cfg 
```