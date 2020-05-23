# 简介

这里是使用caffe作为部署框架，来实验各种模型



# 前提

- 安装caffe





# 编译

```shell
mkdir build && cd build
cmake ..
make
```



# 目录说明





# 模型说明



## 语义分割



## 图像分类

bvlc_reference_caffenet模型

运行命令：

```shell
./semantic_division_test ../model/bvlc_refernce_caffenet/deploy.prototxt ../model/bvlc_refernce_caffenet/bvlc_reference_caffenet.caffemodel ../model/bvlc_refernce_caffenet/imagenet_mean.binaryproto ../data/synset_words.txt ../data/cat.jpg
```

运行的结果：

```shell
---------- Prediction for ../data/cat.jpg ----------
0.3134 - "n02123045 tabby, tabby cat"
0.2380 - "n02123159 tiger cat"
0.1235 - "n02124075 Egyptian cat"
0.1003 - "n02119022 red fox, Vulpes vulpes"
0.0715 - "n02127052 lynx, catamount"

```



## 目标识别

