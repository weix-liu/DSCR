# Unsupervised Domain Adaptation for Remote Sensing Vehicle Detection using Domain-specific Channel Recalibration
A Pytorch Implementation of Unsupervised Domain Adaptation for Remote Sensing Vehicle Detection using Domain-specific Channel Recalibration. 

## Introduction
Please follow [DA_Detection](https://github.com/VisionLearningGroup/DA_Detection.git) respository to setup the environment. In this project, we use Pytorch 0.4.0. 

## Datasets
### Datasets Preparation
* **GTAV10k dataset:** Download our [GTAV10k](https://drive.google.com/file/d/1dlGy5L7ko_I8qdPRTWhK5wc1Z-VXfK7a/view) dataset, see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data).


### Datasets Format
All codes are written to fit for the **format of PASCAL_VOC**.  
If you want to use this code on your own dataset, please arrange the dataset in the format of PASCAL, make dataset class in ```lib/datasets/```, and add it to ```lib/datasets/factory.py```, ```lib/datasets/config_dataset.py```. Then, add the dataset option to ```lib/model/utils/parser_func.py```.

## Models
### Pre-trained Models
In our experiments, we used two pre-trained models on ImageNet, i.e., VGG16 and ResNet101. Please download these two models from:
* **VGG16:** [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and write the path in **__C.VGG_PATH** and **__C.RESNET_PATH** at ```lib/model/utils/config.py```.

## Train
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
python trainval_DSCR_DWPA.py --gc --cuda --lr 0.001  --net res101 --dataset  gta_car   --dataset_t  ucas_car  --save_dir path_to_save
```
## Test
```
CUDA_VISIBLE_DEVICES=$GPU_ID \
python test_DSCR_DWPA.py --gc --dataset ucas_car --net res101 --cuda --load_name path_to_model
```
