# SuperCuts: Fine-Grained Image Segmentation

![image](https://github.com/user-attachments/assets/c14621e7-0fb7-4ec6-926f-dd40fb3e2986)


### Abstract
<div align="justify">
This project aims to perform image segmentation at various levels of granularity in an unsupervised way. Several powerful methods exist for performing instance, semantic, and panoptic segmentation. A relatively less popular task in computer vision is object part segmentation, where the task is to identify different meaningful parts belonging to the same object. We wish to perform image segmentation at a fine-grained level, i.e., segmenting the image into several perceptually similar regions, which can then be combined to achieve segmentation at a coarser level. 

<br/><br/>
We first look at popular image segmentation models and Vision Transformers (ViTs) and how they help in identifying perceptually similar regions within images. Next, we explore the task of superpixel segmentation, which is a way of grouping perceptually similar pixels into individual regions. It substantially reduces the number of primitives, reducing the computation  equirement for subsequent downstream tasks. To this end, we take a look at classic as well as deep learning techniques. Deep learning based methods allow us to learn representations and embeddings for individual superpixels in higher dimensional spaces. We try to improve upon existing methods using pre-trained Convolutional Neural Networks (CNNs) and ViTs and integrating them in the superpixel segmentation pipeline. Finally, we will explore methods of combining neighboring superpixels to achieve a coarse segmentation of the target image. To this end, we will explore classical techniques like spectral clustering, and modern methods based on Graph Neural Networks. By combining the methods for superpixel segmentation and  ine-grained segmentation, we can perform both tasks using a single model. These techniques will allow us to achieve meaningful segmentations of images in an unsupervised or semi-supervised fashion. Our final goal is to devise a model capable of performing superpixel segmentation and, as a downstream task, can also perform segmentation at various levels of granularity.

<br/><br/>
This project is divided into 3 sections: superpixel segmentation, object localization and segmentation, and finally part segmentation.
</div>

## Superpixel Segmentation
We train a Superpixel Sampling Network (SSN) with a DINO backbone for superpixel segmentation. We finetune only the last few layers on the BSDS500 dataset. We call this model SSN_DINO. We have made use of the code provided by [^1] and [^2].

To train your own model, download the BSDS500 dataset <a href='https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500'>here</a>. Your directory structure should look like below:
```
project_root/
└──data/
  └── BSDS500/
    └── BSDS500/
      └── data/
          ├── groundTruth/
          │   ├── train/
          │   ├── val/
          │   └── test/
          └── images/
              ├── train/
              ├── val/
              └── test/
```

Run the following code to train your own SSN_DINO:<br></br>

```
python train_ssn_dino.py
```

You can also use our trained model from the directory `model_checkpoints/ssn_dino.pth`

```
python infer_ssn_dino.py --image /path/to/image
```

Sample results: <br></br>
<img src="https://github.com/user-attachments/assets/a5931fc6-7c9d-4203-8543-a26df4090c22" alt="myplabels"  width="450px" height="300px">
<img src="https://github.com/user-attachments/assets/a78eae07-ac62-4cc9-b742-47edc3692ec2" alt="001"  width="450px" height="300px">


## Object Localization and Segmentation
We use SSN_DINO to extract individual superpixels along with their features. Then we follow the Deep Spectral Methods [^1] [^2] approach at the superpixel level to construct an affinity matrix of superpixels and then discretize the superpixels which belong to the dominant object in the image. 

We can run the object localization task on PASCAL VOC 2007 and PASCAL VOC 2012 trainval datasets. You can also pass additional parameters for e.g. the number of superpixels, model checkpoint and the dataset.

### PASCAL VOC 2007 (Download <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"> here</a>).
Your directory structure should look like below:
```
project_root/
└──data/
  └── voc2007trainval/
    └── VOCdevkit/
      └── VOC2007/
          ├── Annotations/
          ├── ImageSets/
          ├── JPEGImages/
          ├── SegmentationClass/
          └── SegmentationObject/
```
Install the requirements:
```
pip install -r requirements_ssn.txt
```
Run the below command:
```
python object_localization.py --weight 'model_checkpoints/ssn_dino.pth' --dataset 'VOC07' --nspix '100'
```

### PASCAL VOC 2012 (Download <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"> here</a>).
Your directory structure should look like below:
```
project_root/
└──data/
  └── voc2012trainval/
    └── VOCdevkit/
      └── VOC2012/
          ├── Annotations/
          ├── ImageSets/
          ├── JPEGImages/
          ├── SegmentationClass/
          └── SegmentationObject/
```
Run the below command:
```
python object_localization.py --weight 'model_checkpoints/ssn_dino.pth' --dataset 'VOC12' --nspix '100'
```

Sample object localization results: <br></br>
<img src="https://github.com/user-attachments/assets/3a14afa0-3fd2-4dd0-9b00-e0f9e560a142" alt="myplabels"  width="450px" height="300px">
<img src="https://github.com/user-attachments/assets/ece7caf7-f4df-4983-a48c-ed2ae53690cc" alt="001"  width="450px" height="300px">

We can run the object segmentation task on the CUB-200-2011, ECSSD, DUTS and DUT-OMRON datasets using the below commands. You can also pass additional parameters for e.g. the number of superpixels, model checkpoint and the dataset.

### CUB-200-2011 (Download <a href="https://www.vision.caltech.edu/datasets/cub_200_2011/"> here</a>).
Your directory structure should look like below:
```
project_root/
└──data/
  └── CUB/
    └── CUB_200_2011/
        ├── attributes/
        ├── images/
        ├── parts/
        ├── segmentations/
        ├── bounding_boxes
        ├── classes
        ├── image_class_labels
        ├── images
        ├── README
        └── train_test_split
```
Run the below command:
```
python object_segmentation.py --weight 'model_checkpoints/ssn_dino.pth' --dataset 'CUB' --nspix '100'
```

### ECSSD (Download <a href="https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html">here</a>).
Your directory structure should look like below:
```
project_root/
└──data/
  └── ECSSD-data/
      ├── images/
      └── ground_truth_mask/
```
Run the below command:
```
python object_segmentation.py --weight 'model_checkpoints/ssn_dino.pth' --dataset 'ECSSD' --nspix '100'
```

### DUTS (Download <a href="http://saliencydetection.net/duts/">here</a>).
Your directory structure should look like below:
```
project_root/
└──data/
  └── DUTS/
      ├── DUTS-TE-Image/
      └── DUTS-TE-Mask/
```
Run the below command:
```
python object_segmentation.py --weight 'model_checkpoints/ssn_dino.pth' --dataset 'DUTS' --nspix '100'
```

### DUT-OMRON (Download <a href="http://saliencydetection.net/dut-omron/"> here</a>).
Your directory structure should look like below:
```
project_root/
└──data/
  └── DUT-OMRON/
      ├── DUT-OMRON-image/
      └── pixelwiseGT-new-PNG/
```
Run the below command:
```
python object_segmentation.py --weight 'model_checkpoints/ssn_dino.pth' --dataset 'DUT-OMRON' --nspix '100'
```

Sample object segmentation results:

<img src="https://github.com/user-attachments/assets/ed132a88-294d-4fd5-be95-5b72c88b15f6"  alt="os1"  width="450px" height="300px">
<img src="https://github.com/user-attachments/assets/4180a033-2cd3-4bfa-93a3-58ee2e3ac90a"  alt="os2"  width="450px" height="300px">

<br><br>

To perform localization and segmentation inference on your own image:
```
python infer_localization_and_segmentation.py --weight 'model_checkpoints/ssn_dino.pth' --image /path/to/image
```




## Part Segmentation


## References
[^1]: https://github.com/NVlabs/ssn_superpixels
[^2]: https://github.com/perrying/ssn-pytorch
[^3]:https://github.com/lukemelas/unsupervised-image-segmentation
[^4]:https://github.com/lukemelas/deep-spectral-segmentation
