# SuperCuts: Fine-Grained Image Segmentation

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

Sample results: <br></br>
![myplabels](https://github.com/user-attachments/assets/a5931fc6-7c9d-4203-8543-a26df4090c22) ![001](https://github.com/user-attachments/assets/a78eae07-ac62-4cc9-b742-47edc3692ec2)



## Object Localization and Segmentation

## Part Segmentation


## References
[1^]: https://github.com/NVlabs/ssn_superpixels
[^2]: https://github.com/perrying/ssn-pytorch
