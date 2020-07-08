# SCAN: Learning to Classify Images without Labels



This repo contains the Pytorch implementation of our paper:
> [**SCAN: Learning to Classify Images without Labels**](https://arxiv.org/pdf/2005.12320.pdf)
>
> [Wouter Van Gansbeke](https://twitter.com/WGansbeke), [Simon Vandenhende](https://twitter.com/svandenh1), [Stamatios Georgoulis](https://twitter.com/stam_g), Marc Proesmans and Luc Van Gool.

__Accepted at ECCV 2020. Watch the explanation of our paper by Yannick Kilcher on [YouTube](https://www.youtube.com/watch?v=hQEnzdLkPj4).__
<p align="center">
    <img src="images/teaser.jpg" />

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-to-classify-images-without-labels/unsupervised-image-classification-on-imagenet)](https://paperswithcode.com/sota/unsupervised-image-classification-on-imagenet?p=learning-to-classify-images-without-labels)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-to-classify-images-without-labels/unsupervised-image-classification-on-cifar-10)](https://paperswithcode.com/sota/unsupervised-image-classification-on-cifar-10?p=learning-to-classify-images-without-labels)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-to-classify-images-without-labels/unsupervised-image-classification-on-stl-10)](https://paperswithcode.com/sota/unsupervised-image-classification-on-stl-10?p=learning-to-classify-images-without-labels)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-to-classify-images-without-labels/unsupervised-image-classification-on-cifar-20)](https://paperswithcode.com/sota/unsupervised-image-classification-on-cifar-20?p=learning-to-classify-images-without-labels)
</p>

## Installation
The code runs with recent Pytorch versions, e.g. 1.4. 
Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/), the most important packages can be installed as:
```shell
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch
conda install matplotlib scipy scikit-learn   # For evaluation and confusion matrix visualization
conda install faiss-gpu                       # For efficient nearest neighbors search 
conda install pyyaml easydict                 # For using config files
conda install termcolor                       # For colored print statements
```
We refer to the `requirements.txt` file for an overview of the packages in the environment we used to produce our results.

## Training

### Setup
The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the datasets in `utils/mypath.py`, e.g. `/path/to/cifar10`.
- Specify the output directory in `configs/env.yml`. All results will be stored under this directory. 

Our experimental evaluation includes the following datasets: CIFAR10, CIFAR100-20, STL10 and ImageNet. The ImageNet dataset should be downloaded separately and saved to the path described in `utils/mypath.py`. Other datasets will be downloaded automatically and saved to the correct path when missing.

### Train model
The configuration files can be found in the `configs/` directory. The training procedure consists of the following steps:
- __STEP 1__: Solve the pretext task i.e. `simclr.py`
- __STEP 2__: Perform the clustering step i.e. `scan.py`
- __STEP 3__: Perform the self-labeling step i.e. `selflabel.py`

For example, run the following commands sequentially to perform our method on CIFAR10:
```shell
python simclr.py --config_env configs/your_env.yml --config_exp configs/pretext/simclr_cifar10.yml
python scan.py --config_env configs/your_env.yml --config_exp configs/scan/scan_cifar10.yml
python selflabel.py --config_env configs/your_env.yml --config_exp configs/selflabel/selflabel_cifar10.yml
```
### Remarks
The provided hyperparameters are identical for CIFAR10, CIFAR100-20 and STL10. However, fine-tuning the hyperparameters can further improve the results. We list the most important hyperparameters of our method below:
- Entropy weight: Can be adapted when the of clusters changed. In general, try to avoid imbalanced clusters during training. 
- Confidence threshold: When every cluster contains a sufficiently large amount of confident samples, it can be beneficial to increase the threshold. This generally helps to decrease the noise.
- Number of neighbors in SCAN: The dependency on this hyperparameter is rather small as shown in the paper. 

## Model Zoo
### Pretext tasks
We perform the instance discrimination task in accordance with the scheme from [SimCLR](https://arxiv.org/abs/2002.05709) on CIFAR10, CIFAR100 and STL10. Pretrained models can be downloaded from the links listed below. On ImageNet, we use the pretrained weights provided by [MoCo](https://github.com/facebookresearch/moco) and transfer them to be compatible with our code repository. 

| Dataset          | Download link |
|------------------|---------------| 
|CIFAR10           | [Download](https://drive.google.com/file/d/1Cl5oAcJKoNE5FSTZsBSAKLcyA5jXGgTT/view?usp=sharing)  |
|CIFAR100          | [Download](https://drive.google.com/file/d/1huW-ChBVvKcx7t8HyDaWTQB5Li1Fht9x/view?usp=sharing)  |
|STL10             | [Download](https://drive.google.com/file/d/1261NDFfXuKR2Dh4RWHYYhcicdcPag9NZ/view?usp=sharing)  |

### Clustering
We provide the following pretrained models after training with the __SCAN-loss__, and after the __self-labeling__ step. The best models can be found here and we futher refer to the paper for the averages and standard deviations.

| Dataset          | Step              | ACC                   |  NMI            |  ARI      |Download link |
|------------------|-------------------|---------------------- |-----------------|-----------|--------------|
| CIFAR10          |  SCAN-loss        | 81.6                  | 71.5            | 66.5      |[Download](https://drive.google.com/file/d/1v6b6jJY5M4-duSqWpGFmdf9e9T3dPrx0/view?usp=sharing)  |
|                  |  Self-labeling    | 88.3                  | 79.7            | 77.2      |[Download](https://drive.google.com/file/d/18gITFzAbQsGS5vt8hyi5HjbeRDsVLihw/view?usp=sharing) |
| CIFAR100         |  SCAN-loss        | 44.0                  | 44.9            | 28.3      |[Download](https://drive.google.com/file/d/1pPCi1QG05kP_JdoX29dxEhVddIRk68Sd/view?usp=sharing)  |
|                  |  Self-labeling    | 50.7                  | 48.6            | 33.3      |[Download](https://drive.google.com/file/d/11mEmpDMyq63pM4kmDy6ItHouI6Q__uB7/view?usp=sharing)  |
| STL10            |  SCAN-loss        | 79.2                  | 67.3            | 61.8      |[Download](https://drive.google.com/file/d/1y1cnGLpeTVo80cnWhAJy-B72FYs2AjZ_/view?usp=sharing)  |
|                  |  Self-labeling    | 80.9                  | 69.8            | 64.6      |[Download](https://drive.google.com/file/d/1uNYN9XOMIPb40hmxOzALg4PWhU_xwkEF/view?usp=sharing)  |
| ImageNet-50      |  SCAN-loss        | 75.1                  | 80.5            | 63.5      |[Download](https://drive.google.com/file/d/1UdBtvCHVGd08x8SiH6Cuh6mQmqsADg0t/view?usp=sharing)  |
|                  |  Self-labeling    | 76.8                  | 82.2            | 66.1      |[Download](https://drive.google.com/file/d/1iOE4_lQ4w7CGPLU4algBDG34nz68eN8o/view?usp=sharing)  |
| ImageNet-100     |  SCAN-loss        | 66.2                  | 78.7            | 54.4      |[Download](https://drive.google.com/file/d/1tcROQ3wc_MbxmLr05qt-UvF9yVrBwBq9/view?usp=sharing)  |
|                  |  Self-labeling    | 68.9                  | 80.8            | 57.6      |[Download](https://drive.google.com/file/d/1VVgRpJ9DJn9dNrbAKbfPer2FllTvP6Cs/view?usp=sharing)  |
| ImageNet-200     |  SCAN-loss        | 56.3                  | 75.7            | 44.1      |[Download](https://drive.google.com/file/d/1oO-OCW2MiXmNC4sD6pkw8PurYScX7oVW/view?usp=sharing)  |
|                  |  Self-labeling    | 58.1                  | 77.2            | 47.0      |[Download](https://drive.google.com/file/d/11dfobUwy6ragh7PoqFagoEns5-teWalm/view?usp=sharing)  |


### Evaluation
Pretrained models from the model zoo can be evaluated using the `eval.py` script. For example, the model on cifar-10 can be evaluated as follows:
```shell
python eval.py --config_exp configs/scan/scan_cifar10.yml --model $MODEL_PATH 
```
Visualizing the prototype images is easily done by setting the `--visualize_prototypes` flag. For example on cifar-10:
<p align="center">
    <img src="images/prototypes_cifar10.jpg" width="600" />
</p>


## Citation

If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{wvangansbeke2020learning,
  title={SCAN: Learning to Classify Images without Labels},
  author={Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
For any enquiries, please contact the main authors.

## License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).

## Acknoledgements
This work was supported by Toyota, and was carried out at the TRACE Lab at KU Leuven (Toyota Research on Automated Cars in Europe - Leuven).
