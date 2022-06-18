# Multi-grained Spatio-Temporal Features Perceived Network for Event-based Lip-Reading (CVPR 2022)
## Introduction
In this [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.pdf),
we introduce a novel type of optical sensor, event cameras, to tackle automatic lip-reading
problem. Event cameras are biologically inspired optical
sensors. Unlike conventional cameras that capture images
at a fixed rate, event cameras capture per-pixel brightness
changes asynchronously in the microsecond level. For the
ALR task that requires the perception of fine-grained spatiotemporal features, event cameras have significant advantages over conventional cameras in terms of technology and
applications: 1) the high temporal resolution of event cameras allow them to record finer-grained movements; 2) their
output does not contain much redundant visual information
since only brightness changes of the scene are recorded; 3)
they are low-power and can work on challenging lighting
conditions which are essential in real-world applications.
 This code is the Pytorch implementation of our work.
![image](https://github.com/tgc1997/event-based-lip-reading/blob/main/misc/framework.jpg)


## Dependencies
* Python 3.7
* Pytorch 1.6.0

## Prepare
1. Create a new folder and name it `log`;
2. Download [DVS-Lip](https://drive.google.com/file/d/1dBEgtmctTTWJlWnuWxFtk8gfOdVVpkQ0/view) dataset, and put it in `data` folder;
3. Download the pre-trained model [MSTP](https://drive.google.com/drive/folders/1xi9qoQ0LjEoo6SvWOH2pSXrdjia9_jJC?usp=sharing), and put it in `log`. 

## Test
You can test our provided pre-trained model by running
```python
python main.py --gpus=0 --num_bins=1+7 --test=True --alpha=4 --beta=7 --weights=mstp
```

## Training
You can also train your own model by running
```python
python main.py --gpus=0 --num_bins=1+4 --test=False --alpha=4 --beta=4 --log_dir=debug
```

## Citation
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```
@InProceedings{Tan_2022_CVPR,
    author    = {Tan, Ganchao and Wang, Yang and Han, Han and Cao, Yang and Wu, Feng and Zha, Zheng-Jun},
    title     = {Multi-Grained Spatio-Temporal Features Perceived Network for Event-Based Lip-Reading},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {20094-20103}
}
```
