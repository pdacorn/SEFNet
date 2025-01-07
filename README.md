# SEFNet

## 

- CUDA/CUDNN
- Python3
- Packages found in requirements.txt

## Datasets

### NYUv2 

```
mkdir ../data/new_data/
```

Download the dataset from [here](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).

### CityScapes

```
mkdir ../data/CityScapes/
```

Download the dataset from [here](https://www.cityscapes-dataset.com/).

### SUNRGBD

```
mkdir ../data/SUNRGBD/
```

Download the dataset from [here](https://rgbd.cs.princeton.edu/).

## Experiments

#### For example, for NYUv2:

```
python3 Flowdecoder_train.py
python3 Flowdecoder_val.py
```
## Citation
If you find this repo useful for your research, please consider citing the paper as follows:
```
@article{xiang2024self,
  title={Self-Enhanced Feature Fusion for RGB-D Semantic Segmentation},
  author={Xiang, Pengcheng and Yao, Baochen and Jiang, Zefeng and Peng, Chengbin},
  journal={IEEE Signal Processing Letters},
  year={2024},
  publisher={IEEE}
}
```


