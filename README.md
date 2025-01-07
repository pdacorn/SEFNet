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



