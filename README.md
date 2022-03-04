# Attributes-Guided and Pure-Visual Attention Alignment for Few-Shot Recognition

PyTorch implementation of the paper:

* **Title**: Attributes-Guided and Pure-Visual Attention Alignment for Few-Shot Recognition
* **Author**: [Siteng Huang](https://kyonhuang.top/), [Min Zhang](https://remimz.github.io/), [Yachen Kang](https://yachenkang.github.io/), [Donglin Wang](https://milab.westlake.edu.cn/)
* **Conference**: Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI 2021)
* **More details**: [[arXiv]](https://arxiv.org/abs/2009.04724) | [[homepage]](https://kyonhuang.top/publication/attributes-guided-attention-module)

![](https://kyonhuang.top/files/AGAM/AGAM-model-structure.png)

## Requirements

The code runs correctly with

* Python 3.6
* PyTorch 1.2
* Torchvision 0.4

## How to run

```bash
# clone project
git clone https://github.com/bighuang624/AGAM.git
cd AGAM/models/agam_protonet

# download data and run on multiple GPUs with special settings
python train.py --train-data [train_data] --test-data [test_data] --backbone [backbone] --num-shots [num_shots] --train-tasks [train_tasks] --semantic-type [semantic_type] --multi-gpu --download

# Example: run on CUB dataset, Conv-4 backbone, 1 shot, single GPU
python train.py --train-data cub --test-data cub --backbone conv4 --num-shots 1 --train-tasks 50000 --semantic-type class_attributes
# Example: run on SUN dataset, ResNet-12 backbone, 5 shot, multiple GPUs
python train.py --train-data sun --test-data sun --backbone resnet12 --num-shots 5 --train-tasks 40000  --semantic-type image_attributes --multi-gpu
```

### Data Preparation

You can download datasets automatically by adding `--download` when running the program. However, here we give steps to manually download datasets to prevent problems such as poor network connection:

**CUB**:

1. Create the dir `AGAM/datasets/cub`;
2. Download `CUB_200_2011.tgz` from [here](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view), and put the archive into `AGAM/datasets/cub`;
3. Running the program with `--download`.

Update: As the CUB dataset need to be converted into hdf5 format with`--download`, we now release the generated hdf5 and json files for your convenience. You can download these files from [here](https://drive.google.com/drive/folders/1OTEIOyki35K1XQ_qRUZJTG0h_q7MDtDR?usp=sharing) and put them into `AGAM/datasets/cub`, and then `--download` is no longer needed when running the program.

**SUN**:

1. Create the dir `AGAM/datasets/sun`;
2. Download the archive of images from [here](http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz), and put the archive into `AGAM/datasets/sun`;
3. Download the archive of attributes from [here](http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB.tar.gz), and put the archive into `AGAM/datasets/sun`;
4. Running the program with `--download`.

## Citation

If our code is helpful for your research, please cite our paper:

```
@inproceedings{Huang2021AGAM,
  author = {Siteng Huang and Min Zhang and Yachen Kang and Donglin Wang},
  title = {Attributes-Guided and Pure-Visual Attention Alignment for Few-Shot Recognition},
  booktitle = {Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI 2021)},
  month = {February},
  year = {2021}
}
```

## Acknowledgement

Our code references the following projects:

* [Torchmeta](https://github.com/tristandeleu/pytorch-meta)
* [FEAT](https://github.com/Sha-Lab/FEAT)