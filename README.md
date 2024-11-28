<div align="center">

# [P2PFormer: A Primitive-to-polygon Method for Regular Building Contour Extraction from Remote Sensing Images](https://arxiv.org/pdf/2406.02930)
[Tao Zhang](https://scholar.google.com/citations?user=3xu4a5oAAAAJ&hl=zh-CN), Shiqing Wei, Yikang Zhou, Muying Luo, Wenling Yu, [ShunPing Ji](https://scholar.google.com/citations?user=FjoRmF4AAAAJ&hl=zh-CN)
</div>

## Install 
```shell
# create the conda env
conda create -n p2pformer python=3.9
conda activate p2pformer

# install the pytorch 1.x, 
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# install the mmcv
pip install opemim
mim install mmcv-full==1.7.2

# install the mmdet and p2pformer
pip install -e .
```

## Prepare datas
Download the WHU, WHU-Mix and CrowdAI datastes, then change the data path in [whu_line.py](./p2pformer/configs/_base_/datasets/whu_line.py), [whu-mix_line.py](./p2pformer/configs/_base_/datasets/whu-mix_line.py) and [crowdAI_line](./p2pformer/configs/_base_/datasets/crowdAI_line.py).

## Getting Started

### train 

```shell
# on whu dataset
PYTHONPATH=. bash tools/dist_train.sh p2pformer/configs/retrain/p2p2former_corner_whu.py 8
# on whu-mix dataset
PYTHONPATH=. bash tools/dist_train.sh p2pformer/configs/retrain/p2p2former_corner_whu-mix.py 8
# on crowdai dataset
PYTHONPATH=. bash tools/dist_train.sh p2pformer/configs/retrain/crowdai.py 8
```

### test
```shell
# on whu dataset
PYTHONPATH=. bash tools/dist_test.sh p2pformer/configs/retrain/p2p2former_corner_whu.py /path/to/model.pth 8
# on whu-mix dataset
PYTHONPATH=. bash tools/dist_train.sh p2pformer/configs/retrain/p2p2former_corner_whu-mix.py /path/to/model.pth 8
# on crowdai dataset
PYTHONPATH=. bash tools/dist_train.sh p2pformer/configs/retrain/crowdai.py /path/to/model.pth 8
```

### visualization
1. Please uncomment lines 229-231 in [p2pformer.py](./p2pformer/models/p2pformer.py), and the polygon prediction results will be stored in `work_dirs/json_preds/`.
2. Run the test script or demo script.
3. Run the [polygon_show.py](./tools/polygon_visualize/polygon_show.py) to obtain the visualization results.


```BibTeX
@article{zhang2024p2pformer,
  title={P2PFormer: A Primitive-to-polygon Method for Regular Building Contour Extraction from Remote Sensing Images},
  author={Zhang, Tao and Wei, Shiqing and Zhou, Yikang and Luo, Muying and Yu, Wenling and Ji, Shunping},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
