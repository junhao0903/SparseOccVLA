# Data Preparation

1. Download [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

2. Download [Info Files](https://huggingface.co/MSunDYY2001/SparseOccVLA/tree/main/data_infos). Unzip the data_nusc.tar.gz and move the files into `./data/nuscenes`.

3. Download occupancy labels 'gts' from [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D).

4. Download [checkpoints](https://huggingface.co/MSunDYY2001/SparseOccVLA/tree/main/ckpts) and save in `./ckpts`.

5. Download [pretrained LLM](https://huggingface.co/OpenGVLab/InternVL2-2B) and save in `./ckpts`.

**Folder structure**
```
SparseOccVLA
├── projects/
|   ├── configs
├── mmdetection3d/
├── tools/
├── ckpts/
|   ├── InternVL2-2B
|   ├── cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
|   ├── stage1_3d_600q_lidar.pth
|   ├── stage1_4d_600q.pth
|   ├── stage3_4d_600q_forecasting.pth
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── gts/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── conv/
│   │   ├── desc/
│   │   ├── keywords/
│   │   ├── vqa/
│   │   ├── nuscenes2d_ego_temporal_infos_train.pkl
│   │   ├── nuscenes2d_ego_temporal_infos_val.pkl
│   │   ├── data_dict_sample.pkl
│   │   ├── data_dict_subset_B.json
│   │   ├── data_dict_subset_B_val.pkl
```

