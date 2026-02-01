# Train & inference
## Train SparseOccVLA
Train the sparse occupancy encoder first:

```
bash tools/dist_train.sh projects/configs/OmniDrive/sparseoccvla_stage1_4d_600q.py 8 --work-dir work_dirs/stage1_4d_600q/
```

Then fine-tune the entire model end-to-end, without any additional operations:
```
bash tools/dist_train.sh projects/configs/OmniDrive/sparseoccvla_stage3_4d_600q_forecasting.py 8 --work-dir work_dirs/stage3_4d_600q_forecasting/
```

## Evaluation SparseOccVLA
**1. Scene Understanding**

```bash
bash tools/dist_test.sh projects/configs/SparseOccVLA/sparseoccvla_stage3_4d_600q_forecasting.py ckpts/stage3_4d_600q_forecasting.pth 8 --save_path results/stage3_4d_600q_forecasting

cd evaluation
python eval_language.py ../results/stage3_4d_600q_forecasting
```
**2. Occupancy Forecasting**

```bash
python tools/test.py projects/configs/SparseOccVLA/sparseoccvla_stage3_4d_600q_forecasting.py ckpts/stage3_4d_600q_forecasting.pth --eval_occ
```



## (Optional)Train SparseOccVLA-Lidar
We also explore using raw point clouds as a form of weak supervision, which removes the reliance on dense semantic occupancy labels and improves the practical applicability of SparseOccVLA.

```
bash tools/dist_train.sh projects/configs/SparseOccVLA/sparseoccvla_stage1_3d_600q_lidar.py 8 --work-dir work_dirs/stage1_3d_600q_lidar/

bash tools/dist_train.sh projects/configs/SparseOccVLA/sparseoccvla_stage3_4d_600q_forecasting.py 8 --work-dir work_dirs/stage3_3d_600q_lidar/
```