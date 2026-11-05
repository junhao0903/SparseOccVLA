# SparseOccVLA 数据转换

实现位置在外接盘，仓库内 `data_conversion/` 是软链接。

## 运行

```bash
/usr/bin/python3 data_conversion/convert_rosbag.py --config data_conversion/config.yaml
```

配置默认只转换指定的 `zd1_w_rosbag_out_000007.bag`。一个输入目录生成一个 scene；目录中的 bag 会按文件名处理。当前输出为 nuScenes 风格 `samples/`、`sweeps/LIDAR_TOP/` 和 `nuscenes_infos_test.pkl`。

同步主时钟为 `camera34/compressed`，输出采样 10 Hz；每一路相机、三路 LiDAR 和定位均按 0.1 秒容差做最近邻匹配，缺任一路则丢帧。相机先按标定畸变参数去畸变，再缩放到 1600×900 并同步更新内参。三个 LiDAR 点云按 xyz/intensity 拼接。

## 当前验证边界

转换流程已在指定单个 bag 上跑通，产出 300 帧。当前 pkl 的 localization 原始字段已保留，但模型使用的 `can_bus`/ego pose 尚未从定位定义正确换算；LiDAR 到六相机的统一外参也需进一步确认。因此当前输出可用于检查提取、时间匹配、图像尺寸/内参和点云文件，不应视为已可直接用官方权重完成可靠推理。`visualize_projection.py` 目前打印内参和尺寸；投影叠加待外参方向确认后补齐。
