# SparseOccVLA 数据转换

转换脚本位于仓库内 `data_conversion/`；大体积 rosbag 和转换数据保存在外接盘。

## 运行

```bash
/usr/bin/python3 data_conversion/convert_rosbag.py --config data_conversion/config.yaml
```

配置默认只转换指定的 `zd1_w_rosbag_out_000007.bag`。一个输入目录生成一个 scene；目录中的 bag 会按文件名处理。当前输出为 nuScenes 风格 `samples/`、`sweeps/LIDAR_TOP/` 和含 `infos`/`metadata` 的 `nuscenes_infos_test.pkl`。当前帧和历史相机按 SparseOccVLA 读取器约定顺序保存；关键帧之间保留约 30 Hz 的相机 sweep 图像。

同步主时钟为 `camera34/compressed`，输出采样 10 Hz；每一路相机、三路 LiDAR 和定位均按 0.1 秒容差做最近邻匹配，缺任一路则丢帧。相机先按标定畸变参数去畸变，再缩放到 1600×900 并同步更新内参。三个 LiDAR 点云先根据 `lidar_f/lidar_r/lidar_l.yml` 的 RT 变换到 `base_link`，再按 xyz/intensity 拼接。

## 当前验证边界

转换流程已在指定单个 bag 上跑通，产出 300 帧。已将定位姿态写入 `ego2global_translation` / `ego2global_rotation`，并按 `[position(3), quaternion_wxyz(4), acceleration(3), angular_velocity(3)]` 写入 13 维 `can_bus`。该 bag 的 Mercator 位置和速度在全程恒定，pkl 标记了 `pose_translation_is_constant_in_bag=True`；因此平移时序无效。规划命令仍是显式占位，六路相机的 `sensor2ego` / `sensor2lidar` / `sensor2global` 外参已写入每帧 `cams`，来自对应 `camera/mid_*.yml` 中的 R/T 和 `sensor_link`；`visualize_projection.py` 已实现按 `sensor2lidar` 和目标内参投影，并已对 frame 10 输出六路图像叠加；画面中地面/场景点云大致重合，但仍需人工确认标定方向和成像边缘误差。工程测试 pipeline 默认 `test_interval=6`，当前 30 Hz sweep + 10 Hz keyframe 下实际选择的历史间隔约为 0.2 秒，和 nuScenes 训练关键帧历史间隔不同，推理配置需评估是否调大到 12（约 0.4 秒）。该 bag 的定位平移恒定、规划命令仍为占位，当前不应视为已能可靠使用官方权重推理。
