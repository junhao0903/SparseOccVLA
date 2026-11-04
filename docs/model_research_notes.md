# SparseOccVLA 模型研究笔记

记录阅读和运行 SparseOccVLA 时值得保留的结论；后续可继续追加。

## Forward 耗时、参数量与模块职责

### 当前单帧实测

测试使用 Stage3 forecasting 配置、batch size 1、RTX 4090 和一帧带语言标注的验证样本。计时发生在完整模型已构建并加载权重后。Stage3 测试调用了 4 次 `lm_head.chat`。

| 阶段/模块 | 单帧耗时 | 参数量（约） | 说明 |
| --- | ---: | ---: | --- |
| Stage1：OPUS 主预测 | 44 ms | 698.3M checkpoint 张量元素 | `pts_bbox_head`；这是 Stage3 中 OPUS 主预测模块的计时，含主占用预测，不是独立 Stage1 配置的端到端计时。元素数按 Stage3 checkpoint 中 `pts_bbox_head` 张量统计，含少量 buffer，未单独统计 `named_parameters()`。 |
| Stage2：VL / InternVL2-2B | 约 3.58 s | 约 2.206B 参数 | `lm_head.chat` 4 次调用的累计墙钟时间；语言生成逐 token 进行，耗时会随提问次数和生成长度变化。参数量按本地 InternVL2-2B safetensors 参数计数。 |
| Stage3：新增模块及未来预测 | 约 30 ms | 约 142M BF16 等效权重 | 根据本次计时中 global attention、特征投影/解码、未来占用解码等新增部分汇总估计；非严格独占计时，部分 Python 开销及模块调用与其他阶段相邻。参数量由 Stage3 相对 Stage1/InternVL 的新增 checkpoint 权重字节数折算，属于估值。 |
| 完整 Stage3 forward | 约 3.69 s | — | 单帧完整测试 forward；包含 4 次 VL chat、占用预测和 forecasting。 |
