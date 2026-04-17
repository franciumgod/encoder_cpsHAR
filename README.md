# EncoderBlock 最小可运行方案（CPS HAR）

这个目录用于快速做“手工特征 + Encoder + 树模型”的组合实验，重点支持：

- `step=500` 默认切窗（可改）
- `special signal` 开关
- `Acc.norm / Gyro.norm` 合成信号开关
- 域选择：`time / freq / time_freq`
- 频域方法：`rfft / welch_psd / stft / dwt`
- 旋转增强：`xy / xz / yz / xyz`
- 特征融合：手工特征、Encoder 特征、两者融合
- 采样视图：原始采样、降采样、两者并用
- Encoder 轴模式：全轴联合（joint）/ 每轴独立（per_axis）

## 目录结构

- `encoderblock/config.py`：默认配置
- `encoderblock/data.py`：窗口数据加载与 split
- `encoderblock/sampling.py`：采样视图与降采样
- `encoderblock/domain.py`：域特征构建
- `encoderblock/augment.py`：旋转增强
- `encoderblock/encoder.py`：Encoder 模块
- `encoderblock/tree_models.py`：LightGBM / XGBoost 多标签分类
- `encoderblock/metrics.py`：评估与绘图
- `encoderblock/run.py`：主入口
- `run_encoderblock.py`：快捷入口

## 数据目录

默认数据目录是：

- `encoderblock/data`

你也可以通过 `--data_dir` 指定别的目录。

## 快速运行

```bash
conda run -n p2s python run_encoderblock.py --model lightgbm --step 500
```

## 常见命令

1) 原始采样 + 手工特征 + MLP Encoder

```bash
conda run -n p2s python run_encoderblock.py \
  --sampling_feature_mode raw \
  --feature_fusion_mode hybrid \
  --encoder_backend mlp \
  --encoder_output_dim 128
```

2) 仅降采样视图（100Hz）+ 融合特征

```bash
conda run -n p2s python run_encoderblock.py \
  --sampling_feature_mode downsampled \
  --downsample_target_hz 100 \
  --downsample_method interval \
  --feature_fusion_mode hybrid
```

3) 原始 + 降采样 两路同时用

```bash
conda run -n p2s python run_encoderblock.py \
  --sampling_feature_mode both \
  --downsample_target_hz 100 \
  --downsample_method mean_pool \
  --feature_fusion_mode hybrid
```

4) 滑动窗口平均降采样（先平滑再抽点）

```bash
conda run -n p2s python run_encoderblock.py \
  --sampling_feature_mode downsampled \
  --downsample_target_hz 100 \
  --downsample_method sliding_window \
  --downsample_window_size 40 \
  --downsample_window_step 20 \
  --feature_fusion_mode hybrid
```

## 输出

`--output` 目录下会有：

- 每折混淆矩阵图
- 每折二分类混淆矩阵图
- 每折时间线图
- overall 汇总图
- `run_summary.json`

`run_summary.json` 会记录：

- 实际采样分支（`raw/downsampled/both`）
- 每个分支的输入形状、采样率、降采样因子
- 每个分支的 encoder 实际后端（含回退信息）
- 每折与汇总指标

## Torch 说明

当前环境如果没有 `torch`：

- `encoder_backend=mlp` 可正常跑
- `encoder_backend=cnn1d/rescnn1d` 会自动回退到 PCA（会写入 summary）
