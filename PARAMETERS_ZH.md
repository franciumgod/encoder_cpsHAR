# EncoderBlock 参数使用说明（详细版）

本文档对应入口脚本：

```bash
conda run -n p2s python run_encoderblock.py [参数...]
```

---

## 1. 参数分组总览

### 1.1 数据与切窗

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--data_dir` | str | `encoderblock/data` | 数据目录，放 `.pkl` 数据文件。 |
| `--raw_dataset_file` | str | `cps_data_multi_label.pkl` | 原始数据文件名。 |
| `--step` | int | `500` | 切窗步长（不是降采样率）。 |
| `--window_seconds` | float | `2.0` | 窗口时长（秒）。 |
| `--sample_rate_hz` | int | `2000` | 输入数据原始采样率（Hz）。 |
| `--folds` | str | `1,2,3,4` | 训练折号（留一实验）。 |
| `--output` | str | 自动生成 | 输出目录。 |
| `--show_images` | bool | `False` | 是否显示图像窗口。 |

### 1.2 信号开关

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--special_signal` | bool | `False` | 使用预设的类专属信号组合。 |
| `--use_synth_signals` | bool | `True` | 是否保留 `Acc.norm/Gyro.norm` 两个合成轴。 |

### 1.3 采样视图（新）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--sampling_feature_mode` | enum | `raw` | 特征提取使用的数据视图：`raw` / `downsampled` / `both`。 |
| `--downsample_target_hz` | int | `100` | 降采样目标频率（Hz）。 |
| `--downsample_method` | enum | `interval` | 降采样方法：`interval` / `mean_pool` / `sliding_window`。 |
| `--downsample_window_size` | int | `40` | 仅 `sliding_window` 使用：滑动窗口大小。 |
| `--downsample_window_step` | int | `20` | 仅 `sliding_window` 使用：窗口滑动步长。 |

说明：
- `raw`：只用原始采样率特征。
- `downsampled`：只用降采样后的特征。
- `both`：两路都提特征，最后拼接。
- `interval`：按固定步长抽点（无平均）。
- `mean_pool`：不重叠平均池化（窗口=步长=factor）。
- `sliding_window`：按你指定的 `window_size` 和 `window_step` 做滑动窗口均值（默认 40/20）。

### 1.4 域与特征工程

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--feature_domain` | enum | `time_freq` | `time` / `freq` / `time_freq`。 |
| `--spectrum_method` | enum | `welch_psd` | `rfft` / `welch_psd` / `stft` / `dwt`。 |
| `--use_feature_engineering` | bool | `True` | 开启 basic FE（均值、方差、RMS、lag 等）。 |
| `--include_handcrafted_features` | bool | `True` | 是否把手工特征纳入最终输入。 |
| `--feature_fusion_mode` | enum | `auto` | `auto` / `handcrafted_only` / `encoder_only` / `hybrid`。 |

`feature_fusion_mode` 优先级高于单独开关：
- `handcrafted_only`：仅手工特征。
- `encoder_only`：仅 encoder latent。
- `hybrid`：手工 + encoder 一起。

### 1.5 数据增强

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--use_rotation_augment` | bool | `True` | 是否使用旋转增强。 |
| `--augment_count` | int | `2` | 每个匹配样本增强次数。 |
| `--augment_target` | str | `multilabel,Lifting(raising),Lifting(lowering)` | 增强目标，可多项逗号分隔。 |
| `--rotation_plane` | enum | `xz` | `xy` / `xz` / `yz` / `xyz`。 |
| `--rotation_max_degrees` | float | `15.0` | 最大旋转角度。 |

### 1.6 Encoder

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--use_encoder` | bool | `True` | 是否启用 encoder。 |
| `--encoder_backend` | enum | `mlp` | `none` / `mlp` / `cnn1d` / `rescnn1d`。 |
| `--encoder_axis_mode` | enum | `joint` | `joint`（全轴联合）/ `per_axis`（每轴独立）。 |
| `--encoder_output_dim` | int | `128` | encoder 输出维度目标。 |
| `--encoder_hidden_dim` | int | `256` | 隐层维度。 |
| `--encoder_channels` | csv int | `64,128` | CNN 通道配置。 |
| `--encoder_kernels` | csv int | `3,5,7` | CNN 卷积核配置。 |
| `--encoder_use_se` | bool | `False` | 是否使用 SE 模块。 |
| `--encoder_dropout` | float | `0.1` | dropout。 |
| `--encoder_epochs` | int | `15` | encoder 训练轮数。 |
| `--encoder_batch_size` | int | `128` | batch size。 |
| `--encoder_lr` | float | `1e-3` | 学习率。 |

### 1.7 分类器

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--model` | enum | `lightgbm` | `lightgbm` / `xgboost`。 |
| `--train_with_val` | bool | `True` | 是否并入 val 训练。 |
| `--seed` | int | `42` | 随机种子。 |

#### LightGBM 超参

`--lgbm_n_estimators`, `--lgbm_learning_rate`, `--lgbm_num_leaves`, `--lgbm_max_depth`, `--lgbm_subsample`, `--lgbm_colsample_bytree`, `--lgbm_min_child_samples`

#### XGBoost 超参

`--xgb_n_estimators`, `--xgb_learning_rate`, `--xgb_max_depth`, `--xgb_subsample`, `--xgb_colsample_bytree`

---

## 2. 你最关心的三类组合

### 2.1 原始采样特征（raw）

```bash
conda run -n p2s python run_encoderblock.py \
  --sampling_feature_mode raw \
  --feature_fusion_mode hybrid
```

### 2.2 降采样特征（downsampled）

```bash
conda run -n p2s python run_encoderblock.py \
  --sampling_feature_mode downsampled \
  --downsample_target_hz 100 \
  --downsample_method interval \
  --feature_fusion_mode hybrid
```

### 2.3 原始 + 降采样同时使用（both）

```bash
conda run -n p2s python run_encoderblock.py \
  --sampling_feature_mode both \
  --downsample_target_hz 100 \
  --downsample_method mean_pool \
  --feature_fusion_mode hybrid
```

### 2.4 滑动窗口平均降采样（sliding_window）

```bash
conda run -n p2s python run_encoderblock.py \
  --sampling_feature_mode downsampled \
  --downsample_target_hz 100 \
  --downsample_method sliding_window \
  --downsample_window_size 40 \
  --downsample_window_step 20 \
  --feature_fusion_mode hybrid
```

---

## 3. 每轴独立 CNN vs 全轴联合 CNN

### 全轴联合（joint）

```bash
--encoder_backend cnn1d --encoder_axis_mode joint
```

适合直接学习轴间耦合关系。

### 每轴独立（per_axis）

```bash
--encoder_backend cnn1d --encoder_axis_mode per_axis
```

适合做“每轴单独提取 + 后融合”的对照实验。

---

## 4. 输出维度是否固定

是可以固定的，规则如下：

1. 每个 encoder 分支输出目标由 `encoder_output_dim` 控制。  
2. 手工特征维度在给定配置下固定。  
3. `sampling_feature_mode=both` 时，最终维度是各分支拼接总和。  

也就是说：输入时间长度可以变（例如 4000 -> 200），最终模型输入依然是“配置可控的固定维度”。

---

## 5. run_summary.json 怎么看

每折里重点看：

- `sampling`：本次用的采样模式和降采样参数
- `feature_branches`：每个分支（raw/downsampled）的输入形状、采样率、特征维度、encoder后端
- `feature_shape`：最终拼接后的总维度
- `metrics`：每折指标
- `plots`：对应混淆矩阵图路径

---

## 6. GPU 建议（本项目）

- LightGBM/XGBoost：CPU 可跑。  
- MLP encoder：CPU 可跑。  
- CNN/ResCNN：建议 GPU（但不是必须）。  
- 后续 TCN：建议 GPU。  
- 后续 Transformer（长序列）：基本建议 GPU，否则速度很慢。  

---

## 7. 常见报错与处理

1. `feature_fusion_mode requires encoder backend not equal to 'none'`
- 原因：你选了 `encoder_only/hybrid`，但 `encoder_backend=none`。
- 处理：把 `encoder_backend` 改为 `mlp/cnn1d/rescnn1d`。

2. `No features available. Enable handcrafted features or encoder.`
- 原因：手工特征和 encoder 都被关闭了。
- 处理：打开任一分支，或用 `feature_fusion_mode` 强制设置。

3. `torch` 不存在
- 行为：`cnn1d/rescnn1d` 自动回退 PCA，并在 summary 里记录 `effective_backend`。
