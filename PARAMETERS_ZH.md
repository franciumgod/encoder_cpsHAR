# EncoderBlock 参数说明

入口文件：

```bash
python run.py [参数...]
```

## 1. 数据与路径

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--data_dir` | str | `None` | 首选数据目录。查找顺序：`--data_dir -> 当前目录/data -> 上一级目录/cpsHAR/data(如果存在)` |
| `--raw_dataset_file` | str | `cps_data_multi_label.pkl` | 原始数据文件。只有在需要自动生成窗口数据时才会使用。 |
| `--window_dataset_file` | str | `None` | 直接指定已经切好的窗口数据文件。 |
| `--step` | int | `500` | 窗口步长，对应约定命名文件 `cps_windows_2s_2000hz_step_<step>.pkl`。 |
| `--window_seconds` | float | `2.0` | 单个窗口时长，单位秒。 |
| `--sample_rate_hz` | int | `2000` | 原始采样率。 |
| `--folds` | str | `1,2,3,4` | 使用哪些 fold。 |
| `--output` | str | 自动生成 | 输出目录。默认在当前目录的 `output/` 下。 |
| `--show_images` | bool | `False` | 是否显示图像窗口。 |

### 1.1 已切割数据怎么用

如果你已经有窗口数据，推荐：

```bash
python run.py --window_dataset_file your_windows.pkl
```

如果你的文件名就是：

```text
cps_windows_2s_2000hz_step_500.pkl
```

也可以直接：

```bash
python run.py --step 500
```

注意：

- `step` 在这里主要是“选择哪个窗口数据文件”
- 不是训练时再切一遍窗

## 2. 信号开关

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--special_signal` | bool | `False` | 是否启用预定义的 special signal 组合。 |
| `--use_synth_signals` | bool | `True` | 是否保留 `Acc.norm` 和 `Gyro.norm`。 |

## 3. 采样视图与降采样

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--sampling_feature_mode` | enum | `raw` | 特征提取视图：`raw` / `downsampled` / `both` |
| `--downsample_target_hz` | int | `100` | 降采样目标频率。 |
| `--downsample_method` | enum | `interval` | `interval` / `mean_pool` / `sliding_window` / `sliding_mean` |
| `--downsample_window_size` | int | `40` | 仅 `sliding_window` 使用。 |
| `--downsample_window_step` | int | `20` | 仅 `sliding_window` 使用。 |

### 3.1 `sliding_mean` 和 `sliding_window`

当前实现里两者没有区别：

- `sliding_mean` 是兼容写法
- 最终都会归一化成 `sliding_window`

### 3.2 三种主要降采样方式

- `interval`：每隔若干点直接取样，不做平均
- `mean_pool`：不重叠均值池化
- `sliding_window`：滑动窗口均值，可以重叠

## 4. 特征与融合

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--feature_domain` | enum | `time_freq` | `time` / `freq` / `time_freq` |
| `--spectrum_method` | enum | `welch_psd` | `rfft` / `welch_psd` / `stft` / `dwt` |
| `--use_feature_engineering` | bool | `True` | 是否启用特征工程。 |
| `--include_handcrafted_features` | bool | `True` | 是否保留手工特征。 |
| `--feature_fusion_mode` | enum | `auto` | `auto` / `handcrafted_only` / `encoder_only` / `hybrid` |

融合模式优先级高于单独开关：

- `handcrafted_only`：只保留手工特征
- `encoder_only`：只保留 encoder 特征
- `hybrid`：两者一起用

## 5. 数据增强

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--use_rotation_augment` | bool | `True` | 是否启用旋转增强。 |
| `--augment_count` | int | `2` | 每个样本增强次数。 |
| `--augment_target` | str | `multilabel,Lifting(raising),Lifting(lowering)` | 需要增强的目标类。 |
| `--rotation_plane` | enum | `xz` | `xy` / `xz` / `yz` / `xyz` |
| `--rotation_max_degrees` | float | `15.0` | 最大旋转角度。 |

## 6. Encoder

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--use_encoder` | bool | `True` | 是否启用 encoder。 |
| `--encoder_backend` | enum | `mlp` | `none` / `mlp` / `cnn1d` / `rescnn1d` |
| `--encoder_axis_mode` | enum | `joint` | `joint` 或 `per_axis` |
| `--encoder_output_dim` | int | `128` | encoder 输出维度。 |
| `--encoder_hidden_dim` | int | `256` | 隐层维度。 |
| `--encoder_channels` | csv int | `64,128` | CNN 通道数。 |
| `--encoder_kernels` | csv int | `3,5,7` | CNN 卷积核。 |
| `--encoder_use_se` | bool | `False` | 是否使用 SE。 |
| `--encoder_dropout` | float | `0.1` | dropout。 |
| `--encoder_epochs` | int | `15` | 训练轮数。 |
| `--encoder_batch_size` | int | `128` | batch size。 |
| `--encoder_lr` | float | `1e-3` | 学习率。 |

## 7. 分类器

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `--model` | enum | `lightgbm` | `lightgbm` / `xgboost` |
| `--train_with_val` | bool | `True` | 是否把验证集合并进训练。 |
| `--seed` | int | `42` | 随机种子。 |

## 8. 常用命令

### 8.1 直接使用现成窗口数据

```bash
python run.py \
  --window_dataset_file cps_windows_2s_2000hz_step_500.pkl \
  --sampling_feature_mode raw \
  --feature_fusion_mode hybrid
```

### 8.2 只用降采样特征

```bash
python run.py \
  --window_dataset_file cps_windows_2s_2000hz_step_500.pkl \
  --sampling_feature_mode downsampled \
  --downsample_target_hz 100 \
  --downsample_method interval
```

### 8.3 同时使用原始和降采样视图

```bash
python run.py \
  --window_dataset_file cps_windows_2s_2000hz_step_500.pkl \
  --sampling_feature_mode both \
  --downsample_target_hz 100 \
  --downsample_method mean_pool
```

### 8.4 滑动窗口均值降采样

```bash
python run.py \
  --window_dataset_file cps_windows_2s_2000hz_step_500.pkl \
  --sampling_feature_mode downsampled \
  --downsample_method sliding_window \
  --downsample_window_size 40 \
  --downsample_window_step 20
```

## 9. 结果文件怎么看

`run_summary.json` 里重点看：

- `data_loading`：这次实际读取了哪个窗口数据文件
- `sampling`：本次采样视图和降采样参数
- `feature_branches`：每个分支的输入形状和输出维度
- `feature_shape`：最终拼接后的总维度
- `metrics`：每折指标
- `plots`：图像输出路径
