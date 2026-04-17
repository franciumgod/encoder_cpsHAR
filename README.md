# EncoderBlock

`encoderblock/` 现在按“可单独搬走的最小目录单元”来组织。

也就是说：

- 入口就是当前目录下的 `run.py`
- 默认数据目录就是当前目录下的 `data/`
- 默认输出目录就是当前目录下的 `output/`
- 你把整个 `encoderblock/` 文件夹拷走后，优先保证 `python run.py ...` 还能直接工作

## 目录说明

- `run.py`
  主入口。直接运行这个文件即可。
- `config.py`
  默认配置。
- `window_dataset.py`
  窗口数据加载、自动查找、按 experiment 划分 train / val / test。
- `data.py`
  兼容层，保留旧名字；新代码请看 `window_dataset.py`。
- `sampling.py`
  构造 `raw / downsampled / both` 三种特征视图。
- `domain.py`
  构造时域 / 频域 / 工程特征。
- `augment.py`
  旋转增强。
- `encoder.py`
  encoder 投影模块。
- `tree_models.py`
  LightGBM / XGBoost 多标签模型。
- `metrics.py`
  评估与绘图。
- `data/`
  默认数据目录。
- `output/`
  默认输出目录。

## 运行方式

在 `encoderblock/` 目录里直接运行：

```bash
python run.py --step 500
```

如果你是在上一级目录运行，也可以：

```bash
python encoderblock/run.py --step 500
```

## 数据路径规则

现在的查找顺序是：

1. `--data_dir`
2. 当前目录下的 `data/`
3. 上一级目录下的 `cpsHAR/data/`，如果这个目录存在

所以在“单独搬走 `encoderblock/`”的场景下，最稳妥的放法就是：

```text
encoderblock/
  run.py
  config.py
  window_dataset.py
  ...
  data/
    cps_windows_2s_2000hz_step_500.pkl
```

## `from .data` 到底是什么

以前 `run.py` 里写的：

```python
from .data import load_window_payload, split_window_payload
```

这里的 `.data` 指的是 Python 模块 `data.py`，不是 `data/` 文件夹。

现在主逻辑已经改成：

```python
from .window_dataset import load_window_payload, split_window_payload
```

这样模块名和数据目录不再混淆。

## 已切割数据怎么用

### 方式 1：直接指定文件

如果你已经有切好的窗口数据，推荐直接写：

```bash
python run.py --window_dataset_file your_windows.pkl
```

### 方式 2：按 `step` 自动找约定文件名

如果你的文件名就是：

```text
cps_windows_2s_2000hz_step_500.pkl
```

那就可以直接：

```bash
python run.py --step 500
```

这里的 `step` 是“选择哪个窗口数据文件”的依据，不是训练时再切一次窗。

## `sliding_mean` 和 `sliding_window`

当前实现里两者没有区别：

- `sliding_mean` 是别名
- 最终都会归一化成 `sliding_window`

真正有区别的是下面三种降采样：

- `interval`：隔点抽样，不做平均
- `mean_pool`：不重叠窗口均值
- `sliding_window`：滑动窗口均值，可以重叠，更平滑

## 常用命令

### 1. 直接使用现成窗口数据

```bash
python run.py \
  --window_dataset_file cps_windows_2s_2000hz_step_500.pkl \
  --sampling_feature_mode raw \
  --feature_fusion_mode hybrid
```

### 2. 只使用降采样视图

```bash
python run.py \
  --window_dataset_file cps_windows_2s_2000hz_step_500.pkl \
  --sampling_feature_mode downsampled \
  --downsample_target_hz 100 \
  --downsample_method interval
```

### 3. 使用滑动窗口均值降采样

```bash
python run.py \
  --window_dataset_file cps_windows_2s_2000hz_step_500.pkl \
  --sampling_feature_mode downsampled \
  --downsample_method sliding_window \
  --downsample_window_size 40 \
  --downsample_window_step 20
```

## 输出

默认输出到当前目录下的 `output/`。

`run_summary.json` 里会记录：

- 这次实际读到的窗口数据文件路径
- 数据来源文件名
- payload 类型
- step 和窗口大小

这样你能直接确认程序到底读的是哪个 `.pkl`。
