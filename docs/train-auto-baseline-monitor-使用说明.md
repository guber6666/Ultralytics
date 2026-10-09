# train_auto 自动训练与基线提前淘汰使用说明

这份文档说明本项目新增的 `train_auto.py` 怎么用。

它主要解决两个问题：

1. 一次顺序训练多个模型配置文件，不需要手动一个个运行。
2. 可选开启“基线提前淘汰”：如果当前实验明显落后于已有 baseline 实验，就提前停止，节省训练时间。

## 一、最简单用法：只做批量训练

假设你要依次训练两个 YAML：

```bash
python train_auto.py \
  --models ultralytics/cfg/models/11/yolo11n.yaml ultralytics/cfg/models/12/yolo12n.yaml \
  --data /root/dataset/dataset_visdrone/data.yaml \
  --epochs 300 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

它会依次执行类似下面的 YOLO 训练命令：

```bash
python -m ultralytics.cfg train model=xxx.yaml data=xxx.yaml epochs=300 batch=16 imgsz=640 device=0
```

## 二、先 dry-run 检查命令

正式训练前，建议先加 `--dry-run` 看看会执行什么命令：

```bash
python train_auto.py \
  --models a.yaml b.yaml \
  --data /root/dataset/dataset_visdrone/data.yaml \
  --epochs 300 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --dry-run
```

`--dry-run` 只打印命令，不会启动训练。

## 三、文件内配置模式

如果不想每次写很长命令，可以直接编辑 `train_auto.py` 顶部的 `AUTO_TRAIN_CONFIG`：

```python
AUTO_TRAIN_CONFIG = AutoTrainConfig(
    models=[
        "ultralytics/cfg/models/11/yolo11n.yaml",
        "ultralytics/cfg/models/12/yolo12n.yaml",
    ],
    data="/root/dataset/dataset_visdrone/data.yaml",
    epochs=300,
    batch=16,
    imgsz=640,
    device="0",
    project="train_auto",
    extra={
        "optimizer": "SGD",
        "cache": False,
        "workers": 4,
        "amp": True,
    },
)
```

然后直接运行：

```bash
python train_auto.py
```

命令行参数会覆盖文件内配置。例如文件里写了 `epochs=300`，但运行时传了 `--epochs 10`，最终会用 10。

## 四、多个 YAML 的保存名称与 SwanLab 名称

每个 YAML 都会根据 `name_template` 生成独立的 YOLO `name`，默认模板是 `{stem}`，即 YAML 文件名去掉扩展名。

例如依次训练 `yolo11n.yaml` 与 `yolo12n.yaml` 时：

| YAML | 本地保存 name | SwanLab 实验名 |
| --- | --- | --- |
| `yolo11n.yaml` | `yolo11n` | `yolo11n` |
| `yolo12n.yaml` | `yolo12n` | `yolo12n` |

SwanLab 的 `swanlab_run_name=None`（默认值）会自动沿用本次训练的 `name`。因此批量训练时，建议保持它为 `None`，只用 `--name-template` 统一命名：

```bash
python train_auto.py \
  --models a.yaml b.yaml \
  --data data.yaml \
  --name-template "visdrone-{index}-{stem}" \
  --swanlab \
  --swanlab-project Ultralytics-YOLO
```

上例会生成 `visdrone-1-a`、`visdrone-2-b`，本地训练目录与对应的 SwanLab 实验同名，便于比较和定位。

不要在训练多个 YAML 时使用固定的 `--swanlab-run-name my-experiment`，否则所有 SwanLab 实验都会使用同一个显示名称。当前 `swanlab_run_name` 不支持 `{stem}` 等模板占位符。

`--name-template` 支持 `{index}`（从 1 开始的序号）、`{stem}`（去扩展名文件名）、`{name}`（带扩展名文件名）和 `{model}`（完整模型路径）。

## 五、训练失败后是否继续

默认情况下，如果某个模型训练失败，`train_auto.py` 还会继续训练后面的模型。

如果希望第一个失败后就停止，添加：

```bash
--stop-on-fail
```

示例：

```bash
python train_auto.py \
  --models a.yaml b.yaml c.yaml \
  --data /root/dataset/dataset_visdrone/data.yaml \
  --stop-on-fail
```

## 六、开启基线提前淘汰

先准备一个已有实验的 `results.csv`，比如：

```text
runs/detect/baseline/results.csv
```

然后运行：

```bash
python train_auto.py \
  --models a.yaml b.yaml \
  --data /root/dataset/dataset_visdrone/data.yaml \
  --epochs 300 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --baseline-monitor \
  --baseline-path runs/detect/baseline/results.csv
```

开启后，训练过程中会读取 baseline 的 `results.csv`，用当前实验的指标和 baseline 同一 epoch 的指标对比。

如果当前实验连续落后太多，就会提前停止这个实验，然后继续下一个模型。

## 七、基线提前淘汰的常用参数

```bash
--baseline-monitor
```

开启基线监控。

```bash
--baseline-path runs/detect/baseline/results.csv
```

指定 baseline 实验的 `results.csv` 路径。开启 `--baseline-monitor` 时必须传这个参数。

```bash
--baseline-metric metrics/mAP50-95\(B\)
```

指定要对比的指标。默认是：

```text
metrics/mAP50-95(B)
```

推荐指标：

- 检测 detect：`metrics/mAP50-95(B)`
- 旋转框 obb：`metrics/mAP50-95(B)`
- 分割 segment：`metrics/mAP50-95(M)`，如果只看框也可以用 `metrics/mAP50-95(B)`
- 姿态 pose：`metrics/mAP50-95(P)`

```bash
--baseline-start-ratio 0.5
```

从总 epoch 的多少比例后开始检查。默认 `0.5`，也就是前一半 epoch 不淘汰。

```bash
--baseline-min-gap 0.03
```

允许当前实验比 baseline 低多少。默认 `0.03`。

例如 baseline 当前 epoch 的 mAP 是 `0.50`，当前实验是 `0.45`，差距是 `0.05`，超过 `0.03`，这一轮会记为一次落后。

```bash
--baseline-patience 2
```

连续落后多少次才停止。默认 `1`。

如果设置成 `2`，表示连续 2 次落后超过阈值才提前停止。

## 八、文件内开启基线监控

也可以直接在 `AUTO_TRAIN_CONFIG` 里配置：

```python
AUTO_TRAIN_CONFIG = AutoTrainConfig(
    models=[
        "ultralytics/cfg/models/11/yolo11n.yaml",
        "ultralytics/cfg/models/12/yolo12n.yaml",
    ],
    data="/root/dataset/dataset_visdrone/data.yaml",
    epochs=300,
    batch=16,
    imgsz=640,
    device="0",
    baseline_monitor=True,
    baseline_path="runs/detect/baseline/results.csv",
    baseline_metric="metrics/mAP50-95(B)",
    baseline_start_ratio=0.5,
    baseline_min_gap=0.03,
    baseline_patience=1,
)
```

然后运行：

```bash
python train_auto.py
```

## 九、传递更多 YOLO 参数

`train_auto.py` 支持用 `--extra` 传递额外的 YOLO 参数：

```bash
python train_auto.py \
  --models a.yaml \
  --data /root/dataset/dataset_visdrone/data.yaml \
  --extra optimizer=SGD amp=False workers=4 cache=False
```

这些参数会直接拼到 YOLO 命令后面。

## 十、常见问题

### 1. baseline_path 应该填什么？

填一个已经训练完成的实验目录里的 `results.csv`，例如：

```text
runs/detect/train/results.csv
```

或者你自己整理的 baseline 路径：

```text
runs/detect/baseline/results.csv
```

### 2. baseline 的 epoch 和当前实验怎么对齐？

按 `results.csv` 里的 `epoch` 列对齐。

例如当前训练到第 50 轮，就会拿 baseline CSV 中 `epoch=50` 的指标来对比。

### 3. baseline 缺少某个 epoch 会怎样？

不会停止。

如果 baseline CSV 里找不到当前 epoch，监控器会跳过这一轮。

### 4. 当前指标还没产生会怎样？

不会停止。

只有验证产生了对应指标，并且 baseline 也有同一 epoch 的指标时，才会比较。

### 5. 不开启 baseline_monitor 会影响普通训练吗？

不会。

默认 `baseline_monitor=False`，普通 `YOLO(...).train(...)` 和正常 YOLO CLI 训练行为保持不变。

## 十一、推荐工作流

1. 先单独训练一个可信 baseline，得到 `results.csv`。
2. 用 `train_auto.py --dry-run` 检查多模型训练命令。
3. 确认没问题后去掉 `--dry-run` 正式训练。
4. 如果要节省时间，开启 `--baseline-monitor` 和 `--baseline-path`。
