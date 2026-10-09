# legacy 参数配置说明

本文说明 `ultralytics/nn/tasks.py` 中 `parse_model()` 对 `legacy` 的原始判断逻辑，以及在 YOLO11、YOLO12、YOLO26 改进配置文件中显式添加 `legacy` 参数的建议。

## legacy 的作用

`legacy` 是 Detect、Segment、Pose、OBB 等任务头的兼容性开关。`parse_model()` 在解析模型 YAML 时，会把当前的 `legacy` 值赋给检测头类，例如 `Detect.legacy`。

在 `Detect` 中，`legacy` 会影响分类分支 `cv3` 的构建方式：

- `legacy: True`：使用旧版兼容分支，主要面向 v3/v5/v8/v9 风格模型。
- `legacy: False`：使用新版分支，主要面向 YOLO11、YOLO12、YOLO26 等新结构。

因此，`legacy` 不只是一个显示参数，它会改变 Detect 模块内部结构。

## 原来的自动判断逻辑

在没有显式配置 `legacy` 时，`parse_model()` 的默认行为是：

```python
legacy = True
```

随后在解析 backbone/head 时，如果遇到下列模块，会自动切换为新版 head：

```python
C3k2
C3k2_Block
A2C2f
C2fCIB
```

对应逻辑可以理解为：

```python
legacy = True

if model_contains(C3k2, C3k2_Block, A2C2f, C2fCIB):
    legacy = False
```

最后在解析 Detect、Segment、Pose、OBB 等 head 时，将该值写入 head 类：

```python
m.legacy = legacy
```

这个自动判断对官方基线 YAML 通常是可用的，但对改进模型存在隐患。

## 为什么改进 YAML 建议显式写 legacy

在做模块替换、结构消融或 neck/head 改进时，可能出现下面两类情况：

1. 原模型中触发 `legacy=False` 的模块被替换掉。
2. 改进模块新增或移除了 `C3k2`、`A2C2f`、`C2fCIB` 等触发条件。

如果 YAML 不显式写 `legacy`，Detect 的性质会跟着自动判断结果变化。这样会导致实验变量不纯：你以为只替换了某个 backbone、neck 或 block 模块，但 Detect head 的内部结构也可能一起变了。

这会影响参数量、计算量、训练行为和实验对比结论。

## YOLO11、YOLO12、YOLO26 的推荐写法

YOLO11、YOLO12、YOLO26 基线都属于新版结构，正常情况下应使用：

```yaml
legacy: False
```

建议在改进 YAML 的 Parameters 区域显式添加，例如：

```yaml
# Parameters
nc: 80
legacy: False # keep the YOLO11/YOLO12/YOLO26 Detect head behavior fixed
scales:
  n: [0.50, 0.25, 1024]
```

对以下场景尤其建议添加：

- 基于 `ultralytics/cfg/models/11/` 的 YOLO11 改进配置。
- 基于 `ultralytics/cfg/models/12/` 的 YOLO12 改进配置。
- 基于 `ultralytics/cfg/models/26/` 的 YOLO26 改进配置。
- 对 backbone、neck、head 中的 `C3k2`、`A2C2f`、`C2fCIB` 做替换或消融。
- 做论文实验、消融实验或多模型公平对比。

## 什么时候使用 legacy: True

只有在明确希望保持 v8/v10 风格 Detect head 时，才建议写：

```yaml
legacy: True
```

如果模型是从 YOLO11、YOLO12、YOLO26 基线改进而来，一般不要写 `legacy: True`，否则 Detect head 会变成旧版兼容分支，和原基线行为不一致。

## 当前兼容策略

当前项目支持 YAML 显式控制 `legacy`：

- YAML 中存在 `legacy`：以 YAML 的值为准，不再由模块触发条件覆盖。
- YAML 中不存在 `legacy`：保持原来的自动判断逻辑，保证旧配置文件兼容。

也就是说，旧 YAML 不需要强制修改；但新建 YOLO11、YOLO12、YOLO26 改进 YAML 时，建议显式添加：

```yaml
legacy: False
```

这样可以固定 Detect head 行为，避免因为替换模块导致 Detect 模块性质发生非预期变化。
