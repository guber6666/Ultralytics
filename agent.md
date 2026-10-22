# 项目代理指南

## 项目概况

- 本仓库是扩展版 Ultralytics YOLO，支持 detect、segment、classify、pose、obb，以及训练、验证、预测、跟踪和导出。除上游功能外，重点维护大量研究性模型 YAML、自定义网络模块、损失函数和实验脚本。
- Python 包入口在 `ultralytics/`；`pyproject.toml` 定义依赖、`yolo` CLI、pytest 和 Ruff 配置。项目要求 Python >=3.8，但部分附加模块和实验可能要求更高版本或特定 CUDA 环境。
- 阅读 `UserGuide.md` 了解本仓库环境和脚本用法；损失函数看 `LOSS-UserGuide.md`。`README.md` 和 `README.zh-CN.md` 主要介绍上游通用功能。

## 代码地图

- `ultralytics/engine/`：模型 API、训练器、验证器、预测器及导出流程。
- `ultralytics/models/`：YOLO 等模型和各任务的 trainer、validator、predictor；AFSS 训练器位于 `ultralytics/models/yolo/<task>/afss_train.py`。
- `ultralytics/nn/tasks.py`：由 YAML 解析、构建网络的关键位置；`parse_model()` 按模块类别处理输入输出通道、重复次数及嵌套模块参数。
- `ultralytics/nn/modules/`：基础网络层；`ultralytics/nn/extra_modules/`：本仓库的注意力、卷积、CSP block、neck、head、Mamba 等扩展模块，统一从其 `__init__.py` 导出。
- `ultralytics/cfg/models/`：模型结构 YAML；`improve/` 按改进位置分类，包含大量实验配置。通用训练参数见 `ultralytics/cfg/default.yaml`。
- `ultralytics/utils/`、`ultralytics/data/`、`ultralytics/trackers/`：损失与工具、数据处理、跟踪。
- 根目录 `train.py`、`train_auto.py`、`val.py`、`detect.py`、`track.py`、`export.py`、`main_profile.py` 是实验入口；`tests/` 是 pytest 测试，`compile_module/` 是可选 CUDA/C++ 扩展源码。

## 修改模型时

1. 先找相近的模块实现及 YAML，确认张量形状、通道约定、stride、重复次数和任务头的接口。
2. 新模块若要在 YAML 中按名称使用，应在 `ultralytics/nn/extra_modules/__init__.py` 导出，并确认 `ultralytics/nn/tasks.py` 已导入且 `parse_model()` 能按该模块的构造参数正确计算通道和处理重复次数；仅有 Python 类或 YAML 名称并不够。
3. `C3_Block`、`C2f_Block`、`C3k2_Block` 等 YAML 中的 `{'module': 名称, 'param': {...}}` 由 `parse_model()` 递归解析。修改其参数格式时检查嵌套模块解析，并验证模型实际构建。
4. YAML 的 `nc`、`scales`、`legacy`、backbone/head 的层索引和 Detect/Segment/Pose/OBB 输出层都可能影响行为。不要批量改写无关实验 YAML。
5. 新模块可能依赖 `compile_module/` 内的扩展或 `ultralytics-requirements.txt` 的附加包。缺少编译产物时先确认依赖，再区分环境问题与代码问题。

## 开发与验证

- 在仓库根目录安装本地包：`python -m pip install -e .`。需要附加实验依赖时参考 `UserGuide.md` 和 `ultralytics-requirements.txt`；GPU/CUDA 扩展按各子目录 README 或 `make.sh` 单独处理。
- 修改 Python 逻辑后，先运行相关测试，例如 `python -m pytest -q tests/test_parse_model_nested_params.py`；再按影响范围扩大到 `python -m pytest -q tests/`。pytest 配置在 `pyproject.toml`，默认跳过标记为 `slow` 的测试；`--slow` 会启用它们。
- 修改 YAML 或模块后，用 `YOLO("相应模型.yaml")` 做模型构建检查，必要时再用小输入做 forward；只有确实需要时才运行完整训练、验证或 `main_profile.py`。后者会对每层多次执行前向，开销较大。
- 代码风格遵循现有文件，Ruff 行宽 120，新增公共函数和类使用 Google 风格 docstring。避免只为格式化而改动大批历史文件。
- `train.py`、`val.py`、`main_profile.py` 等脚本含本机路径和 GPU 选择设置；运行前检查并按当前环境配置。训练、验证、导出和部分测试可能下载权重、使用 GPU、修改 `runs/` 或 `exp.csv`。

## 工作区约束

- 保留用户已有的未提交改动；开始前查看 `git status --short`，不要重置、覆盖或清理与本任务无关的文件。
- `dataset/`、`runs/`、权重、导出模型和实验记录属于数据或生成物；不要为了代码修改而删除或提交这些内容。不要假设数据集和 CUDA 在每个环境中可用。
- 修复问题时优先针对相关模块、配置和测试做小范围修改，并在交付时说明实际运行过的验证及未运行的 GPU/数据集步骤。
