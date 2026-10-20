# 数据集1：[agu2](dataset%2Fagu2)
## 绝缘子缺陷类别
nc: 4
names: ['insulator', 'explose', 'pollution-flashover', 'broken']
## 数据集1实验记录
基线：yolo11
多尺度融合：[yolo11-BiFPN.yaml] 
小目标分支：[yolo11-P2.yaml]
特征融合：[yolo11-CGAFusion.yaml]
上采样：[yolo11-DySample.yaml]
CSP-PKIBlock
CSP-EFCM
CSP-MSCB
-------------------------------------------------------------------
# 数据集2：[dataset721](dataset%2Fdataset721)
## 输电线缺陷类别
nc: 4
names: ['insulator_breakage', 'insulator_flashover', 'damper_defect', 'nest']

## 数据集1实验记录
基线：yolo11
多尺度融合：[yolo11-BiFPN.yaml]
小目标分支：[yolo11-P2.yaml]
特征融合：[yolo11-CGAFusion.yaml]
上采样：[yolo11-DySample.yaml]
CSP-MSBlock