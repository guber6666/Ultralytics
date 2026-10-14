import warnings, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 指定使用第一张显卡

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    yaml_path = r'D:\A92\Ultralytics_2269237804\ultralytics\cfg\models\improve\csp-block\2026\yolo26-csp-PKIBlockV2.yaml'

    # 初始化 YOLO 模型，从 yaml 配置文件构建网络结构
    model = YOLO(yaml_path)
    # model.load('yolo26n.pt') # 加载预训练权重，一般都不建议加载
    model.train(data=r'D:\A92\Ultralytics_2269237804\dataset\dataset721\data.yaml', # 数据集配置文件路径
                cache=False, # 是否缓存图像到内存以加快训练速度。False=不缓存，True=缓存到RAM(很吃内存，内存少的慎开)，'disk'=缓存到磁盘(吃硬盘空间)
                imgsz=640, # 输入图像尺寸（像素）
                epochs=300, # 训练总轮数
                batch=16, # 批次大小
                # close_mosaic=0, # 最后多少个 epoch 关闭 Mosaic 数据增强。设置 0 代表全程开启 Mosaic 训练
                workers=4, # 数据加载的工作线程数。Windows 下出现卡顿或奇怪错误可尝试设置为 0
                device=os.environ.get("CUDA_VISIBLE_DEVICES", 0), # 训练设备选择，不在这里设置，在头部设置，详细可以看UserGuide.md中的常见问题第4点
                # optimizer='MuSGD' if 'yolo26' in yaml_path else 'SGD', # 优化器选择。YOLO26 使用官方推荐的 MuSGD，其他模型使用 SGD
                patience=50, # 早停机制的耐心值。连续 50 个 epoch 验证指标未提升则停止训练。设置 0 关闭早停
                # amp='fp16', # 是否启用自动混合精度（Automatic Mixed Precision）训练，默认为 True | loss出现nan可以关闭amp
                cos_lr=False, # 是否使用余弦退火学习率调度器，默认为 False
                save_period=-1, # 每隔多少个 epoch 保存一次 checkpoint（默认 -1 表示禁用，仅保存最好和最后的）
                project='train', # 训练结果保存的项目目录
                name='exp', # 本次实验的名称，（若已存在则自动创建 exp2, exp3...）

 )
