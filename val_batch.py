import warnings, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 指定使用第一张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2' # 指定使用第三张显卡
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from val import run_val

# 批量验证的父目录，自动依次处理其下每个实验目录（如 yolo11-BiFPN、yolo11）中的 weights/best.pt
ROOT = r'/media/dell/Data1/cheng/zhangxuan/zyf/Ultralytics/runs/detect/train_auto'

if __name__ == '__main__':
    # 设置用于计算指标的图像尺寸
    imgsz = 640

    model_paths = []
    for exp_name in sorted(os.listdir(ROOT)):
        weight_path = os.path.join(ROOT, exp_name, 'weights', 'best.pt')
        if os.path.isfile(weight_path):
            model_paths.append(weight_path)

    if not model_paths:
        raise FileNotFoundError(f'{ROOT} 下未找到 */weights/best.pt')

    for idx, model_path in enumerate(model_paths, 1):
        print(f'\n{"="*30} [{idx}/{len(model_paths)}] {model_path} {"="*30}\n')
        try:
            run_val(model_path, imgsz)
        except Exception as e:
            print(f'[跳过] {model_path} 验证失败: {e}')
