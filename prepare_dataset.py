import os
import pickle
import numpy as np
import SimpleITK as sitk
from PIL import Image
import glob
from skimage.transform import resize

def load_dicom(dicom_path):
    """加载DICOM图像"""
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_path)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)[0]  # 获取第一个切片

def load_tiff(tiff_path):
    """加载TIFF图像"""
    return np.array(Image.open(tiff_path))

def normalize_to_uint8(image):
    """将图像归一化到0-255范围并转换为uint8类型"""
    if image.dtype != np.uint8:
        # 处理可能的无效值
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 确保图像有有效的值范围
        if image.max() == image.min():
            image = np.zeros_like(image, dtype=np.uint8)
        else:
            # 归一化到0-1范围
            image = (image - image.min()) / (image.max() - image.min())
            # 转换到0-255范围
            image = (image * 255).astype(np.uint8)
    return image

def resize_image(image, target_size):
    """调整图像大小，保持宽高比"""
    h, w = image.shape
    scale = min(target_size[0]/h, target_size[1]/w)
    new_h, new_w = int(h*scale), int(w*scale)
    return resize(image, (new_h, new_w), preserve_range=True)

def create_patches(image, patch_size=(256, 256), stride=256):
    """使用滑动窗口创建图像块，默认不重叠"""
    patches = []
    h, w = image.shape
    
    # 计算可以生成多少个完整的块
    n_h = h // patch_size[0]
    n_w = w // patch_size[1]
    
    # 计算中心点
    center_y = (h - n_h * patch_size[0]) // 2
    center_x = (w - n_w * patch_size[1]) // 2
    
    # 从中心开始生成块
    for y in range(center_y, center_y + n_h * patch_size[0], patch_size[0]):
        for x in range(center_x, center_x + n_w * patch_size[1], patch_size[1]):
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            patches.append(patch)
    
    return patches

def get_view_type(filename):
    """从文件名中提取视图类型"""
    if 'L_CC' in filename:
        return 'L_CC'
    elif 'R_CC' in filename:
        return 'R_CC'
    elif 'L_MLO' in filename:
        return 'L_MLO'
    elif 'R_MLO' in filename:
        return 'R_MLO'
    return None

def get_patient_id(filename):
    """从文件名中提取病人ID"""
    parts = filename.split('/')
    for part in parts:
        if part.startswith('SID-'):
            return part
    return None

def prepare_dataset(A_dir, B_dir, output_pickle, patch_size=(256, 256), stride=256):
    """准备训练数据集"""
    dataset = []
    
    # 获取所有A域图像
    A_files = glob.glob(os.path.join(A_dir, '**/*.dcm'), recursive=True)
    print(f"找到 {len(A_files)} 个DICOM文件")
    
    for A_path in A_files:
        print(f"\n处理文件: {A_path}")
        
        # 从文件名中提取视图类型和病人ID
        view_type = get_view_type(A_path)
        patient_id = get_patient_id(A_path)
        
        print(f"视图类型: {view_type}")
        print(f"病人ID: {patient_id}")
        
        if not view_type or not patient_id:
            print(f"无法识别视图类型或病人ID: {A_path}")
            continue
            
        # 构造对应的B域图像路径
        B_path = os.path.join(B_dir, f"{patient_id}.{view_type}.tiff")
        print(f"查找对应的B域图像: {B_path}")
        
        if not os.path.exists(B_path):
            print(f"找不到对应的B域图像: {B_path}")
            continue
            
        print(f"找到匹配的图像对: {A_path} -> {B_path}")
            
        try:
            # 加载图像
            print("加载A域图像...")
            A_image = load_dicom(A_path)
            print(f"A域图像形状: {A_image.shape}")
            
            print("加载B域图像...")
            B_image = load_tiff(B_path)
            print(f"B域图像形状: {B_image.shape}")
            
            # 检查并调整图像大小
            if A_image.shape != B_image.shape:
                print("图像大小不一致，进行调整...")
                # 获取较小的尺寸
                target_size = (min(A_image.shape[0], B_image.shape[0]),
                             min(A_image.shape[1], B_image.shape[1]))
                print(f"调整到目标尺寸: {target_size}")
                
                # 调整图像大小
                A_image = resize_image(A_image, target_size)
                B_image = resize_image(B_image, target_size)
                print(f"调整后A域图像形状: {A_image.shape}")
                print(f"调整后B域图像形状: {B_image.shape}")
            
            # 创建图像块
            print("创建图像块...")
            A_patches = create_patches(A_image, patch_size, stride)
            B_patches = create_patches(B_image, patch_size, stride)
            print(f"生成了 {len(A_patches)} 个图像块")
            
            # 保存图像块
            for i, (A_patch, B_patch) in enumerate(zip(A_patches, B_patches)):
                patch_dir = os.path.join('patches', f"{patient_id}_{view_type}_{i}")
                os.makedirs(patch_dir, exist_ok=True)
                
                # 转换数据类型并保存A域图像块
                A_patch = normalize_to_uint8(A_patch)
                A_patch_path = os.path.join(patch_dir, 'A.png')
                Image.fromarray(A_patch).save(A_patch_path)
                
                # 转换数据类型并保存B域图像块
                B_patch = normalize_to_uint8(B_patch)
                B_patch_path = os.path.join(patch_dir, 'B.png')
                Image.fromarray(B_patch).save(B_patch_path)
                
                # 添加到数据集
                dataset.append({
                    'A': A_patch_path,
                    'B': B_patch_path
                })
                
            print(f"成功处理图像对，当前数据集大小: {len(dataset)}")
            
        except Exception as e:
            print(f"处理图像时出错: {str(e)}")
            continue
    
    # 保存数据集到pickle文件
    with open(output_pickle, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n数据集准备完成，共生成 {len(dataset)} 对图像块")
    print(f"数据集保存在: {output_pickle}")

if __name__ == '__main__':
    A_dir = 'testset/A'
    B_dir = 'testset/B'
    output_pickle = 'dataset.pickle'
    
    prepare_dataset(A_dir, B_dir, output_pickle) 