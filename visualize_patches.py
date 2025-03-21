import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.rcParams['axes.unicode_minus'] = False

def visualize_patches(pickle_path, num_samples=4):
    """可视化数据集中的图像块"""
    # 加载数据集
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"数据集中共有 {len(dataset)} 对图像块")
    
    # 随机选择一些样本进行可视化
    import random
    samples = random.sample(dataset, min(num_samples, len(dataset)))
    
    # 创建子图
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        # 读取A域图像
        A_img = np.array(Image.open(sample['A']))
        axes[i, 0].imshow(A_img, cmap='gray')
        axes[i, 0].set_title('A域图像 (DICOM)', fontsize=12)
        axes[i, 0].axis('off')
        
        # 读取B域图像
        B_img = np.array(Image.open(sample['B']))
        axes[i, 1].imshow(B_img, cmap='gray')
        axes[i, 1].set_title('B域图像 (TIFF)', fontsize=12)
        axes[i, 1].axis('off')
        
        # 添加图像信息
        info_text = f'尺寸: {A_img.shape}\n像素范围: [{A_img.min()}, {A_img.max()}]'
        axes[i, 0].text(0.02, 0.98, info_text, 
                       transform=axes[i, 0].transAxes, 
                       verticalalignment='top',
                       fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        info_text = f'尺寸: {B_img.shape}\n像素范围: [{B_img.min()}, {B_img.max()}]'
        axes[i, 1].text(0.02, 0.98, info_text, 
                       transform=axes[i, 1].transAxes, 
                       verticalalignment='top',
                       fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('patches_visualization.png', dpi=300, bbox_inches='tight')
    print("可视化结果已保存为 patches_visualization.png")

if __name__ == '__main__':
    pickle_path = 'dataset.pickle'
    visualize_patches(pickle_path) 