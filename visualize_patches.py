import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle

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
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        # 读取A域图像
        A_img = Image.open(sample['A'])
        axes[i, 0].imshow(A_img, cmap='gray')
        axes[i, 0].set_title('A域图像')
        axes[i, 0].axis('off')
        
        # 读取B域图像
        B_img = Image.open(sample['B'])
        axes[i, 1].imshow(B_img, cmap='gray')
        axes[i, 1].set_title('B域图像')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('patches_visualization.png')
    print("可视化结果已保存为 patches_visualization.png")

if __name__ == '__main__':
    pickle_path = 'dataset.pickle'
    visualize_patches(pickle_path) 