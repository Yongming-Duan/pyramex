"""
示例1: 基础数据分析流程

本示例展示如何使用PyRamEx进行基本的拉曼光谱分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyramex import Ramanome
from pyramex.preprocessing import smooth, remove_baseline, normalize
from pyramex.qc import quality_control
from pyramex.features import reduce
from pyramex.visualization import plot_spectra, plot_reduction

# 1. 创建模拟数据
print("=" * 60)
print("示例1: 基础数据分析流程")
print("=" * 60)

# 生成两个类别的人工光谱数据
np.random.seed(42)
n_samples_per_class = 25
wavenumbers = np.linspace(400, 4000, 500)

# 类别A: 在1000 cm^-1附近有峰
spectra_a = []
for i in range(n_samples_per_class):
    baseline = 0.001 * wavenumbers + np.random.randn(500) * 0.05
    peak1 = 10 * np.exp(-((wavenumbers - 1000)**2) / (2 * 50**2))
    peak2 = 8 * np.exp(-((wavenumbers - 1600)**2) / (2 * 60**2))
    spectrum = baseline + peak1 + peak2
    spectra_a.append(spectrum)

# 类别B: 在1500 cm^-1附近有峰
spectra_b = []
for i in range(n_samples_per_class):
    baseline = 0.001 * wavenumbers + np.random.randn(500) * 0.05
    peak1 = 6 * np.exp(-((wavenumbers - 1000)**2) / (2 * 50**2))
    peak2 = 12 * np.exp(-((wavenumbers - 1500)**2) / (2 * 60**2))
    spectrum = baseline + peak1 + peak2
    spectra_b.append(spectrum)

# 合并数据
all_spectra = np.vstack([np.array(spectra_a), np.array(spectra_b)])
labels = ['Class A'] * n_samples_per_class + ['Class B'] * n_samples_per_class

metadata = pd.DataFrame({
    'sample_id': range(len(all_spectra)),
    'label': labels,
    'class_id': [0] * n_samples_per_class + [1] * n_samples_per_class
})

print(f"\n数据集信息:")
print(f"  样本数: {len(all_spectra)}")
print(f"  波数点数: {len(wavenumbers)}")
print(f"  类别数: 2")
print(f"  每类样本数: {n_samples_per_class}")

# 2. 创建Ramanome对象
ramanome = Ramanome(all_spectra, wavenumbers, metadata)

print(f"\nRamanome对象:")
print(f"  {ramanome}")

# 3. 数据预处理
print("\n" + "=" * 60)
print("步骤1: 数据预处理")
print("=" * 60)

# 截取感兴趣区域
ramanome.cutoff((500, 3500))
print(f"✓ 截取波数范围: 500-3500 cm⁻¹")

# 平滑去噪
ramanome.smooth(window_size=9, polyorder=3)
print(f"✓ Savitzky-Golay平滑 (window=9, polyorder=3)")

# 去除基线
ramanome.remove_baseline(method='polyfit', degree=2)
print(f"✓ 多项式基线去除 (degree=2)")

# 归一化
ramanome.normalize(method='minmax')
print(f"✓ Min-Max归一化")

print(f"\n预处理步骤: {ramanome.processed}")

# 4. 质量控制
print("\n" + "=" * 60)
print("步骤2: 质量控制")
print("=" * 60)

qc_result = quality_control(ramanome, method='dis', threshold=0.05)

print(f"QC结果: {qc_result}")
print(f"  好样本: {qc_result.n_good}")
print(f"  坏样本: {qc_result.n_bad}")
print(f"  好样本率: {qc_result.good_rate:.1%}")

# 5. 降维分析
print("\n" + "=" * 60)
print("步骤3: 降维分析")
print("=" * 60)

# PCA降维
result = ramanome.reduce(method='pca', n_components=2)

print(f"PCA结果:")
print(f"  解释方差比: {result['explained_variance']}")
print(f"  累积方差: {result['cumulative_variance']}")

# 6. 可视化
print("\n" + "=" * 60)
print("步骤4: 可视化")
print("=" * 60)

# 绘制原始光谱（示例：每类前5个样本）
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 原始光谱
for i in range(5):
    axes[0, 0].plot(wavenumbers, all_spectra[i], alpha=0.7, label='Class A' if i < 5 else 'Class B')
    axes[0, 0].plot(wavenumbers, all_spectra[n_samples_per_class + i], alpha=0.7)

axes[0, 0].set_title('原始光谱（每类前5个样本）')
axes[0, 0].set_xlabel('波数 (cm⁻¹)')
axes[0, 0].set_ylabel('强度')
axes[0, 0].invert_xaxis()
axes[0, 0].legend()

# 预处理后的光谱
for i in range(5):
    axes[0, 1].plot(ramanome.wavenumbers, ramanome.spectra[i], alpha=0.7)
    axes[0, 1].plot(ramanome.wavenumbers, ramanome.spectra[n_samples_per_class + i], alpha=0.7)

axes[0, 1].set_title('预处理后的光谱')
axes[0, 1].set_xlabel('波数 (cm⁻¹)')
axes[0, 1].set_ylabel('强度（归一化）')
axes[0, 1].invert_xaxis()

# PCA结果
transformed = result['transformed']
colors = ['red' if label == 'Class A' else 'blue' for label in labels]
axes[1, 0].scatter(transformed[:, 0], transformed[:, 1], c=colors, alpha=0.6)
axes[1, 0].set_xlabel('PC1 (%.1f%%)' % (result['explained_variance'][0] * 100))
axes[1, 0].set_ylabel('PC2 (%.1f%%)' % (result['explained_variance'][1] * 100))
axes[1, 0].set_title('PCA降维结果')
axes[1, 0].grid(True, alpha=0.3)

# 平均光谱
mean_a = ramanome.spectra[:n_samples_per_class].mean(axis=0)
std_a = ramanome.spectra[:n_samples_per_class].std(axis=0)
mean_b = ramanome.spectra[n_samples_per_class:].mean(axis=0)
std_b = ramanome.spectra[n_samples_per_class:].std(axis=0)

axes[1, 1].plot(ramanome.wavenumbers, mean_a, 'r-', label='Class A', linewidth=2)
axes[1, 1].fill_between(ramanome.wavenumbers, mean_a - std_a, mean_a + std_a, alpha=0.3, color='red')
axes[1, 1].plot(ramanome.wavenumbers, mean_b, 'b-', label='Class B', linewidth=2)
axes[1, 1].fill_between(ramanome.wavenumbers, mean_b - std_b, mean_b + std_b, alpha=0.3, color='blue')
axes[1, 1].set_title('平均光谱 ± 标准差')
axes[1, 1].set_xlabel('波数 (cm⁻¹)')
axes[1, 1].set_ylabel('强度（归一化）')
axes[1, 1].invert_xaxis()
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('examples/ex1_basic_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ 保存图像: examples/ex1_basic_analysis.png")

# 7. 输出统计信息
print("\n" + "=" * 60)
print("统计摘要")
print("=" * 60)

print(f"\n类别A统计:")
print(f"  样本数: {n_samples_per_class}")
print(f"  平均强度: {ramanome.spectra[:n_samples_per_class].mean():.4f}")
print(f"  标准差: {ramanome.spectra[:n_samples_per_class].std():.4f}")

print(f"\n类别B统计:")
print(f"  样本数: {n_samples_per_class}")
print(f"  平均强度: {ramanome.spectra[n_samples_per_class:].mean():.4f}")
print(f"  标准差: {ramanome.spectra[n_samples_per_class:].std():.4f}")

print("\n" + "=" * 60)
print("示例1完成！")
print("=" * 60)
