"""
示例4: 降维和可视化比较

比较不同降维方法的效果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyramex import Ramanome
from pyramex.features import reduce

print("=" * 60)
print("示例4: 降维和可视化比较")
print("=" * 60)

# 1. 创建多类别数据
print("\n步骤1: 创建多类别数据集")

np.random.seed(42)
n_classes = 4
n_samples_per_class = 30
wavenumbers = np.linspace(400, 4000, 500)

spectra_list = []
labels = []
peak_positions = [800, 1200, 1600, 2000]

for class_id, peak_pos in enumerate(peak_positions):
    for i in range(n_samples_per_class):
        # 基线
        baseline = 0.001 * wavenumbers

        # 主峰
        main_peak = 10 * np.exp(-((wavenumbers - peak_pos)**2) / (2 * 50**2))

        # 次峰（类别特征）
        if class_id == 0:
            secondary_peak = 5 * np.exp(-((wavenumbers - 1600)**2) / (2 * 40**2))
        elif class_id == 1:
            secondary_peak = 5 * np.exp(-((wavenumbers - 800)**2) / (2 * 40**2))
        elif class_id == 2:
            secondary_peak = 5 * np.exp(-((wavenumbers - 1200)**2) / (2 * 40**2))
        else:
            secondary_peak = 5 * np.exp(-((wavenumbers - 1800)**2) / (2 * 40**2))

        # 噪声
        noise = np.random.randn(500) * 0.3

        spectrum = baseline + main_peak + secondary_peak + noise
        spectra_list.append(spectrum)
        labels.append(f'Class_{class_id}')

all_spectra = np.vstack(spectra_list)
metadata = pd.DataFrame({
    'sample_id': range(len(all_spectra)),
    'label': labels
})

print(f"  样本数: {len(all_spectra)}")
print(f"  类别数: {n_classes}")
print(f"  峰位: {peak_positions}")

# 2. 预处理
print("\n步骤2: 数据预处理")

ramanome = Ramanome(all_spectra, wavenumbers, metadata)
ramanome.cutoff((500, 3500))
ramanome.smooth(window_size=7, polyorder=2)
ramanome.normalize(method='area')

print("  ✓ 预处理完成")

# 3. 应用不同降维方法
print("\n步骤3: 应用降维方法")

methods_to_test = ['pca', 'pcoa']
results = {}

for method in methods_to_test:
    print(f"\n  {method.upper()}...")
    try:
        result = reduce(ramanome.spectra, method=method, n_components=2)
        results[method] = result
        print(f"    完成")
    except Exception as e:
        print(f"    错误: {e}")

# 4. 可视化比较
print("\n步骤4: 可视化比较")

n_methods = len(results)
fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 12))

if n_methods == 1:
    axes = axes.reshape(-1, 1)

# 绘制降维结果
for idx, (method, result) in enumerate(results.items()):
    transformed = result['transformed']

    # 2D散点图
    colors = plt.cm.tab10(range(n_classes))
    for class_id in range(n_classes):
        class_mask = (metadata['label'] == f'Class_{class_id}').values
        axes[0, idx].scatter(transformed[class_mask, 0],
                            transformed[class_mask, 1],
                            c=[colors[class_id]],
                            label=f'Class_{class_id}',
                            alpha=0.6)

    axes[0, idx].set_xlabel(f'{method.upper()} 1')
    axes[0, idx].set_ylabel(f'{method.upper()} 2')
    axes[0, idx].set_title(f'{method.upper()} - 2D')
    axes[0, idx].legend()
    axes[0, idx].grid(True, alpha=0.3)

# 绘制解释方差（PCA）
if 'pca' in results:
    pca_result = results['pca']

    # 碎石图
    axes[1, 0].plot(range(1, 11),
                    pca_result['explained_variance'][:10],
                    'bo-')
    axes[1, 0].set_xlabel('主成分数')
    axes[1, 0].set_ylabel('解释方差比')
    axes[1, 0].set_title('PCA碎石图')
    axes[1, 0].grid(True)

    # 累积方差
    cumulative = pca_result['cumulative_variance'][:10]
    axes[1, 1].plot(range(1, 11), cumulative, 'ro-')
    axes[1, 1].axhline(y=0.8, color='gray', linestyle='--', label='80%')
    axes[1, 1].axhline(y=0.9, color='gray', linestyle=':', label='90%')
    axes[1, 1].set_xlabel('主成分数')
    axes[1, 1].set_ylabel('累积解释方差')
    axes[1, 1].set_title('累积解释方差')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('examples/ex4_dim_reduction_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存图像: examples/ex4_dim_reduction_comparison.png")

# 5. 定量分析
print("\n步骤5: 定量分析")

if 'pca' in results:
    pca_result = results['pca']
    print("\nPCA分析:")
    print(f"  PC1解释方差: {pca_result['explained_variance'][0]:.2%}")
    print(f"  PC2解释方差: {pca_result['explained_variance'][1]:.2%}")
    print(f"  前2成分累积: {pca_result['cumulative_variance'][1]:.2%}")

    # 计算保留80%和90%方差需要的成分数
    n_components_80 = np.argmax(pca_result['cumulative_variance'] >= 0.8) + 1
    n_components_90 = np.argmax(pca_result['cumulative_variance'] >= 0.9) + 1

    print(f"  保留80%方差: {n_components_80}个成分")
    print(f"  保留90%方差: {n_components_90}个成分")

# 6. 类别可分性分析
print("\n步骤6: 类别可分性分析")

from scipy.stats import f_oneway

if 'pca' in results:
    pca_data = results['pca']['transformed']

    # 对每个主成分进行ANOVA
    for pc_idx in [0, 1]:
        pc_data = pca_data[:, pc_idx]
        groups = [pc_data[metadata['label'] == f'Class_{i}']
                  for i in range(n_classes)]

        f_stat, p_value = f_oneway(*groups)
        print(f"\n  PC{pc_idx+1} ANOVA:")
        print(f"    F统计量: {f_stat:.4f}")
        print(f"    p值: {p_value:.6f}")

        if p_value < 0.001:
            print(f"    结论: 类别间差异显著 (***p<0.001)")
        elif p_value < 0.01:
            print(f"    结论: 类别间差异显著 (**p<0.01)")
        elif p_value < 0.05:
            print(f"    结论: 类别间差异显著 (*p<0.05)")
        else:
            print(f"    结论: 类别间差异不显著")

print("\n" + "=" * 60)
print("示例4完成！")
print("=" * 60)
print("\n关键要点:")
print("  1. PCA适合线性降维和特征提取")
print("  2. PCoA基于距离矩阵，适合非线性数据")
print("  3. 碎石图帮助选择最优成分数")
print("  4. 累积方差指导降维维度选择")
print("  5. ANOVA评估类别可分性")
