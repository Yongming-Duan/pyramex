"""
示例5: 批量处理工作流

展示如何批量处理多个光谱文件
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyramex import Ramanome, load_spectra
from pyramex.preprocessing import smooth, remove_baseline, normalize
from pyramex.qc import quality_control
from pyramex.features import reduce

print("=" * 60)
print("示例5: 批量处理工作流")
print("=" * 60)

# 1. 创建模拟数据集
print("\n步骤1: 创建模拟数据文件")

data_dir = Path('examples/data/batch_data')
data_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n_files = 20
wavenumbers = np.linspace(400, 4000, 500)

created_files = []

for i in range(n_files):
    # 创建不同的光谱
    baseline = 0.001 * wavenumbers
    peak1 = 10 * np.exp(-((wavenumbers - 1000)**2) / (2 * 50**2))
    peak2 = (5 + i * 0.3) * np.exp(-((wavenumbers - 1600)**2) / (2 * 60**2))
    noise = np.random.randn(500) * 0.5
    spectrum = baseline + peak1 + peak2 + noise

    # 保存为双列格式
    file_path = data_dir / f'spectrum_{i:03d}.txt'
    data = pd.DataFrame({
        'wavenumber': wavenumbers,
        'intensity': spectrum
    })
    data.to_csv(file_path, sep='\t', header=False, index=False)
    created_files.append(file_path)

print(f"  创建了 {len(created_files)} 个文件")
print(f"  目录: {data_dir}")

# 2. 批量加载
print("\n步骤2: 批量加载光谱")

ramanome = load_spectra(data_dir)
print(f"  加载了 {ramanome.n_samples} 个光谱")
print(f"  波数点数: {ramanome.n_wavenumbers}")

# 3. 批量预处理
print("\n步骤3: 批量预处理")

ramanome.cutoff((500, 3500))
ramanome.smooth(window_size=7, polyorder=2)
ramanome.remove_baseline(method='polyfit', degree=2)
ramanome.normalize(method='minmax')

print("  ✓ 预处理完成")

# 4. 批量质量控制
print("\n步骤4: 批量质量控制")

qc_result = quality_control(ramanome, method='dis', threshold=0.1)

print(f"  好样本: {qc_result.n_good}")
print(f"  坏样本: {qc_result.n_bad}")
print(f"  好样本率: {qc_result.good_rate:.1%}")

# 5. 批量特征提取
print("\n步骤5: 批量特征提取")

# PCA降维
ramanome.reduce(method='pca', n_components=3)

# 提取特定波段强度
from pyramex.features import extract_band_intensity

bands = [(800, 900), (1000, 1100), (1500, 1600), (2000, 2100)]
band_intensities = extract_band_intensity(ramanome, bands)

print(f"  提取了 {band_intensities.shape[1]} 个波段特征")
print(f"  特征矩阵形状: {band_intensities.shape}")

# 6. 创建结果报告
print("\n步骤6: 创建结果报告")

# 汇总统计
report = pd.DataFrame({
    'file_name': [f'spectrum_{i:03d}.txt' for i in range(n_files)],
    'quality_score': qc_result.quality_scores,
    'is_good': qc_result.good_samples,
    'band_800_900': band_intensities[:, 0],
    'band_1000_1100': band_intensities[:, 1],
    'band_1500_1600': band_intensities[:, 2],
    'band_2000_2100': band_intensities[:, 3],
})

# 添加PCA分数
report['pca_1'] = ramanome.reductions['pca']['transformed'][:, 0]
report['pca_2'] = ramanome.reductions['pca']['transformed'][:, 1]
report['pca_3'] = ramanome.reductions['pca']['transformed'][:, 2]

report.to_csv('examples/batch_processing_report.csv', index=False)
print("  ✓ 保存报告: examples/batch_processing_report.csv")

# 7. 可视化批量处理结果
print("\n步骤7: 可视化")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 质量分数分布
axes[0, 0].hist(qc_result.quality_scores, bins=20, edgecolor='black')
axes[0, 0].axvline(x=qc_result.threshold, color='red', linestyle='--', label='阈值')
axes[0, 0].set_xlabel('质量分数')
axes[0, 0].set_ylabel('频数')
axes[0, 0].set_title('质量分数分布')
axes[0, 0].legend()

# 好坏样本对比
good_indices = np.where(qc_result.good_samples)[0]
bad_indices = np.where(~qc_result.good_samples)[0]

if len(bad_indices) > 0:
    axes[0, 1].plot(ramanome.wavenumbers,
                    ramanome.spectra[good_indices].mean(axis=0),
                    'g-', label='好样本平均', linewidth=2)
    axes[0, 1].plot(ramanome.wavenumbers,
                    ramanome.spectra[bad_indices].mean(axis=0),
                    'r-', label='坏样本平均', linewidth=2)
    axes[0, 1].set_xlabel('波数 (cm⁻¹)')
    axes[0, 1].set_ylabel('强度（归一化）')
    axes[0, 1].set_title('好坏样本对比')
    axes[0, 1].invert_xaxis()
    axes[0, 1].legend()

# PCA结果
colors = ['green' if qc_result.good_samples[i] else 'red' for i in range(len(qc_result.good_samples))]
axes[0, 2].scatter(report['pca_1'], report['pca_2'], c=colors, alpha=0.6)
axes[0, 2].set_xlabel('PC1')
axes[0, 2].set_ylabel('PC2')
axes[0, 2].set_title('PCA结果（绿=好，红=坏）')
axes[0, 2].grid(True, alpha=0.3)

# 波段强度分布
band_data = [report[f'band_{bands[i][0]}_{bands[i][1]}'].values for i in range(4)]
axes[1, 0].boxplot(band_data, labels=[f'{b[0]}-{b[1]}' for b in bands])
axes[1, 0].set_xlabel('波段 (cm⁻¹)')
axes[1, 0].set_ylabel('强度')
axes[1, 0].set_title('波段强度分布')
axes[1, 0].tick_params(axis='x', rotation=45)

# 波段强度相关性
import seaborn as sns
band_corr = report[[f'band_{bands[i][0]}_{bands[i][1]}' for i in range(4)]].corr()
sns.heatmap(band_corr, annot=True, cmap='coolwarm', center=0,
            xticklabels=[f'{b[0]}-{b[1]}' for b in bands],
            yticklabels=[f'{b[0]}-{b[1]}' for b in bands],
            ax=axes[1, 1])
axes[1, 1].set_title('波段强度相关性')

# 处理流程摘要
summary_text = f"""
批量处理摘要
{'='*30}
总样本数: {len(report)}
好样本: {qc_result.n_good} ({qc_result.good_rate:.1%})
坏样本: {qc_result.n_bad}

预处理步骤:
{'  ' + chr(10).join(ramanome.processed)}

波段特征:
{chr(10).join([f'  {bands[i][0]}-{bands[i][1]}: {band_intensities[:, i].mean():.4f} ± {band_intensities[:, i].std():.4f}' for i in range(4)])}

降维:
  PCA前3成分累积方差: {ramanome.reductions['pca']['cumulative_variance'][2]:.2%}
"""

axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('examples/ex5_batch_processing.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存图像: examples/ex5_batch_processing.png")

# 8. 保存处理后的数据
print("\n步骤8: 保存处理后的数据")

# 保存好样本
if qc_result.n_good > 0:
    good_ramanome = Ramanome(
        ramanome.spectra[good_indices],
        ramanome.wavenumbers,
        metadata.iloc[good_indices].reset_index(drop=True)
    )

    # 保存为CSV格式
    good_data = pd.DataFrame(
        good_ramanome.spectra.T,
        index=[f'{wn:.1f}' for wn in good_ramanome.wavenumbers],
        columns=[f'Sample_{i}' for i in range(good_ramanome.n_samples)]
    )
    good_data.to_csv('examples/processed_good_spectra.csv')
    print("  ✓ 保存好样本: examples/processed_good_spectra.csv")

print("\n" + "=" * 60)
print("示例5完成！")
print("=" * 60)
print("\n关键要点:")
print("  1. 使用load_spectra()批量加载文件")
print("  2. 链式预处理简化批量处理")
print("  3. QC自动过滤低质量数据")
print("  4. 批量提取特征用于下游分析")
print("  5. 生成结构化报告便于追踪")
print("\n生成的文件:")
print("  - batch_processing_report.csv")
print("  - processed_good_spectra.csv")
print("  - ex5_batch_processing.png")
