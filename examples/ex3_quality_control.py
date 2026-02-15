"""
示例3: 质量控制和异常检测

展示如何使用PyRamEx识别异常光谱
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyramex import Ramanome
from pyramex.qc import quality_control
from pyramex.preprocessing import smooth, normalize

print("=" * 60)
print("示例3: 质量控制和异常检测")
print("=" * 60)

# 1. 创建包含异常值的数据集
print("\n步骤1: 创建数据集（包含异常样本）")

np.random.seed(42)
n_good_samples = 45
n_bad_samples = 5
wavenumbers = np.linspace(400, 4000, 500)

# 生成正常样本
good_spectra = []
for i in range(n_good_samples):
    baseline = 0.001 * wavenumbers
    peak = 10 * np.exp(-((wavenumbers - 1000)**2) / (2 * 50**2))
    noise = np.random.randn(500) * 0.5
    spectrum = baseline + peak + noise
    good_spectra.append(spectrum)

# 生成异常样本
bad_spectra = []
# 类型1: 高噪声
for i in range(2):
    baseline = 0.001 * wavenumbers
    peak = 10 * np.exp(-((wavenumbers - 1000)**2) / (2 * 50**2))
    noise = np.random.randn(500) * 5.0  # 高噪声
    spectrum = baseline + peak + noise
    bad_spectra.append(spectrum)

# 类型2: 基线漂移
for i in range(2):
    baseline = 0.005 * wavenumbers + 50  # 严重基线漂移
    peak = 10 * np.exp(-((wavenumbers - 1000)**2) / (2 * 50**2))
    noise = np.random.randn(500) * 0.5
    spectrum = baseline + peak + noise
    bad_spectra.append(spectrum)

# 类型3: 峰位偏移
for i in range(1):
    baseline = 0.001 * wavenumbers
    peak = 10 * np.exp(-((wavenumbers - 1500)**2) / (2 * 50**2))  # 峰位偏移
    noise = np.random.randn(500) * 0.5
    spectrum = baseline + peak + noise
    bad_spectra.append(spectrum)

# 合并数据
all_spectra = np.vstack([np.array(good_spectra), np.array(bad_spectra)])
labels = (['Good'] * n_good_samples +
          ['High Noise'] * 2 +
          ['Baseline Drift'] * 2 +
          ['Peak Shift'] * 1)

metadata = pd.DataFrame({
    'sample_id': range(len(all_spectra)),
    'true_label': labels
})

print(f"  总样本数: {len(all_spectra)}")
print(f"  正常样本: {n_good_samples}")
print(f"  异常样本: {n_bad_samples}")

# 2. 创建Ramanome
ramanome = Ramanome(all_spectra, wavenumbers, metadata)

# 3. 预处理
print("\n步骤2: 预处理")

ramanome.cutoff((500, 3500))
ramanome.smooth(window_size=7, polyorder=2)
ramanome.normalize(method='minmax')

print("  ✓ 预处理完成")

# 4. 应用多种QC方法
print("\n步骤3: 应用质量控制方法")

qc_methods = ['dis', 'snr', 'icod']
qc_results = {}

for method in qc_methods:
    print(f"\n  {method.upper()}方法:")
    try:
        if method == 'snr':
            qc_result = quality_control(ramanome, method=method, threshold=15.0)
        else:
            qc_result = quality_control(ramanome, method=method, threshold=0.1)

        qc_results[method] = qc_result

        print(f"    好样本: {qc_result.n_good}")
        print(f"    坏样本: {qc_result.n_bad}")
        print(f"    好样本率: {qc_result.good_rate:.1%}")
    except Exception as e:
        print(f"    错误: {e}")

# 5. 分析结果
print("\n步骤4: 分析QC结果")

# 使用距离法的结果作为主要结果
primary_qc = qc_results['dis']

# 检测效果
good_indices = np.where(primary_qc.good_samples)[0]
bad_indices = np.where(~primary_qc.good_samples)[0]

print(f"\n检测到的异常样本索引: {bad_indices.tolist()}")

# 计算检测准确率
true_bad_indices = metadata[metadata['true_label'] != 'Good'].index.values
detected_bad = set(bad_indices) & set(true_bad_indices)
false_positive = set(bad_indices) - set(true_bad_indices)
false_negative = set(true_bad_indices) - set(bad_indices)

print(f"\n检测性能:")
print(f"  真阳性（正确检测的异常）: {len(detected_bad)}/{len(true_bad_indices)}")
print(f"  假阳性（误判的正常样本）: {len(false_positive)}")
print(f"  假阴性（未检测的异常）: {len(false_negative)}")

# 6. 可视化
print("\n步骤5: 可视化")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 原始光谱（区分好坏）
colors = ['green' if primary_qc.good_samples[i] else 'red' for i in range(len(all_spectra))]
for i in range(len(all_spectra)):
    axes[0, 0].plot(wavenumbers, all_spectra[i], color=colors[i], alpha=0.5, linewidth=0.5)
axes[0, 0].set_title('原始光谱（绿=好，红=坏）')
axes[0, 0].set_xlabel('波数 (cm⁻¹)')
axes[0, 0].set_ylabel('强度')
axes[0, 0].invert_xaxis()

# 预处理后的光谱
for i in range(len(all_spectra)):
    axes[0, 1].plot(ramanome.wavenumbers, ramanome.spectra[i],
                    color=colors[i], alpha=0.5, linewidth=0.5)
axes[0, 1].set_title('预处理后（绿=好，红=坏）')
axes[0, 1].set_xlabel('波数 (cm⁻¹)')
axes[0, 1].set_ylabel('强度（归一化）')
axes[0, 1].invert_xaxis()

# 质量分数
axes[0, 2].bar(range(len(primary_qc.quality_scores)),
               primary_qc.quality_scores,
               color=colors)
axes[0, 2].axhline(y=primary_qc.threshold, color='red', linestyle='--', label='阈值')
axes[0, 2].set_xlabel('样本索引')
axes[0, 2].set_ylabel('质量分数')
axes[0, 2].set_title('质量分数（距离法）')
axes[0, 2].legend()

# 平均光谱对比
mean_good = ramanome.spectra[good_indices].mean(axis=0)
std_good = ramanome.spectra[good_indices].std(axis=0)
if len(bad_indices) > 0:
    mean_bad = ramanome.spectra[bad_indices].mean(axis=0)
    std_bad = ramanome.spectra[bad_indices].std(axis=0)

axes[1, 0].plot(ramanome.wavenumbers, mean_good, 'g-', label='好样本', linewidth=2)
axes[1, 0].fill_between(ramanome.wavenumbers,
                         mean_good - std_good,
                         mean_good + std_good,
                         alpha=0.3, color='green')
if len(bad_indices) > 0:
    axes[1, 0].plot(ramanome.wavenumbers, mean_bad, 'r-', label='坏样本', linewidth=2)
    axes[1, 0].fill_between(ramanome.wavenumbers,
                             mean_bad - std_bad,
                             mean_bad + std_bad,
                             alpha=0.3, color='red')
axes[1, 0].set_xlabel('波数 (cm⁻¹)')
axes[1, 0].set_ylabel('强度（归一化）')
axes[1, 0].set_title('平均光谱对比')
axes[1, 0].invert_xaxis()
axes[1, 0].legend()

# 不同QC方法对比
method_comparison = []
method_names = []
for method, qc_result in qc_results.items():
    method_comparison.append(qc_result.good_rate)
    method_names.append(method.upper())

axes[1, 1].bar(method_names, method_comparison)
axes[1, 1].set_ylabel('好样本率')
axes[1, 1].set_title('不同QC方法对比')
axes[1, 1].set_ylim([0, 1])

# 异常类型分布
if len(bad_indices) > 0:
    bad_labels = metadata.iloc[bad_indices]['true_label'].value_counts()
    axes[1, 2].pie(bad_labels.values, labels=bad_labels.index, autopct='%1.1f%%')
    axes[1, 2].set_title('检测到的异常类型')
else:
    axes[1, 2].text(0.5, 0.5, '未检测到异常',
                    ha='center', va='center', fontsize=14)
    axes[1, 2].set_title('检测到的异常类型')

plt.tight_layout()
plt.savefig('examples/ex3_quality_control.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存图像: examples/ex3_quality_control.png")

# 7. 导出异常样本报告
print("\n步骤6: 导出异常样本报告")

if len(bad_indices) > 0:
    bad_samples_report = metadata.iloc[bad_indices].copy()
    bad_samples_report['quality_score'] = primary_qc.quality_scores[bad_indices]
    bad_samples_report.to_csv('examples/bad_samples_report.csv', index=False)
    print("  ✓ 保存报告: examples/bad_samples_report.csv")
else:
    print("  ✓ 未检测到异常样本")

print("\n" + "=" * 60)
print("示例3完成！")
print("=" * 60)
print("\n关键要点:")
print("  1. 使用多种QC方法综合评估")
print("  2. 距离法适合快速筛查")
print("  3. SNR方法关注信号质量")
print("  4. ICOD方法检测多元异常")
print("  5. 可视化帮助理解检测原因")
