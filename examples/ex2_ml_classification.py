"""
示例2: 机器学习分类工作流

展示如何使用PyRamEx进行光谱分类
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyramex import Ramanome
from pyramex.ml import to_sklearn_format
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("=" * 60)
print("示例2: 机器学习分类工作流")
print("=" * 60)

# 1. 加载或创建数据
print("\n步骤1: 准备数据")

# 使用示例1的数据生成函数
np.random.seed(42)
n_samples_per_class = 50
wavenumbers = np.linspace(400, 4000, 500)

spectra_list = []
labels = []

# 创建3个类别
for class_id, peak_pos in enumerate([1000, 1500, 2000]):
    for i in range(n_samples_per_class):
        baseline = 0.001 * wavenumbers + np.random.randn(500) * 0.1
        peak = 10 * np.exp(-((wavenumbers - peak_pos)**2) / (2 * 50**2))
        spectrum = baseline + peak
        spectra_list.append(spectrum)
        labels.append(f'Class_{class_id}')

all_spectra = np.vstack(spectra_list)
metadata = pd.DataFrame({
    'sample_id': range(len(all_spectra)),
    'label': labels
})

print(f"  样本数: {len(all_spectra)}")
print(f"  类别数: 3")
print(f"  波数点数: {len(wavenumbers)}")

# 2. 创建Ramanome并预处理
print("\n步骤2: 数据预处理")

ramanome = Ramanome(all_spectra, wavenumbers, metadata)

# 预处理流程
ramanome.cutoff((500, 3500))
ramanome.smooth(window_size=9, polyorder=3)
ramanome.remove_baseline(method='polyfit', degree=2)
ramanome.normalize(method='minmax')

print("  ✓ 完成预处理")

# 3. 划分数据集
print("\n步骤3: 划分训练集和测试集")

X_train, X_test, y_train, y_test = to_sklearn_format(
    ramanome,
    test_size=0.2,
    random_state=42
)

print(f"  训练集大小: {X_train.shape[0]}")
print(f"  测试集大小: {X_test.shape[0]}")
print(f"  特征数: {X_train.shape[1]}")

# 4. 训练分类器
print("\n步骤4: 训练随机森林分类器")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("  ✓ 模型训练完成")

# 5. 评估模型
print("\n步骤5: 模型评估")

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\n准确率:")
print(f"  训练集: {train_score:.2%}")
print(f"  测试集: {test_score:.2%}")

# 交叉验证
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\n5折交叉验证:")
print(f"  平均准确率: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
print(f"  各折分数: {[f'{s:.2%}' for s in cv_scores]}")

# 详细分类报告
y_pred = model.predict(X_test)
print(f"\n分类报告:")
print(classification_report(y_test, y_pred))

# 6. 特征重要性分析
print("\n步骤6: 特征重要性分析")

feature_importance = model.feature_importances_
top_indices = np.argsort(feature_importance)[-10:]  # 最重要的10个特征

print("\n最重要的10个波数点:")
for idx in reversed(top_indices):
    wn = ramanome.wavenumbers[idx]
    importance = feature_importance[idx]
    print(f"  {wn:.1f} cm⁻¹: {importance:.4f}")

# 7. 可视化
print("\n步骤7: 可视化结果")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_,
            yticklabels=model.classes_, cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('混淆矩阵')
axes[0, 0].set_xlabel('预测类别')
axes[0, 0].set_ylabel('真实类别')

# 特征重要性
axes[0, 1].plot(ramanome.wavenumbers, feature_importance)
axes[0, 1].set_xlabel('波数 (cm⁻¹)')
axes[0, 1].set_ylabel('重要性')
axes[0, 1].set_title('特征重要性')
axes[0, 1].invert_xaxis()

# 平均光谱
for i, label in enumerate(['Class_0', 'Class_1', 'Class_2']):
    class_mask = (metadata['label'] == label).values
    mean_spectrum = ramanome.spectra[class_mask].mean(axis=0)
    axes[1, 0].plot(ramanome.wavenumbers, mean_spectrum,
                    label=label, linewidth=2)

axes[1, 0].set_xlabel('波数 (cm⁻¹)')
axes[1, 0].set_ylabel('强度（归一化）')
axes[1, 0].set_title('各类别平均光谱')
axes[1, 0].invert_xaxis()
axes[1, 0].legend()

# 交叉验证分数
axes[1, 1].bar(range(5), cv_scores)
axes[1, 1].axhline(y=cv_scores.mean(), color='r', linestyle='--', label='平均值')
axes[1, 1].set_xlabel('折数')
axes[1, 1].set_ylabel('准确率')
axes[1, 1].set_title('交叉验证分数')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('examples/ex2_ml_classification.png', dpi=300, bbox_inches='tight')
print("  ✓ 保存图像: examples/ex2_ml_classification.png")

# 8. 保存模型
print("\n步骤8: 保存模型")

import joblib
joblib.dump(model, 'examples/rf_model.pkl')
joblib.dump(ramanome.wavenumbers, 'examples/wavenumbers.pkl')
print("  ✓ 保存模型: examples/rf_model.pkl")

print("\n" + "=" * 60)
print("示例2完成！")
print("=" * 60)
print("\n关键要点:")
print("  1. 使用to_sklearn_format()快速转换为ML格式")
print("  2. 预处理对分类性能至关重要")
print("  3. 交叉验证评估模型稳定性")
print("  4. 特征重要性揭示关键波数位置")
