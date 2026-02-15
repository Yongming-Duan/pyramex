# PyRamEx 用户指南

深入了解PyRamEx的高级功能和最佳实践。

## 目录

1. [数据准备](#数据准备)
2. [预处理策略](#预处理策略)
3. [质量控制](#质量控制)
4. [降维分析](#降维分析)
5. [机器学习工作流](#机器学习工作流)
6. [可视化技巧](#可视化技巧)
7. [性能优化](#性能优化)
8. [最佳实践](#最佳实践)

---

## 数据准备

### 支持的数据格式

PyRamEx支持多种拉曼光谱数据格式：

#### 1. 双列格式（Two-Column Format）

最常见的格式，每行包含波数和强度：

```
400    1200.5
401    1205.3
402    1198.7
...
```

#### 2. 矩阵格式（Matrix Format）

第一列是波数，其余列是不同样本：

```
Wavenumber    Sample1    Sample2    Sample3
400           1200.5     1198.3     1205.7
401           1205.3     1202.1     1208.9
...
```

#### 3. 坐标扫描格式（Coordinate Scan Format）

包含空间坐标的光谱映射：

```
X    Y    Wavenumber    Intensity
1    1    400           1200.5
1    1    401           1205.3
...
```

### 数据加载最佳实践

```python
from pyramex import load_spectra

# 自动检测格式
ramanome = load_spectra('data/')

# 指定格式
ramanome = load_spectra('data/', format='two_col')

# 加载单个文件
ramanome = load_spectra('data/spectrum.txt')

# 验证数据
print(f"样本数: {ramanome.n_samples}")
print(f"波数点数: {ramanome.n_wavenumbers}")
print(f"波数范围: {ramanome.wavenumbers.min():.1f} - {ramanome.wavenumbers.max():.1f}")
```

---

## 预处理策略

### 预处理流程选择

预处理顺序很重要。推荐的流程：

#### 基础流程

```python
# 1. 截取感兴趣区域
ramanome.cutoff((500, 3500))

# 2. 平滑去噪
ramanome.smooth(window_size=7, polyorder=3)

# 3. 去除基线
ramanome.remove_baseline(method='polyfit', degree=3)

# 4. 归一化
ramanome.normalize(method='minmax')
```

#### 高级流程

```python
# 针对高质量分析
ramanome.cutoff((500, 3500)) \
        .smooth(window_size=9, polyorder=3) \
        .remove_baseline(method='als', lam=1e5, p=0.01) \
        .normalize(method='area')
```

### 平滑方法选择

| 方法 | 适用场景 | 参数建议 |
|------|---------|---------|
| Savitzky-Golay | 通用平滑 | window_size=7-11, polyorder=2-3 |
| Moving Average | 快速平滑 | window_size=5-15 |

**参数调优：**
```python
# 小窗口保留细节
ramanome.smooth(window_size=5, polyorder=2)

# 大窗口平滑度高
ramanome.smooth(window_size=15, polyorder=3)
```

### 基线去除选择

| 方法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| polyfit | 简单基线 | 快速 | 复杂基线效果差 |
| als | 复杂基线 | 效果好 | 需要调参 |
| airpls | 自适应 | 自动化 | 计算量大 |

**示例：**
```python
# 简单基线
ramanome.remove_baseline(method='polyfit', degree=2)

# 复杂基线
ramanome.remove_baseline(method='als', lam=1e5, p=0.01)

# 荧光背景
ramanome.remove_baseline(method='airpls', lam=1e6)
```

### 归一化选择

| 方法 | 公式 | 适用场景 |
|------|------|---------|
| minmax | (x-min)/(max-min) | 强度比较 |
| zscore | (x-μ)/σ | 统计分析 |
| area | x/∫\|x\|dx | 定量分析 |
| vecnorm | x/\|\|x\|\|₂ | 机器学习 |

**示例：**
```python
# 机器学习预处理
ramanome.normalize(method='zscore')

# 定量比较
ramanome.normalize(method='area')

# 强度范围归一化
ramanome.normalize(method='minmax', feature_range=(0, 1))
```

---

## 质量控制

### QC方法选择指南

#### ICOD（逆协方差异常检测）

**适用：** 高维数据、多变量异常

```python
qc = ramanome.quality_control(method='icod', threshold=0.05)

# 更严格
qc = ramanome.quality_control(method='icod', threshold=0.01)
```

#### MCD（最小协方差行列式）

**适用：** 多元正态分布数据

```python
qc = ramanome.quality_control(method='mcd', threshold=0.95)
```

#### SNR（信噪比）

**适用：** 低质量数据、噪声识别

```python
qc = ramanome.quality_control(method='snr', threshold=10.0)

# 高质量要求
qc = ramanome.quality_control(method='snr', threshold=20.0)
```

#### 距离法（Distance-based）

**适用：** 快速筛选、初步检查

```python
qc = ramanome.quality_control(method='dis', threshold=0.05)
```

### QC结果分析

```python
# 执行QC
qc_result = ramanome.quality_control(method='icod', threshold=0.05)

# 查看结果
print(f"好样本: {qc_result.n_good}")
print(f"坏样本: {qc_result.n_bad}")
print(f"好样本率: {qc_result.good_rate:.2%}")

# 获取质量分数
scores = qc_result.quality_scores

# 可视化
from pyramex.visualization import plot_quality_control
plot_quality_control(ramanome, method='icod')
```

### 处理异常样本

```python
# 方法1：删除坏样本
good_indices = np.where(qc_result.good_samples)[0]
clean_spectra = ramanome.spectra[good_indices]

# 方法2：标记并单独分析
ramanome.metadata['quality'] = 'good'
ramanome.metadata.loc[~qc_result.good_samples, 'quality'] = 'bad'

# 方法3：重新测量
bad_samples = ramanome.metadata[~qc_result.good_samples]
print("需要重新测量的样本:", bad_samples['sample_id'].tolist())
```

---

## 降维分析

### PCA（主成分分析）

**适用：** 线性降维、特征提取

```python
# 执行PCA
result = ramanome.reduce(method='pca', n_components=5)

# 查看解释方差
print("解释方差比:", result['explained_variance'])
print("累积方差:", result['cumulative_variance'])

# 选择组件数
# 通常保留80-95%的方差
n_components = np.argmax(result['cumulative_variance'] >= 0.9) + 1
print(f"需要{n_components}个主成分保留90%方差")
```

### UMAP

**适用：** 非线性降维、可视化

```python
# 执行UMAP
result = ramanome.reduce(
    method='umap',
    n_components=2,
    n_neighbors=15,  # 影响局部/全局平衡
    min_dist=0.1     # 影响点分布
)

# 参数调优
# n_neighbors大 → 全局结构
# n_neighbors小 → 局部结构
# min_dist大 → 点分散
# min_dist小 → 点聚集
```

### t-SNE

**适用：** 可视化、聚类探索

```python
# 执行t-SNE
result = ramanome.reduce(
    method='tsne',
    n_components=2,
    perplexity=30  # 通常5-50
)

# 注意：t-SNE每次运行结果不同
# 设置random_state获得可重复结果
```

### 降维结果可视化

```python
# 2D可视化
ramanome.plot_reduction(method='pca', color_by='label')

# 3D可视化
ramanome.reduce(method='pca', n_components=3)
ramanome.plot_reduction(method='pca', color_by='group')

# 保存结果
import matplotlib.pyplot as plt
fig = ramanome.plot_reduction(method='pca', return_fig=True)
fig.savefig('pca_plot.png', dpi=300)
```

---

## 机器学习工作流

### 分类任务

```python
from pyramex.ml import to_sklearn_format
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. 数据准备
ramanome.smooth(window_size=5).normalize(method='minmax')

# 2. 编码标签
le = LabelEncoder()
ramanome.metadata['label_encoded'] = le.fit_transform(ramanome.metadata['label'])

# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    ramanome.spectra,
    ramanome.metadata['label_encoded'],
    test_size=0.2,
    random_state=42,
    stratify=ramanome.metadata['label_encoded']
)

# 4. 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 评估
print(f"训练集准确率: {model.score(X_train, y_train):.2%}")
print(f"测试集准确率: {model.score(X_test, y_test):.2%}")

# 6. 交叉验证
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"交叉验证准确率: {scores.mean():.2%} ± {scores.std():.2%}")
```

### 回归任务

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 回归任务
X_train, X_test, y_train, y_test = to_sklearn_format(
    ramanome,
    test_size=0.2
)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### 深度学习（PyTorch）

```python
from pyramex.ml import to_torch_dataset, create_cnn_model
from torch.utils.data import DataLoader

# 1. 创建数据集
dataset = to_torch_dataset(ramanome)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 2. 创建模型
model = create_cnn_model(
    input_length=ramanome.n_wavenumbers,
    n_classes=len(ramanome.metadata['label'].unique()),
    dropout=0.3
)

# 3. 训练循环
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for spectra, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(spectra)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## 可视化技巧

### 自定义光谱图

```python
import matplotlib.pyplot as plt

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 原始光谱
axes[0, 0].plot(ramanome.wavenumbers, ramanome.spectra.T)
axes[0, 0].set_title('原始光谱')
axes[0, 0].invert_xaxis()

# 预处理后的光谱
axes[0, 1].plot(ramanome.wavenumbers, ramanome.spectra.T)
axes[0, 1].set_title('预处理后')
axes[0, 1].invert_xaxis()

# 平均光谱
mean_spectrum = ramanome.spectra.mean(axis=0)
std_spectrum = ramanome.spectra.std(axis=0)
axes[1, 0].plot(ramanome.wavenumbers, mean_spectrum)
axes[1, 0].fill_between(
    ramanome.wavenumbers,
    mean_spectrum - std_spectrum,
    mean_spectrum + std_spectrum,
    alpha=0.3
)
axes[1, 0].set_title('平均光谱 ± 标准差')
axes[1, 0].invert_xaxis()

# 差异光谱
axes[1, 1].plot(ramanome.wavenumbers, (ramanome.spectra - mean_spectrum).T)
axes[1, 1].set_title('差异光谱')
axes[1, 1].invert_xaxis()

plt.tight_layout()
plt.savefig('spectral_analysis.png', dpi=300)
```

### 交互式可视化

```python
# Plotly交互式图表
try:
    import plotly.graph_objects as go

    fig = go.Figure()

    for i in range(ramanome.n_samples):
        fig.add_trace(go.Scatter(
            x=ramanome.wavenumbers,
            y=ramanome.spectra[i],
            name=f'Sample {i}',
            opacity=0.7
        ))

    fig.update_layout(
        title='交互式光谱',
        xaxis_title='波数 (cm⁻¹)',
        yaxis_title='强度',
        hovermode='x unified'
    )

    fig.show()
except ImportError:
    print("Plotly未安装")
```

---

## 性能优化

### 大数据集处理

```python
# 1. 分批处理
def process_in_batches(ramanome, batch_size=100):
    results = []
    for i in range(0, ramanome.n_samples, batch_size):
        batch = ramanome.spectra[i:i+batch_size]
        # 处理batch
        processed = smooth(batch, window_size=5)
        results.append(processed)
    return np.vstack(results)

# 2. 使用降采样
if ramanome.n_wavenumbers > 1000:
    # 降采样到1000个点
    indices = np.linspace(0, ramanome.n_wavenumbers-1, 1000, dtype=int)
    ramanome.spectra = ramanome.spectra[:, indices]
    ramanome.wavenumbers = ramanome.wavenumbers[indices]

# 3. 使用稀疏矩阵（如果适用）
from scipy.sparse import csr_matrix
sparse_spectra = csr_matrix(ramanome.spectra)
```

### 并行处理

```python
from joblib import Parallel, delayed

def parallel_preprocessing(spectra, n_jobs=-1):
    def process_spectrum(spec):
        return smooth(spec.reshape(1, -1), window_size=5)[0]

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_spectrum)(spectra[i])
        for i in range(len(spectra))
    )
    return np.array(results)
```

---

## 最佳实践

### 1. 数据管理

```python
# 保存处理后的数据
import pickle

with open('processed_ramanome.pkl', 'wb') as f:
    pickle.dump(ramanome, f)

# 或使用HDF5格式（适合大数据）
import h5py
with h5py.File('spectra.h5', 'w') as f:
    f.create_dataset('spectra', data=ramanome.spectra)
    f.create_dataset('wavenumbers', data=ramanome.wavenumbers)
```

### 2. 可重复性

```python
# 设置随机种子
np.random.seed(42)

# 记录处理步骤
processing_log = {
    'smoothing': {'window_size': 5, 'polyorder': 2},
    'baseline': {'method': 'polyfit', 'degree': 3},
    'normalization': {'method': 'minmax'}
}

import json
with open('processing_log.json', 'w') as f:
    json.dump(processing_log, f)
```

### 3. 错误处理

```python
try:
    ramanome.smooth(window_size=5)
except ValueError as e:
    print(f"平滑失败: {e}")
    # 尝试替代方法
    ramanome.smooth(window_size=7)
```

### 4. 验证

```python
# 检查数据完整性
assert not np.isnan(ramanome.spectra).any(), "包含NaN值"
assert not np.isinf(ramanome.spectra).any(), "包含无穷值"
assert ramanome.spectra.min() >= 0, "包含负值（如果预期非负）"

# 检查预处理效果
original_var = spectra.var()
processed_var = ramanome.spectra.var()
print(f"方差变化: {original_var:.2f} → {processed_var:.2f}")
```

---

## 故障排除

### 常见问题

**Q1: 内存不足**
```python
# 解决方案：减少数据大小或分批处理
ramanome.cutoff((500, 2000))  # 减少波数范围
```

**Q2: 降维失败**
```python
# 解决方案：先进行预处理
ramanome.smooth().normalize()
ramanome.reduce(method='pca', n_components=2)
```

**Q3: 可视化显示问题**
```python
# 解决方案：使用非交互式后端
import matplotlib
matplotlib.use('Agg')
```

---

## 下一步

- 查看[API参考](api.md)了解详细API
- 浏览[示例代码](../examples/)学习实际应用
- 参与[GitHub讨论](https://github.com/Yongming-Duan/pyramex/discussions)

---

**版本：** 0.1.0-beta
**最后更新：** 2026-02-15
