# PyRamEx 快速开始教程

本教程将帮助您快速上手PyRamEx，学习基本的拉曼光谱分析流程。

## 安装

首先安装PyRamEx：

```bash
pip install pyramex
```

## 基本概念

PyRamEx的核心数据结构是`Ramanome`类，它封装了拉曼光谱数据和相关操作。

```python
from pyramex import Ramanome
```

## 示例1：加载和查看数据

### 从文件加载

PyRamEx支持多种文件格式：

```python
from pyramex import load_spectra

# 加载单个文件
ramanome = load_spectra('path/to/spectrum.txt')

# 加载目录中的所有文件
ramanome = load_spectra('path/to/spectra_directory/')
```

### 从NumPy数组创建

```python
import numpy as np
import pandas as pd

# 创建示例数据
spectra = np.random.randn(10, 100)  # 10个样本，100个波数点
wavenumbers = np.linspace(400, 4000, 100)  # 400-4000 cm^-1
metadata = pd.DataFrame({
    'sample_id': range(10),
    'label': ['A', 'B'] * 5
})

# 创建Ramanome对象
ramanome = Ramanome(spectra, wavenumbers, metadata)

print(ramanome)
# 输出: Ramanome(n_samples=10, n_wavenumbers=100, processed=0 steps)
```

## 示例2：数据预处理

PyRamEx提供链式API进行数据预处理：

```python
# 链式预处理
ramanome = (ramanome
            .smooth(window_size=5, polyorder=2)      # 平滑
            .remove_baseline(method='polyfit', degree=3)  # 去基线
            .normalize(method='minmax')              # 归一化
            .cutoff((500, 3500)))                    # 截取波数范围

print(f"预处理步骤: {ramanome.processed}")
# 输出: ['smooth(w=5, p=2)', 'baseline(method=polyfit)', 'normalize(method=minmax)', 'cutoff((500, 3500))']
```

### 可用的预处理方法

**平滑（Smoothing）:**
```python
# Savitzky-Golay平滑
ramanome.smooth(window_size=7, polyorder=3)
```

**基线去除（Baseline Removal）:**
```python
# 多项式拟合
ramanome.remove_baseline(method='polyfit', degree=3)

# 不对称最小二乘法
ramanome.remove_baseline(method='als', lam=1e5, p=0.01)

# airPLS算法
ramanome.remove_baseline(method='airpls')
```

**归一化（Normalization）:**
```python
# Min-Max归一化
ramanome.normalize(method='minmax')

# Z-score归一化
ramanome.normalize(method='zscore')

# 面积归一化
ramanome.normalize(method='area')

# 向量归一化
ramanome.normalize(method='vecnorm')
```

## 示例3：可视化

### 绘制光谱

```python
# 绘制所有光谱
ramanome.plot()

# 绘制特定样本
ramanome.plot(samples=[0, 1, 2])

# 保存图形
import matplotlib.pyplot as plt
fig = ramanome.plot(return_fig=True)
fig.savefig('spectra.png')
```

### 降维可视化

```python
# 首先进行降维
ramanome.reduce(method='pca', n_components=2)

# 绘制PCA结果，按标签着色
ramanome.plot_reduction(method='pca', color_by='label')
```

## 示例4：质量控制

识别异常样本：

```python
# 使用距离法进行质量控制
qc_result = ramanome.quality_control(method='dis', threshold=0.05)

print(qc_result)
# 输出: QualityResult(method=dis, good=8/10, rate=80.00%)

# 获取好样本的索引
good_indices = np.where(qc_result.good_samples)[0]
print(f"好样本: {good_indices}")
```

### 可用的QC方法

```python
# 逆协方差异常检测
qc1 = ramanome.quality_control(method='icod', threshold=0.05)

# 最小协方差行列式
qc2 = ramanome.quality_control(method='mcd', threshold=0.95)

# Hotelling T²检验
qc3 = ramanome.quality_control(method='t2', alpha=0.95)

# 信噪比
qc4 = ramanome.quality_control(method='snr', threshold=10.0)
```

## 示例5：降维和特征提取

### PCA降维

```python
# 执行PCA
result = ramanome.reduce(method='pca', n_components=3)

# 查看解释方差
print("解释方差比:", result['explained_variance'])
print("累积方差:", result['cumulative_variance'])

# 获取降维后的数据
transformed = ramanome.reductions['pca']['transformed']
print("降维后形状:", transformed.shape)  # (10, 3)
```

### 提取特征波段

```python
from pyramex.features import extract_band_intensity

# 提取特定波段的强度
bands = [(500, 600), (1000, 1100), (1500, 1600)]
intensities = extract_band_intensity(ramanome, bands)

print("波段强度形状:", intensities.shape)  # (10, 3)
```

### 计算CDR（胞质比）

```python
from pyramex.features import calculate_cdr

cdr = calculate_cdr(ramanome, band1=(500, 600), band2=(1000, 1100))
print("CDR值:", cdr)
```

## 示例6：机器学习集成

### 准备sklearn数据

```python
from pyramex.ml import to_sklearn_format
from sklearn.ensemble import RandomForestClassifier

# 转换为sklearn格式
X_train, X_test, y_train, y_test = to_sklearn_format(
    ramanome,
    test_size=0.2,
    random_state=42
)

# 训练模型
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 评估
score = model.score(X_test, y_test)
print(f"准确率: {score:.2%}")
```

### PyTorch集成

```python
from pyramex.ml import to_torch_dataset
from torch.utils.data import DataLoader

# 转换为PyTorch数据集
dataset = to_torch_dataset(ramanome)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 训练循环
for spectra_batch, labels_batch in dataloader:
    # spectra_batch形状: (batch_size, 1, n_wavenumbers)
    # 使用模型训练...
    pass
```

### TensorFlow集成

```python
from pyramex.ml import to_tf_dataset

# 转换为TensorFlow数据集
dataset = to_tf_dataset(ramanome, batch_size=4, shuffle=True)

# 训练
for spectra_batch, labels_batch in dataset:
    # spectra_batch形状: (batch_size, 1, n_wavenumbers)
    # 使用模型训练...
    pass
```

## 示例7：完整分析流程

```python
from pyramex import load_spectra
from pyramex.ml import to_sklearn_format
from sklearn.ensemble import RandomForestClassifier

# 1. 加载数据
ramanome = load_spectra('data/spectra/')

# 2. 预处理
ramanome = (ramanome
            .smooth(window_size=5)
            .remove_baseline(method='polyfit', degree=2)
            .normalize(method='minmax'))

# 3. 质量控制
qc_result = ramanome.quality_control(method='dis', threshold=0.05)
good_indices = np.where(qc_result.good_samples)[0]

# 4. 过滤好样本
from pyramex.core.ramanome import Ramanome
clean_ramanome = Ramanome(
    ramanome.spectra[good_indices],
    ramanome.wavenumbers,
    ramanome.metadata.iloc[good_indices].reset_index(drop=True)
)

# 5. 降维
clean_ramanome.reduce(method='pca', n_components=2)

# 6. 可视化
clean_ramanome.plot_reduction(method='pca', color_by='label')

# 7. 机器学习
X_train, X_test, y_train, y_test = to_sklearn_format(
    clean_ramanome,
    test_size=0.2
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print(f"测试集准确率: {model.score(X_test, y_test):.2%}")
```

## 下一步

恭喜！您已经掌握了PyRamEx的基本用法。

继续学习：

- [用户指南](user_guide.md) - 深入了解高级功能
- [API参考](api.md) - 查看完整的API文档
- [示例代码](../examples/) - 更多实际应用案例

## 常见问题

### 如何重置预处理？

```python
# 重置到原始数据
ramanome.reset()
```

### 如何复制Ramanome对象？

```python
# 深拷贝
ramanome2 = ramanome.copy()
```

### 如何导出数据？

```python
# 转换为ML格式
X = ramanome.to_ml_format()

# 转换为张量（深度学习）
tensor = ramanome.to_tensor(add_channel=True)

# 导出到CSV
import pandas as pd
df = pd.DataFrame(ramanome.spectra, columns=ramanome.wavenumbers)
df.to_csv('spectra.csv', index=False)
```

---

**版本：** 0.1.0-beta
**最后更新：** 2026-02-15
