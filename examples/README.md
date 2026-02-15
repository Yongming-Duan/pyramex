# PyRamEx 示例代码

本目录包含PyRamEx的完整示例代码，展示各种功能和使用场景。

## 示例列表

### 示例1: 基础数据分析流程
**文件:** `ex1_basic_analysis.py`

**内容:**
- 创建模拟拉曼光谱数据
- 数据预处理（平滑、基线去除、归一化）
- 质量控制
- PCA降维分析
- 可视化结果

**运行:**
```bash
python examples/ex1_basic_analysis.py
```

**输出:**
- `ex1_basic_analysis.png` - 包含4个子图的综合分析图

---

### 示例2: 机器学习分类工作流
**文件:** `ex2_ml_classification.py`

**内容:**
- 创建多类别光谱数据
- 预处理和数据集划分
- 训练随机森林分类器
- 交叉验证和模型评估
- 特征重要性分析
- 混淆矩阵可视化

**运行:**
```bash
python examples/ex2_ml_classification.py
```

**输出:**
- `ex2_ml_classification.png` - 分类结果可视化
- `rf_model.pkl` - 训练好的模型
- `wavenumbers.pkl` - 波数数据

---

### 示例3: 质量控制和异常检测
**文件:** `ex3_quality_control.py`

**内容:**
- 创建包含异常值的数据集
- 应用多种QC方法（距离法、SNR、ICOD）
- 检测性能评估
- 异常类型分析
- 可视化对比

**运行:**
```bash
python examples/ex3_quality_control.py
```

**输出:**
- `ex3_quality_control.png` - QC结果可视化
- `bad_samples_report.csv` - 异常样本报告

---

### 示例4: 降维和可视化比较
**文件:** `ex4_dimensionality_reduction.py`

**内容:**
- 创建多类别数据
- 比较PCA和PCoA降维方法
- 碎石图分析
- 累积方差分析
- 类别可分性评估（ANOVA）

**运行:**
```bash
python examples/ex4_dimensionality_reduction.py
```

**输出:**
- `ex4_dim_reduction_comparison.png` - 降维方法比较图

---

### 示例5: 批量处理工作流
**文件:** `ex5_batch_processing.py`

**内容:**
- 创建多个光谱文件
- 批量加载和处理
- 批量质量控制
- 批量特征提取
- 生成处理报告

**运行:**
```bash
python examples/ex5_batch_processing.py
```

**输出:**
- `ex5_batch_processing.png` - 批量处理结果
- `batch_processing_report.csv` - 详细报告
- `processed_good_spectra.csv` - 处理后的好样本
- `data/batch_data/` - 模拟数据文件

---

### 示例6: 验证对比测试 ✨
**文件:** `ex6_validation.py`

**内容:**
- 9项算法正确性验证
- 性能基准测试
- 数值稳定性验证
- 边界情况测试
- 结果可重现性验证

**运行:**
```bash
python examples/ex6_validation.py
```

**输出:**
- `validation_report.csv` - 验证结果汇总

**验证项目：**
- 预处理算法正确性
- 质量控制算法
- PCA降维正确性
- 方法链式调用
- 数据转换功能
- 边界情况处理
- 性能基准
- 数值稳定性
- 结果可重现性

---

### 示例7: 代码优化 ✨
**文件:** `ex7_optimization.py`

**内容:**
- 性能基准测试
- 算法复杂度分析
- 内存使用分析
- 向量化优化
- 批量处理优化

**运行:**
```bash
python examples/ex7_optimization.py
```

**输出:**
- `optimization_report.csv` - 优化建议汇总

**优化方向：**
- 预处理性能改进
- QC算法优化
- 降维算法优化
- 内存使用优化
- 代码质量改进

---

## 快速开始

### 运行所有示例

```bash
# 运行单个示例
python examples/ex1_basic_analysis.py

# 运行所有示例
for i in {1..5}; do
    python examples/ex$i*.py
done
```

### 使用Jupyter Notebook

如果您更喜欢交互式环境，可以参考 `tutorial.ipynb`。

---

## 示例数据结构

### data/batch_data/

批量处理示例创建的模拟数据：

```
data/batch_data/
├── spectrum_000.txt
├── spectrum_001.txt
├── ...
└── spectrum_019.txt
```

每个文件是双列格式（波数、强度）。

---

## 常见使用模式

### 模式1: 基本分析流程

```python
from pyramex import Ramanome

# 1. 加载数据
ramanome = load_spectra('data/')

# 2. 预处理
ramanome.smooth().remove_baseline().normalize()

# 3. 分析
ramanome.reduce(method='pca', n_components=2)
ramanome.plot()
```

### 模式2: 机器学习工作流

```python
from pyramex.ml import to_sklearn_format
from sklearn.ensemble import RandomForestClassifier

# 预处理
X_train, X_test, y_train, y_test = to_sklearn_format(ramanome)

# 训练
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### 模式3: 批量处理

```python
# 批量加载
ramanome = load_spectra('data_directory/')

# 批量预处理
ramanome.smooth().normalize()

# 批量QC
qc = ramanome.quality_control(method='dis')

# 过滤好样本
good_samples = ramanome.spectra[qc.good_samples]
```

---

## 扩展阅读

- [安装指南](../docs/installation.md)
- [快速开始教程](../docs/tutorial.md)
- [用户指南](../docs/user_guide.md)
- [API参考](../docs/api.md)

---

## 贡献示例

欢迎贡献更多示例！

### 提交示例的指南：

1. **清晰的主题** - 每个示例应该展示一个明确的功能
2. **可运行代码** - 确保代码可以直接运行
3. **详细注释** - 解释每个步骤的目的
4. **可视化结果** - 提供清晰的图表输出
5. **文档说明** - 在README中添加示例说明

### 示例模板：

```python
"""
示例X: [标题]

[简短描述]
"""

import numpy as np
from pyramex import Ramanome

print("=" * 60)
print("示例X: [标题]")
print("=" * 60)

# 1. 准备数据
print("\n步骤1: ...")

# 2. 处理
print("\n步骤2: ...")

# 3. 可视化
print("\n步骤3: ...")

print("\n" + "=" * 60)
print("示例X完成！")
print("=" * 60)
```

---

## 许可证

这些示例代码遵循项目的GPL许可证。

---

**最后更新:** 2026-02-15
**版本:** 0.1.0-beta
