# PyRamEx API参考

完整的PyRamEx API文档。

## 核心类

### Ramanome

PyRamEx的核心数据结构，用于封装和管理拉曼光谱数据。

#### 初始化

```python
from pyramex import Ramanome

ramanome = Ramanome(
    spectra: np.ndarray,      # 光谱矩阵 (n_samples, n_wavenumbers)
    wavenumbers: np.ndarray,  # 波数数组 (n_wavenumbers,)
    metadata: pd.DataFrame    # 元数据 (n_samples, n_metadata)
)
```

**参数：**
- `spectra`: 光谱数据矩阵，形状为(n_samples, n_wavenumbers)
- `wavenumbers`: 波数轴数组
- `metadata`: 样本元数据DataFrame

**示例：**
```python
import numpy as np
import pandas as pd

spectra = np.random.randn(10, 100)
wavenumbers = np.linspace(400, 4000, 100)
metadata = pd.DataFrame({'id': range(10)})

ramanome = Ramanome(spectra, wavenumbers, metadata)
```

#### 属性

- `n_samples`: 样本数量
- `n_wavenumbers`: 波数点数量
- `shape`: (n_samples, n_wavenumbers)
- `spectra`: 光谱数据矩阵
- `wavenumbers`: 波数数组
- `metadata`: 元数据
- `processed`: 预处理步骤列表
- `quality`: 质量控制结果字典
- `reductions`: 降维结果字典

#### 方法

##### copy()

创建深拷贝。

```python
ramanome2 = ramanome.copy()
```

##### reset()

重置到原始数据。

```python
ramanome.reset()
```

##### smooth()

应用Savitzky-Golay平滑。

```python
ramanome.smooth(window_size: int = 5, polyorder: int = 2) -> Ramanome
```

**参数：**
- `window_size`: 窗口大小（必须为奇数）
- `polyorder`: 多项式阶数

**返回：** self（支持链式调用）

##### remove_baseline()

去除基线。

```python
ramanome.remove_baseline(method: str = 'polyfit', **kwargs) -> Ramanome
```

**参数：**
- `method`: 基线去除方法
  - `'polyfit'`: 多项式拟合
  - `'als'`: 不对称最小二乘法
  - `'airpls'`: 自适应迭代重加权惩罚最小二乘法
- `**kwargs`: 方法特定参数

**返回：** self

##### normalize()

归一化光谱。

```python
ramanome.normalize(method: str = 'minmax', **kwargs) -> Ramanome
```

**参数：**
- `method`: 归一化方法
  - `'minmax'`: Min-Max归一化
  - `'zscore'`: Z-score归一化
  - `'area'`: 面积归一化
  - `'max'`: 最大值归一化
  - `'vecnorm'`: 向量归一化
- `**kwargs`: 方法特定参数

**返回：** self

##### cutoff()

截取波数范围。

```python
ramanome.cutoff(wavenumber_range: Tuple[float, float]) -> Ramanome
```

**参数：**
- `wavenumber_range`: (min_wavenumber, max_wavenumber)

**返回：** self

##### quality_control()

执行质量控制。

```python
qc_result = ramanome.quality_control(method: str = 'icod', **kwargs) -> QualityResult
```

**参数：**
- `method`: QC方法
  - `'icod'`: 逆协方差异常检测
  - `'mcd'`: 最小协方差行列式
  - `'t2'`: Hotelling's T²检验
  - `'snr'`: 信噪比
  - `'dis'`: 距离异常检测
- `**kwargs`: 方法特定参数

**返回：** QualityResult对象

##### reduce()

降维。

```python
ramanome.reduce(method: str = 'pca', n_components: int = 2, **kwargs) -> Ramanome
```

**参数：**
- `method`: 降维方法
  - `'pca'`: 主成分分析
  - `'umap'`: UMAP
  - `'tsne'`: t-SNE
  - `'pcoa'`: 主坐标分析
- `n_components`: 组件数量
- `**kwargs`: 方法特定参数

**返回：** self

##### to_ml_format()

转换为机器学习格式。

```python
X = ramanome.to_ml_format(return_metadata: bool = False)
X, metadata = ramanome.to_ml_format(return_metadata: bool = True)
```

**参数：**
- `return_metadata`: 是否返回元数据

**返回：** X 或 (X, metadata)

##### to_tensor()

转换为深度学习张量格式。

```python
tensor = ramanome.to_tensor(add_channel: bool = True)
```

**参数：**
- `add_channel`: 是否添加通道维度

**返回：** tensor数组

##### plot()

绘制光谱。

```python
ramanome.plot(samples: Optional[List[int]] = None)
fig = ramanome.plot(return_fig=True)
```

**参数：**
- `samples`: 要绘制的样本索引列表
- `return_fig`: 是否返回图形对象

##### plot_reduction()

绘制降维结果。

```python
ramanome.plot_reduction(method: str = 'pca', color_by: Optional[str] = None)
```

### QualityResult

质量控制结果类。

#### 属性

- `good_samples`: 布尔数组，标记好样本
- `quality_scores`: 质量分数数组
- `method`: 使用的QC方法
- `threshold`: 应用的阈值
- `params`: 参数字典
- `n_good`: 好样本数量
- `n_bad`: 坏样本数量
- `good_rate`: 好样本比例

## 预处理函数

### smooth()

Savitzky-Golay平滑。

```python
from pyramex.preprocessing import smooth

smoothed = smooth(
    spectra: np.ndarray,
    window_size: int = 5,
    polyorder: int = 2,
    axis: int = 1
) -> np.ndarray
```

### remove_baseline()

基线去除。

```python
from pyramex.preprocessing import remove_baseline

corrected = remove_baseline(
    spectra: np.ndarray,
    method: str = 'polyfit',
    degree: int = 3,
    **kwargs
) -> np.ndarray
```

### normalize()

归一化。

```python
from pyramex.preprocessing import normalize

normalized = normalize(
    spectra: np.ndarray,
    method: str = 'minmax',
    **kwargs
) -> np.ndarray
```

### cutoff()

截取波数范围。

```python
from pyramex.preprocessing import cutoff

spectra_cut, wavenumbers_cut = cutoff(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    wavenumber_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]
```

### derivative()

计算导数。

```python
from pyramex.preprocessing import derivative

deriv = derivative(
    spectra: np.ndarray,
    order: int = 1,
    window_size: int = 5,
    polyorder: int = 2
) -> np.ndarray
```

## 质量控制函数

### quality_control()

主质量控制函数。

```python
from pyramex.qc import quality_control

qc_result = quality_control(
    ramanome: Ramanome,
    method: str = 'icod',
    **kwargs
) -> QualityResult
```

## 降维函数

### reduce()

主降维函数。

```python
from pyramex.features import reduce

result = reduce(
    spectra: np.ndarray,
    method: str = 'pca',
    n_components: int = 2,
    **kwargs
) -> Dict
```

**返回字典包含：**
- `transformed`: 降维后的数据
- `method`: 使用的方法
- `explained_variance`: 解释方差（PCA）
- `model`: 拟合的模型
- `n_components`: 组件数量

### extract_band_intensity()

提取波段强度。

```python
from pyramex.features import extract_band_intensity

intensities = extract_band_intensity(
    ramanome: Ramanome,
    bands: List  # [(min1, max1), (min2, max2), ...] 或 [wn1, wn2, ...]
) -> np.ndarray
```

### calculate_cdr()

计算CDR（胞质比）。

```python
from pyramex.features import calculate_cdr

cdr = calculate_cdr(
    ramanome: Ramanome,
    band1: Tuple[float, float],
    band2: Tuple[float, float]
) -> np.ndarray
```

## 可视化函数

### plot_spectra()

绘制光谱。

```python
from pyramex.visualization import plot_spectra

fig = plot_spectra(
    ramanome: Ramanome,
    samples: Optional[List[int]] = None,
    n_samples: int = 10,
    figsize: Tuple[int, int] = (12, 6),
    return_fig: bool = False
)
```

### plot_reduction()

绘制降维结果。

```python
from pyramex.visualization import plot_reduction

fig = plot_reduction(
    ramanome: Ramanome,
    method: str = 'pca',
    color_by: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    return_fig: bool = False
)
```

### plot_quality_control()

绘制QC结果。

```python
from pyramex.visualization import plot_quality_control

fig = plot_quality_control(
    ramanome: Ramanome,
    method: str = 'icod',
    figsize: Tuple[int, int] = (12, 5),
    return_fig: bool = False
)
```

## 数据加载函数

### load_spectra()

加载数据。

```python
from pyramex import load_spectra

ramanome = load_spectra(
    path: Union[str, Path],
    format: str = 'auto',
    **kwargs
) -> Ramanome
```

**支持的格式：**
- `'auto'`: 自动检测
- `'two_col'`: 双列格式（波数，强度）
- `'matrix'`: 矩阵格式（第一列是波数）
- `'coords'`: 坐标扫描格式

## ML集成函数

### to_sklearn_format()

转换为scikit-learn格式。

```python
from pyramex.ml import to_sklearn_format

X_train, X_test, y_train, y_test = to_sklearn_format(
    ramanome: Ramanome,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple
```

### to_torch_dataset()

转换为PyTorch数据集。

```python
from pyramex.ml import to_torch_dataset

dataset = to_torch_dataset(
    ramanome: Ramanome,
    transform: Optional[Callable] = None
)
```

### to_tf_dataset()

转换为TensorFlow数据集。

```python
from pyramex.ml import to_tf_dataset

dataset = to_tf_dataset(
    ramanome: Ramanome,
    batch_size: int = 32,
    shuffle: bool = True
)
```

### create_dataloader()

创建数据加载器。

```python
from pyramex.ml import create_dataloader

data = create_dataloader(
    ramanome: Ramanome,
    framework: str = 'sklearn',
    **kwargs
)
```

### create_cnn_model()

创建CNN模型（PyTorch）。

```python
from pyramex.ml import create_cnn_model

model = create_cnn_model(
    input_length: int,
    n_classes: Optional[int] = None,
    dropout: float = 0.2
)
```

### create_mlp_model()

创建MLP模型（scikit-learn）。

```python
from pyramex.ml import create_mlp_model

model = create_mlp_model(
    input_length: int,
    n_classes: Optional[int] = None,
    hidden_dims: List[int] = [256, 128, 64]
)
```

## 异常

### ValueError

无效的参数或输入数据。

### ImportError

缺少可选依赖。

---

**版本：** 0.1.0-beta
**最后更新：** 2026-02-15
