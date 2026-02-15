"""
核心模块测试
测试Ramanome核心类和关键功能
"""

import pytest
import numpy as np
import pandas as pd
from pyramex.core.ramanome import Ramanome, QualityResult


class TestRamanomeInit:
    """测试Ramanome初始化"""

    def test_init_basic(self, sample_spectra):
        """测试基本初始化"""
        # 从sample_spectra创建必要的输入
        wavelength = sample_spectra['wavelength'].values
        intensity = sample_spectra['intensity'].values.reshape(1, -1)
        metadata = pd.DataFrame({'sample': ['test_0']})

        ramanome = Ramanome(
            spectra=intensity,
            wavenumbers=wavelength,
            metadata=metadata
        )

        assert ramanome is not None
        assert ramanome.n_samples == 1
        assert ramanome.n_wavenumbers == len(wavelength)
        assert ramanome.shape == (1, len(wavelength))

    def test_init_multiple_samples(self, sample_dataset):
        """测试多样本初始化"""
        # 重组数据
        grouped = sample_dataset.groupby('label')
        spectra_list = []
        metadata_list = []

        for label, group in grouped:
            spectra_list.append(group['intensity'].values)
            metadata_list.append({'label': label})

        spectra = np.vstack(spectra_list)
        wavenumbers = sample_dataset['wavelength'].unique()

        metadata = pd.DataFrame(metadata_list)

        ramanome = Ramanome(
            spectra=spectra,
            wavenumbers=wavenumbers,
            metadata=metadata
        )

        assert ramanome.n_samples == len(spectra)
        assert ramanome.n_wavenumbers == len(wavenumbers)

    def test_init_validation_mismatch(self, sample_spectra):
        """测试验证：光谱和波数不匹配"""
        wavelength = sample_spectra['wavelength'].values
        intensity = sample_spectra['intensity'].values.reshape(1, -1)
        wrong_wavenumbers = wavelength[:-10]  # 少10个点

        metadata = pd.DataFrame({'sample': ['test_0']})

        with pytest.raises(ValueError, match="Wavenumber mismatch"):
            Ramanome(
                spectra=intensity,
                wavenumbers=wrong_wavenumbers,
                metadata=metadata
            )

    def test_init_validation_metadata_mismatch(self, sample_spectra):
        """测试验证：元数据和样本数不匹配"""
        wavelength = sample_spectra['wavelength'].values
        intensity = sample_spectra['intensity'].values.reshape(1, -1)

        # 创建2行元数据但只有1个样本
        metadata = pd.DataFrame({'sample': ['test_0', 'test_1']})

        with pytest.raises(ValueError, match="Metadata mismatch"):
            Ramanome(
                spectra=intensity,
                wavenumbers=wavelength,
                metadata=metadata
            )

    def test_init_sort_wavenumbers(self):
        """测试自动排序波数"""
        # 创建未排序的波数
        wavenumbers = np.array([1000, 500, 1500, 2000, 2500])
        spectra = np.random.randn(3, 5)
        metadata = pd.DataFrame({'sample': ['a', 'b', 'c']})

        ramanome = Ramanome(
            spectra=spectra,
            wavenumbers=wavenumbers,
            metadata=metadata
        )

        # 波数应该被排序
        assert np.all(np.diff(ramanome.wavenumbers) > 0)


class TestRamanomeProperties:
    """测试Ramanome属性"""

    def test_n_samples(self):
        """测试样本数属性"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)
        assert ramanome.n_samples == 5

    def test_n_wavenumbers(self):
        """测试波数点数属性"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)
        assert ramanome.n_wavenumbers == 100

    def test_shape(self):
        """测试形状属性"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)
        assert ramanome.shape == (5, 100)


class TestRamanomeMethods:
    """测试Ramanome方法"""

    def test_copy(self):
        """测试复制功能"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome1 = Ramanome(spectra, wavenumbers, metadata)
        ramanome2 = ramanome1.copy()

        assert ramanome1 is not ramanome2
        assert ramanome1.spectra is not ramanome2.spectra
        assert ramanome1.wavenumbers is not ramanome2.wavenumbers
        np.testing.assert_array_equal(ramanome1.spectra, ramanome2.spectra)

    def test_reset(self):
        """测试重置功能"""
        spectra = np.random.randn(5, 100).copy()
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)
        original_spectra = ramanome.spectra.copy()

        # 修改光谱
        ramanome.spectra *= 2
        ramanome.processed.append("test")

        # 重置
        ramanome.reset()

        np.testing.assert_array_equal(ramanome.spectra, original_spectra)
        assert len(ramanome.processed) == 0

    def test_to_ml_format(self):
        """测试转换为ML格式"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5), 'label': ['A', 'B', 'A', 'B', 'A']})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 不返回元数据
        X = ramanome.to_ml_format(return_metadata=False)
        assert X.shape == (5, 100)

        # 返回元数据
        X, meta = ramanome.to_ml_format(return_metadata=True)
        assert X.shape == (5, 100)
        assert isinstance(meta, pd.DataFrame)
        assert len(meta) == 5

    def test_to_tensor(self):
        """测试转换为张量格式"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 不添加通道维度
        tensor = ramanome.to_tensor(add_channel=False)
        assert tensor.shape == (5, 100)

        # 添加通道维度
        tensor = ramanome.to_tensor(add_channel=True)
        assert tensor.shape == (5, 1, 100)

    def test_repr(self):
        """测试字符串表示"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)
        repr_str = repr(ramanome)

        assert 'Ramanome' in repr_str
        assert 'n_samples=5' in repr_str
        assert 'n_wavenumbers=100' in repr_str


class TestRamanomePreprocessing:
    """测试预处理方法链"""

    def test_smooth(self):
        """测试平滑"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)
        result = ramanome.smooth(window_size=5, polyorder=2)

        # 应该返回自身（方法链）
        assert result is ramanome
        assert 'smooth' in ramanome.processed[-1]

    def test_normalize(self):
        """测试归一化"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)
        result = ramanome.normalize(method='minmax')

        assert result is ramanome
        # 检查是否归一化到[0, 1]
        assert ramanome.spectra.min() >= 0
        assert ramanome.spectra.max() <= 1

    def test_cutoff(self):
        """测试截取波数范围"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)
        original_n_wavenumbers = ramanome.n_wavenumbers

        result = ramanome.cutoff((500, 3000))

        assert result is ramanome
        assert ramanome.n_wavenumbers < original_n_wavenumbers
        assert ramanome.wavenumbers.min() >= 500
        assert ramanome.wavenumbers.max() <= 3000


class TestQualityResult:
    """测试QualityResult类"""

    def test_quality_result_properties(self):
        """测试QualityResult属性"""
        good_samples = np.array([True, True, False, True, False])
        quality_scores = np.array([0.9, 0.8, 0.1, 0.85, 0.2])

        result = QualityResult(
            good_samples=good_samples,
            quality_scores=quality_scores,
            method='test',
            threshold=0.5
        )

        assert result.n_good == 3
        assert result.n_bad == 2
        assert result.good_rate == 0.6

    def test_quality_result_repr(self):
        """测试QualityResult字符串表示"""
        good_samples = np.array([True, True, False, True, False])
        quality_scores = np.array([0.9, 0.8, 0.1, 0.85, 0.2])

        result = QualityResult(
            good_samples=good_samples,
            quality_scores=quality_scores,
            method='test',
            threshold=0.5
        )

        repr_str = repr(result)
        assert 'QualityResult' in repr_str
        assert 'good=3/5' in repr_str
        assert 'rate=60.00%' in repr_str


class TestRamanomeIntegration:
    """集成测试"""

    def test_method_chaining(self):
        """测试方法链式调用"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 链式调用
        result = (ramanome
                  .smooth(window_size=5, polyorder=2)
                  .normalize(method='minmax')
                  .cutoff((500, 3500)))

        assert result is ramanome
        assert len(ramanome.processed) == 3
