"""
特征工程模块测试
测试降维和特征提取功能
"""

import pytest
import numpy as np
import pandas as pd
from pyramex.core.ramanome import Ramanome
from pyramex.features.reduction import (
    reduce,
    extract_band_intensity,
    calculate_cdr,
    _reduce_pca,
    _reduce_umap,
    _reduce_tsne,
    _reduce_pcoa
)


@pytest.fixture
def sample_ramanome():
    """创建示例Ramanome对象"""
    spectra = np.random.randn(10, 100)
    wavenumbers = np.linspace(400, 4000, 100)
    metadata = pd.DataFrame({
        'id': range(10),
        'label': ['A'] * 5 + ['B'] * 5
    })

    return Ramanome(spectra, wavenumbers, metadata)


class TestReduceMain:
    """测试主降维函数"""

    def test_reduce_pca(self, sample_ramanome):
        """测试PCA降维"""
        result = reduce(sample_ramanome.spectra, method='pca', n_components=2)

        assert result is not None
        assert 'transformed' in result
        assert result['transformed'].shape == (10, 2)
        assert result['method'] == 'pca'
        assert 'explained_variance' in result
        assert 'model' in result

    def test_reduce_pca_3d(self, sample_ramanome):
        """测试PCA降维到3D"""
        result = reduce(sample_ramanome.spectra, method='pca', n_components=3)

        assert result['transformed'].shape == (10, 3)
        assert result['n_components'] == 3

    def test_reduce_invalid_method(self, sample_ramanome):
        """测试无效的降维方法"""
        with pytest.raises(ValueError, match="Unknown method"):
            reduce(sample_ramanome.spectra, method='invalid', n_components=2)

    def test_reduce_umap(self, sample_ramanome):
        """测试UMAP降维"""
        try:
            result = reduce(sample_ramanome.spectra, method='umap', n_components=2)

            assert result is not None
            assert result['transformed'].shape == (10, 2)
            assert result['method'] == 'umap'

        except ImportError:
            pytest.skip("UMAP not installed")

    def test_reduce_tsne(self, sample_ramanome):
        """测试t-SNE降维"""
        result = reduce(sample_ramanome.spectra, method='tsne', n_components=2)

        assert result is not None
        assert result['transformed'].shape == (10, 2)
        assert result['method'] == 'tsne'

    def test_reduce_pcoa(self, sample_ramanome):
        """测试PCoA降维"""
        result = reduce(sample_ramanome.spectra, method='pcoa', n_components=2)

        assert result is not None
        assert result['transformed'].shape == (10, 2)
        assert result['method'] == 'pcoa'


class TestPCAReduction:
    """测试PCA降维详细功能"""

    def test_pca_explained_variance(self, sample_ramanome):
        """测试PCA解释方差"""
        result = reduce(sample_ramanome.spectra, method='pca', n_components=5)

        assert 'explained_variance' in result
        assert 'cumulative_variance' in result

        # 解释方差应该在[0, 1]范围内
        assert np.all(result['explained_variance'] >= 0)
        assert np.all(result['explained_variance'] <= 1)

        # 累积方差应该是递增的
        assert np.all(np.diff(result['cumulative_variance']) >= 0)

    def test_pca_scaler(self, sample_ramanome):
        """测试PCA的标准化器"""
        result = reduce(sample_ramanome.spectra, method='pca', n_components=2)

        assert 'scaler' in result
        assert 'components' in result

        # 组件应该有正确的形状
        assert result['components'].shape == (2, 100)

    def test_pca_predict(self, sample_ramanome):
        """测试使用PCA模型进行预测"""
        # 先训练
        result = reduce(sample_ramanome.spectra, method='pca', n_components=2)

        # 使用模型转换新数据
        model = result['model']
        scaler = result['scaler']

        new_data = sample_ramanome.spectra[:3]
        new_scaled = scaler.transform(new_data)
        new_transformed = model.transform(new_scaled)

        assert new_transformed.shape == (3, 2)


class TestUMAPReduction:
    """测试UMAP降维详细功能"""

    def test_umap_parameters(self, sample_ramanome):
        """测试UMAP参数"""
        try:
            result = reduce(
                sample_ramanome.spectra,
                method='umap',
                n_components=2,
                n_neighbors=10,
                min_dist=0.2
            )

            assert result is not None
            assert result['method'] == 'umap'

        except ImportError:
            pytest.skip("UMAP not installed")

    def test_umap_model(self, sample_ramanome):
        """测试UMAP模型"""
        try:
            result = reduce(sample_ramanome.spectra, method='umap', n_components=2)

            assert 'model' in result

        except ImportError:
            pytest.skip("UMAP not installed")


class TestTSNEReduction:
    """测试t-SNE降维详细功能"""

    def test_tsne_parameters(self, sample_ramanome):
        """测试t-SNE参数"""
        result = reduce(
            sample_ramanome.spectra,
            method='tsne',
            n_components=2,
            perplexity=20.0
        )

        assert result is not None
        assert result['method'] == 'tsne'

    def test_tsne_model(self, sample_ramanome):
        """测试t-SNE模型"""
        result = reduce(sample_ramanome.spectra, method='tsne', n_components=2)

        assert 'model' in result


class TestExtractBandIntensity:
    """测试波段强度提取"""

    def test_extract_single_band(self, sample_ramanome):
        """测试提取单个波段"""
        # 创建已知特征的光谱
        wavenumbers = sample_ramanome.wavenumbers
        spectra = sample_ramanome.spectra.copy()

        # 在特定位置添加峰
        idx_1000 = np.argmin(np.abs(wavenumbers - 1000))
        spectra[:, idx_1000] += 10

        intensities = extract_band_intensity(
            sample_ramanome,
            bands=[(950, 1050)]
        )

        assert intensities.shape == (10, 1)

    def test_extract_multiple_bands(self, sample_ramanome):
        """测试提取多个波段"""
        intensities = extract_band_intensity(
            sample_ramanome,
            bands=[(500, 600), (1000, 1100), (1500, 1600)]
        )

        assert intensities.shape == (10, 3)

    def test_extract_single_wavenumber(self, sample_ramanome):
        """测试提取单个波数点"""
        intensities = extract_band_intensity(
            sample_ramanome,
            bands=[1000.0]
        )

        assert intensities.shape == (10, 1)

    def test_extract_mixed_bands(self, sample_ramanome):
        """测试混合波段和波数点"""
        intensities = extract_band_intensity(
            sample_ramanome,
            bands=[(500, 600), 1000.0, (1500, 1600)]
        )

        assert intensities.shape == (10, 3)

    def test_extract_invalid_band(self, sample_ramanome):
        """测试无效的波段"""
        with pytest.raises(ValueError):
            extract_band_intensity(sample_ramanome, bands=["invalid"])


class TestCalculateCDR:
    """测试CDR计算"""

    def test_calculate_cdr_basic(self, sample_ramanome):
        """测试基本CDR计算"""
        cdr = calculate_cdr(
            sample_ramanome,
            band1=(500, 600),
            band2=(1000, 1100)
        )

        assert cdr.shape == (10,)
        assert np.all(cdr >= 0)
        assert np.all(cdr <= 1)

    def test_calculate_cdr_values(self, sample_ramanome):
        """测试CDR值的合理性"""
        # 修改光谱以创建已知模式
        spectra = sample_ramanome.spectra.copy()
        wavenumbers = sample_ramanome.wavenumbers

        # 在band1添加强峰
        idx1 = np.argmin(np.abs(wavenumbers - 550))
        spectra[:, idx1] += 10

        # 在band2添加弱峰
        idx2 = np.argmin(np.abs(wavenumbers - 1050))
        spectra[:, idx2] += 1

        sample_ramanome.spectra = spectra

        cdr = calculate_cdr(
            sample_ramanome,
            band1=(500, 600),
            band2=(1000, 1100)
        )

        # band1更强，CDR应该接近1
        assert cdr.mean() > 0.5


class TestRamanomeReductionIntegration:
    """测试Ramanome降维集成"""

    def test_ramanome_reduce(self, sample_ramanome):
        """测试Ramanome.reduce()方法"""
        result = sample_ramanome.reduce(method='pca', n_components=2)

        assert result is sample_ramanome  # 应该返回自身
        assert 'pca' in sample_ramanome.reductions

    def test_ramanome_multiple_reductions(self, sample_ramanome):
        """测试多次降维"""
        sample_ramanome.reduce(method='pca', n_components=2)
        sample_ramanome.reduce(method='pcoa', n_components=3)

        assert 'pca' in sample_ramanome.reductions
        assert 'pcoa' in sample_ramanome.reductions

    def test_ramanome_reduction_persistence(self, sample_ramanome):
        """测试降维结果持久化"""
        sample_ramanome.reduce(method='pca', n_components=2)

        # 复制对象，降维结果应该被复制
        ramanome2 = sample_ramanome.copy()

        assert 'pca' in ramanome2.reductions
        np.testing.assert_array_equal(
            sample_ramanome.reductions['pca']['transformed'],
            ramanome2.reductions['pca']['transformed']
        )


class TestFeatureEngineeringEdgeCases:
    """测试特征工程的边界情况"""

    def test_reduce_n_components_too_large(self, sample_ramanome):
        """测试n_components过大"""
        # n_components不能超过样本数
        result = reduce(
            sample_ramanome.spectra,
            method='pca',
            n_components=10  # 等于样本数
        )

        assert result is not None

    def test_reduce_single_sample(self):
        """测试单样本降维"""
        spectra = np.random.randn(1, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': [0]})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # PCA需要至少2个样本
        with pytest.raises(Exception):
            reduce(spectra, method='pca', n_components=1)

    def test_extract_band_outside_range(self, sample_ramanome):
        """测试提取超出范围的波段"""
        intensities = extract_band_intensity(
            sample_ramanome,
            bands=[(100, 200)]  # 超出波数范围
        )

        # 应该返回全零或很小的值
        assert intensities.shape == (10, 1)
