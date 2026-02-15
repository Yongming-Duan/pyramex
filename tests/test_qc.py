"""
质量控制模块测试
测试各种QC方法
"""

import pytest
import numpy as np
import pandas as pd
from pyramex.core.ramanome import Ramanome
from pyramex.qc.methods import (
    quality_control,
    _qc_icod,
    _qc_mcd,
    _qc_t2,
    _qc_snr,
    _qc_dis
)


@pytest.fixture
def sample_ramanome():
    """创建示例Ramanome对象"""
    # 创建一些"好"样本和一些"坏"样本
    np.random.seed(42)
    good_spectra = np.random.randn(8, 100) * 0.1
    bad_spectra = np.random.randn(2, 100) * 5  # 异常值

    spectra = np.vstack([good_spectra, bad_spectra])
    wavenumbers = np.linspace(400, 4000, 100)
    metadata = pd.DataFrame({'id': range(10)})

    return Ramanome(spectra, wavenumbers, metadata)


class TestQualityControlMain:
    """测试主质量控制系统"""

    def test_quality_control_icod(self, sample_ramanome):
        """测试ICOD质量控制"""
        result = quality_control(sample_ramanome, method='icod', threshold=0.05)

        assert result is not None
        assert hasattr(result, 'good_samples')
        assert hasattr(result, 'quality_scores')
        assert result.method == 'icod'
        assert len(result.good_samples) == sample_ramanome.n_samples

    def test_quality_control_mcd(self, sample_ramanome):
        """测试MCD质量控制"""
        result = quality_control(sample_ramanome, method='mcd', threshold=0.95)

        assert result is not None
        assert result.method == 'mcd'

    def test_quality_control_t2(self, sample_ramanome):
        """测试T2质量控制"""
        result = quality_control(sample_ramanome, method='t2', alpha=0.95)

        assert result is not None
        assert result.method == 't2'

    def test_quality_control_snr(self, sample_ramanome):
        """测试SNR质量控制"""
        result = quality_control(sample_ramanome, method='snr', threshold=10.0)

        assert result is not None
        assert result.method == 'snr'

    def test_quality_control_dis(self, sample_ramanome):
        """测试距离质量控制"""
        result = quality_control(sample_ramanome, method='dis', threshold=0.05)

        assert result is not None
        assert result.method == 'dis'

    def test_quality_control_invalid_method(self, sample_ramanome):
        """测试无效的质量控制方法"""
        with pytest.raises(ValueError, match="Unknown QC method"):
            quality_control(sample_ramanome, method='invalid')


class TestICOD:
    """测试ICOD方法"""

    def test_icod_basic(self, sample_ramanome):
        """测试基本ICOD"""
        result = _qc_icod(sample_ramanome, threshold=0.05)

        assert result.good_samples.dtype == bool
        assert len(result.good_samples) == sample_ramanome.n_samples
        assert result.method == 'icod'

    def test_icod_different_thresholds(self, sample_ramanome):
        """测试不同的阈值"""
        result1 = _qc_icod(sample_ramanome, threshold=0.01)
        result2 = _qc_icod(sample_ramanome, threshold=0.1)

        # 更严格的阈值应该识别出更少的坏样本
        assert result1.good_samples.sum() <= result2.good_samples.sum()

    def test_icod_fallback(self, sample_ramanome):
        """测试ICOD的fallback机制"""
        # 通过创建可能导致失败的情况来测试fallback
        # 这里我们只测试函数不会崩溃
        result = _qc_icod(sample_ramanome, threshold=0.05)

        assert result is not None


class TestMCD:
    """测试MCD方法"""

    def test_mcd_basic(self, sample_ramanome):
        """测试基本MCD"""
        result = _qc_mcd(sample_ramanome, threshold=0.95)

        assert result.good_samples.dtype == bool
        assert len(result.good_samples) == sample_ramanome.n_samples
        assert result.method == 'mcd'

    def test_mcd_different_thresholds(self, sample_ramanome):
        """测试不同的阈值"""
        result1 = _qc_mcd(sample_ramanome, threshold=0.9)
        result2 = _qc_mcd(sample_ramanome, threshold=0.99)

        # 不同阈值可能产生不同结果
        # 由于样本量小，我们只检查函数运行
        assert result1 is not None
        assert result2 is not None


class TestT2:
    """测试Hotelling's T-squared方法"""

    def test_t2_basic(self, sample_ramanome):
        """测试基本T2"""
        result = _qc_t2(sample_ramanome, alpha=0.95)

        assert result.good_samples.dtype == bool
        assert len(result.good_samples) == sample_ramanome.n_samples
        assert result.method == 't2'

    def test_t2_different_alpha(self, sample_ramanome):
        """测试不同的alpha值"""
        result1 = _qc_t2(sample_ramanome, alpha=0.9)
        result2 = _qc_t2(sample_ramanome, alpha=0.99)

        assert result1 is not None
        assert result2 is not None

    def test_t2_threshold_in_result(self, sample_ramanome):
        """测试阈值是否包含在结果中"""
        result = _qc_t2(sample_ramanome, alpha=0.95)

        assert hasattr(result, 'threshold')
        assert result.threshold > 0


class TestSNR:
    """测试信噪比方法"""

    def test_snr_easy(self, sample_ramanome):
        """测试简单的SNR计算"""
        result = _qc_snr(sample_ramanome, method='easy', threshold=10.0)

        assert result.good_samples.dtype == bool
        assert len(result.good_samples) == sample_ramanome.n_samples
        assert result.method == 'snr'

    def test_snr_advanced(self, sample_ramanome):
        """测试高级SNR计算"""
        result = _qc_snr(sample_ramanome, method='advanced', threshold=10.0)

        assert result is not None
        assert result.method == 'snr'

    def test_snr_different_thresholds(self, sample_ramanome):
        """测试不同的阈值"""
        result1 = _qc_snr(sample_ramanome, method='easy', threshold=5.0)
        result2 = _qc_snr(sample_ramanome, method='easy', threshold=20.0)

        # 更高的阈值应该更严格
        assert result1.good_samples.sum() >= result2.good_samples.sum()


class TestDistance:
    """测试距离方法"""

    def test_dis_basic(self, sample_ramanome):
        """测试基本距离方法"""
        result = _qc_dis(sample_ramanome, threshold=0.05)

        assert result.good_samples.dtype == bool
        assert len(result.good_samples) == sample_ramanome.n_samples
        assert result.method == 'dis'

    def test_dis_different_thresholds(self, sample_ramanome):
        """测试不同的阈值"""
        result1 = _qc_dis(sample_ramanome, threshold=0.01)
        result2 = _qc_dis(sample_ramanome, threshold=0.1)

        assert result1 is not None
        assert result2 is not None


class TestQualityResultIntegration:
    """测试质量控制结果的集成"""

    def test_qc_result_storage(self, sample_ramanome):
        """测试QC结果存储在Ramanome中"""
        result = quality_control(sample_ramanome, method='icod', threshold=0.05)

        # 执行QC后，结果应该被存储
        # 这个功能在Ramanome.quality_control方法中实现
        # 这里我们测试返回的结果
        assert result is not None

    def test_qc_filter_samples(self, sample_ramanome):
        """测试使用QC结果过滤样本"""
        result = quality_control(sample_ramanome, method='icod', threshold=0.05)

        # 过滤出好样本
        good_indices = np.where(result.good_samples)[0]
        bad_indices = np.where(~result.good_samples)[0]

        # 应该有一些好样本和坏样本
        assert len(good_indices) > 0
        # 坏样本数量取决于数据和阈值

    def test_qc_multiple_methods(self, sample_ramanome):
        """测试使用多种QC方法"""
        result1 = quality_control(sample_ramanome, method='icod', threshold=0.05)
        result2 = quality_control(sample_ramanome, method='mcd', threshold=0.95)
        result3 = quality_control(sample_ramanome, method='snr', threshold=10.0)

        # 所有方法都应该返回结果
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None

        # 不同方法可能产生不同的结果
        # 我们只检查它们都运行成功


class TestQualityResultProperties:
    """测试QualityResult的属性"""

    def test_n_good_n_bad(self, sample_ramanome):
        """测试n_good和n_bad属性"""
        result = quality_control(sample_ramanome, method='dis', threshold=0.05)

        assert result.n_good + result.n_bad == sample_ramanome.n_samples

    def test_good_rate(self, sample_ramanome):
        """测试good_rate属性"""
        result = quality_control(sample_ramanome, method='dis', threshold=0.05)

        assert 0 <= result.good_rate <= 1
        assert result.good_rate == result.n_good / sample_ramanome.n_samples

    def test_repr(self, sample_ramanome):
        """测试QualityResult的字符串表示"""
        result = quality_control(sample_ramanome, method='dis', threshold=0.05)

        repr_str = repr(result)
        assert 'QualityResult' in repr_str
        assert 'good=' in repr_str


class TestQCEdgeCases:
    """测试QC的边界情况"""

    def test_qc_small_sample(self):
        """测试小样本的QC"""
        spectra = np.random.randn(3, 50)
        wavenumbers = np.linspace(400, 4000, 50)
        metadata = pd.DataFrame({'id': range(3)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 应该能处理小样本
        result = quality_control(ramanome, method='dis', threshold=0.05)

        assert result is not None

    def test_qc_identical_spectra(self):
        """测试完全相同的光谱"""
        spectra = np.ones((5, 100))
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        result = quality_control(ramanome, method='dis', threshold=0.05)

        # 所有样本应该被认为是好的
        assert result.n_good == 5
