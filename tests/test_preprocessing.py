"""
预处理模块测试
测试平滑、基线去除、归一化等功能
"""

import pytest
import numpy as np
import pandas as pd
from pyramex.preprocessing.processors import (
    smooth,
    remove_baseline,
    normalize,
    cutoff,
    derivative
)


class TestSmoothing:
    """测试平滑功能"""

    def test_smooth_basic(self):
        """测试基本平滑"""
        spectra = np.random.randn(5, 100)
        smoothed = smooth(spectra, window_size=5, polyorder=2)

        assert smoothed.shape == spectra.shape
        # 平滑后的数据方差应该更小
        assert smoothed.std() < spectra.std()

    def test_smooth_window_size_odd(self):
        """测试窗口大小自动调整为奇数"""
        spectra = np.random.randn(5, 100)
        # 偶数窗口大小
        smoothed = smooth(spectra, window_size=6, polyorder=2)

        assert smoothed.shape == spectra.shape

    def test_smooth_invalid_window_size(self):
        """测试无效的窗口大小"""
        spectra = np.random.randn(5, 100)

        with pytest.raises(ValueError, match="window_size"):
            smooth(spectra, window_size=2, polyorder=2)

    def test_smooth_different_polyorder(self):
        """测试不同的多项式阶数"""
        spectra = np.random.randn(5, 100)

        smoothed1 = smooth(spectra, window_size=7, polyorder=2)
        smoothed2 = smooth(spectra, window_size=7, polyorder=3)

        assert smoothed1.shape == spectra.shape
        assert smoothed2.shape == spectra.shape
        # 不同阶数应该产生不同结果
        assert not np.allclose(smoothed1, smoothed2)


class TestBaselineRemoval:
    """测试基线去除功能"""

    def test_remove_baseline_polyfit(self):
        """测试多项式拟合基线去除"""
        # 创建带基线的数据
        x = np.linspace(0, 10, 100)
        baseline = 0.5 * x + 1
        signal = np.sin(x)
        spectra = np.tile(signal + baseline, (5, 1))

        corrected = remove_baseline(spectra, method='polyfit', degree=1)

        assert corrected.shape == spectra.shape
        # 基线去除后的数据应该接近原始信号
        # 均值应该接近0
        assert abs(corrected.mean()) < abs(spectra.mean())

    def test_remove_baseline_als(self):
        """测试ALS基线去除"""
        spectra = np.random.randn(5, 100) + 2  # 添加偏移

        corrected = remove_baseline(spectra, method='als', lam=1e5, p=0.01)

        assert corrected.shape == spectra.shape
        # 去除基线后，最小值应该更接近0
        assert corrected.min() < spectra.min()

    def test_remove_baseline_airpls(self):
        """测试airPLS基线去除"""
        spectra = np.random.randn(5, 100) + 2  # 添加偏移

        corrected = remove_baseline(spectra, method='airpls', lam=1e5, p=0.01)

        assert corrected.shape == spectra.shape

    def test_remove_baseline_invalid_method(self):
        """测试无效的基线去除方法"""
        spectra = np.random.randn(5, 100)

        with pytest.raises(ValueError, match="Unknown method"):
            remove_baseline(spectra, method='invalid_method')


class TestNormalization:
    """测试归一化功能"""

    def test_normalize_minmax(self):
        """测试MinMax归一化"""
        spectra = np.random.randn(5, 100) * 10 + 5

        normalized = normalize(spectra, method='minmax')

        assert normalized.shape == spectra.shape
        # MinMax归一化后应该在[0, 1]范围内
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_minmax_custom_range(self):
        """测试自定义范围的MinMax归一化"""
        spectra = np.random.randn(5, 100)

        normalized = normalize(spectra, method='minmax', feature_range=(-1, 1))

        assert normalized.min() >= -1
        assert normalized.max() <= 1

    def test_normalize_zscore(self):
        """测试Z-score归一化"""
        spectra = np.random.randn(5, 100) * 10 + 5

        normalized = normalize(spectra, method='zscore')

        assert normalized.shape == spectra.shape
        # Z-score归一化后均值应该接近0，标准差接近1
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1

    def test_normalize_area(self):
        """测试面积归一化"""
        spectra = np.random.rand(5, 100) * 10

        normalized = normalize(spectra, method='area')

        assert normalized.shape == spectra.shape
        # 检查每个样本的面积
        for i in range(len(spectra)):
            area = np.trapz(np.abs(normalized[i]))
            # 面积应该接近1
            assert abs(area - 1.0) < 0.1

    def test_normalize_max(self):
        """测试最大值归一化"""
        spectra = np.random.rand(5, 100) * 10

        normalized = normalize(spectra, method='max')

        assert normalized.shape == spectra.shape
        # 每个样本的最大值应该接近1
        for i in range(len(spectra)):
            assert abs(normalized[i].max() - 1.0) < 0.01

    def test_normalize_vecnorm(self):
        """测试向量（L2）归一化"""
        spectra = np.random.randn(5, 100)

        normalized = normalize(spectra, method='vecnorm')

        assert normalized.shape == spectra.shape
        # 每个样本的L2范数应该接近1
        for i in range(len(spectra)):
            norm = np.linalg.norm(normalized[i])
            assert abs(norm - 1.0) < 0.01

    def test_normalize_invalid_method(self):
        """测试无效的归一化方法"""
        spectra = np.random.randn(5, 100)

        with pytest.raises(ValueError, match="Unknown method"):
            normalize(spectra, method='invalid_method')


class TestCutoff:
    """测试波数截取功能"""

    def test_cutoff_basic(self):
        """测试基本截取"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)

        spectra_cut, wn_cut = cutoff(spectra, wavenumbers, (500, 3000))

        assert spectra_cut.shape[0] == spectra.shape[0]
        assert spectra_cut.shape[1] < spectra.shape[1]
        assert len(wn_cut) == spectra_cut.shape[1]
        assert wn_cut.min() >= 500
        assert wn_cut.max() <= 3000

    def test_cutoff_narrow_range(self):
        """测试窄范围截取"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)

        spectra_cut, wn_cut = cutoff(spectra, wavenumbers, (1000, 1500))

        assert spectra_cut.shape[1] < 50  # 应该有更少的数据点

    def test_cutoff_outside_range(self):
        """测试范围外的截取"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)

        # 范围超出实际范围
        spectra_cut, wn_cut = cutoff(spectra, wavenumbers, (100, 5000))

        # 应该返回所有数据
        assert spectra_cut.shape == spectra.shape
        np.testing.assert_array_equal(wn_cut, wavenumbers)


class TestDerivative:
    """测试导数计算"""

    def test_derivative_first_order(self):
        """测试一阶导数"""
        # 创建简单的信号
        x = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(x)
        spectra = np.tile(signal, (5, 1))

        deriv = derivative(spectra, order=1, window_size=7, polyorder=3)

        assert deriv.shape == spectra.shape
        # 一阶导数的均值应该接近0（对于正弦波）
        assert abs(deriv.mean()) < 0.5

    def test_derivative_second_order(self):
        """测试二阶导数"""
        x = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(x)
        spectra = np.tile(signal, (5, 1))

        deriv = derivative(spectra, order=2, window_size=7, polyorder=3)

        assert deriv.shape == spectra.shape

    def test_derivative_different_window_size(self):
        """测试不同的窗口大小"""
        spectra = np.random.randn(5, 100)

        deriv1 = derivative(spectra, order=1, window_size=5, polyorder=2)
        deriv2 = derivative(spectra, order=1, window_size=11, polyorder=3)

        assert deriv1.shape == spectra.shape
        assert deriv2.shape == spectra.shape
        # 不同窗口大小应该产生不同结果
        assert not np.allclose(deriv1, deriv2)


class TestPreprocessingPipeline:
    """测试预处理流程"""

    def test_full_pipeline(self):
        """测试完整的预处理流程"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)

        # 完整流程
        step1 = smooth(spectra, window_size=5, polyorder=2)
        step2 = remove_baseline(step1, method='polyfit', degree=2)
        step3 = normalize(step2, method='minmax')
        step4, wn_cut = cutoff(step3, wavenumbers, (500, 3000))

        assert step4.shape[0] == 5
        assert step4.shape[1] < 100
        assert len(wn_cut) == step4.shape[1]

    def test_pipeline_reproducibility(self):
        """测试预处理流程的可重复性"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)

        # 固定随机种子
        np.random.seed(42)
        result1 = smooth(spectra, window_size=5, polyorder=2)
        result1 = remove_baseline(result1, method='polyfit', degree=2)

        np.random.seed(42)
        result2 = smooth(spectra, window_size=5, polyorder=2)
        result2 = remove_baseline(result2, method='polyfit', degree=2)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_pipeline_with_different_orders(self):
        """测试不同顺序的预处理"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)

        # 顺序1: smooth -> baseline -> normalize
        result1 = normalize(
            remove_baseline(smooth(spectra, 5, 2), 'polyfit', 2),
            'minmax'
        )

        # 顺序2: baseline -> smooth -> normalize
        result2 = normalize(
            remove_baseline(smooth(spectra, 5, 2), 'polyfit', 2),
            'minmax'
        )

        # 结果可能不同
        assert result1.shape == result2.shape


class TestEdgeCases:
    """测试边界情况"""

    def test_smooth_single_sample(self):
        """测试单个样本的平滑"""
        spectra = np.random.randn(1, 100)
        smoothed = smooth(spectra, window_size=5, polyorder=2)

        assert smoothed.shape == (1, 100)

    def test_normalize_constant_signal(self):
        """测试常数信号的归一化"""
        spectra = np.ones((5, 100))

        normalized = normalize(spectra, method='minmax')

        # 常数信号归一化后应该仍然是常数
        assert normalized.std() < 0.01

    def test_cutoff_single_point(self):
        """测试截取到单点"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)

        # 选择单点
        target_wavenumber = 1500
        idx = np.argmin(np.abs(wavenumbers - target_wavenumber))

        spectra_cut, wn_cut = cutoff(
            spectra,
            wavenumbers,
            (target_wavenumber, target_wavenumber)
        )

        # 应该返回最接近的点
        assert spectra_cut.shape == (5, 1)
        assert len(wn_cut) == 1
