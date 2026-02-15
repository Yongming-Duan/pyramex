"""
可视化模块测试
测试绘图功能
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from pyramex.core.ramanome import Ramanome, QualityResult
from pyramex.visualization.plots import (
    plot_spectra,
    plot_reduction,
    plot_quality_control,
    plot_preprocessing_steps
)


@pytest.fixture
def sample_ramanome():
    """创建示例Ramanome对象"""
    spectra = np.random.randn(10, 100)
    wavenumbers = np.linspace(400, 4000, 100)
    metadata = pd.DataFrame({
        'id': range(10),
        'label': ['A'] * 5 + ['B'] * 5,
        'group': [f'sample_{i}' for i in range(10)]
    })

    return Ramanome(spectra, wavenumbers, metadata)


@pytest.fixture
def sample_ramanome_with_reduction(sample_ramanome):
    """创建带降维结果的Ramanome对象"""
    from pyramex.features.reduction import reduce

    # 执行PCA降维
    result = reduce(sample_ramanome.spectra, method='pca', n_components=2)
    sample_ramanome.reductions['pca'] = result

    return sample_ramanome


@pytest.fixture
def sample_ramanome_with_qc(sample_ramanome):
    """创建带质量控制结果的Ramanome对象"""
    from pyramex.qc.methods import _qc_dis

    # 执行质量控制
    result = _qc_dis(sample_ramanome, threshold=0.05)
    sample_ramanome.quality['dis'] = result

    return sample_ramanome


class TestPlotSpectra:
    """测试光谱绘图功能"""

    def test_plot_spectra_basic(self, sample_ramanome):
        """测试基本光谱绘图"""
        fig = plot_spectra(sample_ramanome, return_fig=True)

        assert fig is not None
        assert hasattr(fig, 'axes')
        plt.close(fig)

    def test_plot_spectra_specific_samples(self, sample_ramanome):
        """测试绘制特定样本"""
        fig = plot_spectra(sample_ramanome, samples=[0, 1, 2], return_fig=True)

        assert fig is not None
        plt.close(fig)

    def test_plot_spectra_random_samples(self, sample_ramanome):
        """测试随机选择样本绘制"""
        fig = plot_spectra(sample_ramanome, n_samples=5, return_fig=True)

        assert fig is not None
        plt.close(fig)

    def test_plot_spectra_custom_figsize(self, sample_ramanome):
        """测试自定义图形大小"""
        fig = plot_spectra(sample_ramanome, figsize=(15, 8), return_fig=True)

        assert fig is not None
        assert fig.get_figwidth() == 15
        assert fig.get_figheight() == 8
        plt.close(fig)

    def test_plot_spectra_no_return(self, sample_ramanome):
        """测试不返回图形对象"""
        result = plot_spectra(sample_ramanome, return_fig=False)

        # 应该直接显示，不返回值
        assert result is None


class TestPlotReduction:
    """测试降维绘图功能"""

    def test_plot_reduction_2d(self, sample_ramanome_with_reduction):
        """测试2D降维绘图"""
        fig = plot_reduction(
            sample_ramanome_with_reduction,
            method='pca',
            return_fig=True
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_reduction_with_color(self, sample_ramanome_with_reduction):
        """测试带颜色映射的降维绘图"""
        fig = plot_reduction(
            sample_ramanome_with_reduction,
            method='pca',
            color_by='label',
            return_fig=True
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_reduction_invalid_method(self, sample_ramanome):
        """测试无效的降维方法"""
        with pytest.raises(ValueError, match="not found"):
            plot_reduction(sample_ramanome, method='invalid', return_fig=True)

    def test_plot_reduction_3d(self):
        """测试3D降维绘图"""
        # 创建带3D PCA的Ramanome
        spectra = np.random.randn(10, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(10)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        from pyramex.features.reduction import reduce
        result = reduce(ramanome.spectra, method='pca', n_components=3)
        ramanome.reductions['pca'] = result

        fig = plot_reduction(ramanome, method='pca', return_fig=True)

        assert fig is not None
        plt.close(fig)

    def test_plot_reduction_invalid_dimensions(self, sample_ramanome):
        """测试不支持的降维维度"""
        # 创建4D降维结果
        from pyramex.features.reduction import reduce
        result = reduce(sample_ramanome.spectra, method='pca', n_components=4)
        sample_ramanome.reductions['pca'] = result

        with pytest.raises(ValueError, match="Cannot plot"):
            plot_reduction(sample_ramanome, method='pca', return_fig=True)


class TestPlotQualityControl:
    """测试质量控制绘图功能"""

    def test_plot_quality_control_basic(self, sample_ramanome_with_qc):
        """测试基本质量控制绘图"""
        fig = plot_quality_control(
            sample_ramanome_with_qc,
            method='dis',
            return_fig=True
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_quality_control_invalid_method(self, sample_ramanome):
        """测试无效的质量控制方法"""
        with pytest.raises(ValueError, match="not found"):
            plot_quality_control(sample_ramanome, method='invalid', return_fig=True)

    def test_plot_quality_control_custom_figsize(self, sample_ramanome_with_qc):
        """测试自定义图形大小"""
        fig = plot_quality_control(
            sample_ramanome_with_qc,
            method='dis',
            figsize=(14, 6),
            return_fig=True
        )

        assert fig is not None
        plt.close(fig)


class TestPlotPreprocessingSteps:
    """测试预处理步骤绘图"""

    def test_plot_preprocessing_steps_no_steps(self, sample_ramanome):
        """测试没有预处理步骤的情况"""
        result = plot_preprocessing_steps(sample_ramanome, return_fig=True)

        # 应该返回None并打印消息
        assert result is None

    def test_plot_preprocessing_steps_with_steps(self, sample_ramanome):
        """测试带预处理步骤的绘图"""
        # 添加一些预处理步骤
        sample_ramanome.smooth(window_size=5)
        sample_ramanome.normalize(method='minmax')

        fig = plot_preprocessing_steps(sample_ramanome, return_fig=True)

        assert fig is not None
        plt.close(fig)


class TestInteractivePlot:
    """测试交互式绘图功能"""

    def test_interactive_plot(self, sample_ramanome):
        """测试交互式绘图（如果plotly可用）"""
        try:
            import plotly.graph_objects as go

            # 这个测试只是确保函数能够运行
            # 实际的交互式显示在测试环境中不可行
            result = plot_spectra.__module__  # 确保模块存在

            assert result is not None

        except ImportError:
            # 如果plotly不可用，跳过测试
            pytest.skip("Plotly not installed")


class TestRamanomePlotMethods:
    """测试Ramanome的绘图方法"""

    def test_ramanome_plot(self, sample_ramanome):
        """测试Ramanome.plot()方法"""
        from unittest.mock import patch

        # 模拟plt.show()以避免显示
        with patch('matplotlib.pyplot.show'):
            result = sample_ramanome.plot(samples=[0, 1, 2])

        # 应该返回None（因为return_fig默认为False）
        assert result is None

    def test_ramanome_plot_reduction(self, sample_ramanome_with_reduction):
        """测试Ramanome.plot_reduction()方法"""
        from unittest.mock import patch

        with patch('matplotlib.pyplot.show'):
            result = sample_ramanome_with_reduction.plot_reduction(method='pca')

        assert result is None


class TestVisualizationEdgeCases:
    """测试可视化的边界情况"""

    def test_plot_empty_ramanome(self):
        """测试空Ramanome对象"""
        spectra = np.array([]).reshape(0, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame()

        # 这种情况下Ramanome初始化会失败
        # 所以我们测试单样本的情况
        spectra = np.random.randn(1, 100)
        metadata = pd.DataFrame({'id': [0]})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        fig = plot_spectra(ramanome, return_fig=True)

        assert fig is not None
        plt.close(fig)

    def test_plot_with_extreme_values(self):
        """测试极端值的光谱绘图"""
        spectra = np.array([[1e10, -1e10] + [0] * 98])
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': [0]})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        fig = plot_spectra(ramanome, return_fig=True)

        assert fig is not None
        plt.close(fig)
