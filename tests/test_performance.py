"""
性能测试
测试计算性能和资源使用
"""

import pytest
import numpy as np
import pandas as pd
import time
from pyramex.core.ramanome import Ramanome
from pyramex.preprocessing.processors import smooth, normalize, remove_baseline
from pyramex.qc.methods import quality_control
from pyramex.features.reduction import reduce


@pytest.fixture
def small_dataset():
    """创建小数据集（10个样本，100个波数点）"""
    spectra = np.random.randn(10, 100)
    wavenumbers = np.linspace(400, 4000, 100)
    metadata = pd.DataFrame({'id': range(10)})
    return Ramanome(spectra, wavenumbers, metadata)


@pytest.fixture
def medium_dataset():
    """创建中等数据集（100个样本，500个波数点）"""
    spectra = np.random.randn(100, 500)
    wavenumbers = np.linspace(400, 4000, 500)
    metadata = pd.DataFrame({'id': range(100)})
    return Ramanome(spectra, wavenumbers, metadata)


@pytest.fixture
def large_dataset():
    """创建大数据集（1000个样本，1000个波数点）"""
    spectra = np.random.randn(1000, 1000)
    wavenumbers = np.linspace(400, 4000, 1000)
    metadata = pd.DataFrame({'id': range(1000)})
    return Ramanome(spectra, wavenumbers, metadata)


class TestPreprocessingPerformance:
    """测试预处理性能"""

    @pytest.mark.slow
    def test_smooth_performance_small(self, small_dataset):
        """测试小数据集的平滑性能"""
        start_time = time.time()

        smoothed = smooth(small_dataset.spectra, window_size=5, polyorder=2)

        elapsed = time.time() - start_time

        # 应该在100ms内完成
        assert elapsed < 0.1
        assert smoothed.shape == small_dataset.spectra.shape

    @pytest.mark.slow
    def test_smooth_performance_medium(self, medium_dataset):
        """测试中等数据集的平滑性能"""
        start_time = time.time()

        smoothed = smooth(medium_dataset.spectra, window_size=5, polyorder=2)

        elapsed = time.time() - start_time

        # 应该在1秒内完成
        assert elapsed < 1.0
        assert smoothed.shape == medium_dataset.spectra.shape

    @pytest.mark.slow
    def test_normalize_performance(self, medium_dataset):
        """测试归一化性能"""
        start_time = time.time()

        normalized = normalize(medium_dataset.spectra, method='minmax')

        elapsed = time.time() - start_time

        # 应该在100ms内完成
        assert elapsed < 0.1
        assert normalized.shape == medium_dataset.spectra.shape

    @pytest.mark.slow
    def test_baseline_removal_performance(self, medium_dataset):
        """测试基线去除性能"""
        start_time = time.time()

        corrected = remove_baseline(medium_dataset.spectra, method='polyfit', degree=2)

        elapsed = time.time() - start_time

        # 应该在5秒内完成
        assert elapsed < 5.0
        assert corrected.shape == medium_dataset.spectra.shape


class TestQCPerformance:
    """测试质量控制性能"""

    @pytest.mark.slow
    def test_qc_performance_small(self, small_dataset):
        """测试小数据集的QC性能"""
        start_time = time.time()

        qc_result = quality_control(small_dataset, method='dis', threshold=0.05)

        elapsed = time.time() - start_time

        # 应该在100ms内完成
        assert elapsed < 0.1
        assert qc_result is not None

    @pytest.mark.slow
    def test_qc_performance_large(self, large_dataset):
        """测试大数据集的QC性能"""
        start_time = time.time()

        qc_result = quality_control(large_dataset, method='dis', threshold=0.05)

        elapsed = time.time() - start_time

        # 应该在5秒内完成
        assert elapsed < 5.0
        assert qc_result is not None


class TestReductionPerformance:
    """测试降维性能"""

    @pytest.mark.slow
    def test_pca_performance_small(self, small_dataset):
        """测试小数据集的PCA性能"""
        start_time = time.time()

        result = reduce(small_dataset.spectra, method='pca', n_components=2)

        elapsed = time.time() - start_time

        # 应该在100ms内完成
        assert elapsed < 0.1
        assert result['transformed'].shape == (10, 2)

    @pytest.mark.slow
    def test_pca_performance_medium(self, medium_dataset):
        """测试中等数据集的PCA性能"""
        start_time = time.time()

        result = reduce(medium_dataset.spectra, method='pca', n_components=2)

        elapsed = time.time() - start_time

        # 应该在1秒内完成
        assert elapsed < 1.0
        assert result['transformed'].shape == (100, 2)

    @pytest.mark.slow
    def test_pca_performance_large(self, large_dataset):
        """测试大数据集的PCA性能"""
        start_time = time.time()

        result = reduce(large_dataset.spectra, method='pca', n_components=2)

        elapsed = time.time() - start_time

        # 应该在5秒内完成
        assert elapsed < 5.0
        assert result['transformed'].shape == (1000, 2)


class TestMemoryUsage:
    """测试内存使用"""

    @pytest.mark.slow
    def test_memory_efficient_copy(self, medium_dataset):
        """测试内存高效的复制"""
        import sys

        # 获取原始对象大小
        original_size = sys.getsizeof(medium_dataset.spectra)

        start_time = time.time()
        copied = medium_dataset.copy()
        elapsed = time.time() - start_time

        copied_size = sys.getsizeof(copied.spectra)

        # 复制应该在合理时间内完成
        assert elapsed < 1.0

        # 大小应该相似
        assert abs(copied_size - original_size) < original_size * 0.1

    @pytest.mark.slow
    def test_memory_with_multiple_reductions(self, medium_dataset):
        """测试多次降维的内存使用"""
        import sys

        initial_size = sys.getsizeof(medium_dataset)

        # 添加多种降维
        medium_dataset.reduce(method='pca', n_components=2)
        medium_dataset.reduce(method='pca', n_components=5)
        medium_dataset.reduce(method='pcoa', n_components=3)

        final_size = sys.getsizeof(medium_dataset)

        # 大小增长应该合理（不应该指数增长）
        assert final_size < initial_size * 10


class TestScalability:
    """测试可扩展性"""

    @pytest.mark.slow
    def test_scalability_smooth(self):
        """测试平滑的可扩展性"""
        sizes = [10, 50, 100]
        times = []

        for size in sizes:
            spectra = np.random.randn(size, 100)
            wavenumbers = np.linspace(400, 4000, 100)
            metadata = pd.DataFrame({'id': range(size)})

            ramanome = Ramanome(spectra, wavenumbers, metadata)

            start_time = time.time()
            smooth(ramanome.spectra, window_size=5, polyorder=2)
            elapsed = time.time() - start_time

            times.append(elapsed)

        # 时间应该线性增长
        # 最后一个应该比第一个慢，但不应该超过10倍
        assert times[-1] < times[0] * 10

    @pytest.mark.slow
    def test_scalability_pca(self):
        """测试PCA的可扩展性"""
        sizes = [10, 50, 100]
        times = []

        for size in sizes:
            spectra = np.random.randn(size, 100)
            wavenumbers = np.linspace(400, 4000, 100)
            metadata = pd.DataFrame({'id': range(size)})

            ramanome = Ramanome(spectra, wavenumbers, metadata)

            start_time = time.time()
            reduce(ramanome.spectra, method='pca', n_components=2)
            elapsed = time.time() - start_time

            times.append(elapsed)

        # PCA的时间复杂度接近O(n^2)
        # 但对于小数据集，应该保持合理
        assert times[-1] < times[0] * 20


class TestBenchmarks:
    """性能基准测试"""

    @pytest.mark.slow
    def test_benchmark_full_pipeline(self):
        """测试完整流程的性能基准"""
        # 创建中等规模数据集
        spectra = np.random.randn(100, 500)
        wavenumbers = np.linspace(400, 4000, 500)
        metadata = pd.DataFrame({'id': range(100)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        start_time = time.time()

        # 完整流程
        ramanome.smooth(window_size=5)
        ramanome.remove_baseline(method='polyfit', degree=2)
        ramanome.normalize(method='minmax')
        ramanome.quality_control(method='dis')
        ramanome.reduce(method='pca', n_components=2)

        elapsed = time.time() - start_time

        # 完整流程应该在10秒内完成
        assert elapsed < 10.0

    @pytest.mark.slow
    def test_benchmark_ml_conversion(self):
        """测试ML格式转换的性能基准"""
        from pyramex.ml.integration import to_sklearn_format

        spectra = np.random.randn(1000, 1000)
        wavenumbers = np.linspace(400, 4000, 1000)
        metadata = pd.DataFrame({'id': range(1000)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        start_time = time.time()
        X_train, X_test = to_sklearn_format(ramanome, test_size=0.2)
        elapsed = time.time() - start_time

        # 应该在1秒内完成
        assert elapsed < 1.0


class TestOptimization:
    """测试优化机会"""

    def test_vectorized_vs_loop(self):
        """比较向量化和循环的性能"""
        spectra = np.random.randn(100, 500)

        # 向量化操作
        start_time = time.time()
        mean_vec = spectra.mean(axis=1)
        time_vec = time.time() - start_time

        # 循环操作（不推荐）
        start_time = time.time()
        mean_loop = np.array([spectra[i].mean() for i in range(len(spectra))])
        time_loop = time.time() - start_time

        # 向量化应该更快
        assert time_vec < time_loop

        # 结果应该相同
        np.testing.assert_array_almost_equal(mean_vec, mean_loop)


class TestPerformanceProfiling:
    """性能剖析"""

    @pytest.mark.slow
    def test_profiling_bottleneck_identification(self):
        """识别性能瓶颈"""
        spectra = np.random.randn(100, 500)
        wavenumbers = np.linspace(400, 4000, 500)
        metadata = pd.DataFrame({'id': range(100)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 测量每个步骤的时间
        steps = {}

        start = time.time()
        smooth(ramanome.spectra, window_size=5, polyorder=2)
        steps['smooth'] = time.time() - start

        start = time.time()
        remove_baseline(ramanome.spectra, method='polyfit', degree=2)
        steps['baseline'] = time.time() - start

        start = time.time()
        normalize(ramanome.spectra, method='minmax')
        steps['normalize'] = time.time() - start

        start = time.time()
        reduce(ramanome.spectra, method='pca', n_components=2)
        steps['pca'] = time.time() - start

        # 所有步骤都应该在合理时间内完成
        for step, elapsed in steps.items():
            assert elapsed < 5.0, f"{step} took too long: {elapsed}s"


class TestConcurrency:
    """测试并发性能"""

    @pytest.mark.slow
    def test_parallel_qc(self):
        """测试并行QC（如果可用）"""
        # 这个测试只是确保串行版本工作正常
        # 未来可以添加并行实现
        spectra = np.random.randn(50, 500)
        wavenumbers = np.linspace(400, 4000, 500)
        metadata = pd.DataFrame({'id': range(50)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        start_time = time.time()
        quality_control(ramanome, method='dis', threshold=0.05)
        elapsed = time.time() - start_time

        assert elapsed < 5.0
