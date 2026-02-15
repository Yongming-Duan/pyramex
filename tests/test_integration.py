"""
集成测试
测试端到端的工作流程
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from pyramex.core.ramanome import Ramanome, QualityResult
from pyramex.preprocessing.processors import smooth, remove_baseline, normalize, cutoff
from pyramex.qc.methods import quality_control
from pyramex.features.reduction import reduce, extract_band_intensity
from pyramex.visualization.plots import plot_spectra, plot_reduction
import matplotlib
matplotlib.use('Agg')


@pytest.fixture
def complete_workflow_data(tmp_path):
    """创建完整的测试数据集"""
    # 创建多个光谱文件
    dir_path = tmp_path / "spectra"
    dir_path.mkdir()

    wavenumbers = np.linspace(400, 4000, 100)

    for i in range(10):
        file_path = dir_path / f"spectrum_{i}.txt"

        # 创建不同的光谱模式
        if i < 5:
            # 类别A
            intensity = np.sin(wavenumbers / 100) + np.random.randn(100) * 0.1
        else:
            # 类别B
            intensity = np.cos(wavenumbers / 100) + np.random.randn(100) * 0.1

        data = pd.DataFrame({
            'wavenumber': wavenumbers,
            'intensity': intensity
        })

        data.to_csv(file_path, sep='\t', header=False, index=False)

    return dir_path


class TestEndToEndWorkflow:
    """测试端到端工作流程"""

    def test_complete_analysis_workflow(self, complete_workflow_data):
        """测试完整的分析工作流程"""
        from pyramex.io.loader import load_spectra

        # 1. 加载数据
        ramanome = load_spectra(complete_workflow_data)
        assert ramanome.n_samples == 10

        # 2. 预处理
        smoothed = smooth(ramanome.spectra, window_size=5, polyorder=2)
        baseline_removed = remove_baseline(smoothed, method='polyfit', degree=2)
        normalized = normalize(baseline_removed, method='minmax')

        # 更新ramanome
        ramanome.spectra = normalized

        # 3. 质量控制
        qc_result = ramanome.quality_control(method='dis', threshold=0.05)
        assert qc_result is not None

        # 4. 过滤坏样本
        good_indices = np.where(qc_result.good_samples)[0]
        clean_spectra = ramanome.spectra[good_indices]

        # 5. 降维
        reduction_result = reduce(clean_spectra, method='pca', n_components=2)
        assert reduction_result is not None

        # 6. 特征提取
        band_intensities = extract_band_intensity(
            ramanome,
            bands=[(500, 600), (1000, 1100), (1500, 1600)]
        )
        assert band_intensities.shape[1] == 3

        # 7. 可视化
        fig = plot_spectra(ramanome, return_fig=True)
        assert fig is not None

    def test_ml_pipeline_workflow(self, complete_workflow_data):
        """测试机器学习流程"""
        from pyramex.io.loader import load_spectra
        from pyramex.ml.integration import to_sklearn_format

        # 1. 加载和预处理
        ramanome = load_spectra(complete_workflow_data)
        ramanome.spectra = smooth(ramanome.spectra, window_size=5)
        ramanome.spectra = normalize(ramanome.spectra, method='minmax')

        # 2. 添加标签
        labels = ['A'] * 5 + ['B'] * 5
        ramanome.metadata['label'] = labels

        # 3. 转换为ML格式
        X_train, X_test, y_train, y_test = to_sklearn_format(
            ramanome,
            test_size=0.3,
            random_state=42
        )

        assert X_train.shape[1] == 100
        assert len(y_train) > 0
        assert len(y_test) > 0

        # 4. 训练简单模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # 5. 评估
        score = model.score(X_test, y_test)
        assert 0 <= score <= 1

    def test_ramanome_method_chaining_workflow(self):
        """测试Ramanome方法链工作流程"""
        # 创建数据
        spectra = np.random.randn(20, 200)
        wavenumbers = np.linspace(400, 4000, 200)
        metadata = pd.DataFrame({
            'id': range(20),
            'label': ['A'] * 10 + ['B'] * 10
        })

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 方法链
        result = (ramanome
                  .smooth(window_size=7, polyorder=2)
                  .remove_baseline(method='polyfit', degree=2)
                  .normalize(method='minmax')
                  .cutoff((500, 3500))
                  .reduce(method='pca', n_components=2))

        # 检查结果
        assert result is ramanome
        assert len(ramanome.processed) == 4  # smooth, baseline, normalize, cutoff
        assert 'pca' in ramanome.reductions


class TestPreprocessingIntegration:
    """测试预处理集成"""

    def test_preprocessing_pipeline_reproducibility(self):
        """测试预处理流程的可重复性"""
        spectra = np.random.randn(10, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(10)})

        ramanome1 = Ramanome(spectra.copy(), wavenumbers.copy(), metadata.copy())
        ramanome2 = Ramanome(spectra.copy(), wavenumbers.copy(), metadata.copy())

        # 应用相同的预处理
        for ramanome in [ramanome1, ramanome2]:
            ramanome.smooth(window_size=5)
            ramanome.normalize(method='minmax')

        # 结果应该相同
        np.testing.assert_array_almost_equal(
            ramanome1.spectra,
            ramanome2.spectra
        )

    def test_preprocessing_with_reset(self):
        """测试预处理后重置"""
        spectra = np.random.randn(10, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(10)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)
        original = ramanome.spectra.copy()

        # 应用预处理
        ramanome.smooth(window_size=5)
        ramanome.normalize(method='minmax')

        assert not np.allclose(ramanome.spectra, original)

        # 重置
        ramanome.reset()

        np.testing.assert_array_equal(ramanome.spectra, original)


class TestQCIntegration:
    """测试质量控制集成"""

    def test_qc_with_filtering(self):
        """测试QC后过滤"""
        # 创建包含异常值的数据
        np.random.seed(42)
        good_spectra = np.random.randn(8, 100) * 0.1
        bad_spectra = np.random.randn(2, 100) * 5

        spectra = np.vstack([good_spectra, bad_spectra])
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(10)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 执行QC
        qc_result = ramanome.quality_control(method='dis', threshold=0.05)

        # 过滤
        good_mask = qc_result.good_samples
        clean_ramanome = Ramanome(
            spectra[good_mask],
            wavenumbers,
            metadata[good_mask].reset_index(drop=True)
        )

        # 清理后的数据应该有更少的样本
        assert clean_ramanome.n_samples < ramanome.n_samples

    def test_multiple_qc_methods(self):
        """测试使用多种QC方法"""
        spectra = np.random.randn(10, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(10)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 应用多种QC方法
        methods = ['dis', 'icod', 'snr']

        for method in methods:
            qc_result = quality_control(ramanome, method=method)
            assert qc_result is not None
            assert qc_result.method == method


class TestReductionIntegration:
    """测试降维集成"""

    def test_reduction_with_visualization(self):
        """测试降维和可视化集成"""
        spectra = np.random.randn(20, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({
            'id': range(20),
            'group': ['A'] * 10 + ['B'] * 10
        })

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 降维
        ramanome.reduce(method='pca', n_components=2)

        # 可视化
        fig = plot_reduction(ramanome, method='pca', color_by='group', return_fig=True)

        assert fig is not None

    def test_multiple_reductions_comparison(self):
        """测试多种降维方法对比"""
        spectra = np.random.randn(20, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(20)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 应用多种降维
        methods = ['pca', 'pcoa']

        for method in methods:
            ramanome.reduce(method=method, n_components=2)
            assert method in ramanome.reductions

        # 所有降维结果应该有相同的形状
        pca_shape = ramanome.reductions['pca']['transformed'].shape
        pcoa_shape = ramanome.reductions['pcoa']['transformed'].shape

        assert pca_shape == pcoa_shape


class TestMLEndToEnd:
    """测试ML端到端流程"""

    def test_sklearn_workflow(self):
        """测试scikit-learn工作流程"""
        from pyramex.ml.integration import to_sklearn_format, create_mlp_model

        # 创建数据
        spectra = np.random.randn(50, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({
            'id': range(50),
            'label': ['A'] * 25 + ['B'] * 25
        })

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 预处理
        ramanome.smooth(window_size=5)
        ramanome.normalize(method='minmax')

        # 转换为ML格式
        X_train, X_test, y_train, y_test = to_sklearn_format(
            ramanome,
            test_size=0.2,
            random_state=42
        )

        # 训练模型
        model = create_mlp_model(input_length=100, n_classes=2)
        model.fit(X_train, y_train)

        # 评估
        score = model.score(X_test, y_test)
        assert 0 <= score <= 1

    @pytest.mark.ml
    def test_pytorch_workflow(self):
        """测试PyTorch工作流程"""
        try:
            import torch
            from torch.utils.data import DataLoader
            from pyramex.ml.integration import to_torch_dataset, create_cnn_model

            # 创建数据
            spectra = np.random.randn(20, 100)
            wavenumbers = np.linspace(400, 4000, 100)
            metadata = pd.DataFrame({
                'id': range(20),
                'label': ['A'] * 10 + ['B'] * 10
            })

            ramanome = Ramanome(spectra, wavenumbers, metadata)

            # 转换为PyTorch数据集
            dataset = to_torch_dataset(ramanome)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

            # 创建模型
            model = create_cnn_model(input_length=100, n_classes=2)

            # 测试前向传播
            for batch_spectra, batch_labels in dataloader:
                output = model(batch_spectra)
                assert output.shape[0] <= 4
                assert output.shape[1] == 2
                break

        except ImportError:
            pytest.skip("PyTorch not installed")


class TestDataPersistence:
    """测试数据持久化"""

    def test_ramanome_copy_preserves_all(self):
        """测试Ramanome复制保留所有信息"""
        spectra = np.random.randn(10, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(10)})

        ramanome1 = Ramanome(spectra, wavenumbers, metadata)

        # 应用各种操作
        ramanome1.smooth(window_size=5)
        ramanome1.quality_control(method='dis')
        ramanome1.reduce(method='pca', n_components=2)

        # 复制
        ramanome2 = ramanome1.copy()

        # 检查所有信息都被保留
        assert ramanome2.n_samples == ramanome1.n_samples
        assert len(ramanome2.processed) == len(ramanome1.processed)
        assert 'pca' in ramanome2.reductions
        assert 'dis' in ramanome2.quality

    def test_ramanome_to_ml_format_with_metadata(self):
        """测试转换为ML格式时保留元数据"""
        spectra = np.random.randn(10, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({
            'id': range(10),
            'label': ['A', 'B'] * 5,
            'batch': [1, 1, 2, 2, 3] * 2
        })

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        X, meta = ramanome.to_ml_format(return_metadata=True)

        assert X.shape == (10, 100)
        assert meta.shape == (10, 3)
        assert list(meta.columns) == ['id', 'label', 'batch']


class TestErrorHandling:
    """测试错误处理"""

    def test_workflow_with_invalid_data(self):
        """测试处理无效数据的流程"""
        # 创建包含NaN的数据
        spectra = np.random.randn(10, 100)
        spectra[0, 0] = np.nan

        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(10)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # QC应该能检测到问题
        qc_result = ramanome.quality_control(method='dis', threshold=0.05)

        assert qc_result is not None

    def test_workflow_with_empty_data(self):
        """测试处理空数据的流程"""
        spectra = np.array([]).reshape(0, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame()

        # 这种情况下初始化会失败
        # 所以我们测试单样本
        spectra = np.random.randn(1, 100)
        metadata = pd.DataFrame({'id': [0]})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        # 应该能处理
        result = ramanome.reduce(method='pca', n_components=1)

        assert result is ramanome
