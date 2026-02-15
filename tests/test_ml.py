"""
机器学习集成模块测试
测试ML/DL框架集成
"""

import pytest
import numpy as np
import pandas as pd
from pyramex.core.ramanome import Ramanome
from pyramex.ml.integration import (
    to_sklearn_format,
    to_torch_dataset,
    to_tf_dataset,
    create_dataloader,
    create_cnn_model,
    create_mlp_model
)


@pytest.fixture
def sample_ramanome():
    """创建示例Ramanome对象"""
    spectra = np.random.randn(20, 100)
    wavenumbers = np.linspace(400, 4000, 100)
    metadata = pd.DataFrame({
        'id': range(20),
        'label': ['A'] * 10 + ['B'] * 10
    })

    return Ramanome(spectra, wavenumbers, metadata)


@pytest.fixture
def sample_ramanome_no_labels():
    """创建无标签的示例Ramanome对象"""
    spectra = np.random.randn(20, 100)
    wavenumbers = np.linspace(400, 4000, 100)
    metadata = pd.DataFrame({'id': range(20)})

    return Ramanome(spectra, wavenumbers, metadata)


class TestSklearnFormat:
    """测试scikit-learn格式转换"""

    def test_to_sklearn_format_with_labels(self, sample_ramanome):
        """测试带标签的sklearn格式"""
        X_train, X_test, y_train, y_test = to_sklearn_format(
            sample_ramanome,
            test_size=0.2,
            random_state=42
        )

        assert X_train.shape[1] == 100
        assert X_test.shape[1] == 100
        assert len(X_train) + len(X_test) == 20
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_to_sklearn_format_without_labels(self, sample_ramanome_no_labels):
        """测试无标签的sklearn格式"""
        X_train, X_test = to_sklearn_format(
            sample_ramanome_no_labels,
            test_size=0.2,
            random_state=42
        )

        assert X_train.shape[1] == 100
        assert X_test.shape[1] == 100
        assert len(X_train) + len(X_test) == 20

    def test_to_sklearn_format_different_test_size(self, sample_ramanome):
        """测试不同的测试集比例"""
        X_train, X_test, y_train, y_test = to_sklearn_format(
            sample_ramanome,
            test_size=0.3,
            random_state=42
        )

        assert len(X_test) == 6  # 20 * 0.3
        assert len(X_train) == 14


class TestTorchDataset:
    """测试PyTorch数据集"""

    def test_to_torch_dataset(self, sample_ramanome):
        """测试转换为PyTorch数据集"""
        try:
            import torch

            dataset = to_torch_dataset(sample_ramanome)

            assert len(dataset) == 20

            # 测试获取单个样本
            sample = dataset[0]
            assert isinstance(sample, tuple)
            assert len(sample) == 2  # (spectrum, label)

            spectrum, label = sample
            assert spectrum.shape == (1, 100)  # (channel, wavenumbers)

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_to_torch_dataset_no_labels(self, sample_ramanome_no_labels):
        """测试无标签的PyTorch数据集"""
        try:
            import torch

            dataset = to_torch_dataset(sample_ramanome_no_labels)

            # 无标签时应该只返回spectrum
            sample = dataset[0]
            assert sample.shape == (1, 100)

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_to_torch_dataloader(self, sample_ramanome):
        """测试PyTorch DataLoader"""
        try:
            import torch
            from torch.utils.data import DataLoader

            dataset = to_torch_dataset(sample_ramanome)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

            # 测试迭代
            for batch_spectra, batch_labels in dataloader:
                assert batch_spectra.shape[0] <= 4
                assert batch_spectra.shape[1] == 1
                assert batch_spectra.shape[2] == 100
                break

        except ImportError:
            pytest.skip("PyTorch not installed")


class TestTFDataset:
    """测试TensorFlow数据集"""

    def test_to_tf_dataset(self, sample_ramanome):
        """测试转换为TensorFlow数据集"""
        try:
            import tensorflow as tf

            dataset = to_tf_dataset(sample_ramanome, batch_size=4, shuffle=False)

            # 测试迭代
            for batch in dataset.take(1):
                if isinstance(batch, tuple):
                    spectra, labels = batch
                    assert spectra.shape[1] == 1  # channel
                    assert spectra.shape[2] == 100
                else:
                    spectra = batch
                    assert spectra.shape[1] == 1
                    assert spectra.shape[2] == 100

        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_to_tf_dataset_no_labels(self, sample_ramanome_no_labels):
        """测试无标签的TensorFlow数据集"""
        try:
            import tensorflow as tf

            dataset = to_tf_dataset(
                sample_ramanome_no_labels,
                batch_size=4,
                shuffle=False
            )

            for batch in dataset.take(1):
                assert batch.shape[1] == 1
                assert batch.shape[2] == 100

        except ImportError:
            pytest.skip("TensorFlow not installed")


class TestCreateDataloader:
    """测试通用数据加载器创建"""

    def test_create_dataloader_sklearn(self, sample_ramanome):
        """测试创建sklearn数据加载器"""
        data = create_dataloader(
            sample_ramanome,
            framework='sklearn',
            test_size=0.2
        )

        assert isinstance(data, tuple)
        assert len(data) == 4  # X_train, X_test, y_train, y_test

    def test_create_dataloader_torch(self, sample_ramanome):
        """测试创建PyTorch数据加载器"""
        try:
            data = create_dataloader(
                sample_ramanome,
                framework='torch'
            )

            assert len(data) == 20

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_create_dataloader_tensorflow(self, sample_ramanome):
        """测试创建TensorFlow数据加载器"""
        try:
            import tensorflow as tf

            data = create_dataloader(
                sample_ramanome,
                framework='tensorflow',
                batch_size=4
            )

            assert data is not None

        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_create_dataloader_invalid_framework(self, sample_ramanome):
        """测试无效的框架"""
        with pytest.raises(ValueError, match="Unknown framework"):
            create_dataloader(sample_ramanome, framework='invalid')


class TestCNNModel:
    """测试CNN模型创建"""

    def test_create_cnn_model_classification(self):
        """测试创建分类CNN模型"""
        try:
            import torch
            import torch.nn as nn

            model = create_cnn_model(
                input_length=100,
                n_classes=2,
                dropout=0.2
            )

            assert isinstance(model, nn.Module)

            # 测试前向传播
            x = torch.randn(4, 1, 100)
            output = model(x)

            assert output.shape == (4, 2)

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_create_cnn_model_regression(self):
        """测试创建回归CNN模型"""
        try:
            import torch
            import torch.nn as nn

            model = create_cnn_model(
                input_length=100,
                n_classes=None,
                dropout=0.2
            )

            # 测试前向传播
            x = torch.randn(4, 1, 100)
            output = model(x)

            assert output.shape == (4, 1)

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_create_cnn_model_different_input_length(self):
        """测试不同输入长度"""
        try:
            import torch

            model = create_cnn_model(input_length=200, n_classes=3)

            x = torch.randn(2, 1, 200)
            output = model(x)

            assert output.shape == (2, 3)

        except ImportError:
            pytest.skip("PyTorch not installed")


class TestMLPModel:
    """测试MLP模型创建"""

    def test_create_mlp_model_classification(self):
        """测试创建分类MLP模型"""
        model = create_mlp_model(
            input_length=100,
            n_classes=2,
            hidden_dims=[64, 32]
        )

        assert model is not None
        assert hasattr(model, 'fit')

        # 测试预测
        X = np.random.randn(10, 100)
        model.fit(X, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        predictions = model.predict(X[:3])

        assert predictions.shape == (3,)

    def test_create_mlp_model_regression(self):
        """测试创建回归MLP模型"""
        model = create_mlp_model(
            input_length=100,
            n_classes=None,
            hidden_dims=[64, 32]
        )

        assert model is not None

    def test_create_mlp_model_different_hidden_dims(self):
        """测试不同的隐藏层维度"""
        model = create_mlp_model(
            input_length=100,
            n_classes=2,
            hidden_dims=[128, 64, 32]
        )

        assert model is not None


class TestMLEdgeCases:
    """测试ML集成的边界情况"""

    def test_sklearn_small_dataset(self):
        """测试小数据集的sklearn格式"""
        spectra = np.random.randn(5, 100)
        wavenumbers = np.linspace(400, 4000, 100)
        metadata = pd.DataFrame({'id': range(5)})

        ramanome = Ramanome(spectra, wavenumbers, metadata)

        X_train, X_test = to_sklearn_format(ramanome, test_size=0.2)

        # 即使是5个样本，也应该能分割
        assert X_train.shape[0] >= 1
        assert X_test.shape[0] >= 1

    def test_torch_dataset_transform(self):
        """测试PyTorch数据集的变换"""
        try:
            import torch

            spectra = np.random.randn(10, 100)
            wavenumbers = np.linspace(400, 4000, 100)
            metadata = pd.DataFrame({'id': range(10)})

            ramanome = Ramanome(spectra, wavenumbers, metadata)

            def custom_transform(x):
                return x * 2

            dataset = to_torch_dataset(ramanome, transform=custom_transform)

            spectrum = dataset[0]
            assert spectrum is not None

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_tf_dataset_batch_size(self):
        """测试TensorFlow数据集的batch size"""
        try:
            import tensorflow as tf

            spectra = np.random.randn(20, 100)
            wavenumbers = np.linspace(400, 4000, 100)
            metadata = pd.DataFrame({'id': range(20)})

            ramanome = Ramanome(spectra, wavenumbers, metadata)

            dataset = to_tf_dataset(ramanome, batch_size=5, shuffle=False)

            # 检查batch大小
            for i, batch in enumerate(dataset.take(4)):
                if isinstance(batch, tuple):
                    spectra_batch = batch[0]
                else:
                    spectra_batch = batch

                if i < 3:  # 前三个完整batch
                    assert spectra_batch.shape[0] == 5

        except ImportError:
            pytest.skip("TensorFlow not installed")
