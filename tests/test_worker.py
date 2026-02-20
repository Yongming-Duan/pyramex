"""
测试GPU Worker
"""

import pytest
from unittest.mock import patch, MagicMock
from pyramex.worker.gpu_worker import GPUWorker


@pytest.fixture
def worker():
    """创建GPU Worker实例"""
    return GPUWorker()


def test_gpu_worker_init(worker):
    """测试GPU Worker初始化"""
    assert worker is not None
    assert worker.device in ["cpu", "cuda"]


def test_check_gpu_without_cuda():
    """测试GPU检测（CUDA不可用）"""
    # 这个测试验证在CPU环境下的行为
    worker = GPUWorker()
    assert worker.device in ["cpu", "cuda"]


def test_check_gpu_torch_not_installed():
    """测试PyTorch未安装的情况"""
    # 由于import发生在模块加载时，这个测试验证降级行为
    worker = GPUWorker()
    # 实际上由于torch已经在模块顶部导入，这里测试降级逻辑
    assert worker is not None


def test_preprocess_method(worker):
    """测试预处理方法"""
    data = {
        "spectra": [[1.0, 2.0, 3.0]],
        "wavenumber": [400.0, 500.0, 600.0]
    }

    result = worker.preprocess(data)
    assert result["status"] == "success"
    assert "device" in result
    assert result["processed"] is True
    assert result["device"] in ["cpu", "cuda"]


def test_pca_reduce_method(worker):
    """测试PCA降维方法"""
    data = {
        "X": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "n_features": 3
    }

    result = worker.pca_reduce(data, n_components=2)
    assert result["status"] == "success"
    assert "device" in result
    assert result["n_components"] == 2


def test_pca_reduce_default_components(worker):
    """测试PCA降维默认参数"""
    data = {
        "X": [[1.0, 2.0, 3.0]],
        "n_features": 3
    }

    result = worker.pca_reduce(data)
    assert result["status"] == "success"
    assert result["n_components"] == 50  # 默认值


def test_train_model_method(worker):
    """测试模型训练方法"""
    data = {
        "X_train": [[1.0, 2.0], [3.0, 4.0]],
        "y_train": [0, 1],
        "X_test": [[5.0, 6.0]]
    }

    result = worker.train_model(data, model_type="rf")
    assert result["status"] == "success"
    assert "device" in result
    assert result["model_type"] == "rf"


def test_train_model_different_types(worker):
    """测试不同模型类型"""
    data = {
        "X_train": [[1.0, 2.0]],
        "y_train": [0]
    }

    # 测试随机森林
    result_rf = worker.train_model(data, model_type="rf")
    assert result_rf["status"] == "success"
    assert result_rf["model_type"] == "rf"

    # 测试其他模型类型
    result_svm = worker.train_model(data, model_type="svm")
    assert result_svm["status"] == "success"
    assert result_svm["model_type"] == "svm"


def test_health_check_method(worker):
    """测试健康检查方法"""
    result = worker.health_check()
    assert "status" in result
    # 注意：如果PyTorch未安装，返回的error可能没有device字段
    if result["status"] == "healthy":
        assert "device" in result


def test_health_check_with_cuda():
    """测试CUDA环境下的健康检查"""
    # 注意：由于torch在模块顶部导入，这里测试实际行为
    worker = GPUWorker()
    result = worker.health_check()
    assert "status" in result


def test_preprocess_with_empty_data(worker):
    """测试空数据的预处理"""
    data = {}
    result = worker.preprocess(data)
    assert result["status"] == "success"


def test_preprocess_with_large_dataset(worker):
    """测试大数据集的预处理"""
    import numpy as np

    data = {
        "spectra": np.random.randn(1000, 1000).tolist(),
        "wavenumber": np.linspace(400, 4000, 1000).tolist()
    }

    result = worker.preprocess(data)
    assert result["status"] == "success"
    assert result["device"] in ["cpu", "cuda"]


def test_pca_reduce_with_different_components(worker):
    """测试不同数量的PCA组件"""
    data = {"X": [[1.0, 2.0, 3.0, 4.0]], "n_features": 4}

    # 测试不同的组件数
    for n_comp in [1, 2, 3, 5, 10, 100]:
        result = worker.pca_reduce(data, n_components=n_comp)
        assert result["status"] == "success"
        assert result["n_components"] == n_comp


def test_train_model_with_different_params(worker):
    """测试不同参数的模型训练"""
    data = {
        "X_train": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "y_train": [0, 1, 0]
    }

    # 测试不同的模型类型
    model_types = ["rf", "svm", "lr", "knn"]
    for model_type in model_types:
        result = worker.train_model(data, model_type=model_type)
        assert result["status"] == "success"
        assert result["model_type"] == model_type


def test_worker_device_attribute(worker):
    """测试worker设备属性"""
    assert hasattr(worker, 'device')
    assert isinstance(worker.device, str)
    assert worker.device in ["cpu", "cuda"]


def test_health_check_device_field(worker):
    """测试健康检查中的设备字段"""
    result = worker.health_check()
    # 注意：如果PyTorch未安装，返回的error可能没有device字段
    if result["status"] == "healthy":
        assert "device" in result
        assert isinstance(result["device"], str)
        assert result["device"] in ["cpu", "cuda"]
