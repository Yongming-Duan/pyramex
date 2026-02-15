"""
PyRamEx测试配置
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_spectra():
    """示例拉曼光谱数据"""
    np.random.seed(42)
    wavelength = np.linspace(400, 4000, 1000)
    intensity = np.random.randn(1000) * 0.1 + np.sin(wavelength / 100)
    return pd.DataFrame({
        'wavelength': wavelength,
        'intensity': intensity
    })


@pytest.fixture
def sample_dataset():
    """示例数据集（多个样本）"""
    np.random.seed(42)
    samples = []
    for i in range(10):
        wavelength = np.linspace(400, 4000, 1000)
        intensity = np.random.randn(1000) * 0.1 + np.sin(wavelength / 100)
        samples.append(pd.DataFrame({
            'wavelength': wavelength,
            'intensity': intensity,
            'label': f'sample_{i}'
        }))
    return pd.concat(samples, ignore_index=True)


@pytest.fixture
def temp_dir():
    """临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data_file(temp_dir, sample_dataset):
    """示例数据文件"""
    data_file = temp_dir / "sample_data.csv"
    sample_dataset.to_csv(data_file, index=False)
    return data_file


@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录（session级别）"""
    test_data = Path(__file__).parent / "data"
    test_data.mkdir(exist_ok=True)
    return test_data


# 配置pytest标记
def pytest_configure(config):
    """配置pytest自定义标记"""
    config.addinivalue_line(
        "markers", "slow: 标记慢速测试"
    )
    config.addinivalue_line(
        "markers", "integration: 标记集成测试"
    )
    config.addinivalue_line(
        "markers", "ml: 标记机器学习相关测试"
    )
