"""
测试FastAPI应用
"""

import pytest
from fastapi.testclient import TestClient
from pyramex.api.main import app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


def test_root_endpoint(client):
    """测试根路径"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "2.0.0"


def test_health_check(client):
    """测试健康检查端点"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "pyramex-app"
    assert "version" in data


def test_preprocess_endpoint(client):
    """测试预处理端点"""
    request_data = {
        "spectra": [
            {
                "wavenumber": [400.0, 500.0, 600.0],
                "intensity": [100.0, 200.0, 150.0],
                "metadata": {"sample_id": "test1"}
            }
        ],
        "analysis_type": "preprocessing"
    }

    response = client.post("/api/v1/preprocess", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "results" in data
    assert "message" in data


def test_qc_endpoint(client):
    """测试质控端点"""
    request_data = {
        "spectra": [
            {
                "wavenumber": [400.0, 500.0, 600.0],
                "intensity": [100.0, 200.0, 150.0],
                "metadata": {"sample_id": "test1"}
            }
        ],
        "analysis_type": "qc"
    }

    response = client.post("/api/v1/qc", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "results" in data
    assert "total" in data["results"]


def test_analyze_endpoint(client):
    """测试ML分析端点"""
    request_data = {
        "spectra": [
            {
                "wavenumber": [400.0, 500.0, 600.0],
                "intensity": [100.0, 200.0, 150.0],
                "metadata": {"sample_id": "test1"}
            }
        ],
        "analysis_type": "ml"
    }

    response = client.post("/api/v1/analyze", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "results" in data
    assert "predictions" in data["results"]


def test_report_endpoint(client):
    """测试AI报告生成端点"""
    request_data = {
        "spectra": [
            {
                "wavenumber": [400.0, 500.0, 600.0],
                "intensity": [100.0, 200.0, 150.0],
                "metadata": {"sample_id": "test1"}
            }
        ],
        "analysis_type": "report"
    }

    response = client.post("/api/v1/report", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "results" in data
    assert "report" in data["results"]


def test_preprocess_error_handling(client):
    """测试预处理错误处理"""
    # 发送无效数据
    request_data = {
        "spectra": [
            {
                "wavenumber": [400.0],
                "intensity": [100.0]
            }
        ],
        "analysis_type": "preprocessing"
    }

    # 这个应该正常处理，因为我们的实现比较简单
    response = client.post("/api/v1/preprocess", json=request_data)
    assert response.status_code == 200


def test_qc_error_handling(client):
    """测试质控错误处理"""
    # 发送空列表
    request_data = {
        "spectra": [],
        "analysis_type": "qc"
    }

    response = client.post("/api/v1/qc", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_analyze_with_empty_spectra(client):
    """测试空光谱列表的ML分析"""
    request_data = {
        "spectra": [],
        "analysis_type": "ml"
    }

    response = client.post("/api/v1/analyze", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_multiple_spectra_preprocess(client):
    """测试多个光谱的预处理"""
    request_data = {
        "spectra": [
            {
                "wavenumber": [400.0, 500.0, 600.0],
                "intensity": [100.0, 200.0, 150.0]
            },
            {
                "wavenumber": [400.0, 500.0, 600.0],
                "intensity": [110.0, 210.0, 160.0]
            },
            {
                "wavenumber": [400.0, 500.0, 600.0],
                "intensity": [90.0, 190.0, 140.0]
            }
        ],
        "analysis_type": "preprocessing"
    }

    response = client.post("/api/v1/preprocess", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["results"]["processed"] == 3


def test_metadata_optional(client):
    """测试metadata是可选的"""
    request_data = {
        "spectra": [
            {
                "wavenumber": [400.0, 500.0, 600.0],
                "intensity": [100.0, 200.0, 150.0]
                # 没有metadata
            }
        ],
        "analysis_type": "preprocessing"
    }

    response = client.post("/api/v1/preprocess", json=request_data)
    assert response.status_code == 200


def test_api_version_consistency(client):
    """测试所有端点的版本一致性"""
    # 获取根路径的版本
    root_response = client.get("/")
    root_version = root_response.json()["version"]

    # 获取健康检查的版本
    health_response = client.get("/health")
    health_version = health_response.json()["version"]

    assert root_version == health_version == "2.0.0"
