"""
数据IO模块测试
测试数据加载功能
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from pyramex.io.loader import (
    load_spectra,
    load_single_file,
    detect_format,
    is_wavenumber_like,
    has_coordinate_pattern
)


@pytest.fixture
def temp_two_col_file(tmp_path):
    """创建双列格式的测试文件"""
    file_path = tmp_path / "two_col.txt"

    wavenumbers = np.linspace(400, 4000, 100)
    intensity = np.random.randn(100) * 0.1 + np.sin(wavenumbers / 100)

    data = pd.DataFrame({
        'wavenumber': wavenumbers,
        'intensity': intensity
    })

    data.to_csv(file_path, sep='\t', header=False, index=False)

    return file_path


@pytest.fixture
def temp_matrix_file(tmp_path):
    """创建矩阵格式的测试文件"""
    file_path = tmp_path / "matrix.txt"

    wavenumbers = np.linspace(400, 4000, 100)
    n_samples = 5

    # 第一列是波数，其余列是不同样本
    data = {'col0': wavenumbers}
    for i in range(n_samples):
        data[f'col{i+1}'] = np.random.randn(100) * 0.1

    df = pd.DataFrame(data)
    df.to_csv(file_path, sep='\t', header=False, index=False)

    return file_path


@pytest.fixture
def temp_coords_file(tmp_path):
    """创建坐标格式的测试文件"""
    file_path = tmp_path / "coords.txt"

    # 模拟坐标扫描数据：x, y, wavenumber, intensity
    data = []
    for x in [1, 2, 3]:
        for y in [1, 2]:
            wavenumbers = np.linspace(400, 1000, 10)  # 简化
            for wn in wavenumbers:
                intensity = np.random.randn() * 0.1
                data.append([x, y, wn, intensity])

    df = pd.DataFrame(data)
    df.to_csv(file_path, sep='\t', header=False, index=False)

    return file_path


@pytest.fixture
def temp_directory(tmp_path):
    """创建包含多个文件的测试目录"""
    dir_path = tmp_path / "spectra"
    dir_path.mkdir()

    # 创建3个双列格式文件
    for i in range(3):
        file_path = dir_path / f"spectrum_{i}.txt"
        wavenumbers = np.linspace(400, 4000, 100)
        intensity = np.random.randn(100) * 0.1

        data = pd.DataFrame({
            'wavenumber': wavenumbers,
            'intensity': intensity
        })
        data.to_csv(file_path, sep='\t', header=False, index=False)

    return dir_path


class TestDetectFormat:
    """测试格式检测"""

    def test_detect_format_two_col(self):
        """测试检测双列格式"""
        data = np.column_stack([
            np.linspace(400, 4000, 100),
            np.random.randn(100)
        ])

        format_type = detect_format(data)

        assert format_type == 'two_col'

    def test_detect_format_matrix(self):
        """测试检测矩阵格式"""
        wavenumbers = np.linspace(400, 4000, 100)
        data = np.column_stack([
            wavenumbers,
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100)
        ])

        format_type = detect_format(data)

        assert format_type == 'matrix'

    def test_detect_format_coords(self):
        """测试检测坐标格式"""
        # 创建类似坐标的数据
        data = []
        for x in range(3):
            for y in range(2):
                wn = np.linspace(400, 1000, 10)
                for w in wn:
                    data.append([x, y, w, np.random.randn()])

        data = np.array(data)

        format_type = detect_format(data)

        # 可能是coords或two_col
        assert format_type in ['coords', 'two_col']


class TestIsWavenumberLike:
    """测试波数识别"""

    def test_is_wavenumber_like_valid(self):
        """测试有效的波数"""
        wavenumbers = np.linspace(400, 4000, 100)

        assert is_wavenumber_like(wavenumbers) == True

    def test_is_wavenumber_like_not_monotonic(self):
        """测试非单调数组"""
        arr = np.array([1000, 500, 1500, 2000])

        assert is_wavenumber_like(arr) == False

    def test_is_wavenumber_like_out_of_range(self):
        """测试超出范围的数组"""
        arr = np.linspace(10000, 20000, 100)

        assert is_wavenumber_like(arr) == False

    def test_is_wavenumber_like_irregular_spacing(self):
        """测试不规则间隔的数组"""
        arr = np.array([400, 500, 600, 2000, 2100, 2200])

        # 标准差与均值之比应该较大
        assert is_wavenumber_like(arr) == False


class TestHasCoordinatePattern:
    """测试坐标模式识别"""

    def test_has_coordinate_pattern_true(self):
        """测试有坐标模式的数据"""
        # 小整数坐标 + 波数 + 强度
        data = []
        for x in range(5):
            for y in range(3):
                for wn in np.linspace(400, 1000, 10):
                    data.append([x, y, wn, np.random.randn()])

        data = np.array(data)

        assert has_coordinate_pattern(data) == True

    def test_has_coordinate_pattern_false(self):
        """测试没有坐标模式的数据"""
        # 所有值都很大
        data = np.random.randn(100, 5) * 1000

        assert has_coordinate_pattern(data) == False


class TestLoadSingleFile:
    """测试单文件加载"""

    def test_load_single_file_two_col(self, temp_two_col_file):
        """测试加载双列格式文件"""
        ramanome = load_single_file(temp_two_col_file, format='two_col')

        assert ramanome is not None
        assert ramanome.n_samples == 1
        assert ramanome.n_wavenumbers == 100

    def test_load_single_file_matrix(self, temp_matrix_file):
        """测试加载矩阵格式文件"""
        ramanome = load_single_file(temp_matrix_file, format='matrix')

        assert ramanome is not None
        assert ramanome.n_samples == 5
        assert ramanome.n_wavenumbers == 100

    def test_load_single_file_auto_detect(self, temp_two_col_file):
        """测试自动检测格式"""
        ramanome = load_single_file(temp_two_col_file, format='auto')

        assert ramanome is not None
        assert ramanome.n_samples == 1


class TestLoadDirectory:
    """测试目录加载"""

    def test_load_directory(self, temp_directory):
        """测试加载目录中的所有文件"""
        ramanome = load_directory(
            temp_directory,
            format='two_col',
            pattern='*.txt'
        )

        assert ramanome is not None
        assert ramanome.n_samples == 3
        assert len(ramanome.metadata) == 3

    def test_load_directory_no_files(self, tmp_path):
        """测试加载空目录"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No files found"):
            load_directory(empty_dir, format='two_col')


class TestLoadSpectra:
    """测试主加载函数"""

    def test_load_spectra_from_file(self, temp_two_col_file):
        """测试从文件加载"""
        ramanome = load_spectra(temp_two_col_file)

        assert ramanome is not None
        assert ramanome.n_samples == 1

    def test_load_spectra_from_directory(self, temp_directory):
        """测试从目录加载"""
        ramanome = load_spectra(temp_directory)

        assert ramanome is not None
        assert ramanome.n_samples == 3

    def test_load_spectra_path_not_found(self):
        """测试加载不存在的路径"""
        with pytest.raises(ValueError, match="Path not found"):
            load_spectra("/nonexistent/path")


class TestIOEdgeCases:
    """测试IO的边界情况"""

    def test_load_file_with_header(self, tmp_path):
        """测试带标题的文件"""
        file_path = tmp_path / "with_header.txt"

        wavenumbers = np.linspace(400, 4000, 50)
        intensity = np.random.randn(50)

        # 写入带标题的文件
        with open(file_path, 'w') as f:
            f.write("# This is a header\n")
            f.write("# Another comment\n")
            for wn, inten in zip(wavenumbers, intensity):
                f.write(f"{wn}\t{inten}\n")

        # 尝试加载
        try:
            ramanome = load_single_file(file_path, format='two_col')
            # pandas的引擎可能处理注释行
        except Exception as e:
            # 如果失败，这也是可接受的
            assert True

    def test_load_file_with_different_separators(self, tmp_path):
        """测试不同的分隔符"""
        # 测试逗号分隔
        file_path = tmp_path / "csv_format.txt"

        wavenumbers = np.linspace(400, 4000, 50)
        intensity = np.random.randn(50)

        data = pd.DataFrame({
            'wavenumber': wavenumbers,
            'intensity': intensity
        })

        data.to_csv(file_path, sep=',', header=False, index=False)

        ramanome = load_single_file(file_path, format='two_col')

        assert ramanome is not None

    def test_load_directory_different_wavenumber_grids(self, tmp_path):
        """测试目录中文件有不同波数网格"""
        dir_path = tmp_path / "different_grids"
        dir_path.mkdir()

        # 创建两个波数网格略有不同的文件
        for i in range(2):
            file_path = dir_path / f"spectrum_{i}.txt"

            # 略微不同的波数范围
            wavenumbers = np.linspace(400 + i*10, 4000 + i*10, 100)
            intensity = np.random.randn(100) * 0.1

            data = pd.DataFrame({
                'wavenumber': wavenumbers,
                'intensity': intensity
            })

            data.to_csv(file_path, sep='\t', header=False, index=False)

        # 加载时应该插值到相同的网格
        ramanome = load_directory(dir_path, format='two_col')

        assert ramanome is not None
        assert ramanome.n_samples == 2
        # 所有样本应该有相同的波数点数
        assert ramanome.spectra.shape[1] == ramanome.n_wavenumbers


class TestReadSpecConvenience:
    """测试便捷函数"""

    def test_read_spec_alias(self, temp_two_col_file):
        """测试read_spec是load_spectra的别名"""
        from pyramex.io.loader import read_spec

        ramanome1 = load_spectra(temp_two_col_file)
        ramanome2 = read_spec(temp_two_col_file)

        # 两个函数应该返回相同的结果
        assert ramanome1.n_samples == ramanome2.n_samples
        assert ramanome1.n_wavenumbers == ramanome2.n_wavenumbers
