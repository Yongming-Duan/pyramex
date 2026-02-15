"""
数据库初始化脚本
创建PostgreSQL数据库表结构
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from datetime import datetime

Base = declarative_base()


class Spectrum(Base):
    """光谱数据表"""
    __tablename__ = 'spectra'

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    sample_id = Column(String(100))
    wavenumber_min = Column(Float)
    wavenumber_max = Column(Float)
    num_points = Column(Integer)
    preprocessing_status = Column(String(50), default='pending')
    qc_status = Column(String(50), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AnalysisJob(Base):
    """分析任务表"""
    __tablename__ = 'analysis_jobs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_type = Column(String(50), nullable=False)  # 'preprocess', 'qc', 'ml', 'report'
    status = Column(String(50), default='pending')  # 'pending', 'running', 'completed', 'failed'
    input_data = Column(Text)
    results = Column(Text)
    error_message = Column(Text)
    gpu_used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)


class MLModel(Base):
    """机器学习模型表"""
    __tablename__ = 'ml_models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50))  # 'rf', 'svm', 'nn'
    model_version = Column(String(50))
    accuracy = Column(Float)
    parameters = Column(Text)
    file_path = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Report(Base):
    """AI生成报告表"""
    __tablename__ = 'reports'

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, nullable=True)
    report_type = Column(String(50))
    content = Column(Text)
    llm_model = Column(String(50))
    generated_at = Column(DateTime, default=datetime.utcnow)


def init_database():
    """初始化数据库"""
    # 从环境变量获取数据库URL
    database_url = os.getenv(
        'DATABASE_URL',
        'postgresql://pyramex:pyramex123@pyramex-db:5432/pyramex'
    )

    print(f"连接数据库: {database_url}")

    # 创建引擎
    engine = create_engine(database_url, echo=True)

    # 创建所有表
    Base.metadata.create_all(engine)

    print("✅ 数据库表创建完成")

    # 创建初始数据
    with engine.connect() as conn:
        # 插入示例数据
        conn.execute(text("""
            INSERT INTO ml_models (model_name, model_type, model_version, is_active)
            VALUES ('baseline_rf', 'rf', '1.0.0', true)
            ON CONFLICT DO NOTHING
        """))
        conn.commit()

    print("✅ 初始数据插入完成")


if __name__ == "__main__":
    try:
        init_database()
        print("\n🎉 数据库初始化成功！")
    except Exception as e:
        print(f"\n❌ 数据库初始化失败: {e}")
        sys.exit(1)
