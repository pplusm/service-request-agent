# 使用与项目约束一致的 Python 3.11 运行时。
FROM python:3.11-slim

# 保持容器日志实时输出；默认使用不需要外部 API 的演示模型。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LLM_PROVIDER=demo

WORKDIR /app

# 只复制运行所需的源代码、场景配置和演示知识，避免把本地数据库或密钥带入镜像。
COPY pyproject.toml README.md ./
COPY app ./app
COPY scenarios ./scenarios
COPY data/scenic_service/knowledge ./data/scenic_service/knowledge

# 安装项目及其运行依赖。开发测试依赖不放入生产演示镜像。
RUN pip install --no-cache-dir .

EXPOSE 8000 8501

# Compose 中的 Streamlit 服务会覆盖此默认命令。
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
