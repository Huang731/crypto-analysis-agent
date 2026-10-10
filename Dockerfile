FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

COPY requirements.txt ./

# 沿用现有依赖配置，智谱 SDK 单独安装以避免 PyJWT 版本冲突。
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps 'zhipuai>=2.0,<3' \
    && python -m playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY api/ ./api/
COPY app/ ./app/
COPY prompts/ ./prompts/
COPY skills/ ./skills/
COPY static/ ./static/

EXPOSE 8003

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8003"]
