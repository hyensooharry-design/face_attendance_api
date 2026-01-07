FROM python:3.11-slim

# OpenCV/torch 계열에서 종종 필요한 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements 먼저 복사(캐시 활용)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 소스 복사
COPY . /app

# 기본 포트
EXPOSE 8000

# 프로덕션 실행 (reload 없음)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
