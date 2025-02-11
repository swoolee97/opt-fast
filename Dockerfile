# Python 3.10 slim 버전 사용
FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# pip 최신화 및 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face 관련 패키지 설치
RUN pip install huggingface_hub transformers sentence-transformers

# 환경 변수 파일 복사
COPY .env /app/.env

# 환경 변수 로드 (Hugging Face 토큰 포함)
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Hugging Face 로그인 및 모델 다운로드
RUN python -c "from huggingface_hub import login; login('${HUGGINGFACE_TOKEN}')"
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('swoolee97/opt-business-license-ocr', use_auth_token='${HUGGINGFACE_TOKEN}'); model.save_pretrained('/app/model')"

# FastAPI 코드 복사
COPY . .

# FastAPI 실행
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --reload"]
