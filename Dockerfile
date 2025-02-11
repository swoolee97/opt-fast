# Python 3.10 slim 버전 사용
FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# pip 최신화 및 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face 모델 다운로드
RUN pip install huggingface_hub transformers

# 모델을 컨테이너 내부에 다운로드 (캐싱 적용)
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('swoolee97/opt-business-license-ocr').save_pretrained('/app/model')"

# 환경 변수 파일 복사
COPY .env /app/.env

# FastAPI 코드 복사
COPY . .

# 환경 변수 로드
ENV $(cat /app/.env | xargs)

# FastAPI 실행
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --reload"]
