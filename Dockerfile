FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치 (libGL 관련)
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# pip 최신화 및 의존성 설치
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 환경 변수 파일 복사
COPY .env /app/.env

# FastAPI 코드 복사
COPY . .

# 환경 변수 로드
ENV $(cat /app/.env | xargs)

# FastAPI 실행
# CMD ["uvicorn", "main:app", "--host", "${UVICORN_HOST}", "--port", "${UVICORN_PORT}", "--reload"]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --reload"]
