import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# MySQL 연결 URL 설정
# ✅ .env 파일 로드
load_dotenv()

# ✅ 환경 변수에서 DB URL 가져오기
DATABASE_URL = os.getenv("DATABASE_URL")

# SQLAlchemy 비동기 엔진 생성
engine = create_async_engine(DATABASE_URL, echo=True)

# 비동기 세션 팩토리 생성
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# FastAPI 의존성 주입을 위한 함수
async def get_db():
    session = AsyncSessionLocal()  # ✅ 세션을 직접 생성
    try:
        yield session  # ✅ FastAPI가 세션을 자동 관리하도록 유지
    finally:
        await session.close()  # ✅ finally에서 명확하게 세션 닫기
