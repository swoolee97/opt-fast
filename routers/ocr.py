from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from services.ocr_service import process_ocr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from conf.database import get_db
from models.gym import Gym
from services.business_validator import validate_business_info
from services.gym_finder import *

router = APIRouter()

@router.post("/business-license")
async def upload_business_license(
    file: UploadFile = File(...), 
    db: AsyncSession = Depends(get_db)  # ✅ FastAPI가 관리하는 db 세션을 그대로 사용
):
    # 파일 확장자 검증
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PNG, JPG, JPEG are allowed.")

    # 파일을 바이트로 읽기
    file_bytes = await file.read()

    # OCR 처리 호출
    ocr_result = await process_ocr(file_bytes)

    # 사업자등록 유효성 검증
    validated_result = validate_business_info([ocr_result])
    
    # 🔹 valid 값 확인
    valid_status = validated_result["data"][0].get("valid", "")

    if valid_status == "01":
        # ✅ 기존 db 세션을 그대로 전달 (추가 세션 생성 X)
        matched_gym = await find_most_similar_gym(ocr_result, db)

        return {"ocr_result": ocr_result, "matched_gym": matched_gym}
    
    elif valid_status == "02":
        raise HTTPException(status_code=400, detail="Invalid business registration: 폐업된 사업자입니다.")
    
    else:
        raise HTTPException(status_code=400, detail="Invalid business registration: 유효하지 않은 데이터입니다.")


# 📌 특정 ID의 Gym 조회 API
@router.get("/gyms/{gym_id}")
async def get_gym(gym_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Gym).where(Gym.id == gym_id))
    gym = result.scalars().first()
    
    if not gym:
        raise HTTPException(status_code=404, detail="Gym not found")
    
    return {
        "id": gym.id,
        "phone_number": gym.phone_number,
        "full_address": gym.full_address,
        "road_address": gym.road_address,
        "gym_name": gym.gym_name,
        "latitude": float(gym.latitude) if gym.latitude else None,
        "longitude": float(gym.longitude) if gym.longitude else None
    }