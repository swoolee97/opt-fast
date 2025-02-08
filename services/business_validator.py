import os
import requests
from dotenv import load_dotenv

# ✅ 환경 변수 로드
load_dotenv()

# ✅ API 기본 정보
BASE_URL = "https://api.odcloud.kr/api/nts-businessman/v1"
SERVICE_KEY = os.getenv("SERVICE_KEY")  # ✅ 환경 변수에서 API 키 가져오기

# 사업자등록정보 진위확인 API 호출 함수
def validate_business_info(business_list):
    url = f"{BASE_URL}/validate?serviceKey={SERVICE_KEY}&returnType=JSON"
    
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "businesses": business_list  # 리스트 형식
    }

    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"HTTP {response.status_code}", "message": response.text}

