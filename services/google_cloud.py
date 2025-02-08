import os
from dotenv import load_dotenv
from google.cloud import documentai_v1beta3 as documentai
import cv2

# ✅ 환경 변수 로드
load_dotenv()

# ✅ GCP 프로젝트 정보 설정
PROJECT_ID = os.getenv("PROJECT_ID")  # ✅ 환경 변수에서 GCP 프로젝트 ID 가져오기
LOCATION = os.getenv("LOCATION")  # ✅ 환경 변수에서 LOCATION 가져오기
PROCESSOR_ID = os.getenv("PROCESSOR_ID")  # ✅ 환경 변수에서 PROCESSOR_ID 가져오기

# Document AI 클라이언트 생성
client = documentai.DocumentProcessorServiceClient()

def run_ocr(image):
    """ Google Document AI OCR 호출 (이미지 데이터를 직접 입력) """
    if image is None:
        print("🚨 OCR 실패: 이미지가 None입니다.")
        return None

    # OpenCV 이미지(Numpy 배열) → PNG 바이트 배열로 변환
    _, image_bytes = cv2.imencode(".png", image)  # PNG 형식으로 인코딩
    image_content = image_bytes.tobytes()  # 바이트 변환

    # Document AI 요청 생성
    request = documentai.ProcessRequest(
        name=f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}",
        raw_document=documentai.RawDocument(content=image_content, mime_type="image/png"),
    )

    # OCR 실행
    result = client.process_document(request=request)
    print("📝 OCR 결과:")
    print(result.document.text)

    return result.document.text