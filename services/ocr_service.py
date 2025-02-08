import io
import cv2
import numpy as np
from fastapi import HTTPException
from services.document_aligner import scan_document
from services.google_cloud import run_ocr
from services.business_validator import validate_business_info
from services.semahtic_search import extract_colon_key_values
from services.semahtic_search import match_ocr_keys
from services.semahtic_search import *

async def process_ocr(file: bytes):
    """ OCR 실행 전에 이미지 보정 후, OCR API 호출 """
    # 이미지 변환: 바이트 데이터를 OpenCV 이미지로 변환
    image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "올바른 이미지 파일이 아닙니다."}

    # 이미지 보정 수행
    processed_image = scan_document(image)  # 여기서 image를 넘겨야 함

    if processed_image is None:
        return {"error": "문서 영역을 찾을 수 없습니다."}

    text = run_ocr(processed_image)

    extracted = extract_colon_key_values(text) # :이 들어간 데이터만 뽑은 후 dictionary형태로 변환
    
    result =  match_ocr_keys(extracted)

    return translate_keys(result)

    print(translated_result)

    # OCR 검증 수행
    validated_result = validate_business_info([translated_result])

    return validated_result

    # OCR 실행
    # return run_ocr(processed_image)
