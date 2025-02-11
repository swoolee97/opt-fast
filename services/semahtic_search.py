from sentence_transformers import SentenceTransformer, util
import torch
import re

def extract_colon_key_values(text: str):
    """OCR 결과에서 ':'이 포함된 텍스트만 추출하여 딕셔너리로 변환"""
    result_dict = {}

    for line in text.split("\n"):
        if ":" in line:
            key, value = map(str.strip, line.split(":", 1))  # ':'을 기준으로 첫 번째만 분리
            result_dict[key] = value

    return result_dict


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('swoolee97/opt-business-license-ocr', device=device)
# 한국어 -> 영어 Key 매핑
KEY_TRANSLATION = {
    "등록번호": "b_no",
    "상호": "b_nm",
    "성명": "p_nm",
    "사업장소재지": "business_address",
    "생년월일": "date_of_birth",
    "개업연월일": "start_dt"
}

def match_ocr_keys(ocr_keys: dict, target_keys: list = ["등록번호", "상호", "성명", "사업장소재지", "생년월일", "개업연월일"], threshold: float = 0.85) -> dict:
    """
    OCR 결과에서 추출된 키들을 타겟 키와 매칭.

    :param ocr_keys: OCR 결과에서 추출된 키-값 딕셔너리
    :param target_keys: 우리가 원하는 정확한 키 리스트
    :param threshold: Semantic Search에서 매칭을 인정하는 임계값 (기본값 0.85)
    :return: 타겟 키에 대한 최종 매칭된 OCR 결과 딕셔너리
    """
    matched_results = {}

    # Step 1: 완전 일치(Exact Match) 체크
    for key in target_keys:
        if key in ocr_keys:
            matched_results[key] = ocr_keys[key]  # 완전히 일치하면 바로 저장

    # Step 2: Semantic Search 적용 (완전 일치한 값은 제외하고 나머지 처리)
    remaining_target_keys = [key for key in target_keys if key not in matched_results]
    print(1)
    if remaining_target_keys:
        print(2)
        ocr_key_list = list(ocr_keys.keys())  # OCR 결과에서 추출된 키 리스트
        print(3)
        ocr_embeddings = model.encode(ocr_key_list, convert_to_tensor=True)
        print(4)
        target_embeddings = model.encode(remaining_target_keys, convert_to_tensor=True)
        print(5)
        # 유사도 계산
        cosine_scores = util.pytorch_cos_sim(target_embeddings, ocr_embeddings)
        print(6)
        for i, target_key in enumerate(remaining_target_keys):
            best_match_idx = cosine_scores[i].argmax().item()
            print(7)
            best_match_score = cosine_scores[i][best_match_idx].item()
            print(8)
            if best_match_score >= threshold:
                matched_key = ocr_key_list[best_match_idx]
                matched_results[target_key] = ocr_keys[matched_key]
                print(9)
            else:
                matched_results[target_key] = None  # 유사도가 너무 낮으면 매칭 안 함
                print(10)
    print(22222)
    return matched_results

def translate_keys(matched_results: dict) -> dict:
    print('@@@@@@@@@@@@@ translated_keys 시작 @@@@@@@@@@@@@')
    """
    matched_results의 한글 키를 영어로 변환하고, 키별 값에 규칙을 적용.

    :param matched_results: 매칭된 OCR 결과 딕셔너리
    :return: 키가 영어로 변환되고 규칙이 적용된 딕셔너리
    """
    def clean_b_no(value: str) -> str:
        """등록번호(b_no)는 숫자만 남김"""
        return re.sub(r"[^0-9]", "", value)
    print('11111111111111111111111111111111')
    def format_date(value: str) -> str:
        """날짜 형식을 YYYYMMDD로 변환 (하이픈 제거)"""
        return re.sub(r"[^0-9]", "", value)  # 숫자만 남김

    def format_text(value: str) -> str:
        """텍스트 값의 앞뒤 공백 제거"""
        return value.strip()

    # 키별 변환 로직 적용
    print('222222222222222222222222222222')
    translated_results = {}
    for k, v in matched_results.items():
        key = KEY_TRANSLATION.get(k, k)  # 한글 키를 영어 키로 변환
        value = v.strip()  # 값의 앞뒤 공백 제거

        # 키별로 값 변환 로직 적용
        if key == "b_no":
            value = clean_b_no(value)
        elif key in ["start_dt", "date_of_birth"]:
            value = format_date(value)
        else:
            value = format_text(value)

        translated_results[key] = value
    print('333333333333333333333333333333')
    return translated_results