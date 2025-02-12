from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from Levenshtein import ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models.gym import Gym
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Jaccard 유사도 계산 함수
def jaccard_similarity(a: str, b: str) -> float:
    set_a, set_b = set(a.split()), set(b.split())
    return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0.0

# ✅ Cosine 유사도 계산 함수
def cosine_similarity_score(text1: str, text2: str) -> float:
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vectorizer[0], vectorizer[1])[0][0]

# ✅ 주소 & 상호명 유사도 비교 함수
def calculate_similarity(ocr_name: str, db_name: str, ocr_address: str, db_address: str):
    # 1️⃣ 상호명 유사도 계산 (Levenshtein + Cosine)
    name_levenshtein = ratio(ocr_name, db_name)
    name_cosine = cosine_similarity_score(ocr_name, db_name)
    name_similarity = 0.5 * name_levenshtein + 0.5 * name_cosine

    # 2️⃣ 주소 유사도 계산 (Levenshtein + Jaccard)
    address_levenshtein = ratio(ocr_address, db_address)
    address_jaccard = jaccard_similarity(ocr_address, db_address)
    address_similarity = 0.6 * address_levenshtein + 0.4 * address_jaccard

    return name_similarity, address_similarity

# ✅ 가장 유사한 Gym 데이터 하나만 반환하는 함수
async def find_most_similar_gym(ocr_result: dict, db: AsyncSession):  # ✅ db는 AsyncSession 타입
    ocr_name = ocr_result["b_nm"]
    ocr_address = ocr_result["business_address"]

    # ✅ 트랜잭션 명확하게 관리
    query = select(Gym).where(
        (Gym.gym_name.ilike(f"%{ocr_name}%")) | (Gym.full_address.ilike(f"%{ocr_address}%"))
    )
    # query = select(Gym)
    result = await db.execute(query)

    # fetchall()을 먼저 저장
    rows = result.fetchall()
    # logger.info(f"@@@@@@@@@@@@@@@@ 결과 @@@@@@@@@@@@@@@@")
    # logger.info(rows)
    # logger.info(f"@@@@@@@@@@@@@@@@ 결과 @@@@@@@@@@@@@@@@")

    # scalars()가 아닌, 직접 리스트 변환
    gym_candidates = [row[0] for row in rows]

    logger.info(f"🏋️‍♂️ gym_candidates 개수: {len(gym_candidates)}")

    best_score = 0
    best_match = None

    for gym in gym_candidates:
        # logger.info(f"🔍 현재 비교 대상: {gym.gym_name} ({gym.road_address})")

        name_similarity, address_similarity = calculate_similarity(
            ocr_name, gym.gym_name, ocr_address, gym.road_address
        )

        # logger.info(f"📊 유사도 결과 - 이름: {name_similarity:.2f}, 주소: {address_similarity:.2f}")

        final_score = (name_similarity + address_similarity) / 2
        # logger.info(f"⚖️ 최종 점수: {final_score:.2f}")

        if final_score > best_score and final_score >= 0.75:
            # logger.info(f"✅ 새로운 최적 매칭 발견! {gym.gym_name} (점수: {final_score:.2f})")
            best_score = final_score
            best_match = gym

    logger.info(f"🎯 최종 선택된 매칭: {best_match.gym_name if best_match else '없음'}")
    return best_match
