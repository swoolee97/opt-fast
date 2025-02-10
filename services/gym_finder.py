from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from Levenshtein import ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models.gym import Gym

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
async def find_most_similar_gym(ocr_result: dict, db: AsyncSession):
    ocr_name = ocr_result["b_nm"]
    ocr_address = ocr_result["business_address"]

    # ✅ 트랜잭션 명확하게 관리
    # async with db.begin():
    query = select(Gym).where(
        (Gym.gym_name.ilike(f"%{ocr_name}%")) | (Gym.full_address.ilike(f"%{ocr_address}%"))
    )
    result = await db.execute(query)
    
    gym_candidates = result.scalars().all()

    best_match = None
    best_score = 0.0

    for gym in gym_candidates:
        name_similarity, address_similarity = calculate_similarity(
            ocr_name, gym.gym_name, ocr_address, gym.full_address
        )

        final_score = (name_similarity + address_similarity) / 2

        if final_score > best_score and final_score >= 0.8:
            best_score = final_score
            best_match = gym

    return best_match  # ✅ 가장 유사한 Gym 하나 반환
