from aiokafka import AIOKafkaConsumer
from sqlalchemy.ext.asyncio import AsyncSession
from services.ocr_service import process_ocr
from services.business_validator import validate_business_info
from services.gym_finder import find_most_similar_gym
import json
import asyncio
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaConsumerService:
    def __init__(self, bootstrap_servers: str, group_id: str, db: AsyncSession):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.db = db
        self.consumer = None

    async def start_consumer(self, topic: str):
        """Kafka Consumer 시작 및 메시지 처리"""
        self.consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
        )
        await self.consumer.start()

        logger.info(f"🚀 Kafka Consumer started and subscribed to topic: {topic}")

        try:
            async for message in self.consumer:
                print('#####')
                await self.process_message(message)
        finally:
            await self.consumer.stop()

    async def process_message(self, message):
        """Kafka 메시지 처리 로직"""
        try:
            print(1111)
            data = json.loads(message.value.decode("utf-8"))

            # Base64 디코딩하여 `file_bytes` 변환
            file_bytes = base64.b64decode(data["file"])
            id = int(data["id"])

            # OCR 처리
            print('@@@@@@@@@@@@@@@@@@ translated 결과 @@@@@@@@@@@@@@@@@@@')
            ocr_result = await process_ocr(file_bytes)
            # ocr_result = await asyncio.to_thread(process_ocr, file_bytes)
            print('@@@@@@@@@@@@@@@@@@ translated 결과 @@@@@@@@@@@@@@@@@@@')

            # 사업자등록 유효성 검증
            validated_result = validate_business_info([ocr_result])
            print('@@@@@@@@@@@@@@@@@@ 유효성 검증 결과 @@@@@@@@@@@@@@@@@@@')
            print(validated_result)
            print('@@@@@@@@@@@@@@@@@@ 유효성 검증 결과 @@@@@@@@@@@@@@@@@@@')
            # 유효성 확인 및 로직 실행
            valid_status = validated_result["data"][0].get("valid", "")
            print('@@@@@@@@@@@@@@@@@@ valid status @@@@@@@@@@@@@@@@@@@')
            print(valid_status)
            print('@@@@@@@@@@@@@@@@@@ valid status @@@@@@@@@@@@@@@@@@@')

            if valid_status == "01":
                # 유효한 경우: Gym 정보 매칭
                matched_gym = await find_most_similar_gym(ocr_result, self.db)

                # 처리 결과를 Kafka로 전송 (추가 구현 가능)
                print({"ocr_result": ocr_result, "matched_gym": matched_gym})

            elif valid_status == "02":
                # 폐업된 사업자 처리
                print("Invalid business registration: 폐업된 사업자입니다.")
            
            else:
                # 유효하지 않은 데이터 처리
                print("Invalid business registration: 유효하지 않은 데이터입니다.")

        except Exception as e:
            print(f"Error processing Kafka message: {e}")
