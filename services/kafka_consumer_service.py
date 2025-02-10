from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from sqlalchemy.ext.asyncio import AsyncSession
from services.ocr_service import process_ocr
from services.business_validator import validate_business_info
from services.gym_finder import find_most_similar_gym
import json
import base64
import requests
import logging
import aiohttp
from io import BytesIO

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaConsumerService:
    def __init__(self, bootstrap_servers, group_id, db):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.db = db
        self.consumer = None
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    async def start(self):
        """Kafka Producer 시작"""
        await self.producer.start()

    async def stop(self):
        """Kafka Producer 종료"""
        await self.producer.stop()

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
                await self.process_message(message)
        finally:
            await self.consumer.stop()

    async def process_message(self, message):
        """Kafka 메시지 처리 로직"""
        try:
            logger.info("📩 메시지 수신 완료")

            # JSON 파싱
            data = json.loads(message.value.decode("utf-8"))
            logger.info("✅ Kafka 메시지 파싱 완료")

            # S3에서 이미지 다운로드
            image_url = data["path"]
            user_id = int(data["id"])

            logger.info(f"🌍 S3에서 이미지 다운로드 시작 - URL: {image_url}, 사용자 ID: {user_id}")

            # 이미지 다운로드
            image_data = await self.download_image_from_s3(image_url)

            if not image_data:
                logger.error("❌ S3 이미지 다운로드 실패 - 데이터가 None입니다.")
                return

            logger.info(f"✅ S3 이미지 다운로드 완료 - 크기: {len(image_data)} bytes, 타입: {type(image_data)}")

            # Base64 디코딩 (불필요한 경우 제거 가능)
            try:
                file_bytes = BytesIO(image_data)
                logger.info(f"📂 디코딩 완료 타입: {type(file_bytes)}")
            except Exception as e:
                logger.error(f"❌ Base64 디코딩 실패: {e}")
                return

            # OCR 처리
            logger.info("🔍 OCR 처리 시작")
            ocr_result = await process_ocr(file_bytes)
            logger.info(f"🔍 OCR 처리 완료 - 결과: {ocr_result}")


            # 사업자등록 유효성 검증
            validated_result = validate_business_info([ocr_result])
            logger.info(f"✅ 사업자등록 검증 결과: {validated_result}")

            # 유효성 확인 및 로직 실행
            valid_status = validated_result["data"][0].get("valid", "")

            logger.info(f"📜 유효성 상태 코드: {valid_status}")

            if valid_status == "01":
                logger.info(f"✅ Gym 정보 찾기 시작! ")
                # 유효한 경우: Gym 정보 매칭
                # matched_gym = await find_most_similar_gym(ocr_result, self.db)
                matched_gym = await find_most_similar_gym(ocr_result, self.db)  # ✅ AsyncSession을 직접 사용


                if matched_gym:
                    gym_id = matched_gym.id  # Gym ID 설정
                logger.info(f"✅ 매칭된 Gym 정보: {matched_gym}")
                message = "트레이너 등록에 성공했습니다"

            elif valid_status == "02":
                # 폐업된 사업자 처리
                logger.warning("⚠️ 폐업된 사업자입니다.")
                message = "폐업한 사업자 입니다"
            else:
                # 유효하지 않은 데이터 처리
                logger.warning("⚠️ 유효하지 않은 사업자 등록 정보입니다.")
                message = "유효하지 않은 사업자 등록 정보입니다"
                        # ✅ Kafka 메시지 전송 (business_license_response)
            if matched_gym == None:
                gym_id = None
            response_message = {
                "user_id": user_id,
                "gym_id": gym_id,  # 매칭된 Gym이 없으면 None
                "message" : message
            }

            await self.send_kafka_message("business_license_response", response_message)
            logger.info(f"📤 Kafka 메시지 전송 완료: {response_message}")

        except Exception as e:
            logger.error(f"❌ Kafka 메시지 처리 중 오류 발생: {e}")

    async def download_image_from_s3(self, url: str):
        print(1)
        async with aiohttp.ClientSession() as session:
            print(2)
            async with session.get(url) as response:
                print(3)
                if response.status == 200:
                    print(4)
                    image_data = await response.read()
                    print(5)
                    return image_data
                else:
                    raise Exception(f"Failed to download image: {response.status}")
                
    async def send_kafka_message(self, topic: str, message: dict):
        """Kafka 메시지 전송"""
        try:
            await self.producer.send_and_wait(topic, message)
            logger.info(f"📤 Kafka 메시지 전송 성공: {message}")
        except Exception as e:
            logger.error(f"❌ Kafka 메시지 전송 실패: {e}")