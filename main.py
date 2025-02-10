# from fastapi import FastAPI
# from routers import ocr
# from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
# import asyncio
# import json

# app = FastAPI()
# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# app.include_router(ocr.router, prefix="/ocr", tags=["OCR"])


# # Kafka consumer와 producer 초기화
# async def get_consumer():
#     consumer = AIOKafkaConsumer(
#         'business_license_request',  # 구독할 토픽 이름
#         bootstrap_servers='kafka:9092',
#         group_id="business_license_group",
#     )
#     await consumer.start()
#     return consumer

# async def get_producer():
#     producer = AIOKafkaProducer(
#         bootstrap_servers='kafka:9092'
#     )
#     await producer.start()
#     return producer

# # Kafka 메시지 처리
# async def process_message(message):
#     # 메시지 처리 로직 (여기서 OCR 처리나 인증 등을 진행)
#     print(f"Received message: {message.value}")
    
#     # 메시지 처리 후 'business-license-response' 토픽에 결과 전송
#     producer = await get_producer()
#     response = {"status": "success", "data": "Processed"}
#     await producer.send_and_wait('business_license_response', value=json.dumps(response).encode())
#     await producer.stop()

# @app.on_event("startup")
# async def startup_event():
#     consumer = await get_consumer()
#     loop = asyncio.get_event_loop()

#     # Kafka 메시지 소비
#     async def consume():
#         async for message in consumer:
#             await process_message(message)

#     # 메시지 소비를 비동기로 실행
#     loop.create_task(consume())

# @app.on_event("shutdown")
# async def shutdown_event():
#     # 애플리케이션 종료 시 Kafka 소비자 종료
#     consumer = await get_consumer()
#     await consumer.stop()

# from fastapi import FastAPI
# from routers import ocr
# from services.kafka_service import KafkaService
# import asyncio

# app = FastAPI()

# # KafkaService 인스턴스화
# kafka_service = KafkaService(bootstrap_servers='kafka:9092', group_id="business_license_group")

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# app.include_router(ocr.router, prefix="/ocr", tags=["OCR"])

# @app.on_event("startup")
# async def startup_event():
#     # Kafka 메시지 소비를 비동기적으로 실행
#     loop = asyncio.get_event_loop()
#     loop.create_task(kafka_service.consume_messages('business_license_request'))

# @app.on_event("shutdown")
# async def shutdown_event():
#     # 애플리케이션 종료 시 Kafka 소비자 종료
#     await kafka_service.get_consumer('business_license_request').stop()

from fastapi import FastAPI
from services.kafka_consumer_service import KafkaConsumerService
from conf.database import get_db
import asyncio

app = FastAPI()

# ✅ Kafka Consumer 인스턴스 초기화 (DB는 startup_event에서 설정)
kafka_consumer_service = None

@app.on_event("startup")
async def startup_event():
    """FastAPI 시작 시 Kafka Consumer & Producer 실행"""
    global kafka_consumer_service

    # ✅ 비동기적으로 DB 세션 가져오기
    async for db_session in get_db():
        db = db_session
        break  # 첫 번째 AsyncSession만 사용

    # ✅ Kafka Consumer 서비스 초기화
    kafka_consumer_service = KafkaConsumerService(
        bootstrap_servers="kafka:9092", 
        group_id="business_license_group", 
        db=db  # ✅ 올바른 AsyncSession 전달
    )

    await kafka_consumer_service.start()  # ✅ Kafka Producer 시작
    loop = asyncio.get_event_loop()
    loop.create_task(kafka_consumer_service.start_consumer("business_license_request"))

@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI 종료 시 Kafka Consumer & Producer 정리"""
    if kafka_consumer_service:
        if kafka_consumer_service.consumer:
            await kafka_consumer_service.consumer.stop()
        await kafka_consumer_service.stop()  # ✅ Kafka Producer 종료