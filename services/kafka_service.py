from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import asyncio
import json

class KafkaService:
    def __init__(self, bootstrap_servers: str, group_id: str):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id

    async def get_consumer(self, topic: str) -> AIOKafkaConsumer:
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
        )
        await consumer.start()
        return consumer

    async def get_producer(self) -> AIOKafkaProducer:
        producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
        await producer.start()
        return producer

    async def send_message(self, topic: str, message: dict):
        producer = await self.get_producer()
        await producer.send_and_wait(topic, value=json.dumps(message).encode())
        await producer.stop()

    async def consume_messages(self, topic: str):
        consumer = await self.get_consumer(topic)
        async for message in consumer:
            print(f"Received message: {message.value.decode()}")
            # 메시지 처리 로직
            await consumer.stop()
