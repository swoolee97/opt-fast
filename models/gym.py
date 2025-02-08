from sqlalchemy import Column, Integer, String, Text, DECIMAL
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Gym(Base):
    __tablename__ = "gym"

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone_number = Column(String(20), nullable=True)
    full_address = Column(Text, nullable=True)
    road_address = Column(Text, nullable=True)
    gym_name = Column(String(100), nullable=True)
    latitude = Column(DECIMAL(10, 7), nullable=True)
    longitude = Column(DECIMAL(10, 7), nullable=True)
