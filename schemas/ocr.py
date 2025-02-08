from pydantic import BaseModel

class OcrResponse(BaseModel):
    business_name: str
    business_number: str
    owner_name: str
    registration_date: str
    address: str
    text: str
