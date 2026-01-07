from pydantic import BaseModel
from typing import Optional

class RecognizeResponse(BaseModel):
    employee_id: Optional[int]
    similarity: Optional[float]
    recognized: bool
