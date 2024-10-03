from typing import List
from pydantic import BaseModel, Field
from models.Analyst import *

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations."
    )