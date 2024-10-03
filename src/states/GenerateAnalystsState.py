from typing import List
from typing_extensions import TypedDict
from models.Analyst import *

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]