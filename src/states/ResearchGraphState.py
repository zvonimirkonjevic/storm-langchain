import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from models.Analyst import Analyst

class ResearchGraphState(TypedDict):
    topic: str 
    max_analysts: int 
    human_analyst_feedback: str 
    analysts: List[Analyst] 
    sections: Annotated[list, operator.add]
    introduction: str 
    content: str 
    conclusion: str
    final_report: str