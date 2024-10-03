import operator
from typing import Annotated
from langgraph.graph import MessagesState
from models.Analyst import *

class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: str