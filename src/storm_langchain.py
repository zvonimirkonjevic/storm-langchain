from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import Send
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from states import ResearchGraphState
from langchain_openai import ChatOpenAI
import interview_graph
import create_analysts

class StormGraph:
    def __init__(self):

        self.__llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.__report_writer_instructions = """
        You are a technical writer creating a report on this overall topic: 

        {topic}
            
        You have a team of analysts. Each analyst has done two things: 

        1. They conducted an interview with an expert on a specific sub-topic.
        2. They write up their finding into a memo.

        Your task: 

        1. You will be given a collection of memos from your analysts.
        2. Think carefully about the insights from each memo.
        3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
        4. Summarize the central points in each memo into a cohesive single narrative.

        To format your report:
        
        1. Use markdown formatting. 
        2. Include no pre-amble for the report.
        3. Use no sub-heading. 
        4. Start your report with a single title header: ## Insights
        5. Do not mention any analyst names in your report.
        6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
        7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
        8. List your sources in order and do not repeat.

        [1] Source 1
        [2] Source 2

        Here are the memos from your analysts to build your report from: 

        {context}"""

        intro_conclusion_instructions = """
        You are a technical writer finishing a report on {topic}

        You will be given all of the sections of the report.

        You job is to write a crisp and compelling introduction or conclusion section.

        The user will instruct you whether to write the introduction or conclusion.

        Include no pre-amble for either section.

        Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

        Use markdown formatting. 

        For your introduction, create a compelling title and use the # header for the title.

        For your introduction, use ## Introduction as the section header. 

        For your conclusion, use ## Conclusion as the section header.

        Here are the sections to reflect on for writing: {formatted_str_sections}"""

        def initiate_all_interviews(state: ResearchGraphState):
            """
            This is the "map" step where we run each interview sub-graph using Send API
            """

            human_analyst_feedback=state.get('human_analyst_feedback')
            if human_analyst_feedback:
                return "create_analysts"
            
            else:
                topic = state["topic"]
                return [Send("conduct_interview", {"analyst": analyst,"messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]}) for analyst in state["analysts"]]
            
        def write_report(state: ResearchGraphState):
            sections = state["sections"]
            topic = state["topic"]

            formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
            
            system_message = self.__report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
            report = self.__llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")])
            
            return {"content": report.content}
        
        def write_introduction(state: ResearchGraphState):
            sections = state["sections"]
            topic = state["topic"]

            formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

            instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
            intro = self.__llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 

            return {"introduction": intro.content}
        
        def write_conclusion(state: ResearchGraphState):
            sections = state["sections"]
            topic = state["topic"]

            formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
            
            instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
            conclusion = self.__llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
            
            return {"conclusion": conclusion.content}
        
        def finalize_report(state: ResearchGraphState):
            """ 
            The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion 
            """
            
            content = state["content"]
            if content.startswith("## Insights"):
                content = content.strip("## Insights")
            if "## Sources" in content:
                try:
                    content, sources = content.split("\n## Sources\n")
                except:
                    sources = None
            else:
                sources = None

            final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
            if sources is not None:
                final_report += "\n\n## Sources\n" + sources
                
            return {"final_report": final_report}

    def __build_research_graph(self):
        
        builder = StateGraph(ResearchGraphState)
        builder.add_node("create_analysts", create_analysts)
        builder.add_node("human_feedback", self.__human_feedback)
        builder.add_node("conduct_interview", self.__interview_graph)
        builder.add_node("write_report", self.__write_report)
        builder.add_node("write_introduction", self.__write_introduction)
        builder.add_node("write_conclusion", self.__write_conclusion)
        builder.add_node("finalize_report", self.__finalize_report)

        builder.add_edge(START, "create_analysts")
        builder.add_edge("create_analysts", "human_feedback")
        builder.add_conditional_edges("human_feedback", self.__initiate_all_interviews, ["create_analysts", "conduct_interview"])
        builder.add_edge("conduct_interview", "write_report")
        builder.add_edge("conduct_interview", "write_introduction")
        builder.add_edge("conduct_interview", "write_conclusion")
        builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
        builder.add_edge("finalize_report", END)

        memory = SqliteSaver.from_conn_string(":memory:")
        return builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

    def invoke(self, question, max_analysts=3):
        thread = {"configurable": {"thread_id": "1"}}

        graph = self.__build_research_graph()

        graph.invoke({"topic": question, "max_analysts": max_analysts, "human_analyst_feedback": ""}, thread)

        user_approval = input("Are you satisfied with generated analysts (yes/no): ")

        if user_approval.lower() == "yes":
            graph.invoke(None, thread)  
        else:
            user_feedback = input("How would you improve analyst creation: ")
            graph.invoke({"topic": question + "," + user_feedback, "max_analysts": max_analysts, "human_analyst_feedback": ""}, thread)
            graph.invoke(None, thread)
        
        final_state = graph.get_state(thread)
        report = final_state.values.get('final_report')
        return report

        

        