import os
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import get_buffer_string
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from models import SearchQuery
from models import InterviewState
from langchain_openai import ChatOpenAI

class InterviewGraph:
    def __init__(self):
        self.__llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.__tavily_search = TavilySearchResults(max_results=3)

        self.__question_instructions = """
        You are an analyst tasked with interviewing an expert to learn about a specific topic. 

        Your goal is to gain detailed and tehnical insights related to your topic.

        1. Actionable: Focus on insights that engineers can implement or reference in real-world scenarios.
                
        2. Detailed: Avoid generalities; dig into the expert's code examples, technical specifics, and real-world applications

        Here is your topic of focus and set of goals: {goals}
                
        Begin by introducing yourself using a name that fits your persona, and then ask your question.

        Ask very technical and specific questions related to your topic.
                
        When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

        Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

        self.__search_instructions = SystemMessage(
            content="""
            You will be given a conversation between an analyst and an expert.

            Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

            First, analyze the full conversation.

            Pay particular attention to the final question posed by the analyst.

            Convert this final question into a well-structred web search query""")

        self.__answer_instructions = """
        You are an expert being interview by an analyst.

        Here is analyst area of focus: {goals}.

        Your goal is to answer a question posed by the interviewer.

        To answer question, use this context:
        {context}

        When answering questions, follow these guidelines:

        1. Use only the information provided in the context.

        2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

        3. The context contain sources at the topic of each individual document.

        4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

        5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
                
        6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
                
        [1] assistant/docs/llama3_1.pdf, page 7 
                
        And skip the addition of the brackets as well as the Document source preamble in your citation."""

        self.__section_writer_instructions = """
        You are an expert technical writer. 
                    
        Your task is to create a short, easily digestible section of a report based on a set of source documents.

        1. Analyze the content of the source documents: 
        - The name of each source document is at the start of the document, with the <Document tag.
                
        2. Create a report structure using markdown formatting:
        - Use ## for the section title
        - Use ### for sub-section headers
                
        3. Write the report following this structure:
        a. Title (## header)
        b. Summary (### header)
        c. Sources (### header)

        4. Make your title engaging based upon the focus area of the analyst: 
        {focus}

        5. For the summary section:
        - Set up summary with general background / context related to the focus area of the analyst
        - Emphasize what is novel, interesting, or surprising about insights gathered from the interview
        - Create a numbered list of source documents, as you use them
        - Do not mention the names of interviewers or experts
        - Aim for approximately 400 words maximum
        - Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
                
        6. In the Sources section:
        - Include all sources used in your report
        - Provide full links to relevant websites or specific document paths
        - Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
        - It will look like:

        ### Sources
        [1] Link or Document name
        [2] Link or Document name

        7. Be sure to combine sources. For example this is not correct:

        [3] https://ai.meta.com/blog/meta-llama-3-1/
        [4] https://ai.meta.com/blog/meta-llama-3-1/

        There should be no redundant sources. It should simply be:

        [3] https://ai.meta.com/blog/meta-llama-3-1/
                
        8. Final review:
        - Ensure the report follows the required structure
        - Include no preamble before the title of the report
        - Check that all guidelines have been followed"""

        def generate_question(state: InterviewState):
            """
            Node to generate a question
            """

            analyst = state["analyst"]
            messages = state["messages"]

            system_message = self.__question_instructions.format(goals=analyst.persona)
            question = self.__llm.invoke([SystemMessage(content=system_message)] + messages)

            return {"messages": [question]}

        def search_web(state: InterviewState):
            
            """ 
            Retrieve docs from web search
            """

            structured_llm = self.__llm.with_structured_output(SearchQuery)
            search_query = structured_llm.invoke([self.__search_instructions]+state['messages'])
            
            search_docs = self.__tavily_search.invoke(search_query.search_query)

            formatted_search_docs = "\n\n---\n\n".join(
                [
                    f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                    for doc in search_docs
                ]
            )

            return {"context": [formatted_search_docs]}

        def search_wikipedia(state: InterviewState):
            
            """ 
            Retrieve docs from wikipedia
            """

            structured_llm = self.__llm.with_structured_output(SearchQuery)
            search_query = structured_llm.invoke([self.__search_instructions]+state['messages'])
            
            search_docs = WikipediaLoader(query=search_query.search_query, 
                                        load_max_docs=2).load()

            formatted_search_docs = "\n\n---\n\n".join(
                [
                    f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                    for doc in search_docs
                ]
            )

            return {"context": [formatted_search_docs]}

        def generate_answer(state: InterviewState):
            """
            Node to answer a question
            """

            analyst = state["analyst"]
            messages = state["messages"]
            context = state["context"]

            system_message = self.__answer_instructions.format(goals=analyst.persona, context=context)
            answer = self.__llm.invoke([SystemMessage(system_message)] + messages)

            answer.name = "expert"
            
            return {"messages": [answer]}

        def save_interview(state: InterviewState):
            """
            Save interviews
            """

            messages = state["messages"]
            interview = get_buffer_string(messages)

            return {"interview": interview}

        def route_messages(state: InterviewState, name: str = "expert"):
            """
            Route between question and answer
            """

            messages = state["messages"]
            max_num_turns = state.get("max_num_turns", 2)

            num_responses = len(
                [m for m in messages if isinstance(m, AIMessage) and m.name == name]
            )

            if num_responses >= max_num_turns:
                return "save_interview"

            last_question = messages[-2]

            if "Thank you so much for your help" in last_question.content:
                return 'save_interview'
            return "ask_question"

        def write_section(state: InterviewState):

            """ 
            Node to answer a question 
            """
            
            interview = state["interview"]
            context = state["context"]
            analyst = state["analyst"]
            
            system_message = self.__section_writer_instructions.format(focus=analyst.description)
            section = self.__llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
                        
            return {"sections": [section.content]}
    

    def build_interview_graph(self):
        """
        Build the interview graph
        """

        interview_builder = StateGraph(InterviewState)
        interview_builder.add_node("ask_question", self.__generate_question)
        interview_builder.add_node("search_web", self.__search_web)
        interview_builder.add_node("search_wikipedia", self.__search_wikipedia)
        interview_builder.add_node("answer_question", self.__generate_answer)
        interview_builder.add_node("save_interview", self.__save_interview)
        interview_builder.add_node("write_section", self.__write_section)

        interview_builder.add_edge(START, "ask_question")
        interview_builder.add_edge("ask_question", "search_web")
        interview_builder.add_edge("ask_question", "search_wikipedia")
        interview_builder.add_edge("search_web", "answer_question")
        interview_builder.add_edge("search_wikipedia", "answer_question")
        interview_builder.add_conditional_edges("answer_question", self.__route_messages,['ask_question','save_interview'])
        interview_builder.add_edge("save_interview", "write_section")
        interview_builder.add_edge("write_section", END)

        memory = SqliteSaver.from_conn_string(":memory:")

        return interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")