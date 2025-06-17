from pprint import pprint
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from utils.state import ImageProcessingState, ReportSchema
from utils.tools import select_relevant_images, analyze_images, ddg_search
from utils.examples import REPORT_EXAMPLE, OUTPUT_FORMAT
from pydantic import BaseModel, Field

class RefiningAgent(Runnable):
    def system_prompt(self) -> list[AnyMessage]:
        system_msg = f"""
        You are an assistant who is helping to refine a report about an object used as collateral. You need to identify parts of the report which contain missing or incomplete information and augment it by looking at images of the object. After identifying missing information, you should look for the relevant images and analyze them to find the missing information.
                                                    
        Here are examples of complete reports:
        {REPORT_EXAMPLE}

        You have access to the following tools:
        - select_relevant_images
        - analyze_images
        - ddg_search

        Always use tools to refine the report. If you use the ddg_search, indicate that the information is from a web search.

        Use the following format:

        Question: the report which you need to refine
        Thought: you should always think about what to do
        Action: the action to take should be one of the following: select_relevant_images, analyze_images, ddg_search
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: ONLY the modified final report in markdown format, without the level 3 headers (###)
        """

        return [{"role": "system", "content": system_msg}]

    def invoke(self, state: ImageProcessingState, config = None) -> ImageProcessingState:
        model = ChatOpenAI(model="o3-mini")
        tools = [select_relevant_images, analyze_images, ddg_search]
        model = model.bind_tools(tools)
        # Create the langgraph react agent
        agent = create_react_agent(model=model, tools=tools, response_format=ReportSchema)
        
        report = state["final_report_markdown"]
        
        system_message = self.system_prompt()
        # In langgraph, the agent is directly invokable
        result = agent.invoke({
            "messages": system_message +[{"role": "user", "content": f"Modify this report based on your findings: {report}"}]
        })
        
        # Extract the final answer from the messages
        print("[DEBUG] Agent tought process:")
        pprint(result["messages"])

        final_answer = result["structured_response"]
        state["final_report_markdown"] = final_answer

        return state