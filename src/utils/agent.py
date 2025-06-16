from pprint import pprint
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from utils.state import ImageProcessingState
from utils.tools import select_relevant_images, analyze_images
from utils.examples import REPORT_EXAMPLE

class RefiningAgent(Runnable):
    def invoke(self, state: ImageProcessingState, config = None) -> ImageProcessingState:
        react_prompt = f"""
        You are an assistant who is helping to refine a report about an object used as collateral. You need to identify parts of the report which contain missing or incomplete information and augment it by looking at images of the object. After identifying missing information, you should look for the relevant images and analyze them to find the missing information.
                                                    
        Here are examples of complete reports:
        {REPORT_EXAMPLE}

        You have access to the following tools:
        - select_relevant_images
        - analyze_images

        Always use tools to refine the report.

        Use the following format:

        Question: the report which you need to refine, followed by the image descriptions
        Thought: you should always think about what to do
        Action: the action to take should be one of the following: select_relevant_images, analyze_images
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: ONLY the modified final report
        """

        model = ChatOpenAI(model="gpt-4o-mini")
        tools = [select_relevant_images, analyze_images]
        model = model.bind_tools(tools)
        # Create the langgraph react agent
        agent = create_react_agent(model=model, tools=tools, prompt=react_prompt)
        
        report = state["final_report_markdown"]
        
        # In langgraph, the agent is directly invokable
        result = agent.invoke({
            "messages": [{"role": "user", "content": f"Modify this report based on your findings: {report}"}]
        })
        
        # Extract the final answer from the messages
        print("[DEBUG] Agent tought process:")
        pprint(result)

        final_answer = result["messages"][-1].content
        final_answer = final_answer[final_answer.find("#"):]
        
        state["final_report_markdown"] = final_answer

        return state