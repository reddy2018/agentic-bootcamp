"""
Pattern 2: Sequential/Chain Pattern

Agents work one after another in a linear sequence.
Each agent's output becomes the next agent's input.

Architecture:
    Input
     │
     ▼
    Agent A
     │
     ▼
    Agent B
     │
     ▼
    Agent C
     │
     ▼
    Output
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------
# State definitions
# -------------------------------------------------
# what is this state made of?
# directory based state / pydantic based state
# example of directory based state is in 8-multiagent-patterns/1-pipeline.py
# below is pydantic based state
class State(BaseModel):
    """state flows sequentially from one agent to the next"""
    user_request: str
    research_results: Optional[str] = None # output from ResearchAgent
    analysis_results: Optional[str] = None # output from AnalysisAgent
    final_result: Optional[str] = None # output from SummaryAgent
    
# ---------------------------------------------------
# Sequential Agents
# ---------------------------------------------------

class ResearchAgent:
    """First agent: Gathers information based on user request."""
    
    # -------------------------------------------------
    # Agent Initialization
    # -------------------------------------------------
    def __init__(self):
        self.llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)
        
    def research(self, user_request: str) -> str:
        """Research based on user request."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research assistant. gather detailed information on the given topic."),
            ("user", "Research the following topic and provide detailed information:\n\n{user_request}")
        ])
        
        # Create the chain and invoke it
        # explain in detail about chain
        # chain is a sequence of operations that process the input step by step
        chain = prompt | self.llm
        response = chain.invoke({"user_request": user_request})
        return response.content
    
class AnalysisAgent:
    """Second agent: Analyzes research results."""
    
    # -------------------------------------------------
    # Agent Initialization
    # -------------------------------------------------
    def __init__(self):
        self.llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)
        
    def analyze(self, user_request: str, research: str) -> str: # add user_request as input and get the research agent output
        """Analyze the research results."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an analysis expert. Analyze the provided research results."),
            ("user", "Analyze the following research results and provide insights:\n\n{user_request}\n\nResearch findings: {research})")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"user_request": user_request, "research": research})
        return response.content
    
class SummaryAgent:
    """Third agent: Summarizes the analysis results."""
    
    # -------------------------------------------------
    # Agent Initialization
    # -------------------------------------------------
    def __init__(self):
        self.llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)
        
    def summarize(self, user_request: str, research: str, analysis: str) -> str: # add user_request as input and get the analysis agent output
        """Summarize the analysis results."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a summary expert. Summarize the provided analysis results."),
            ("user", "user request:\n\n{user_request}"
             "\n\nResearch findings: {research}"
             "\n\nAnalysis findings: {analysis}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"user_request": user_request,
                                 "research": research,
                                 "analysis": analysis})
        return response.content
        
# ---------------------------------------------------
# workflow definition
# ---------------------------------------------------

def create_workflow():
    """Create the sequential workflow."""
    workflow = StateGraph(State)
    
    research_agent = ResearchAgent()
    analysis_agent = AnalysisAgent()
    summary_agent = SummaryAgent()
    
    # step 1: ResearchAgent
    def research_node(state: State) -> dict:
        print("\n" + "="*60)
        print("Research Agent Processing...")
        print("="*60 + "\n")
        
        research = research_agent.research(state.user_request)
        print("research completed.")
        return {"research_results": research}
        
        # step 2: AnalysisAgent
    def analysis_node(state: State) -> dict:
        print("\n" + "="*60)
        print("Analysis Agent Processing...")
        print("="*60 + "\n")
        
        analysis = analysis_agent.analyze(state.user_request, state.research_results)
        print("analysis completed.")
        return {"analysis_results": analysis}
    
    # step 3: SummaryAgent
    def summary_node(state: State) -> dict:
        print("\n" + "="*60)
        print("Summary Agent Processing...")
        print("="*60 + "\n")
        
        summary = summary_agent.summarize(state.user_request, state.research_results, state.analysis_results)
        print("summary completed.")
        return {"final_result": summary}
    
    # build sequential graph
    #workflow = StateGraph(State)
    workflow.add_node("research_node", research_node)
    workflow.add_node("analysis_node", analysis_node)
    workflow.add_node("summary_node", summary_node)

    # sequential flow
    workflow.set_entry_point("research_node")
    workflow.add_edge("research_node", "analysis_node")
    workflow.add_edge("analysis_node", "summary_node")
    workflow.add_edge("summary_node", END)

    return workflow.compile()
    
# ---------------------------------------------------
# Main execution
# ---------------------------------------------------

def main():
    print("\n" + "="*60)
    print("Starting Sequential Multi-Agent Workflow")
    print("="*60 + "\n")
    
    user_request = "Explain the impact of climate change on global agriculture."
    print(f"User Request: {user_request}\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    app = create_workflow()
    app.get_graph().draw_mermaid_png(output_file_path="sequential_workflow.png")
    
    initial_state = State(user_request=user_request)
    print("Executing workflow...\n")
    
    try:
        final_state_dict = app.invoke(initial_state.model_dump())
        final_state = State(**final_state_dict)
        
        print("\n" + "="*60)
        print("Final Result:")
        print("="*60 + "\n")
        print(f"\n{final_state.final_result}")
        
    except Exception as e:
        print(f"An error occurred during workflow execution: {e}")
        raise
        
if __name__ == "__main__":
    main()