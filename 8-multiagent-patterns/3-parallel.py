"""
Pattern 3: Parallel Pattern

Multiple agents work simultaneously on independent tasks.
Results are aggregated after all agents complete.

Architecture:
         Input
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
  Agent Agent Agent
    A     B     C
    │     │     │
    └─────┼─────┘
          ▼
      Aggregator
          │
          ▼
       Output
"""
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------
# State definitions
# ---------------------------------------------------------
class State(BaseModel):
    """State with parallel agent results."""
    user_request: str
    agent_a_result: Optional[str] = None # Result from Agent A
    agent_b_result: Optional[str] = None # Result from Agent B
    agent_c_result: Optional[str] = None # Result from Agent C
    final_output: Optional[str] = None  # Aggregated final output
    
# ---------------------------------------------------------
# parallel agents
# ---------------------------------------------------------
class ParallelAgent():
    """agent that provides a specific perspective on the user request."""
    
    def __init__(self, perspective: str):
        self.perspective = perspective # e.g., "technical", "creative", "analytical"
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
    def analyze(self, user_request: str) -> str:
        """Analyze the user request from the agent's perspective."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an expert {self.perspective} analyst."),
            ("user", f"Analyze this request from a {self.perspective} perspective: {user_request}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"user_request": user_request, "perspective": self.perspective})
        return response.content
    
class AggregatorAgent():
    """combines parellel agent results into a final output."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
    def aggregate(self, user_request: str, perspectives: List[str]) -> str:
        """Aggregate results from multiple agents into a final output."""
        combined_results = "\n\n".join([f"perspective {i+1}:\n{p}" for i, p in enumerate(perspectives)]) # Combine all agent results
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert aggregator. combine multiple analyses into a coherent final output."),
            ("user", """ original request: {user_request} 
             multiple perspectives: {perspectives}
            """)
        ])

        chain = prompt | self.llm
        response = chain.invoke({"user_request": user_request, "perspectives": perspectives})
        return response.content

# ---------------------------------------------------------
# workflow
# ---------------------------------------------------------

def create_workflow():
    """create the parallel pattern workflow."""
    
    #create agents with different perspectives
    agent_a = ParallelAgent(perspective="technical")
    agent_b = ParallelAgent(perspective="creative")
    agent_c = ParallelAgent(perspective="analytical")
    aggregator = AggregatorAgent()
    
    # parallel nodes - agents working simultaneously
    def agent_a_node(state: State) -> dict:
        print("\n" + "="*60)
        print("Agent A (Technical) processing...")
        print("="*60 + "\n")
        
        result = agent_a.analyze(state.user_request)
        print("Technical analysis completed")
        return {"agent_a_result": result}
    
    def agent_b_node(state: State) -> dict:
        print("\n" + "="*60)
        print("Agent B (Creative) processing...")
        print("="*60 + "\n")
        
        result = agent_b.analyze(state.user_request)
        print("Creative analysis completed")
        return {"agent_b_result": result}
    
    def agent_c_node(state: State) -> dict:
        print("\n" + "="*60)
        print("Agent C (Analytical) processing...")
        print("="*60 + "\n")
        
        result = agent_c.analyze(state.user_request)
        print("Analytical analysis completed")
        return {"agent_c_result": result}
    
    # aggregator node - combines results from all agents
    def aggregator_node(state: State) -> dict:
        print("\n" + "="*60)
        print("Aggregator processing...")
        print("="*60 + "\n")
        
        perspectives = [
            state.agent_a_result,
            state.agent_b_result,
            state.agent_c_result
        ]
        final_output = aggregator.aggregate(state.user_request, perspectives)
        print("Aggregation completed")
        return {"final_output": final_output}
    
    # build the state graph
    workflow = StateGraph(State)
    
    # add nodes to the workflow
    workflow.add_node("agent_a", agent_a_node)
    workflow.add_node("agent_b", agent_b_node)
    workflow.add_node("agent_c", agent_c_node)
    workflow.add_node("aggregator", aggregator_node)

    # connect nodes in parallel
    workflow.set_entry_point("agent_a")
    workflow.add_edge("agent_a", "agent_b")
    workflow.add_edge("agent_b", "agent_c")
    workflow.add_edge("agent_c", "aggregator")
    workflow.add_edge("aggregator", END)

    # Note: In a true parallel system, all three would run simultaneously
    # LangGraph processes them sequentially, but they're independent
    # For true parallelism, you'd use async/threading
    return workflow.compile()

# ---------------------------------------------------------
# main execution
# ---------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("Starting Parallel Pattern Workflow")
    print("="*60 + "\n")
    
    user_request = "Provide a comprehensive analysis of the impact of AI on modern education."
    print(f"User Request: {user_request}\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    app = create_workflow()
    #app.get_graph().draw_mermaid_png(output_file_path="parallel_pattern_workflow.png", max_retries=5,
    #retry_delay=2.0,)

    initial_state = State(user_request=user_request)
    print("Executing workflow...\n")
    print("Agents working in parallel...\n")
    
    try:
        final_state_dict = app.invoke(initial_state.model_dump()) # what is model_dump? 
        # model_dump() converts the Pydantic model to a dictionary
        final_state = State(**final_state_dict)
        print("\n" + "="*60)
        print("Workflow completed successfully!")
        print("="*60 + "\n")
        print("Final Output:\n")
        print(final_state.final_output)
    except Exception as e:
        print(f"An error occurred during workflow execution: {e}")
        raise
    
if __name__ == "__main__":
    main()