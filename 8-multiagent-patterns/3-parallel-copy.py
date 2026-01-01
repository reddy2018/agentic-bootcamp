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

"""
Pattern 3: Parallel Pattern (TRUE parallel via asyncio.gather)

- Three perspective agents run concurrently (async LLM calls)
- Then an aggregator combines outputs
"""
import os
import asyncio
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
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an expert {perspective} analyst."),
            ("user", f"Analyze this request from a {perspective} perspective: {{user_request}}")
        ])
        self.chain = self.prompt | self.llm
        
        # Note: In a true async implementation, the invoke method should be async
        # invoke - blocks the thread until the response is ready - SequentialIO
        # ainvoke - non-blocking, returns a coroutine - ParallelIO
        
    async def analyze_async(self, user_request: str) -> str:
        """Asynchronously analyze the user request from the agent's perspective."""
        response = await self.chain.ainvoke({"user_request": user_request, "perspective": self.perspective})
        return response.content

class AggregatorAgent():
    """combines parellel agent results into a final output."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert aggregator. combine multiple analyses into a coherent final output."),
            ("user", """ original request: {user_request} 
             multiple perspectives: {perspectives}
            """)
        ])

        self.chain = self.prompt | self.llm
    async def aggregate_async(self, user_request: str, perspectives: List[str]) -> str:
        """Asynchronously aggregate results from multiple agents into a final output."""
        perspective_text = "\n\n".join(f"perspective {i+1}:\n{p}" for i, p in enumerate(perspectives)) # Format perspectives
        response = await self.chain.ainvoke({"user_request": user_request, "perspectives": perspective_text})
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
    
    # true parallel: one node runs 3 async tasks concurrently
    async def parallel_agents(state: State) -> dict:
        print("\n" + "="*60)
        print("Running parallel agents...")
        print("="*60 + "\n")
        
        t1 = agent_a.analyze_async(state.user_request)
        t2 = agent_b.analyze_async(state.user_request)
        t3 = agent_c.analyze_async(state.user_request)
        
        p1, p2, p3 = await asyncio.gather(t1, t2, t3)
        print("All parallel agents completed.\n")
        
        return {"agent_a_result": p1,
                "agent_b_result": p2,
                "agent_c_result": p3,}
    
    async def aggregator_node(state: State) -> dict:
        print("\n" + "="*60)
        print("Aggregator processing...")
        print("="*60 + "\n")
        
        perspectives = [state.agent_a_result, state.agent_b_result, state.agent_c_result]
        final_output = await aggregator.aggregate_async(state.user_request, perspectives)
        print("Aggregation completed.\n")
        
        return {"final_output": final_output}
    
    # build the state graph
    workflow = StateGraph(State)
    
    # add nodes to the workflow
    workflow.add_node("parallel_agents", parallel_agents)
    workflow.add_node("aggregator", aggregator_node)

    # connect nodes in parallel
    workflow.set_entry_point("parallel_agents")
    workflow.add_edge("parallel_agents", "aggregator")
    workflow.add_edge("aggregator", END)

    # Note: In a true parallel system, all three would run simultaneously
    # LangGraph processes them sequentially, but they're independent
    # For true parallelism, you'd use async/threading
    return workflow.compile()

# ---------------------------------------------------------
# main execution
# ---------------------------------------------------------

async def main():
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
    final_out = await app.ainvoke(initial_state) # await needed for async invoke
    final_state = State.model_validate(final_out)
    
    print("\n" + "="*60)
    print("Final Aggregated Output:")
    print("="*60 + "\n")
    print(final_state.final_output)
    print("\n" + "="*60 + "\n")
    
if __name__ == "__main__":
    asyncio.run(main())