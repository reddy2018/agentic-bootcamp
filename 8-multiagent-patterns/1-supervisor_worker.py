"""
Pattern 1: Supervisor-Worker Pattern

A central Supervisor agent receives tasks and delegates them to specialized Worker agents.
The Supervisor coordinates the workflow and aggregates results.

Architecture:
    User Request
         │
         ▼
    Supervisor (coordinates)
         │
    ┌────┴────┐
    ▼         ▼
  Worker1  Worker2
    │         │
    └────┬────┘
         ▼
    Supervisor (aggregates)
         │
         ▼
    Final Result
"""

import os
import json
from typing import List, Optional, Dict, Any
from unittest import result
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

#-------------------------------------------------
# STATE DEFINITIONS
# -------------------------------------------------
# what is this class for?
# it holds the task information
class Task(BaseModel):
    """ A task assigned to a worker. """
    task_id: int
    description: str
    assigned_to: Optional[str] = None # which worker is assigned
    result: Optional[str] = None # result from the worker

# what is this class for?
# it holds the shared state between supervisor and workers
class State(BaseModel):
    """ shared state between supervisor and workers """
    user_request: str
    tasks: List[Task] = Field(default_factory=list)
    final_result: Optional[str] = None


#-------------------------------------------------
# SUPERVISOR AGENT
# -------------------------------------------------

class SupervisorAgent:
    """ central coordinator that delegates tasks to workers """
    
    # -------------------------------------------------
    # INITIALIZATION METHOD
    # -------------------------------------------------
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # -------------------------------------------------
    # TASK PLANNING METHOD
    # ------------------------------------------------- 
    def plan(self, user_request: str) -> List[Task]:
        """Break down user request into tasks. and assign to workers."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a supervisor agent. Break down the user request into specific tasks and
             assign them to appropriate workers. (writer, Researcher, analyst)"""),
            ("user", """User Request: {user_request}
             
             create 2-3 tasks and assign to workers.
             - writer: creates content
             - researcher: gathers information
             - analyst: analyzes data
             
             Return JSON:
             
             {{
                "tasks": [
                    {{"task_id": 1, "description": "task", "assigned_to": "Writer"}},
                    {{"task_id": 2, "description": "task", "assigned_to": "Researcher"}}
                ]
                }}
             """)
        ])

        # invoke LLM to get tasks
        chain = prompt | self.llm
        response = chain.invoke({"user_request": user_request})
        content = response.content
        
        # extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        try:
            data = json.loads(content)
            return [Task(**task) for task in data["tasks"]]
        except:
            return [
                Task(task_id=1, description="Research the topic", assigned_to="Researcher"),
                Task(task_id=2, description="Create content", assigned_to="Writer")
            ]
    
    # -------------------------------------------------
    # AGGREGATION METHOD
    # -------------------------------------------------
    def aggregate(self, user_request: str, tasks: List[Task]) -> str:
        """Combine results from workers into a final output."""
      
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a supervisor agent. Aggregate the results from various workers into a final output."""),
            ("user", """User Request: {user_request}
            
            Here are the results from the workers:
            {task_results}
            
            Please provide a coherent final result based on these inputs.
            """)
        ])
        
        # format task results for prompt insertion
        task_results = "\n".join([f"Task {task.task_id} ({task.assigned_to}): {task.result}" for task in tasks])
        
        # invoke LLM to aggregate results
        chain = prompt | self.llm
        response = chain.invoke({
            "user_request": user_request,
            "task_results": task_results
        })
        
        return response.content
    
#-------------------------------------------------
# WORKER AGENT
# -------------------------------------------------

class WorkerAgent:
    """ specialized worker that performs assigned tasks """
    
    # -------------------------------------------------
    # INITIALIZATION METHOD
    # -------------------------------------------------
    def __init__(self, worker_type: str):
        self.worker_type = worker_type # e.g., Writer, Researcher, Analyst
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # -------------------------------------------------
    # TASK EXECUTION METHOD

    # -------------------------------------------------
    def execute(self, user_request: str, task: Task) -> str:
        """Execute a task based on worker specialization."""
        
        prompts = {
            "Writer": "You are a skilled writer. Create engaging and well-structured content based on the task description.",
            "Researcher": "You are a diligent researcher. Gather accurate and relevant information based on the task description.",
            "Analyst": "You are a sharp analyst. Analyze the provided data and extract meaningful insights based on the task description."
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.get(self.worker_type, "You are a versatile worker. Perform the task as described.")),
            ("user", """ original User Request: {user_request}
             
             your Task: {task_description}
             
             complete this task as {worker_type} """)
        ])
        
        # invoke LLM to perform task
        chain = prompt | self.llm
        response = chain.invoke({
            "task_description": task.description,
            "user_request": user_request,
            "worker_type": self.worker_type
            })
        
        return response.content
    
#-------------------------------------------------
# workflow DEFINITION
# -------------------------------------------------

def create_workflow():
    """ create supervisor - worker workflow """
    
    supervisor = SupervisorAgent()
    workers = {
        "Writer": WorkerAgent("Writer"),
        "Researcher": WorkerAgent("Researcher"),
        "Analyst": WorkerAgent("Analyst")
    }
    
    # supervisor: plan and delegate tasks
    def supervisor_plan_node(state: State) -> dict:
        print("\n" + "="*60)
        print("Supervisor: Planning and delegating tasks...")
        print("="*60 + "\n")
        
        tasks = supervisor.plan(state.user_request)
        
        print(f"created {len(tasks)} tasks:")
        for task in tasks:
            print(f"  - Task {task.task_id}: {task.description} (assigned to: {task.assigned_to})")
        
        return {"tasks": [task.model_dump() for task in tasks]}
    
    
    # worker: execute assigned task
    def worker_execute_node(state: State) -> dict:
        print("\n" + "="*60)
        print("Workers: Executing assigned tasks...")
        print("="*60 + "\n")
        
        tasks = [Task(**t) if isinstance(t, dict) else t for t in state.tasks]
        
        # execute each task with the appropriate worker
        for task in tasks:
            if task.assigned_to in workers:
                print(f" -> {task.assigned_to} working on Task {task.task_id}...")
                worker = workers[task.assigned_to]
                task.result = worker.execute(state.user_request, task)
                print(f" Task {task.task_id} completed.")
                
        return {"tasks": [task.model_dump() for task in tasks]} # update tasks with results
    
    # supervisor: aggregate results
    def supervisor_aggregate_node(state: State) -> dict:
        print("\n" + "="*60)
        print("Supervisor: Aggregating results from workers...")
        print("="*60 + "\n")
        
        tasks = [Task(**t) if isinstance(t, dict) else t for t in state.tasks]
        
        final_result = supervisor.aggregate(state.user_request, tasks)
        
        print("Final result aggregated.")
        
        return {"final_result": final_result}
    
    # build the state graph
    workflow = StateGraph(State)    
    workflow.add_node("supervisor_plan", supervisor_plan_node)
    workflow.add_node("worker_execute", worker_execute_node)
    workflow.add_node("supervisor_aggregate", supervisor_aggregate_node)
    
    workflow.set_entry_point("supervisor_plan")
    workflow.add_edge("supervisor_plan", "worker_execute")
    workflow.add_edge("worker_execute", "supervisor_aggregate")
    workflow.add_edge("supervisor_aggregate", END)
    return workflow.compile()



#-------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------
def main():
    print("\n" + "#"*60)
    print(" Supervisor-Worker Multi-Agent Pattern Demo ")
    print("#"*60 + "\n")
    
    user_request = "Create a blog post about agentic ai."
    print(f"User Request: {user_request}\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable.")
        return
    
    app = create_workflow()
    app.get_graph().draw_mermaid_png(output_file_path="supervisor_worker_pattern.png")
    initial_state = State(user_request=user_request)
    
    
    print("Executing workflow...\n")
    
    try:
        final_state_dict = app.invoke(initial_state.model_dump())
        final_state = State(**final_state_dict)
        
        print("\n" + "#"*60)
        print(" Final Result from Supervisor-Agent Pattern ")
        print("#"*60 + "\n")
        print(final_state.final_result)
        print("\n" + "#"*60 + "\n")
        
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        raise


if __name__ == "__main__":
    main()