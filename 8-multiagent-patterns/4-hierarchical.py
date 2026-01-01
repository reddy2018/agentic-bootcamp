"""
Pattern 4: Hierarchical Pattern

Multi-level organization with managers and workers.
Top-level manager delegates to middle managers,
who coordinate their own workers.

Architecture:
        Manager (Top Level)
            │
    ┌──────┴──────┐
    ▼             ▼
  Manager A    Manager B
    │             │
  ┌─┴─┐         ┌─┴─┐
  ▼   ▼         ▼   ▼
Worker Worker Worker Worker
  A1   A2      B1   B2
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------
# State definitions
# --------------------------------------------------------------

class Task(BaseModel):
    """Task in the hierarchical multi-agent system."""
    task_id: str # unique identifier
    description: str # task details
    department: str # which manager is responsible
    result : Optional[str] = None # result of the task
    
class State(BaseModel):
    """State of the hierarchical multi-agent system."""
    user_request: str
    department_a_tasks: List[Task] = Field(default_factory=list) # tasks for department A
    department_b_tasks: List[Task] = Field(default_factory=list) # tasks for department B
    department_a_results: List[str] = Field(default_factory=list) # results from department A
    department_b_results: List[str] = Field(default_factory=list) # results from department B
    final_report: Optional[str] = None # final aggregated report
    
# --------------------------------------------------------------
# hierarchical agents
# --------------------------------------------------------------
class TopLevelManager():
    """top-level manager that coordinates middle managers."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    def delegate(self, user_request: str) -> dict: # why dict? because we need to return multiple task lists
        """Delegate tasks to middle managers based on user request."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the top-level manager.delegate tasks to departments:
            1. Department A: Handles tasks related to technical issues.
            2. Department B: Handles tasks related to customer/bussiness service. """),
            ("human", """Request: {user_request}

Create tasks for each department. Return JSON:
{{
  "department_a_tasks": [
    {{"task_id": "1", "description": "technical task"}}
  ],
  "department_b_tasks": [
    {{"task_id": "1", "description": "business task"}}
  ]
}}
""")

        ])
    
        chain = prompt | self.llm
        response = chain.invoke({"user_request": user_request})
        content = response.content
    
        # what is this below for? to parse the json response into Task objects
        # parse response content
        
        import json
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        try:
            data = json.loads(content)

            dept_a = [Task(**t, department="A") for t in data.get("department_a_tasks", [])]
            dept_b = [Task(**t, department="B") for t in data.get("department_b_tasks", [])]

            return {
                "department_a_tasks": dept_a,
                "department_b_tasks": dept_b
            }

        except:
            return {"department_a_tasks": [Task(task_id=1, description="Default task for department A", department="A")],
                    "department_b_tasks": [Task(task_id=1, description="Default task for department B", department="B")]
                    }

class MiddleManager():
    """Middle manager that coordinates workers."""
    
    def __init__(self, department: str):
        self.department = department
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    def coordinate(self, user_request: str, tasks: List[Task]) -> List[str]:
        """Coordinate workers to complete tasks."""
        task_text = "\n".join([f"Task- {task.description} (ID: {task.task_id})" for task in tasks]) # format tasks for prompt
        
        prompt = ChatPromptTemplate.from_messages([
    ("system", f"You are the {self.department} department manager."),
    ("human", """User request: {user_request}

Department tasks:
{tasks}

Provide concise results for each task.""")
])

        
        chain = prompt | self.llm
        response = chain.invoke({"user_request": user_request, "tasks": task_text})
        
        #split response into individual results(simplified)
        results = [response.content] # in real scenario, parse properly based on response format
        return results
    
class TopLevelAggregator():
    """Top manager that aggregates results from middle managers."""
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    def aggregate(self, user_request: str, dept_a_results: List[str], dept_b_results: List[str]) -> str:
        """Aggregate results from all departments into a final report."""
        results_text = f"""
        Department A Results:
        {chr(10).join(dept_a_results)}
        
        Department B Results:
        {chr(10).join(dept_b_results)}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the top-level manager. Aggregate results from all departments into a final report."),
            ("human", f"""original user request: {user_request}\n\n
             {results_text}\n\n
            
            Create a final report summarizing the outcomes.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"user_request": user_request, "results": results_text})
        return response.content
    
# --------------------------------------------------------------
# workflow definition
# --------------------------------------------------------------

def create_workflow():
    """create hierarchical multi-agent workflow."""
    
    top_manager = TopLevelManager()
    middle_manager_a = MiddleManager("Technical")
    middle_manager_b = MiddleManager("Marketing")
    top_aggregator = TopLevelAggregator()
    
    # level 1: top-level manager delegates tasks
    def top_manager_node(state: state) -> dict:
        print("\n" + "="*60)
        print("Top-Level Manager Delegating Tasks...")
        print("="*60 + "\n")
        
        delegation = top_manager.delegate(state.user_request)
        
        print(f"Delegated {len(delegation['department_a_tasks'])} tasks to Department A")
        print(f"Delegated {len(delegation['department_b_tasks'])} tasks to Department B")
        
        return {"department_a_tasks": [t.model_dump() for t in delegation["department_a_tasks"]],
                "department_b_tasks": [t.model_dump() for t in delegation["department_b_tasks"]]
                }
        
    # level 2: middle managers coordinate workers
    def dept_a_node(state: state) -> dict:
        print("\n" + "="*60)
        print("Middle Manager A Coordinating Workers...")
        print("="*60 + "\n")
        
        tasks = [Task(**t) if isinstance(t, dict) else t for t in state.department_a_tasks]
        results = middle_manager_a.coordinate(state.user_request, tasks)
        
        print(f"Department A completed {len(results)} tasks.")
        
        return {"department_a_results": results}
    
    def dept_b_node(state: state) -> dict:
        print("\n" + "="*60)
        print("Middle Manager B Coordinating Workers...")
        print("="*60 + "\n")
        
        tasks = [Task(**t) if isinstance(t, dict) else t for t in state.department_b_tasks]
        results = middle_manager_b.coordinate(state.user_request, tasks)
        
        print(f"Department B completed {len(results)} tasks.")
        
        return {"department_b_results": results}
    
    # level 3: top-level aggregator compiles final report
    def aggregator_node(state: state) -> dict:
        print("\n" + "="*60)
        print("Top-Level Aggregator Compiling Final Report...")
        print("="*60 + "\n")
        
        final_report = top_aggregator.aggregate(
            state.user_request,
            state.department_a_results,
            state.department_b_results
        )
        
        print("Final report compiled.")
        
        return {"final_report": final_report}
    
    # build the state graph
    workflow = StateGraph(state)
    workflow.add_node("top_manager_node", top_manager_node)
    workflow.add_node("dept_a_node", dept_a_node)
    workflow.add_node("dept_b_node", dept_b_node)
    workflow.add_node("aggregator_node", aggregator_node)
    
    workflow.set_entry_point("top_manager_node")
    workflow.add_edge("top_manager_node", "dept_a_node")
    workflow.add_edge("top_manager_node", "dept_b_node")
    workflow.add_edge("dept_a_node", "aggregator_node")
    workflow.add_edge("dept_b_node", "aggregator_node")
    workflow.add_edge("aggregator_node", END)
    
    return workflow.compile()


# --------------------------------------------------------------
# main execution
# --------------------------------------------------------------
def main():
    print("\n" + "#"*80)
    print("Starting Hierarchical Multi-Agent Workflow Execution")
    print("#"*80 + "\n")
    
    user_request = """Our company is facing several challenges.
    1. Technical issues with our server causing downtime.
    2. Customer inquiries are piling up, and we need better response times.
    Please address these issues effectively."""
    
    print(f"User Request:\n{user_request}\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
    workflow = create_workflow()
    initial_state = state(user_request=user_request)
    
    print("Executing workflow...\n")
    try:
        final_state_dict = workflow.invoke(initial_state.model_dump())
        final_state = state(**final_state_dict)
        print("\n" + "#"*80)
        print("Workflow Execution Completed")
        print("#"*80 + "\n")
        print("Final Report:\n")
        print(final_state.final_report)
    except Exception as e:
        print(f"An error occurred during workflow execution: {e}")
        raise
if __name__ == "__main__":
    main()