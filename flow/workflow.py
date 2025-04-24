from pydantic import BaseModel
from typing import List
from langgraph.graph import StateGraph, END
from flow.nodes import ( 
    node1_input, 
    node2_checkvectorDB, 
    node3_classify_intent, 
    node4_paraphase,
    node5_saveToDB, 
    node6_build_results
)

class QuestionState(BaseModel):
    question: str
    intent: str = ""
    paraphrases: List[str] = []
    action: str = ""
    found_in_db: bool = False
    matched_question: str = ""


def input_step(state: QuestionState):
    return {"question": state.question}

def check_vectorDB_step(state: QuestionState):
    result = node2_checkvectorDB.check_vector_db(state.question)
    if result["found"]:
        return {
            "found_in_db": True,
            "intent": result["intent"],
            "matched_question": result["matched_question"]
        }
    else:
        return {"found_in_db": False}

def classify_step(state: QuestionState):
    intent = node3_classify_intent.classify_intent(state.question)
    return {"intent": intent}

def paraphrase_step(state: QuestionState):
    paraphrases = node4_paraphase.generate_paraphrases(state.question)

    # print generated paraphrases for visuallization
    print("\nGenerated paraphrases:")
    for i, p in enumerate(paraphrases, 1):
        print(f"{i}. {p}")
        
    return {"paraphrases": paraphrases}

def save_step(state: QuestionState):
    node5_saveToDB.save_question_and_paraphrases(
        state.question, state.paraphrases, state.intent
    )
    return {}

def build_results_step(state: QuestionState):
    action = node6_build_results.build_results(state.intent, state.question)
    return {"action": action}

# create LangGraph graph
graph = StateGraph(QuestionState)


graph.add_node("Input", input_step)
graph.add_node("CheckCache", check_vectorDB_step)
graph.add_node("ClassifyIntent", classify_step)
graph.add_node("Paraphrase", paraphrase_step)
graph.add_node("SaveToMilvus", save_step)
graph.add_node("BuildResults", build_results_step)


graph.set_entry_point("Input")

graph.add_edge("Input", "CheckCache")

graph.add_conditional_edges(
    "CheckCache",
    lambda state: "Found" if state.found_in_db else "NotFound",
    {
        "Found": "BuildResults",
        "NotFound": "ClassifyIntent"
    }
)

graph.add_edge("ClassifyIntent", "Paraphrase")
graph.add_edge("Paraphrase", "SaveToMilvus")
graph.add_edge("SaveToMilvus", "BuildResults")
graph.add_edge("BuildResults", END)


app = graph.compile()

