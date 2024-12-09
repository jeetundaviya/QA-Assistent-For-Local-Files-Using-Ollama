import sys
import json
import time
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from typing import List

# Adding folder path to sys.path
sys.path.append('/Users/jeetundaviya/Documents/rag_chatbot')

from src.qa_pipeline.langchain_qa import setup_langchain_qa, query_qs

# Initialize the LLM (LLAMA or any model that supports similarity-based judgments)
llm = OllamaLLM(model="llama3.2", temperature=0.5)

# Define a prompt for the LLM to judge the answers
judgment_prompt = PromptTemplate(
    template="""
        You are a highly skilled judge. Given the predicted answer and the ground truth, 
        please rate how correct the predicted answer is based on the ground truth. 
        You can reply with a number between 0 and 1, where:
        - 0 means the predicted answer is completely incorrect
        - 1 means the predicted answer is exactly correct
        Respond with just the rating, no explanation.
        Question: {question}
        Predicted Answer: {predicted_answer}
        Ground Truth: {ground_truth}
    """
)

# Function to get the judgment score
def get_llm_judgment(question:str,predicted_answer: str, ground_truth: str) -> float:
    # prompt = judgment_prompt.format(question=question,predicted_answer=predicted_answer, ground_truth=ground_truth)
    chain = judgment_prompt|llm
    response = chain.invoke({
        "question":question,
        "predicted_answer":predicted_answer,
        "ground_truth":ground_truth
    })
    try:
        # Assuming the LLM response is a number between 0 and 1
        judgment_score = float(response.strip())  
        return judgment_score
    except ValueError:
        return 0.0  # If the LLM response cannot be parsed as a number, return 0.0

def evaluate_chatbot_performance(vector_store, model="llama3.2", test_file="test_dataset.json"):
    # Load test dataset
    with open(test_file, "r") as f:
        test_data = json.load(f)

    # Initialize the QA chain
    qa_chain = setup_langchain_qa(vector_store, model=model)

    # Metrics to compute
    predicted_answers = []
    ground_truth_answers = []
    total_latency = []
    fallback_count = 0
    judgment_scores = []

    for item in test_data:
        question = item["question"]
        ground_truth = item["expected_answer"]
        ground_truth_answers.append(ground_truth)

        # Measure response time
        start_time = time.time()
        response = query_qs(question, vector_store)
        latency = time.time() - start_time
        total_latency.append(latency)

        predicted_answer = response.content
        predicted_answers.append(predicted_answer)

        # Get judgment score from LLM
        judgment_score = get_llm_judgment(question, predicted_answer, ground_truth)
        judgment_scores.append(judgment_score)

        print(f"[+] Judgement Score :- {judgment_score}")

        # Check fallback
        if predicted_answer.lower() == "i don't know the answer":
            fallback_count += 1

    # Compute metrics
    average_judgment_score = sum(judgment_scores) / len(judgment_scores)
    avg_latency = sum(total_latency) / len(total_latency)
    fallback_rate = fallback_count / len(test_data)

    # Generate evaluation report
    report = {
        "average_judgment_score": average_judgment_score,
        "average_latency": avg_latency,
        "fallback_rate": fallback_rate,
        "total_queries": len(test_data)
    }

    return report
