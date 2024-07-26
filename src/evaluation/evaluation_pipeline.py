import json
from langchain.smith import run_on_dataset
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from retriever.retriever import create_retriever
from generator.generator import format_qa_pair

def load_evaluation_set(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def evaluate_rag(evaluation_set_path, contract_path):
    evaluation_set = load_evaluation_set(evaluation_set_path)
    retriever = create_retriever(contract_path)
    results = []
    for item in evaluation_set:
        question = item['question']
        expected_answer = item['answer']
        queries = generate_queries_decomposition(question)
        q_a_pairs = generate_answers(queries, retriever)
        result = {"question": question, "expected_answer": expected_answer, "generated_answer": q_a_pairs}
        results.append(result)
    return results

# Example usage:
# results = evaluate_rag('../data/evaluation_set/short_contract_questions.json', '../data/contracts/Robinson Advisory.docx')
# print(results)
