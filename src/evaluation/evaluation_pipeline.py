import json
from langchain.smith import run_on_dataset
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from retriever.retriever import create_retriever
from generator.generator import format_qa_pair
import time
import string
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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


def get_rag_response(question, final_rag_chain):
    start_time = time.time()
    response = final_rag_chain.invoke({"question": question})
    end_time = time.time()
    response_time = end_time - start_time
    return response, response_time

def run_automated_tests(evaluation_data, final_rag_chain):
    results = []
    for contract in evaluation_data['contracts']:
        for qa in contract['questions']:
            predicted_answer, response_time = get_rag_response(qa['question'], final_rag_chain)
            results.append({
                "question": qa['question'],
                "predicted_answer": predicted_answer,
                "true_answer": qa['answer'],
                "response_time": response_time
            })
    return results

def normalize_answer(text):
    """Normalize the answer by lowercasing and stripping whitespace and punctuation."""
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def cosine_sim(a, b):
    """Compute cosine similarity between two lists of strings"""
    vectorizer = TfidfVectorizer().fit_transform([a, b])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

def evaluate_results(results, threshold=0.7):
    true_answers = [normalize_answer(result['true_answer']) for result in results]
    predicted_answers = [normalize_answer(result['predicted_answer']) for result in results]
    
    similarity_scores = [cosine_sim(true, pred) for true, pred in zip(true_answers, predicted_answers)]
    binary_predictions = [1 if score >= threshold else 0 for score in similarity_scores]
    binary_true = [1] * len(true_answers)
    
    accuracy = accuracy_score(binary_true, binary_predictions)
    relevance = f1_score(binary_true, binary_predictions, average='weighted')
    
    report = {
        "accuracy": accuracy,
        "relevance": relevance,
        "details": results
    }
    return report


