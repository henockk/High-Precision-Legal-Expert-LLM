from flask import Flask, request, jsonify
from retriever.retriever import create_retriever
from generator.generator import generate_queries_decomposition, generate_answers

app = Flask(__name__)

# Create retriever
retriever = create_retriever('../data/contracts/Robinson Advisory.docx')

@app.route('/generate_queries', methods=['POST'])
def generate_queries():
    data = request.get_json()
    question = data['question']
    queries = generate_queries_decomposition(question)
    return jsonify(queries)

@app.route('/generate_answers', methods=['POST'])
def generate_answers_api():
    data = request.get_json()
    questions = data['questions']
    q_a_pairs = generate_answers(questions, retriever)
    return jsonify(q_a_pairs)

if __name__ == '__main__':
    app.run(debug=True)
