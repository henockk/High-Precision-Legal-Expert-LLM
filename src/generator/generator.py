from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def generate_queries_decomposition(question):
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0)
    generate_queries_decomposition = (prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))
    return generate_queries_decomposition.invoke({"question":question})

def format_qa_pair(question, answer):
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

def generate_answers(questions, retriever):
    template = """Here is the question you need to answer:
    \n --- \n {question} \n --- \n
    Here is any available background question + answer pairs:
    \n --- \n {q_a_pairs} \n --- \n
    Here is additional context relevant to the question: 
    \n --- \n {context} \n --- \n
    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """
    decomposition_prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    q_a_pairs = ""
    for q in questions:
        rag_chain = (
        {"context": itemgetter("question") | retriever, 
         "question": itemgetter("question"),
         "q_a_pairs": itemgetter("q_a_pairs")} 
        | decomposition_prompt
        | llm
        | StrOutputParser())
        answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
    return q_a_pairs
