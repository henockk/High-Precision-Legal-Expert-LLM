import unittest
from src.generator.generator import create_query_generator, create_final_rag_chain

class TestGenerator(unittest.TestCase):
    def test_create_query_generator(self):
        query_generator = create_query_generator()
        self.assertIsNotNone(query_generator)

    def test_create_final_rag_chain(self):
        from src.retriever.retriever import create_retriever
        retriever = create_retriever('data/contracts/Robinson Advisory.docx')
        final_rag_chain = create_final_rag_chain(retriever)
        self.assertIsNotNone(final_rag_chain)

if __name__ == '__main__':
    unittest.main()
