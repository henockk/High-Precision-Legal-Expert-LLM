import unittest
from src.retriever.retriever import load_docx_document, create_retriever

class TestRetriever(unittest.TestCase):
    def test_load_docx_document(self):
        text = load_docx_document('data/contracts/Robinson Advisory.docx')
        self.assertIn("Termination Notice", text)

    def test_create_retriever(self):
        retriever = create_retriever('data/contracts/Robinson Advisory.docx')
        self.assertIsNotNone(retriever)

if __name__ == '__main__':
    unittest.main()
