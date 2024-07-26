import unittest
from src.integration.api import get_final_answer

class TestIntegration(unittest.TestCase):
    def test_get_final_answer(self):
        response = get_final_answer("What is the termination notice?", 'data/contracts/Robinson Advisory.docx')
        self.assertIn("termination notice", response)

if __name__ == '__main__':
    unittest.main()
