import unittest
from src.evaluation.evaluation_pipeline import load_evaluation_data, evaluate_results, run_automated_tests

class TestEvaluation(unittest.TestCase):
    def test_load_evaluation_data(self):
        data = load_evaluation_data('data/evaluation_set/short_contract_questions.json')
        self.assertIn('contracts', data)

    def test_evaluate_results(self):
        results = [
            {
                "question": "What is the termination notice?",
                "predicted_answer": "The termination notice is fourteen (14) days' prior written notice.",
                "true_answer": "According to section 4:14 days for convenience by both parties.",
                "response_time": 5.534
            }
        ]
        report = evaluate_results(results)
        self.assertGreater(report['accuracy'], 0.0)

if __name__ == '__main__':
    unittest.main()
