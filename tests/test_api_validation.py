import unittest

from app import app, parse_bounded_int


class SimulationValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_rejects_non_object_json(self):
        response = self.client.post('/api/run-simulation', json=['not', 'an', 'object'])
        self.assertEqual(response.status_code, 400)
        self.assertIn('JSON object', response.get_json()['error'])

    def test_rejects_oversized_training_request(self):
        response = self.client.post('/api/run-simulation', json={'n_samples': 1000000})
        self.assertEqual(response.status_code, 400)
        self.assertIn('n_samples', response.get_json()['error'])

    def test_rejects_excessive_random_forest_size(self):
        response = self.client.post('/api/run-simulation', json={'rf_trees': 100000})
        self.assertEqual(response.status_code, 400)
        self.assertIn('rf_trees', response.get_json()['error'])

    def test_rejects_invalid_noise_level(self):
        response = self.client.post('/api/run-simulation', json={'noise_level': 4})
        self.assertEqual(response.status_code, 400)
        self.assertIn('noise_level', response.get_json()['error'])

    def test_rejects_unknown_svm_kernel(self):
        response = self.client.post('/api/run-simulation', json={'svm_kernel': 'unknown'})
        self.assertEqual(response.status_code, 400)
        self.assertIn('svm_kernel', response.get_json()['error'])

    def test_non_string_svm_kernel_returns_400(self):
        for kernel in (['rbf'], {'name': 'rbf'}, 5):
            response = self.client.post('/api/run-simulation', json={'svm_kernel': kernel})
            self.assertEqual(response.status_code, 400, kernel)
            self.assertIn('svm_kernel', response.get_json()['error'])

    def test_infinite_number_returns_400(self):
        # Flask's JSON parser maps out-of-range literals such as 1e999 to float('inf').
        response = self.client.post(
            '/api/run-simulation',
            data='{"n_samples": 1e999}',
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('n_samples must be an integer', response.get_json()['error'])

    def test_boolean_is_not_accepted_as_integer(self):
        with self.assertRaisesRegex(ValueError, 'n_samples must be an integer'):
            parse_bounded_int({'n_samples': True}, 'n_samples', 1000)


if __name__ == '__main__':
    unittest.main()
