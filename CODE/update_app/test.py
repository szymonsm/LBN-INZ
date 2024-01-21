# Test for functions in the module update_script.py

import unittest
import update_script

class TestUpdateScript(unittest.TestCase):
    """
    Class for testing functions in update_script.py
    """

    def test_update_news(self):
        """
        Test for update_news function
        """
        self.assertEqual(update_script.update_news(update_script.symbols, update_script.ticker_names, update_script.api_keys, update_github=False), 1)

    def test_update_finance(self):
        """
        Test for update_finance function
        """
        self.assertEqual(update_script.update_finance(update_script.symbols, update_github=False), 1)

    def test_update_processed(self):
        """
        Test for update_processed function
        """
        self.assertEqual(update_script.update_processed(update_script.symbols, update_github=False), 1)

    def test_update_predictions(self):
        """
        Test for update_model function
        """
        self.assertEqual(update_script.update_predictions(update_script.symbols, update_github=False), 1)

if __name__ == '__main__':
    unittest.main()

        
