import unittest
import os
import sys
import sqlite3

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui.auth import create_usertable, add_userdata, login_user

class TestAuth(unittest.TestCase):
    def setUp(self):
        # Use a temporary database for testing if possible, but the code uses 'data.db' hardcoded.
        # So I will clean up the database before/after tests.
        self.db_name = 'data.db'
        if os.path.exists(self.db_name):
            os.remove(self.db_name)
        create_usertable()

    def tearDown(self):
        if os.path.exists(self.db_name):
            os.remove(self.db_name)

    def test_create_user_and_login(self):
        username = "testuser"
        password = "testpassword"

        # Test adding user
        self.assertTrue(add_userdata(username, password))

        # Test logging in with correct credentials
        result = login_user(username, password)
        self.assertTrue(len(result) > 0)
        self.assertEqual(result[0][0], username)

        # Test logging in with incorrect password
        result = login_user(username, "wrongpassword")
        self.assertEqual(len(result), 0)

        # Test adding duplicate user
        self.assertFalse(add_userdata(username, "newpassword"))

if __name__ == '__main__':
    unittest.main()
