import unittest
import os
import sqlite3
from agent import add_employee, delete_employee, get_employees, init_db, DB_FILE

class TestHRAgent(unittest.TestCase):

    def setUp(self):
        """Run before every test: Reset the database."""
        # Remove existing DB to start fresh
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        init_db()

    def test_1_add_employee(self):
        """Test if we can add a user."""
        print("\n--- Test 1: Adding Employee ---")
        result = add_employee("Alice", "Manager", 80000)
        self.assertIn("Success", result["result"])
        
        # Verify directly in SQL
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employees WHERE name='Alice'")
        data = cursor.fetchone()
        conn.close()
        
        # Check if Name matches 'Alice'
        self.assertEqual(data[1], "Alice")
        print("✅ Alice added successfully.")

    def test_2_delete_employee(self):
        """Test if we can delete a user."""
        print("\n--- Test 2: Deleting Employee ---")
        # Add Bob first
        add_employee("Bob", "Builder", 50000)
        
        # Now delete him
        result = delete_employee("Bob")
        self.assertIn("Success", result["result"])
        
        # Verify he is gone
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employees WHERE name='Bob'")
        data = cursor.fetchone()
        conn.close()
        
        self.assertIsNone(data)
        print("✅ Bob deleted successfully.")

    def test_3_error_handling(self):
        """Test deleting someone who doesn't exist."""
        print("\n--- Test 3: Error Handling ---")
        result = delete_employee("Ghost")
        # Should return an error message, NOT crash
        self.assertIn("Error", result["result"])
        print("✅ Error handled gracefully for non-existent user.")

    def test_4_persistence(self):
        """Test if data stays after function closes."""
        print("\n--- Test 4: Persistence ---")
        add_employee("Charlie", "Intern", 0)
        
        # Simulate 'restarting' by just opening a new connection
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employees WHERE name='Charlie'")
        data = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(data)
        print("✅ Database saved the data correctly.")

if __name__ == '__main__':
    unittest.main()