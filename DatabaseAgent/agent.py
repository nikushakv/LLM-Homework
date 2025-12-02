import os
import sqlite3
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from dotenv import load_dotenv

MODEL_NAME = "gemini-2.0-flash"

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("CRITICAL ERROR: GOOGLE_API_KEY not found.")
    print("Make sure you created the .env file in the same folder.")
    exit()

genai.configure(api_key=api_key)

DB_FILE = "company.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            salary INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def add_employee(name: str, role: str, salary: int):
    """Adds a new employee to the database.
    Args:
        name: The full name of the employee.
        role: The job title or position.
        salary: The annual salary as a number.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO employees (name, role, salary) VALUES (?, ?, ?)", (name, role, salary))
        conn.commit()
        conn.close()
        return {"result": f"Success: Added {name} as {role} with salary ${salary}."}
    except Exception as e:
        return {"result": f"Error adding user: {str(e)}"}

def delete_employee(name: str):
    """Deletes an employee from the database by name.
    Args:
        name: The name of the employee to fire/remove.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM employees WHERE name = ?", (name,))
        if not cursor.fetchone():
            return {"result": f"Error: Employee '{name}' not found in the database."}
            
        cursor.execute("DELETE FROM employees WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        return {"result": f"Success: Removed {name} from records."}
    except Exception as e:
        return {"result": f"Error deleting user: {str(e)}"}

def get_employees():
    """Retrieves a list of all employees in the company."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employees")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return {"result": "The database is currently empty."}
    return {"result": str(rows)}

functions_map = {
    'add_employee': add_employee,
    'delete_employee': delete_employee,
    'get_employees': get_employees
}

print(f"Connecting to AI Model: {MODEL_NAME}...")
try:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        tools=[add_employee, delete_employee, get_employees] 
    )
except Exception as e:
    print(f"Error connecting to model: {e}")
    exit()

def chat_manager():
    init_db()
    print("\n--- HR Database Agent Ready ---")
    print("Commands: 'Hire [Name]', 'Fire [Name]', 'Who works here?'")
    print("Type 'exit' to quit.\n")

    chat = model.start_chat(enable_automatic_function_calling=False)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            response = chat.send_message(user_input)
            part = response.parts[0]
            
            if part.function_call:
                fc = part.function_call
                func_name = fc.name
                func_args = fc.args
                
                print(f"[Agent Internal Thought]: Calling tool '{func_name}' with {func_args}...")

                if func_name in functions_map:
                    args_dict = {key: val for key, val in func_args.items()}
                    tool_result = functions_map[func_name](**args_dict)
                    
                    response = chat.send_message(
                        content=google.generativeai.protos.Content(
                            parts=[google.generativeai.protos.Part(
                                function_response=google.generativeai.protos.FunctionResponse(
                                    name=func_name,
                                    response=tool_result
                                )
                            )]
                        )
                    )
                    
                    print(f"Agent: {response.text}")
                else:
                    print(f"System Error: Agent tried to call unknown function '{func_name}'")
            
            else:
                print(f"Agent: {response.text}")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    import google.generativeai
    chat_manager()