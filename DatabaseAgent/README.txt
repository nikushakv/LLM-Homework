No-SQL Database Agent
Student Name: Nikoloz Kvinikadze

Description:
This assignment implements an AI Agent that manages a local SQLite database (company.db) using natural language commands. It uses Google's Gemini Pro/Flash model to determine which Python functions to execute (Function Calling).

Setup Instructions:
1. Unzip the project folder.
2. Install dependencies: 
   pip install -r requirements.txt
3. Create a .env file in this folder and add your API key: 
   GOOGLE_API_KEY=your_key_here
4. Run the agent: 
   python agent.py

Files included:
- agent.py: The main script containing the Logic Loop and Tool Definitions.
- company.db: The SQLite database file (generated automatically).
- requirements.txt: List of necessary Python libraries.