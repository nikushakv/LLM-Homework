import os
import time
import pandas as pd
import json
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv
import tools

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

SELECTED_MODEL = None

def discover_working_model():
    """Finds a model that actually works for your specific API key."""
    global SELECTED_MODEL
    print("🔍 Searching for available models on your account...")
    try:
        models = client.models.list()
      
        potential = []
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                potential.append(m.name)
        t
        potential.sort(key=lambda x: ("flash" not in x, "lite" not in x, x))
        
        if not potential:
            raise Exception("No generative models found. Check AI Studio project.")
            
        SELECTED_MODEL = potential[0]
        print(f"✅ Found working model: {SELECTED_MODEL}")
        return SELECTED_MODEL
    except Exception as e:
        print(f"❌ Could not list models: {e}")
        
        SELECTED_MODEL = "gemini-2.5-flash-lite"
        return SELECTED_MODEL

def call_gemini(system_msg, user_msg):
    global SELECTED_MODEL
    if not SELECTED_MODEL:
        discover_working_model()
        
    prompt = f"SYSTEM: {system_msg}\n\nUSER: {user_msg}"
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=SELECTED_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            if "429" in str(e):
                print(f"⏳ Rate limit. Waiting 30s...")
                time.sleep(30)
            elif "404" in str(e):
                print("⚠️ Model disappeared. Re-discovering...")
                discover_working_model()
            else:
                raise e
    raise Exception("API failure. Check AI Studio billing/quota.")

def run_cleaner():
    print(f"🕵️ Agent 1 (Cleaner) starting...")
    df = pd.read_csv("raw_data.csv")
    meta = tools.inspect_metadata(df)
    res = call_gemini("Data Cleaner. Return JSON.", f"Meta: {meta}. JSON: {{'drop': [], 'impute': {{}}, 'summary': 'str'}}")
    for col in res.get('drop', []): 
        if col in df.columns: df = tools.drop_column(df, col)
    for col, strat in res.get('impute', {}).items(): 
        if col in df.columns: df = tools.impute_missing(df, col, strat)
    df.to_csv("clean_data.csv", index=False)
    return res.get('summary', 'Cleaned.')

def run_engineer(summary):
    print("🏗️ Agent 2 (Engineer) starting...")
    df = pd.read_csv("clean_data.csv")
    res = call_gemini("Feature Engineer. Return JSON.", f"Data: {df.head(1).to_dict()}. JSON: {{'interaction': {{'name': 'str', 'expr': 'str'}}, 'encode': 'str', 'summary': 'str'}}")
    if res.get('interaction'):
        df = tools.create_interaction(df, res['interaction']['name'], res['interaction']['expr'])
    if res.get('encode') and res['encode'] in df.columns:
        df = tools.encode_categorical(df, res['encode'])
    df.to_csv("engineered_data.csv", index=False)
    return res.get('summary', 'Engineered.')

def run_trainer():
    print("💻 Agent 3 (Trainer) starting...")
    df = pd.read_csv("engineered_data.csv")
    target = "Purchased"
    
    def train(params):
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        X = df.select_dtypes(include=['number']).drop(columns=[target], errors='ignore')
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Use dictionary unpacking for safety
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)
        return accuracy_score(y_test, clf.predict(X_test))

    acc = train({"max_depth": 3, "n_estimators": 50})
    print(f"📊 Accuracy: {acc:.2f}")
    
    if acc < 0.85:
        print("🔄 Tuning hyperparameters...")
        res = call_gemini("XGBoost Expert. Return JSON.", f"Acc: {acc}. JSON: {{'params': {{'max_depth': 6}}}}")
        acc = train(res.get('params', {"max_depth": 5}))
        print(f"📈 Final Accuracy: {acc:.2f}")
    return acc

if __name__ == "__main__":
    import generate_data
    generate_data.create_raw_data()
    try:
        s1 = run_cleaner()
        s2 = run_engineer(s1)
        final_acc = run_trainer()
        with open("Final_Report.md", "w") as f:
            f.write(f"# Multi-Agent AutoML Report\n\nCleaner: {s1}\n\nEngineer: {s2}\n\nAccuracy: {final_acc:.4f}")
        print("\n🏆 Success! Final_Report.md generated using " + SELECTED_MODEL)
    except Exception as e:
        print(f"❌ Script failed: {e}")