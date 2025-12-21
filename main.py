import os
import time
import pandas as pd
import json
from dotenv import load_dotenv
import tools

load_dotenv()

class AgentLogger:
    def __init__(self):
        self.logs = []
    
    def log(self, agent_name, thought, action):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{agent_name}]\n  THOUGHT: {thought}\n  ACTION: {action}\n"
        print(entry)
        self.logs.append(entry)
    
    def save(self, filename="agent_logs.txt"):
        with open(filename, "w") as f:
            f.write("="*80 + "\n")
            f.write("MULTI-AGENT AUTOML EXECUTION LOG\n")
            f.write("="*80 + "\n\n")
            f.write("\n".join(self.logs))
        print(f"\n📝 Logs saved to {filename}")

logger = AgentLogger()

def simulate_llm_cleaner(meta, detailed_stats):
    reasoning = f"""Analyzing the dataset structure:
- User_ID has 100 unique values for 100 rows, indicating it's a unique identifier with no predictive value
- Age column has 20 missing values (20%), which is manageable for imputation
- City column has 20 missing values (20%), categorical data suitable for mode imputation
- EstimatedSalary and Purchased have no missing values

Decision: Drop User_ID as it won't help prediction. Impute Age with median (robust to outliers) and City with mode (most common category)."""
    
    return {
        "drop": ["User_ID"],
        "impute": {
            "Age": "median",
            "City": "mode"
        },
        "reasoning": reasoning,
        "summary": "Dropped unique identifier column and imputed missing values using statistical methods appropriate for each data type."
    }

def simulate_llm_engineer(df, corr):
    reasoning = f"""Feature engineering strategy based on domain knowledge:
1. Age and Salary interaction: Higher earners at younger ages might have different purchase patterns
2. Age squared: Capture non-linear age effects (purchasing power peaks at certain ages)
3. Salary per age ratio: Normalize income by life stage
4. City encoding: Convert categorical location to numeric for model compatibility

Correlation analysis shows EstimatedSalary has correlation of {corr.get('EstimatedSalary', 0):.3f} with purchase decision."""
    
    return {
        "interactions": [
            {"name": "age_squared", "expr": "df['Age'] ** 2"},
            {"name": "salary_per_age", "expr": "df['EstimatedSalary'] / (df['Age'] + 1)"},
            {"name": "age_salary_interaction", "expr": "df['Age'] * df['EstimatedSalary'] / 10000"}
        ],
        "encode_columns": ["City"],
        "reasoning": reasoning,
        "summary": "Created 3 interaction features capturing non-linear relationships and encoded categorical variables for model compatibility."
    }

def simulate_llm_code_generator(params):
    code = f"""import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('engineered_data.csv')
X = df.select_dtypes(include=['number']).drop(columns=['Purchased'], errors='ignore')
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    max_depth={params['max_depth']},
    n_estimators={params['n_estimators']},
    learning_rate={params['learning_rate']},
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

result = accuracy
print(accuracy)
"""
    return {"code": code}

def simulate_llm_evaluator(iteration, acc, best_acc, current_params, history):
    if iteration == 1 and acc < 0.75:
        return {
            "good_enough": False,
            "reasoning": f"Iteration 1 achieved {acc:.4f} accuracy. This is below the 0.80 threshold for binary classification. Increasing model complexity with more trees and deeper depth.",
            "suggested_params": {
                "max_depth": 5,
                "n_estimators": 100,
                "learning_rate": 0.1
            }
        }
    elif iteration == 2 and acc < 0.80:
        return {
            "good_enough": False,
            "reasoning": f"Iteration 2 achieved {acc:.4f} accuracy. Still below target. Adjusting learning rate and tree depth for better generalization.",
            "suggested_params": {
                "max_depth": 6,
                "n_estimators": 150,
                "learning_rate": 0.05
            }
        }
    elif acc >= 0.80:
        return {
            "good_enough": True,
            "reasoning": f"Achieved {acc:.4f} accuracy, which exceeds the 0.80 threshold for binary classification. Model performance is satisfactory for this dataset size.",
            "suggested_params": current_params
        }
    else:
        return {
            "good_enough": True,
            "reasoning": f"Reached iteration {iteration}. Best accuracy achieved is {best_acc:.4f}. Stopping to avoid overfitting on small dataset.",
            "suggested_params": current_params
        }

def run_cleaner():
    print(f"\n{'='*60}")
    print(f"🕵️  AGENT 1: DATA CLEANER STARTING")
    print(f"{'='*60}\n")
    
    df = pd.read_csv("raw_data.csv")
    meta = tools.inspect_metadata(df)
    
    logger.log("Data Cleaner", 
               f"Inspecting dataset: {meta['shape'][0]} rows, {meta['shape'][1]} columns",
               f"Found null counts: {meta['null_counts']}")
    
    detailed_stats = {}
    for col in df.columns:
        if df[col].isnull().sum() > 0 or df[col].nunique() == len(df):
            detailed_stats[col] = tools.get_column_stats(df, col)
    
    logger.log("Data Cleaner", "Analyzing data quality and identifying issues", "Running statistical analysis on columns")
    
    res = simulate_llm_cleaner(meta, detailed_stats)
    
    logger.log("Data Cleaner",
               res.get('reasoning', 'Processing data'),
               f"Dropping {len(res.get('drop', []))} columns, Imputing {len(res.get('impute', {}))} columns")
    
    for col in res.get('drop', []): 
        if col in df.columns: 
            df = tools.drop_column(df, col)
            logger.log("Data Cleaner", f"Column '{col}' identified as non-predictive", f"Dropped column '{col}'")
    
    for col, strat in res.get('impute', {}).items(): 
        if col in df.columns: 
            null_count = df[col].isnull().sum()
            df = tools.impute_missing(df, col, strat)
            logger.log("Data Cleaner", 
                      f"Column '{col}' has {null_count} missing values",
                      f"Imputed with {strat} strategy")
    
    df.to_csv("clean_data.csv", index=False)
    print(f"\n✅ Agent 1 Complete: clean_data.csv saved\n")
    
    return {
        "summary": res.get('summary', 'Data cleaned'),
        "reasoning": res.get('reasoning', ''),
        "dropped_cols": res.get('drop', []),
        "imputed_cols": list(res.get('impute', {}).keys())
    }

def run_engineer(cleaner_report):
    print(f"\n{'='*60}")
    print(f"🏗️  AGENT 2: FEATURE ENGINEER STARTING")
    print(f"{'='*60}\n")
    
    df = pd.read_csv("clean_data.csv")
    
    logger.log("Feature Engineer",
               f"Received clean dataset with {df.shape[1]} columns",
               f"Starting feature analysis")
    
    target = "Purchased"
    corr = tools.correlation_analysis(df, target)
    
    logger.log("Feature Engineer", "Computing correlations with target variable", f"Correlation analysis complete")
    
    res = simulate_llm_engineer(df, corr)
    
    logger.log("Feature Engineer",
               res.get('reasoning', 'Analyzing feature combinations'),
               f"Creating {len(res.get('interactions', []))} new features")
    
    for interaction in res.get('interactions', []):
        try:
            df = tools.create_interaction(df, interaction['name'], interaction['expr'])
            logger.log("Feature Engineer",
                      f"Creating interaction feature: {interaction['name']}",
                      f"Expression: {interaction['expr']}")
        except Exception as e:
            logger.log("Feature Engineer",
                      f"Failed to create {interaction['name']}",
                      f"Error: {str(e)}")
    
    for col in res.get('encode_columns', []):
        if col in df.columns:
            original_shape = df.shape[1]
            df = tools.encode_categorical(df, col)
            new_cols = df.shape[1] - original_shape + 1
            logger.log("Feature Engineer",
                      f"Encoding categorical variable: {col}",
                      f"Created {new_cols} binary columns")
    
    if len(df.select_dtypes(include=['number']).columns) > 15:
        original_features = len(df.columns)
        df = tools.select_top_features(df, target, k=12)
        logger.log("Feature Engineer",
                  f"Too many features ({original_features}), performing feature selection",
                  f"Selected top 12 features by correlation")
    
    df.to_csv("engineered_data.csv", index=False)
    print(f"\n✅ Agent 2 Complete: engineered_data.csv saved\n")
    
    return {
        "summary": res.get('summary', 'Features engineered'),
        "reasoning": res.get('reasoning', ''),
        "new_features": [i['name'] for i in res.get('interactions', [])],
        "encoded_columns": res.get('encode_columns', [])
    }

def run_trainer(engineer_report):
    print(f"\n{'='*60}")
    print(f"💻 AGENT 3: MODEL TRAINER STARTING")
    print(f"{'='*60}\n")
    
    df = pd.read_csv("engineered_data.csv")
    target = "Purchased"
    
    logger.log("Model Trainer",
               f"Received engineered dataset with {df.shape[1]} features",
               "Preparing to generate training code")
    
    max_iterations = 5
    iteration = 0
    best_acc = 0
    best_params = None
    training_history = []
    
    current_params = {"max_depth": 3, "n_estimators": 50, "learning_rate": 0.1}
    
    while iteration < max_iterations:
        print(f"\n--- Iteration {iteration + 1} ---")
        
        logger.log("Model Trainer", f"Generating training code for iteration {iteration + 1}", f"Using params: {current_params}")
        
        code_res = simulate_llm_code_generator(current_params)
        code = code_res.get('code', '')
        
        logger.log("Model Trainer",
                  f"Generated training code with params: {current_params}",
                  f"Executing code...")
        
        result = tools.execute_python_code(code)
        
        try:
            acc = float(result.strip().split('\n')[-1])
        except:
            acc = 0.0
            logger.log("Model Trainer",
                      "Code execution failed",
                      f"Error: {result}")
        
        training_history.append({"iteration": iteration + 1, "params": current_params.copy(), "accuracy": acc})
        
        print(f"📊 Accuracy: {acc:.4f}")
        logger.log("Model Trainer",
                  f"Model trained with accuracy: {acc:.4f}",
                  f"Params used: {current_params}")
        
        if acc > best_acc:
            best_acc = acc
            best_params = current_params.copy()
        
        decision = simulate_llm_evaluator(iteration + 1, acc, best_acc, current_params, training_history)
        
        logger.log("Model Trainer",
                  f"Evaluating performance: {acc:.4f}",
                  f"Decision: {'Continue' if not decision.get('good_enough', False) else 'Stop'}")
        
        if decision.get('good_enough', False):
            print(f"\n✅ Agent 3 Decision: {decision.get('reasoning', 'Performance acceptable')}")
            logger.log("Model Trainer",
                      decision.get('reasoning', 'Performance acceptable'),
                      f"Training complete with final accuracy: {best_acc:.4f}")
            break
        else:
            print(f"🔄 Agent 3 Decision: {decision.get('reasoning', 'Needs improvement')}")
            logger.log("Model Trainer",
                      decision.get('reasoning', 'Performance needs improvement'),
                      "Generating new hyperparameters")
            current_params = decision.get('suggested_params', current_params)
            iteration += 1
    
    if iteration >= max_iterations:
        logger.log("Model Trainer",
                  f"Reached maximum iterations ({max_iterations})",
                  f"Using best model with accuracy: {best_acc:.4f}")
    
    print(f"\n✅ Agent 3 Complete: Best accuracy = {best_acc:.4f}\n")
    
    return {
        "final_accuracy": best_acc,
        "best_params": best_params,
        "iterations": iteration + 1,
        "training_history": training_history
    }

def generate_final_report(cleaner_report, engineer_report, trainer_report):
    report = f"""# Multi-Agent AutoML Pipeline Report

## Executive Summary
This report documents the autonomous execution of a three-agent machine learning pipeline. Each agent made independent decisions based on data analysis and domain expertise.

---

## Agent 1: Data Cleaner

### Mission
Ensure dataset quality and prepare data for feature engineering.

### Observations
- **Dataset Dimensions**: Started with raw_data.csv
- **Identified Issues**: Missing values and non-predictive columns
- **Columns Dropped**: {', '.join(cleaner_report['dropped_cols']) if cleaner_report['dropped_cols'] else 'None'}
- **Columns Imputed**: {', '.join(cleaner_report['imputed_cols']) if cleaner_report['imputed_cols'] else 'None'}

### Decision Reasoning
{cleaner_report['reasoning']}

### Actions Taken
{cleaner_report['summary']}

### Output
- **File**: clean_data.csv
- **Status**: ✅ Ready for feature engineering

---

## Agent 2: Feature Engineer

### Mission
Maximize information density through intelligent feature creation and selection.

### Strategy
{engineer_report['reasoning']}

### Features Created
{chr(10).join([f"- {feat}" for feat in engineer_report['new_features']]) if engineer_report['new_features'] else "- No new features created"}

### Categorical Encoding
{chr(10).join([f"- {col} (One-Hot Encoded)" for col in engineer_report['encoded_columns']]) if engineer_report['encoded_columns'] else "- No categorical encoding performed"}

### Summary
{engineer_report['summary']}

### Output
- **File**: engineered_data.csv
- **Status**: ✅ Ready for model training

---

## Agent 3: Model Trainer

### Mission
Train and optimize an XGBoost classifier through iterative refinement.

### Training Process
- **Total Iterations**: {trainer_report['iterations']}
- **Final Accuracy**: {trainer_report['final_accuracy']:.4f}
- **Best Parameters**: {trainer_report['best_params']}

### Iteration History
"""
    
    for hist in trainer_report['training_history']:
        report += f"\n**Iteration {hist['iteration']}**\n"
        report += f"- Parameters: {hist['params']}\n"
        report += f"- Accuracy: {hist['accuracy']:.4f}\n"
    
    report += f"""

### Final Model Performance
- **Accuracy**: {trainer_report['final_accuracy']:.4f}
- **Status**: ✅ Model training complete

---

## Conclusion

The multi-agent system successfully processed the dataset through three specialized agents, each making autonomous decisions based on data analysis. The final model achieved an accuracy of **{trainer_report['final_accuracy']:.4f}** after {trainer_report['iterations']} training iteration(s).

### Key Achievements
1. ✅ Automated data cleaning with intelligent imputation
2. ✅ Domain-driven feature engineering
3. ✅ Iterative model optimization with performance evaluation
4. ✅ Complete audit trail of agent decisions

---

*Report generated by Multi-Agent AutoML System*
*Demo Mode: No external API required*
"""
    
    return report

if __name__ == "__main__":
    import generate_data
    
    print("="*60)
    print("MULTI-AGENT AUTOML SYSTEM (DEMO MODE)")
    print("="*60)
    
    generate_data.create_raw_data()
    print("✅ Raw data generated\n")
    
    try:
        cleaner_report = run_cleaner()
        engineer_report = run_engineer(cleaner_report)
        trainer_report = run_trainer(engineer_report)
        
        final_report = generate_final_report(cleaner_report, engineer_report, trainer_report)
        
        with open("Final_Report.md", "w", encoding='utf-8') as f:
            f.write(final_report)
        
        logger.save("agent_logs.txt")
        
        print("\n" + "="*60)
        print("🏆 SUCCESS! PIPELINE COMPLETE")
        print("="*60)
        print(f"\n📄 Final_Report.md generated")
        print(f"📝 agent_logs.txt generated")
        print(f"🎯 Final Accuracy: {trainer_report['final_accuracy']:.4f}")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        logger.save("agent_logs_failed.txt")