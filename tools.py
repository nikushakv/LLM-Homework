import pandas as pd
import numpy as np
import io
import sys
from contextlib import redirect_stdout

def inspect_metadata(df):
    return {
        "shape": df.shape,
        "null_counts": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

def get_column_stats(df, col):
    if df[col].dtype in ['int64', 'float64']:
        return {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }
    else:
        return {
            "unique_count": df[col].nunique(),
            "top_values": df[col].value_counts().head(5).to_dict()
        }

def impute_missing(df, col, strategy='median'):
    if strategy == 'mean': 
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'median': 
        df[col] = df[col].fillna(df[col].median())
    elif strategy == 'mode': 
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def drop_column(df, col):
    return df.drop(columns=[col])

def create_interaction(df, new_col, expr):
    try:
        df[new_col] = eval(expr, {"df": df, "np": np, "pd": pd})
        return df
    except:
        return df

def encode_categorical(df, col):
    return pd.get_dummies(df, columns=[col], drop_first=True)

def correlation_analysis(df, target):
    numeric_df = df.select_dtypes(include=['number'])
    if target in numeric_df.columns:
        return numeric_df.corr()[target].sort_values(ascending=False).to_dict()
    return {}

def select_top_features(df, target, k=10):
    numeric_df = df.select_dtypes(include=['number'])
    if target in numeric_df.columns:
        corr = numeric_df.corr()[target].abs().sort_values(ascending=False)
        top_features = corr.head(k+1).index.tolist()
        all_cols = [col for col in df.columns if col in top_features or col not in numeric_df.columns]
        return df[all_cols]
    return df

def execute_python_code(code_string):
    f = io.StringIO()
    local_vars = {}
    try:
        with redirect_stdout(f):
            exec(code_string, {"pd": pd, "np": np, "__builtins__": __builtins__}, local_vars)
        output = f.getvalue()
        if 'result' in local_vars:
            return str(local_vars['result'])
        return output if output else "Code executed successfully"
    except Exception as e:
        return f"ERROR: {str(e)}"