import pandas as pd
import numpy as np

def inspect_metadata(df):
    return {
        "shape": df.shape,
        "null_counts": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

def impute_missing(df, col, strategy='median'):
    if strategy == 'mean': df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'median': df[col] = df[col].fillna(df[col].median())
    elif strategy == 'mode': df[col] = df[col].fillna(df[col].mode()[0])
    return df

def drop_column(df, col):
    return df.drop(columns=[col])

def create_interaction(df, new_col, expr):
    try:
        df[new_col] = eval(expr, {"df": df, "np": np})
        return df
    except:
        return df

def encode_categorical(df, col):
    return pd.get_dummies(df, columns=[col], drop_first=True)