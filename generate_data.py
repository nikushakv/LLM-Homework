import pandas as pd
import numpy as np

def create_raw_data():
    np.random.seed(42)
    data = {
        'User_ID': range(1, 101),
        'Age': [22, 38, np.nan, 35, np.nan, 54, 2, 27, 14, 30] * 10,
        'EstimatedSalary': np.random.uniform(20000, 100000, 100),
        'City': ['London', 'Paris', 'Berlin', 'London', np.nan] * 20,
        'Purchased': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)
    df.to_csv("raw_data.csv", index=False)

if __name__ == "__main__":
    create_raw_data()