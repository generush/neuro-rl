import pandas as pd

def analyze_traj(df: pd.DataFrame):


    # average data across different time steps
    avg_df = df.groupby('TIME').mean().reset_index()
    
    # remove env column
    avg_df = avg_df.drop('ENV', axis=1)

    return avg_df