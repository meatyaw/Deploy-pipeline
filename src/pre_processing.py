import pandas as pd
from pathlib import Path

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Deck']  = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else 'Unknown')
    df['Side']  = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else 'Unknown')
    df['Group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    return df

def load_featured(path: str | Path = 'data/raw/train.csv') -> pd.DataFrame:
    return engineer_features(pd.read_csv(path))
