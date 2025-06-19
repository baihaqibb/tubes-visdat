import pandas as pd
import streamlit as st

@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)

    df.drop(columns=[col for col in ['county'] if col in df.columns], inplace=True)
    df = df[(df['year'] >= 1970) & (df['year'] <= 2022)]
    df = df.dropna(subset=['odometer'])
    df = df[(df['price'] >= 500) & (df['price'] <= 60000)]
    df = df[(df['odometer'] >= 0) & (df['odometer'] <= 300000)]

    fill_map = {
        'condition': 'unknown',
        'transmission': 'other',
        'fuel': 'other',
        'type': 'other',
        'manufacturer': 'other'
    }
    for col, val in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    return df