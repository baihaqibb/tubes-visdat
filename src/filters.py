import streamlit as st

@st.cache_data
def filter_data(df, year_range, price_range, odo_range, trans_filter, fuel_filter, cond_filter, type_filter):
    filtered = df[
        (df['year'] >= year_range[0]) & (df['year'] <= year_range[1]) &
        (df['price'] >= price_range[0]) & (df['price'] <= price_range[1]) &
        (df['odometer'] >= odo_range[0]) & (df['odometer'] <= odo_range[1])
    ]

    if trans_filter != 'all':
        filtered = filtered[filtered['transmission'] == trans_filter]
    
    if fuel_filter != 'all':
        filtered = filtered[filtered['fuel'] == fuel_filter]

    if cond_filter and len(cond_filter) > 0:
        filtered = filtered[filtered['condition'].isin(cond_filter)]

    if type_filter and len(type_filter) > 0:
        filtered = filtered[filtered['type'].isin(type_filter)]

    return filtered
