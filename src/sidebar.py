import streamlit as st

def render_sidebar(df):
    st.sidebar.header("Filters")
    year_min, year_max = int(df['year'].min()), int(df['year'].max())
    price_min, price_max = int(df['price'].min()), int(df['price'].max())
    odo_min, odo_max = int(df['odometer'].min()), int(df['odometer'].max())

    year_range = st.sidebar.slider("Manufactured Year Range", year_min, year_max, (year_min, year_max))
    price_range = st.sidebar.slider("Price Range", price_min, price_max, (price_min, price_max))
    odo_range = st.sidebar.slider("Odometer Range", odo_min, odo_max, (odo_min, odo_max))

    trans_filter = st.sidebar.selectbox(
        "Select Transmission Type:",
        ['all', 'manual', 'automatic', 'other']
    )
    fuel_filter = st.sidebar.selectbox(
        "Select Fuel Type:",
        ['all', 'gas', 'diesel', 'hybrid', 'electric', 'other']
    )
    cond_filter = st.sidebar.multiselect(
        "Select Conditions:",
        ['new', 'like new', 'excellent', 'good', 'fair', 'salvage']
    )
    type_filter = st.sidebar.multiselect(
        "Select Car Types:",
        ['bus', 'convertible', 'coupe', 'hatchback', 'minivan', 'offroad', 'pickup', 'sedan', 'SUV', 'truck', 'van', 'wagon', 'other']
    )

    return year_range, price_range, odo_range, trans_filter, fuel_filter, cond_filter, type_filter
