import streamlit as st
from src.load_data import load_data
from src.filters import filter_data
from src.sidebar import render_sidebar
from src.layout import render_layout

st.set_page_config(page_title="Used Vehicles Dashboard", layout="wide")
st.title("\U0001F4CA\U0001F698 Used Car Dataset Dashboard")
st.subheader("Kelompok 9:")
st.markdown("- Baihaqi Bintang Bahana - 1301223175")
st.markdown("- Putu Arjuna Nurgraha Eka Wana - 1301223039")
st.markdown("- Binta Adimastama - 1301223005")
st.divider()

# Load data
df = load_data("vehicles.csv")

# Sidebar controls
year_range, price_range, odo_range, trans_filter, fuel_filter, cond_filter, type_filter = render_sidebar(df)

# Apply filters
df_filtered = filter_data(df, year_range, price_range, odo_range, trans_filter, fuel_filter, cond_filter, type_filter)

# Layout
render_layout(df_filtered)
