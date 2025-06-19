import streamlit as st
from src.plots import *

def render_layout(df_filtered):
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        st.subheader("Price Distribution")
        st.bokeh_chart(plot_histogram(df_filtered, 'price', title='Price', xlabel='Price'), use_container_width=True)
    with col1_2:
        st.subheader("Odometer Distribution")
        st.bokeh_chart(plot_histogram(df_filtered, 'odometer', color=Oranges256[0], title='Odometer', xlabel='Odometer'), use_container_width=True)

    col2_1, col2_2, col2_3 = st.columns(3)
    with col2_1:
        st.subheader("Price by Manufactured Year")
        st.bokeh_chart(plot_hexbin(df_filtered, 'year', 'price', palette=Blues256[::-1], title="Price by Year", xlabel="Year", ylabel="Price * 1000", y_scale=1000), use_container_width=True)
    with col2_2:
        st.subheader("Odometer by Manufactured Year")
        st.bokeh_chart(plot_hexbin(df_filtered, 'year', 'odometer', palette=Oranges256[::-1], title="Odometer by Year", xlabel="Year", ylabel="Odometer * 5000", y_scale=5000), use_container_width=True)
    with col2_3:
        st.subheader("Price by Odometer")
        st.bokeh_chart(plot_hexbin(df_filtered, 'price', 'odometer', palette=Reds256[::-1], title="Price by Odometer", xlabel="Price * 1000", ylabel="Odometer * 5000", x_scale=1000, y_scale=5000), use_container_width=True)

    st.divider()

    col3_1, col3_2, col3_3 = st.columns(3)
    with col3_1:
        st.subheader("Cars Condition Distribution")
        st.bokeh_chart(plot_condition_pie(df_filtered), use_container_width=True)
    with col3_2:
        st.subheader("Avg. Car Prices by Condition")
        st.bokeh_chart(plot_avg_price_by_condition(df_filtered), use_container_width=True)
    with col3_3:
        st.subheader("Avg. Car Odometer by Condition")
        st.bokeh_chart(plot_avg_odometer_by_condition(df_filtered), use_container_width=True)

    st.subheader("Heatmap of Avg. Price by Man. Year and Condition")
    st.bokeh_chart(plot_heatmap_avg_price_by_year_condition(df_filtered), use_container_width=True)

    st.divider()

    col4_1, col4_2, col4_3 = st.columns(3)
    with col4_1:
        st.subheader("Car Transmission Distribution")
        st.bokeh_chart(plot_pie_transmission(df_filtered), use_container_width=True)
    with col4_2:
        st.subheader("Frequency of Car Types")
        st.bokeh_chart(plot_type_frequency(df_filtered), use_container_width=True)
    with col4_3:
        st.subheader("Car Fuel Type Distribution")
        st.bokeh_chart(plot_pie_fuel(df_filtered), use_container_width=True)

    col5_1, col5_2 = st.columns(2)
    with col5_1:
        st.subheader("Top 10 Manufacturers by Frequency")
        st.bokeh_chart(plot_top_manufacturers(df_filtered), use_container_width=True)
    with col5_2:
        st.subheader("Top 10 Manufacturers by Average Price")
        st.bokeh_chart(plot_top_manufacturers_by_price(df_filtered), use_container_width=True)
