import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Used Vehicles Dashboard", layout="wide")
st.title("Used Car Dataset Dashboard - Interactive Analysis")


# ===============================
# Data Loading and Cleaning
# ===============================
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)

    if 'county' in df.columns:
        df.drop(['county'], axis=1, inplace=True)

    if 'year' in df.columns:
        df = df[(df['year'] >= 1970) & (df['year'] <= 2022)]

    if 'odometer' in df.columns:
        df.dropna(subset=['odometer'], inplace=True)

    # Clean price
    df = df[(df['price'] >= 500) & (df['price'] <= 60000)]

    # Remove outliers from odometer
    Q1, Q3 = df['odometer'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['odometer'] >= lower_bound) & (df['odometer'] <= upper_bound)]

    return df


# ===============================
# Filtering Function
# ===============================
def filter_data(df, year_range, price_range, odo_range):
    return df[
        (df['year'] >= year_range[0]) & (df['year'] <= year_range[1]) &
        (df['price'] >= price_range[0]) & (df['price'] <= price_range[1]) &
        (df['odometer'] >= odo_range[0]) & (df['odometer'] <= odo_range[1])
    ]


# ===============================
# Plotting Functions
# ===============================
def plot_histogram(data, col, bins=25, color='navy', title='Histogram', xlabel='', ylabel='Count'):
    hist, edges = np.histogram(data[col], bins=bins)
    p = figure(
        title=title,
        x_axis_label=xlabel, y_axis_label=ylabel,
        tools="wheel_zoom,pan,reset", active_scroll="wheel_zoom",
        plot_height=300, sizing_mode='scale_width'
    )
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=color, line_color='white', alpha=0.7)
    hover = HoverTool(tooltips=[
        ("Range", "@left{0} - @right{0}"),
        ("Count", "@top")
    ])
    p.add_tools(hover)
    return p


def plot_avg_price_by_year(df_filtered):
    avg_price_by_year = df_filtered.groupby('year')['price'].mean().reset_index()
    source = ColumnDataSource(avg_price_by_year)
    p = figure(
        title="Average Price by Year",
        x_axis_label='Year', y_axis_label='Average Price',
        x_range=(int(df_filtered['year'].min()), int(df_filtered['year'].max())),
        tools="wheel_zoom,pan,reset", active_scroll="wheel_zoom",
        plot_height=300, sizing_mode='scale_width'
    )
    p.line(x='year', y='price', source=source, line_width=2, color='orange')
    p.circle(x='year', y='price', source=source, size=6, color='orange', alpha=0.6)
    hover = HoverTool(tooltips=[
        ("Year", "@year"),
        ("Avg Price", "@price{$0,0}")
    ])
    p.add_tools(hover)
    return p


def plot_hexbin(data, x_col, y_col, title='Hexbin', xlabel='', ylabel='', x_scale=1, y_scale=1):
    x = data[x_col] / x_scale
    y = data[y_col] / y_scale
    p = figure(
        title=title,
        x_axis_label=xlabel, y_axis_label=ylabel,
        x_range=(x.min() - 5, x.max() + 5),
        y_range=(y.min() - 5, y.max() + 5),
        match_aspect=True,
        tools="wheel_zoom,pan,reset", active_scroll="wheel_zoom",
        sizing_mode='scale_width'
    )
    r, bins = p.hexbin(x, y, size=1, hover_color="pink", hover_alpha=0.8, line_color=None)
    hover = HoverTool(tooltips=[("Count", "@c")], mode="mouse", point_policy="follow_mouse", renderers=[r])
    p.add_tools(hover)
    return p

if __name__ == "__main__":
    # ===============================
    # Main Application Logic
    # ===============================
    df = load_data("vehicles.csv")

    # Sidebar filters
    st.sidebar.header("Filters")
    year_min, year_max = int(df['year'].min()), int(df['year'].max())
    price_min, price_max = int(df['price'].min()), int(df['price'].max())
    odo_min, odo_max = int(df['odometer'].min()), int(df['odometer'].max())

    year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))
    price_range = st.sidebar.slider("Price Range", price_min, price_max, (price_min, price_max))
    odo_range = st.sidebar.slider("Odometer Range", odo_min, odo_max, (odo_min, odo_max))

    # Apply filter
    df_filtered = filter_data(df, year_range, price_range, odo_range)


    # ===============================
    # Layout: Visualizations
    # ===============================
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        st.subheader("Price Distribution")
        st.bokeh_chart(plot_histogram(df_filtered, 'price', color='navy', title='Price', xlabel='Price'))
    with col1_2:
        st.subheader("Odometer Distribution")
        st.bokeh_chart(plot_histogram(df_filtered, 'odometer', color='green', title='Odometer', xlabel='Odometer'))

    st.subheader("Average Price by Year")
    st.bokeh_chart(plot_avg_price_by_year(df_filtered))

    col2_1, col2_2, col2_3 = st.columns(3)
    with col2_1:
        st.subheader("Price by Manufactured Year")
        st.bokeh_chart(plot_hexbin(df_filtered, 'year', 'price', title="Price by Year", xlabel="Year", ylabel="Price * 1000", y_scale=1000))
    with col2_2:
        st.subheader("Odometer by Manufactured Year")
        st.bokeh_chart(plot_hexbin(df_filtered, 'year', 'odometer', title="Odometer by Year", xlabel="Year", ylabel="Odometer * 5000", y_scale=5000))
    with col2_3:
        st.subheader("Price by Odometer")
        st.bokeh_chart(plot_hexbin(df_filtered, 'price', 'odometer', title="Price by Odometer", xlabel="Price * 1000", ylabel="Odometer * 5000", x_scale=1000, y_scale=5000))
