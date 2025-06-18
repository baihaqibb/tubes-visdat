import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter
from bokeh.transform import transform, factor_cmap, cumsum
from bokeh.palettes import Blues256, Oranges256, Reds256, RdBu6, Cividis256, Category20
from sklearn.preprocessing import OneHotEncoder
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from math import pi

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Used Vehicles Dashboard", layout="wide")
st.title("📊🚘 Used Car Dataset Dashboard")
st.subheader("Kelompok 9:")
st.markdown("- Baihaqi Bintang Bahana - 1301223175")
st.markdown("- Putu Arjuna Nurgraha Eka Wana - 1301223039")
st.markdown("- Binta Adimastama - 1301223005")
st.divider()

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

    df = df[(df['price'] >= 500) & (df['price'] <= 60000)]

    df = df[(df['odometer'] >= 0) & (df['odometer'] <= 300000)]

    df['condition'] = df['condition'].fillna('unknown')
    df['transmission'] = df['transmission'].fillna('other')
    df['fuel'] = df['fuel'].fillna('other')
    df['type'] = df['type'].fillna('other')
    df['manufacturer'] = df['manufacturer'].fillna('other')

    return df


# ===============================
# Filtering Function
# ===============================
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


# ===============================
# Plotting Functions
# ===============================
def plot_histogram(data, col, bins=25, color=Blues256[0], title='Histogram', xlabel='', ylabel='Count'):
    hist, edges = np.histogram(data[col], bins=bins)
    p = figure(
        title=title,
        x_axis_label=xlabel, y_axis_label=ylabel,
        tools="wheel_zoom,pan,reset", active_scroll="wheel_zoom", toolbar_location='above',
        plot_height=300, sizing_mode='scale_width'
    )
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=color, line_color='white', alpha=0.7)
    hover = HoverTool(tooltips=[
        ("Range", "@left{0} - @right{0}"),
        ("Count", "@top")
    ])
    p.add_tools(hover)
    return p


def plot_avg_price_by_year(data):
    avg_price_by_year = data.groupby('year')['price'].mean().reset_index()
    source = ColumnDataSource(avg_price_by_year)
    p = figure(
        title="Avg. Price by Year",
        x_axis_label='Year', y_axis_label='Average Price',
        x_range=(int(data['year'].min()), int(data['year'].max())),
        tools="wheel_zoom,pan,reset", active_scroll="wheel_zoom", toolbar_location='above',
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


def plot_hexbin(data, x_col, y_col, palette=Blues256, title='Hexbin', xlabel='', ylabel='', x_scale=1, y_scale=1):
    x = data[x_col] / x_scale
    y = data[y_col] / y_scale
    p = figure(
        title=title,
        x_axis_label=xlabel, y_axis_label=ylabel,
        x_range=(x.min() - 5, x.max() + 5),
        y_range=(y.min() - 5, y.max() + 5),
        match_aspect=True,
        tools="wheel_zoom,pan,reset", active_scroll="wheel_zoom", toolbar_location='above',
        sizing_mode='scale_width'
    )
    r, bins = p.hexbin(x, y, size=1, palette=palette, hover_color="pink", hover_alpha=0.8, line_color=None)
    hover = HoverTool(tooltips=[("Count", "@c")], mode="mouse", point_policy="follow_mouse", renderers=[r])
    p.add_tools(hover)
    return p


def plot_type_frequency(df, palette=Category20[20]):
    df_type_count = (
        df['type']
        .value_counts()
        .sort_values(ascending=False)
        .reset_index()
    )
    df_type_count.columns = ['type', 'count']

    source = ColumnDataSource(df_type_count)
    types = df_type_count['type'].tolist()

    p = figure(
        x_range=types,
        title="Frequency by Car Type",
        x_axis_label="Car Type",
        y_axis_label="Count",
        plot_height=500,
        sizing_mode='scale_width',
        tools="pan,wheel_zoom,reset",
        active_scroll="wheel_zoom",
        toolbar_location='above'
    )

    p.vbar(
        x='type',
        top='count',
        width=0.7,
        source=source,
        fill_color=factor_cmap('type', palette=palette, factors=types),
        line_color='white'
    )

    p.xaxis.major_label_orientation = 1.0

    hover = HoverTool(tooltips=[("Type", "@type"), ("Count", "@count")])
    p.add_tools(hover)

    return p


def plot_top_manufacturers(df, top_n=10, palette=Category20[10][::-1]):
    man_counts = df['manufacturer'].value_counts().head(top_n)
    df_man = man_counts.reset_index()
    df_man.columns = ['manufacturer', 'count']

    df_man = df_man.iloc[::-1]

    source = ColumnDataSource(df_man)
    manufacturers = df_man['manufacturer'].tolist()

    p = figure(
        y_range=manufacturers,
        x_axis_label="Frequency",
        title=f"Top {top_n} Manufacturers",
        plot_height=400,
        sizing_mode='scale_width',
        tools="pan,wheel_zoom,reset",active_scroll="wheel_zoom",
        toolbar_location='above'
    )

    p.hbar(
        y='manufacturer',
        right='count',
        height=0.6,
        source=source,
        fill_color=factor_cmap('manufacturer', palette=palette, factors=manufacturers),
        line_color='white'
    )

    hover = HoverTool(tooltips=[("Manufacturer", "@manufacturer"), ("Count", "@count")])
    p.add_tools(hover)

    return p


def plot_top_manufacturers_by_price(df, top_n=10, palette=Category20[10][::-1]):
    df_man = (
        df[['manufacturer', 'price']]
        .dropna()
        .groupby('manufacturer', as_index=False)
        .mean()
        .sort_values('price', ascending=False)
        .head(top_n)
    )

    df_man = df_man.iloc[::-1]

    source = ColumnDataSource(df_man)
    manufacturers = df_man['manufacturer'].tolist()

    p = figure(
        y_range=manufacturers,
        x_axis_label="Average Price",
        title=f"Top {top_n} Manufacturers by Avg Price",
        plot_height=400,
        sizing_mode='scale_width',
        tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
        toolbar_location='above'
    )

    p.hbar(
        y='manufacturer',
        right='price',
        height=0.6,
        source=source,
        fill_color=factor_cmap('manufacturer', palette=palette, factors=manufacturers),
        line_color='white'
    )

    hover = HoverTool(tooltips=[
        ("Manufacturer", "@manufacturer"),
        ("Avg Price", "@price{$0,0}")
    ])
    p.add_tools(hover)

    return p


def plot_condition_pie(df):
    condition_order = ["new", "like new", "excellent", "good", "fair", "salvage"]

    df_cond = (
        df['condition']
        .value_counts()
        .reindex(condition_order)
        .fillna(0)
        .astype(int)
    )

    total = df_cond.sum()
    df_pie = pd.DataFrame({
        'condition': df_cond.index,
        'count': df_cond.values,
    })
    df_pie['angle'] = df_pie['count'] / total * 2 * pi
    df_pie['percent'] = df_pie['count'] / total * 100
    df_pie['label'] = df_pie['percent'].map(lambda x: f"{x:.1f}%")
    df_pie['x'] = 0  # no explode
    df_pie['y'] = 0


    cmap = cm.get_cmap("RdBu")
    rd_bu = [mcolors.to_hex(cmap(x)) for x in np.concatenate([
        np.linspace(1, 0.7, 3), np.linspace(0.3, 0, 3)
    ])]
    df_pie['color'] = rd_bu

    source = ColumnDataSource(df_pie)

    p = figure(
        height=500,
        title="Condition Distribution",
        toolbar_location="above",
        tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
        x_range=(-1.5, 1.5),
        y_range=(-1.2, 1.2)
    )

    p.wedge(
        x='x', y='y',
        radius=1,
        start_angle=cumsum('angle', include_zero=True),
        end_angle=cumsum('angle'),
        line_color="white",
        fill_color='color',
        legend_field='condition',
        source=source
    )

    p.axis.visible = False
    p.grid.visible = False
    p.legend.location = "top_left"

    hover = HoverTool(tooltips=[
        ("Condition", "@condition"),
        ("Count", "@count"),
        ("Percent", "@label")
    ])
    p.add_tools(hover)

    return p


def plot_avg_price_by_condition(df):
    condition_order = ["new", "like new", "excellent", "good", "fair", "salvage"]

    df_cond_price = (
        df[['price', 'condition']]
        .dropna()
        .groupby('condition')
        .mean()
        .rename(columns={'price': 'avg'})
        .reindex(condition_order)
        .dropna()
        .reset_index()
    )

    source = ColumnDataSource(df_cond_price)

    palette = RdBu6

    p = figure(
        y_range=condition_order[::-1],  # top to bottom: new -> salvage
        x_axis_label="Average Price",
        title="Avg. Prices by Condition",
        plot_height=500,
        sizing_mode='scale_width',
        tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
        toolbar_location='above'
    )

    p.hbar(
        y='condition',
        right='avg',
        height=0.6,
        source=source,
        fill_color=factor_cmap('condition', palette=palette, factors=condition_order),
        line_color='white'
    )

    hover = HoverTool(tooltips=[("Condition", "@condition"), ("Avg Price", "@avg{$0,0}")])
    p.add_tools(hover)

    return p


def plot_avg_odometer_by_condition(df):
    condition_order = ["new", "like new", "excellent", "good", "fair", "salvage"]

    df_cond_odo = (
        df[['odometer', 'condition']]
        .dropna()
        .groupby('condition')
        .mean()
        .rename(columns={'odometer': 'avg'})
        .reindex(condition_order)
        .dropna()
        .reset_index()
    )

    source = ColumnDataSource(df_cond_odo)

    palette = RdBu6

    p = figure(
        y_range=condition_order[::-1],  # Top to bottom
        x_axis_label="Average Odometer (miles)",
        title="Avg. Odometer by Condition",
        plot_height=500,
        sizing_mode='scale_width',
        tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
        toolbar_location='above'
    )

    p.hbar(
        y='condition',
        right='avg',
        height=0.6,
        source=source,
        fill_color=factor_cmap('condition', palette=palette, factors=condition_order),
        line_color='white'
    )

    hover = HoverTool(tooltips=[("Condition", "@condition"), ("Avg Odometer", "@avg{0,0} mi")])
    p.add_tools(hover)

    return p


def plot_condition_by_year(df):
    ordered_conditions = ["new", "like new", "excellent", "good", "fair", "salvage"]

    df_filtered = df[['year', 'condition', 'price']].dropna()
    df_filtered = df_filtered[df_filtered['condition'].isin(ordered_conditions)]

    df_grouped = df_filtered.groupby(['year', 'condition'])['price'].mean().unstack(fill_value=0)

    for cond in ordered_conditions:
        if cond not in df_grouped.columns:
            df_grouped[cond] = 0
    df_grouped = df_grouped[ordered_conditions].sort_index()

    cmap = cm.get_cmap("RdBu")
    rd_bu = [mcolors.to_hex(cmap(x)) for x in np.concatenate([
        np.linspace(1, 0.7, 3), np.linspace(0.3, 0, 3)
    ])]
    colors = rd_bu[:len(ordered_conditions)]

    p = figure(
        title="Average Price by Condition and Year",
        x_axis_label='Year',
        y_axis_label='Average Price (USD)',
        tools="pan,wheel_zoom,reset",
        active_scroll="wheel_zoom",
        sizing_mode='scale_width',
        height=400
    )

    for i, col in enumerate(df_grouped.columns):
        data = pd.DataFrame({
            'x': df_grouped.index,
            'y': df_grouped[col].values,
            'condition': [col] * len(df_grouped)
        })
        source = ColumnDataSource(data)
        line = p.line(x='x', y='y', source=source, line_width=2, color=colors[i], legend_label=col)
        p.circle(x='x', y='y', source=source, size=4, color=colors[i], legend_label=col)

        hover = HoverTool(tooltips=[
            ("Year", "@x"),
            ("Condition", "@condition"),
            ("Avg Price", "@y{$0,0}")
        ], mode="vline", renderers=[line])
        p.add_tools(hover)

    p.legend.title = "Condition"
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def plot_heatmap_avg_price_by_year_condition(df):
    df = df[['year', 'condition', 'price']].dropna()
    ordered_conditions = ["new", "like new", "excellent", "good", "fair", "salvage"]
    pivot = df.groupby(['year', 'condition'])['price'].mean().reset_index()
    pivot = pivot.pivot(index='condition', columns='year', values='price').reindex(ordered_conditions)
    pivot = pivot.fillna(0)

    data = {'year': [], 'condition': [], 'avg_price': []}
    for condition in pivot.index:
        for year in pivot.columns:
            data['year'].append(str(year))
            data['condition'].append(condition)
            data['avg_price'].append(pivot.loc[condition, year])

    source = ColumnDataSource(data)
    low = min(data['avg_price'])
    high = max(data['avg_price'])
    if low == high:
        low -= 1
        high += 1
    mapper = LinearColorMapper(palette=Cividis256, low=low, high=high)

    p = figure(title="Average Car Price by Year and Condition",
               x_range=sorted(list(set(data['year']))),
               y_range=list(reversed(ordered_conditions)),
               x_axis_location="above", plot_width=900, plot_height=400,
               tools="hover,save,pan,box_zoom,reset,wheel_zoom",
               tooltips=[('Year', '@year'), ('Condition', '@condition'), ('Avg Price', '@avg_price{$0,0}')],
               toolbar_location='right')

    p.rect(x="year", y="condition", width=1, height=1, source=source,
           fill_color=transform('avg_price', mapper), line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="10px",
                         ticker=BasicTicker(desired_num_ticks=10),
                         formatter=PrintfTickFormatter(format="$%d"),
                         label_standoff=12, border_line_color=None, location=(0, 0))

    p.add_layout(color_bar, 'right')
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0

    return p


def plot_pie_transmission(df):
    transmission_order = ["automatic", "manual", "other"]

    df_trans = (
        df['transmission']
        .value_counts()
        .reindex(transmission_order)
        .fillna(0)
        .astype(int)
    )

    total = df_trans.sum()
    df_pie = pd.DataFrame({
        'transmission': df_trans.index,
        'count': df_trans.values,
    })
    df_pie['angle'] = df_pie['count'] / total * 2 * pi
    df_pie['percent'] = df_pie['count'] / total * 100
    df_pie['label'] = df_pie['percent'].map(lambda x: f"{x:.1f}%")
    df_pie['x'] = 0
    df_pie['y'] = 0

    cmap = cm.get_cmap("Blues")
    blues = [mcolors.to_hex(cmap(x)) for x in np.linspace(1, 0.2, len(df_pie))]
    df_pie['color'] = blues

    source = ColumnDataSource(df_pie)

    p = figure(
        height=500,
        title="Transmission Type Distribution",
        toolbar_location="above",
        tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
        x_range=(-1.5, 1.5),
        y_range=(-1.2, 1.2)
    )

    p.wedge(
        x='x', y='y',
        radius=1,
        start_angle=cumsum('angle', include_zero=True),
        end_angle=cumsum('angle'),
        line_color="white",
        fill_color='color',
        legend_field='transmission',
        source=source
    )

    p.axis.visible = False
    p.grid.visible = False
    p.legend.location = "top_left"

    hover = HoverTool(tooltips=[
        ("Transmission", "@transmission"),
        ("Count", "@count"),
        ("Percent", "@label")
    ])
    p.add_tools(hover)

    return p


def plot_pie_fuel(df):
    df_fuel = (
        df['fuel']
        .value_counts()
        .fillna(0)
        .astype(int)
    )

    total = df_fuel.sum()
    df_pie = pd.DataFrame({
        'fuel': df_fuel.index,
        'count': df_fuel.values,
    })
    df_pie['angle'] = df_pie['count'] / total * 2 * pi
    df_pie['percent'] = df_pie['count'] / total * 100
    df_pie['label'] = df_pie['percent'].map(lambda x: f"{x:.1f}%")
    df_pie['x'] = 0
    df_pie['y'] = 0

    cmap = cm.get_cmap("Blues")
    colors = [mcolors.to_hex(cmap(x)) for x in np.linspace(1, 0.2, len(df_pie))]
    df_pie['color'] = colors

    source = ColumnDataSource(df_pie)

    p = figure(
        height=500,
        title="Fuel Type Distribution",
        toolbar_location="above",
        tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom",
        x_range=(-1.5, 1.5),
        y_range=(-1.2, 1.2)
    )

    p.wedge(
        x='x', y='y',
        radius=1,
        start_angle=cumsum('angle', include_zero=True),
        end_angle=cumsum('angle'),
        line_color="white",
        fill_color='color',
        legend_field='fuel',
        source=source
    )

    p.axis.visible = False
    p.grid.visible = False
    p.legend.location = "top_left"

    hover = HoverTool(tooltips=[
        ("Fuel", "@fuel"),
        ("Count", "@count"),
        ("Percent", "@label")
    ])
    p.add_tools(hover)

    return p


if __name__ == "__main__":

    # ===============================
    # Main Application Logic
    # ===============================
    df = load_data("vehicles.csv")

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
        ['new', 'like new', 'excellent', 'good', 'fair', 'salvage'], 
    )
    type_filter = st.sidebar.multiselect(
        "Select Car Types:", 
        ['bus', 'convertible', 'coupe', 'hatchback', 'minivan', 'offroad', 'pickup', 'sedan', 'SUV', 'truck', 'van', 'wagon', 'other'], 
    )

    df_filtered = filter_data(df, year_range, price_range, odo_range, trans_filter, fuel_filter, cond_filter, type_filter)


    # ===============================
    # Layout: Visualizations
    # ===============================
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        st.subheader("Price Distribution")
        st.bokeh_chart(plot_histogram(df_filtered, 'price', title='Price', xlabel='Price'))
    with col1_2:
        st.subheader("Odometer Distribution")
        st.bokeh_chart(plot_histogram(df_filtered, 'odometer', color=Oranges256[0], title='Odometer', xlabel='Odometer'))

    col2_1, col2_2, col2_3 = st.columns(3)
    with col2_1:
        st.subheader("Price by Manufactured Year")
        st.bokeh_chart(plot_hexbin(df_filtered, 'year', 'price', palette=Blues256[::-1], title="Price by Year", xlabel="Year", ylabel="Price * 1000", y_scale=1000))
    with col2_2:
        st.subheader("Odometer by Manufactured Year")
        st.bokeh_chart(plot_hexbin(df_filtered, 'year', 'odometer', palette=Oranges256[::-1], title="Odometer by Year", xlabel="Year", ylabel="Odometer * 5000", y_scale=5000))
    with col2_3:
        st.subheader("Price by Odometer")
        st.bokeh_chart(plot_hexbin(df_filtered, 'price', 'odometer', palette=Reds256[::-1], title="Price by Odometer", xlabel="Price * 1000", ylabel="Odometer * 5000", x_scale=1000, y_scale=5000))

    st.divider()

    col3_1, col3_2, col3_3 = st.columns(3)
    with col3_1:
        st.subheader("Cars Condition Distribution")
        st.bokeh_chart(plot_condition_pie(df_filtered))
    with col3_2:
        st.subheader("Avg. Car Prices by Condition")
        st.bokeh_chart(plot_avg_price_by_condition(df_filtered))
    with col3_3:
        st.subheader("Avg. Car Odometer by Condition")
        st.bokeh_chart(plot_avg_odometer_by_condition(df_filtered))
    
    st.subheader("Heatmap of Avg. Price by Man. Year and Condition")
    st.bokeh_chart(plot_heatmap_avg_price_by_year_condition(df_filtered))

    st.divider()

    col4_1, col4_2, col4_3 = st.columns(3)
    
    with col4_1:
        st.subheader("Car Transmission Distribution")
        st.bokeh_chart(plot_pie_transmission(df_filtered))
    with col4_2:
        st.subheader("Frequency of Car Types")
        st.bokeh_chart(plot_type_frequency(df_filtered))
    with col4_3:
        st.subheader("Car Fuel Type Distribution")
        st.bokeh_chart(plot_pie_fuel(df_filtered))

    col5_1, col5_2 = st.columns(2)

    with col5_1:
        st.subheader("Top 10 Manufacturers by Frequency")
        st.bokeh_chart(plot_top_manufacturers(df_filtered))
    with col5_2:
        st.subheader("Top 10 Manufacturers by Average Price")
        st.bokeh_chart(plot_top_manufacturers_by_price(df_filtered))