from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter
from bokeh.transform import transform, factor_cmap, cumsum
from bokeh.palettes import Blues256, Oranges256, Reds256, RdBu6, Cividis256, Category20
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from math import pi

def plot_histogram(data, col, bins=25, color=Blues256[0], title='Histogram', xlabel='', ylabel='Count'):
    hist, edges = np.histogram(data[col], bins=bins)
    p = figure(
        title=title,
        x_axis_label=xlabel, y_axis_label=ylabel,
        tools="wheel_zoom,pan,reset", active_scroll="wheel_zoom", toolbar_location='above',
        height=300, sizing_mode='scale_width'
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
        height=300, sizing_mode='scale_width'
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
        height=500,
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
        height=400,
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
        height=400,
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
        title="Condition Distribution",
        height=500,
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
        height=500,
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
        height=500,
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
               x_axis_location="above", plot_width=900, height=400,
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