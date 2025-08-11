import dash
from dash import dcc, html, Input, Output, State, dash_table
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(
    '/Users/chngshuyan/projects/sample_hdb/datasets/train.csv',
    low_memory=False,
    dtype={'vacancy': 'float'}
)

# ----------------------------
# Scoring helpers
# ----------------------------
SCHOOL_CAP = 1500
MRT_CAP = 800
MALL_CAP = 1200

def score_distance(series, cap):
    return np.clip(1 - (series.astype(float) / cap), 0, 1)

def safe_bool(series):
    return series.fillna(False).astype(bool)

# ----------------------------
# Options
# ----------------------------
pri_schools = df[['pri_sch_name', 'pri_sch_affiliation']].drop_duplicates().sort_values('pri_sch_name')
pri_sch_options = [
    {
        'label': f"{row['pri_sch_name']}{' ⭐' if bool(row['pri_sch_affiliation']) else ''}",
        'value': row['pri_sch_name']
    }
    for _, row in pri_schools.iterrows()
]
town_options = [{'label': t, 'value': t} for t in sorted(df['town'].dropna().unique())]

def human_money(x):
    x = float(x)
    if x >= 1_000_000:
        return f"${x/1_000_000:.2f}m"
    return f"${x/1_000:.0f}k"

price_min = float(df['resale_price'].min())
price_max = float(df['resale_price'].max())
budget_marks_positions = np.linspace(price_min, price_max, 5)
budget_marks = {int(v): human_money(v) for v in budget_marks_positions}
budget_marks[int(price_min)] = human_money(price_min)
budget_marks[int(price_max)] = human_money(price_max)

# Family-relevant features for analysis
analysis_features = [
    'resale_price',
    'floor_area_sqm',
    'hdb_age',
    'pri_sch_nearest_distance',
    'mrt_nearest_distance',
    'Mall_Nearest_Distance',
    'bus_stop_nearest_distance',
    'Hawker_Nearest_Distance',
    'Mall_Within_1km',
    'Mall_Within_2km',
    'Hawker_Within_1km',
    'Hawker_Within_2km'
]

# ----------------------------
# App init
# ----------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "HDB Resale Flats Recommendation for Young Families"

# ----------------------------
# Layout
# ----------------------------
app.layout = html.Div(style={
    'display': 'flex',
    'flexDirection': 'column',
    'height': '100vh',
    'fontFamily': 'Arial, sans-serif',
    'backgroundColor': '#f5f5f5'
}, children=[

    # Header
    html.Div(style={
        'height': '160px',
        'padding': '0 20px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'textAlign': 'center',
        'color': '#fff',
        'backgroundImage': 'linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)), url("/assets/singapore_hdb.jpg")',
        'backgroundSize': 'cover',
        'backgroundPosition': 'center',
        'backgroundRepeat': 'no-repeat',
        'borderBottom': '2px solid #444',
        'boxShadow': '0 2px 12px rgba(0,0,0,0.15)'
    }, children=[
        html.Div(style={'maxWidth': '980px'}, children=[
            html.H1("HDB Resale Flats Recommendation for Young Families", style={
                'margin': '0',
                'fontSize': '30px',
                'fontWeight': 'bold',
                'lineHeight': '1.2',
                'textShadow': '0 2px 8px rgba(0,0,0,0.6)'
            }),
            html.Div("Find the best fit balancing schools, commute, and conveniences.", style={
                'fontSize': '15px',
                'marginTop': '8px',
                'color': '#e6e6e6',
                'textShadow': '0 1px 6px rgba(0,0,0,0.6)'
            })
        ])
    ]),

    # Content: sidebar + main
    html.Div(style={'display': 'flex', 'flex': '1', 'minHeight': 0}, children=[

        # Sidebar
        html.Div(style={
            'width': '350px',
            'padding': '16px 20px',
            'backgroundColor': '#f9f9f9',
            'borderRight': '1px solid #ccc',
            'overflowY': 'auto'
        }, children=[
            html.H3("Filters & Priorities", style={'marginTop': 0}),

            # Data scope
            html.Div(style={'marginBottom': '18px'}, children=[
                html.Label("Data scope :"),
                dcc.Dropdown(
                    id='data-range',
                    options=[
                        {'label': 'Latest year', 'value': 'latest1'},
                        {'label': 'Latest 2 years', 'value': 'latest2'},
                        {'label': 'All years', 'value': 'all'}
                    ],
                    value='latest1',
                    clearable=False,
                    style={'width': '100%'},
                    placeholder="Select data scope"
                )
            ]),

            # Budget range
            html.Div(style={'marginBottom': '18px'}, children=[
                html.Label("Budget range (SGD)"),
                dcc.RangeSlider(
                    id='budget-slider',
                    min=price_min,
                    max=price_max,
                    step=10000,
                    value=[price_min, price_max],
                    marks=budget_marks,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),

            # Flat type
            html.Div(style={'marginBottom': '18px'}, children=[
                html.Label("Flat type"),
                dcc.Dropdown(
                    id='flat-type-dropdown',
                    options=[{'label': ft, 'value': ft} for ft in sorted(df['flat_type'].dropna().unique())],
                    multi=True,
                    placeholder="Select flat type(s)"
                )
            ]),

            # Maximum flat age
            html.Div(style={'marginBottom': '18px'}, children=[
                html.Label("Maximum flat age (years)"),
                dcc.Slider(
                    id='max-age-slider',
                    min=0,
                    max=int(df['hdb_age'].max()),
                    step=1,
                    value=int(df['hdb_age'].max()),
                    marks={i: str(i) for i in range(0, int(df['hdb_age'].max())+1, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),

            html.Hr(),

            html.H4("Prioritisation", style={'marginTop': '6px'}),

            # Near Primary School
            html.Div(style={'marginBottom': '12px'}, children=[
                dcc.Checklist(
                    options=[{'label': 'Include Near Primary School', 'value': 'include'}],
                    value=['include'],
                    id='pri-school-toggle',
                    style={'color': '#555'}
                ),
                dcc.Dropdown(
                    id='pri-school-dropdown',
                    options=pri_sch_options,
                    placeholder="Select primary school (optional)"
                ),
                dcc.Slider(
                    id='pri-school-slider',
                    min=0, max=100, step=1, value=40,
                    marks={0: '0', 50: '50', 100: '100'}
                ),
                html.Small("⭐ Stars indicate schools with affiliation", style={'color': '#888', 'display': 'block', 'marginTop': '4px'})
            ]),

            # Near Parents
            html.Div(style={'marginBottom': '12px'}, children=[
                dcc.Checklist(
                    options=[{'label': 'Include Near Parents', 'value': 'include'}],
                    value=['include'],
                    id='parents-toggle',
                    style={'color': '#555'}
                ),
                dcc.Dropdown(
                    id='parents-town-dropdown',
                    options=town_options,
                    placeholder="Select parents' town"
                ),
                dcc.Slider(
                    id='parents-slider',
                    min=0, max=100, step=1, value=10,
                    marks={0: '0', 50: '50', 100: '100'}
                )
            ]),

            # Near MRT
            html.Div(style={'marginBottom': '12px'}, children=[
                dcc.Checklist(
                    options=[{'label': 'Include Near MRT', 'value': 'include'}],
                    value=['include'],
                    id='mrt-toggle',
                    style={'color': '#555'}
                ),
                dcc.Slider(
                    id='mrt-slider',
                    min=0, max=100, step=1, value=30,
                    marks={0: '0', 50: '50', 100: '100'}
                )
            ]),

            # Near Mall
            html.Div(style={'marginBottom': '12px'}, children=[
                dcc.Checklist(
                    options=[{'label': 'Include Near Mall', 'value': 'include'}],
                    value=['include'],
                    id='mall-toggle',
                    style={'color': '#555'}
                ),
                dcc.Slider(
                    id='mall-slider',
                    min=0, max=100, step=1, value=20,
                    marks={0: '0', 50: '50', 100: '100'}
                )
            ]),

            html.Hr(),

            # Top N + Download
            html.Div(style={'marginBottom': '12px'}, children=[
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
                    html.Label("How many results to show", style={'margin': 0}),
                    dcc.Dropdown(
                        id='topn-dropdown',
                        options=[{'label': str(n), 'value': n} for n in [5, 10, 15, 20, 25, 30, 40, 50]],
                        value=10,
                        clearable=False,
                        style={'width': '110px'}
                    )
                ])
            ]),

            html.Button(
                "Show Recommended Flats",
                id='submit-button',
                n_clicks=0,
                style={
                    'marginTop': '6px',
                    'width': '100%',
                    'border': 'none',
                    'color': 'white',
                    'backgroundColor': '#007bff',
                    'borderRadius': '14px',
                    'padding': '8px 16px',
                    'fontSize': '14px',
                    'cursor': 'pointer'
                }
            ),
            html.Button(
                "Download CSV",
                id='download-btn',
                n_clicks=0,
                style={
                    'marginTop': '8px',
                    'width': '100%',
                    'border': '1px solid #007bff',
                    'color': '#007bff',
                    'backgroundColor': 'white',
                    'borderRadius': '14px',
                    'padding': '8px 16px',
                    'fontSize': '14px',
                    'cursor': 'pointer'
                }
            ),
            dcc.Download(id='download-csv')
        ]),

        # Main area with tabs
        html.Div(style={'flex': '1', 'padding': '8px', 'overflowY': 'auto'}, children=[
            dcc.Tabs(
                id='main-tabs',
                value='tab-reco',
                children=[
                    dcc.Tab(label='🏆 Recommended Flats', value='tab-reco'),
                    dcc.Tab(label='📊 Data Analysis & Visualisation', value='tab-ana'),
                ],
                style={'marginBottom': '12px'}
            ),
            html.Div(id='tab-content')
        ])
    ])
])

# ----------------------------
# Tab content rendering
# ----------------------------
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab(tab):
    if tab == 'tab-reco':
        return html.Div(children=[
            html.H3("Top Recommended Flats", style={'marginTop': 0}),
            dash_table.DataTable(
                id='results-table',
                columns=[],
                data=[],
                sort_action='native',
                page_action='none',
                row_selectable='single',          # enable single-row selection
                selected_rows=[],                 # default: none selected
                fixed_rows={'headers': True},
                style_table={'overflowX': 'auto', 'maxHeight': '42vh', 'overflowY': 'auto'},
                style_cell={
                    'fontSize': '13px',
                    'padding': '6px',
                    'minWidth': '90px',
                    'whiteSpace': 'normal'
                },
                style_header={'backgroundColor': '#f0f0f0', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'state': 'selected'},
                        'backgroundColor': '#e7f1ff',
                        'borderLeft': '4px solid #007bff'
                    }
                ]
            ),
            html.Div("Tip: select a row to view on the map.", style={'color': '#666', 'fontSize': '12px', 'marginTop': '6px'}),

            # Mini Infor Card 
            html.Div(id='map-info-card', style={
                'marginTop': '10px',
                'padding': '10px 12px',
                'backgroundColor': 'white',
                'border': '1px solid #ddd',
                'borderRadius': '10px',
                'boxShadow': '0 1px 6px rgba(0,0,0,0.06)',
                'fontSize': '13px',
                'color': '#333'
            }),


            html.Div(style={'marginTop': '12px', 'display': 'flex', 'alignItems': 'center', 'gap': '12px'}, children=[
                html.Div("Map layers:", style={'fontSize': '13px', 'color': '#444'}),
                dcc.Checklist(
                    id='map-layers',
                    options=[
                        {'label': 'MRT', 'value': 'mrt'},
                        {'label': 'Bus', 'value': 'bus'},
                        {'label': 'Primary School', 'value': 'pri'},
                        {'label': 'Secondary School', 'value': 'sec'},
                    ],
                    value=['mrt', 'bus', 'pri', 'sec'],
                    inline=True,
                    style={'fontSize': '13px', 'color': '#444'}
                )
            ]),

            dcc.Graph(id='reco-map', style={'height': '420px', 'marginTop': '8px'}),

        ])
    else:
        return html.Div(children=[
            html.H3("Data Analysis & Visualisation", style={'marginTop': 0}),
            html.Div(children=[
                html.Label("Correlation heatmap (family-relevant features)"),
                html.Div(
                    id='corr-summary',
                    style={'fontSize': '13px', 'color': '#555', 'margin': '8px 0 12px'}
                ),
                dcc.Graph(id='corr-heatmap', style={'height': '500px'})

                
            ]),
            html.Hr(),
            html.H4("📈 Price Trend Timeline"),
            html.Div([
                html.Label("Filter by:"),
                dcc.Dropdown(
                    id='trend-filter-type',
                    options=[
                        {'label': 'Town', 'value': 'town'},
                        {'label': 'Flat Type', 'value': 'flat_type'}
                    ],
                    value='town',
                    clearable=False,
                    style={'width': '200px'}
                ),
                dcc.Dropdown(
                    id='trend-filter-value',
                    options=[],  # populated dynamically
                    placeholder="Select a value",
                    style={'width': '300px', 'marginTop': '8px'}
                ),
                html.Label("Time range:", style={'marginTop': '12px'}),
                dcc.Dropdown(
                    id='trend-time-range',
                    options=[
                        {'label': 'Last 1 year', 'value': '1y'},
                        {'label': 'Last 2 years', 'value': '2y'},
                        {'label': 'All available', 'value': 'all'}
                    ],
                    value='2y',
                    clearable=False,
                    style={'width': '200px'}
                ),
                dcc.Checklist(
                    id='trend-rolling-toggle',
                    options=[{'label': 'Show 3-month moving average', 'value': 'smooth'}],
                    value=[],
                    style={'marginTop': '12px'}
                )
            ]),
            dcc.Graph(id='price-trend-graph', style={'height': '400px', 'marginTop': '16px'})



            
        ])

# ----------------------------
# UI enabling/disabling
# ----------------------------
@app.callback(
    [Output('pri-school-dropdown', 'disabled'),
     Output('parents-town-dropdown', 'disabled')],
    [Input('pri-school-toggle', 'value'),
     Input('parents-toggle', 'value')]
)
def toggle_dropdowns(pri_toggle, parents_toggle):
    pri_on = isinstance(pri_toggle, list) and 'include' in pri_toggle
    par_on = isinstance(parents_toggle, list) and 'include' in parents_toggle
    return (not pri_on, not par_on)

@app.callback(
    [Output('pri-school-slider', 'disabled'),
     Output('mrt-slider', 'disabled'),
     Output('mall-slider', 'disabled'),
     Output('parents-slider', 'disabled')],
    [Input('pri-school-toggle', 'value'),
     Input('mrt-toggle', 'value'),
     Input('mall-toggle', 'value'),
     Input('parents-toggle', 'value')]
)
def toggle_sliders(pri_toggle, mrt_toggle, mall_toggle, parents_toggle):
    def is_on(v): return isinstance(v, list) and 'include' in v
    return [not is_on(pri_toggle), not is_on(mrt_toggle), not is_on(mall_toggle), not is_on(parents_toggle)]




# ----------------------------
# Helpers: data scoping and scoring
# ----------------------------
def apply_data_scope(df_in: pd.DataFrame, scope_value: str) -> pd.DataFrame:
    # Prefer Tranc_Year if exists; otherwise derive from Tranc_YearMonth
    if 'Tranc_Year' in df_in.columns:
        years = df_in['Tranc_Year'].dropna().astype(int)
        if years.empty:
            return df_in
        max_year = years.max()
        if scope_value == 'latest1':
            return df_in[years == max_year]
        elif scope_value == 'latest2':
            return df_in[years.isin([max_year, max_year - 1])]
        else:
            return df_in
    elif 'Tranc_YearMonth' in df_in.columns:
        yrs = pd.to_datetime(df_in['Tranc_YearMonth'], errors='coerce').dt.year
        max_year = yrs.max()
        if scope_value == 'latest1':
            return df_in[yrs == max_year]
        elif scope_value == 'latest2':
            return df_in[yrs.isin([max_year, max_year - 1])]
        else:
            return df_in
    return df_in

# ----------------------------
# Main recommendation logic with dynamic columns
# ----------------------------
@app.callback(
    [Output('results-table', 'data'),
     Output('results-table', 'columns')],
    Input('submit-button', 'n_clicks'),
    State('data-range', 'value'),
    State('budget-slider', 'value'),
    State('flat-type-dropdown', 'value'),
    State('max-age-slider', 'value'),
    State('pri-school-toggle', 'value'),
    State('pri-school-dropdown', 'value'),
    State('pri-school-slider', 'value'),
    State('mrt-toggle', 'value'),
    State('mrt-slider', 'value'),
    State('mall-toggle', 'value'),
    State('mall-slider', 'value'),
    State('parents-toggle', 'value'),
    State('parents-town-dropdown', 'value'),
    State('parents-slider', 'value'),
    State('topn-dropdown', 'value'),
    prevent_initial_call=True
)
def recommend_flats(n_clicks, data_scope, budget_range, flat_types, max_age,
                    pri_toggle, pri_school, pri_weight,
                    mrt_toggle, mrt_weight,
                    mall_toggle, mall_weight,
                    parents_toggle, parents_town, parents_weight,
                    topn):

    if not budget_range:
        return [], []

    # Scope data
    df_view = apply_data_scope(df, data_scope)

    # Base filters
    filtered = df_view[
        (df_view['resale_price'] >= budget_range[0]) &
        (df_view['resale_price'] <= budget_range[1]) &
        (df_view['hdb_age'] <= max_age)
    ]
    if flat_types:
        filtered = filtered[filtered['flat_type'].isin(flat_types)]
    filtered = filtered.copy()

    # Priority toggles
    pri_on = isinstance(pri_toggle, list) and 'include' in pri_toggle
    mrt_on = isinstance(mrt_toggle, list) and 'include' in mrt_toggle
    mall_on = isinstance(mall_toggle, list) and 'include' in mall_toggle
    parents_on = isinstance(parents_toggle, list) and 'include' in parents_toggle

    # Optional school restriction if a school is chosen
    if pri_on and pri_school:
        filtered = filtered[filtered['pri_sch_name'] == pri_school].copy()

    # Compute component scores
    if pri_on:
        school_base = score_distance(filtered['pri_sch_nearest_distance'], SCHOOL_CAP)
        aff_bonus = np.where(safe_bool(filtered['pri_sch_affiliation']), 0.05, 0.0)
        filtered['school_score'] = np.clip(school_base + aff_bonus, 0, 1)
    else:
        filtered['school_score'] = 0.0

    if mrt_on:
        mrt_base = score_distance(filtered['mrt_nearest_distance'], MRT_CAP)
        interchange_bonus = np.where(safe_bool(filtered['mrt_interchange']), 0.05, 0.0)
        bus_bonus = np.where(safe_bool(filtered['bus_interchange']), 0.03, 0.0)
        filtered['mrt_score'] = np.clip(mrt_base + interchange_bonus + bus_bonus, 0, 1)
    else:
        filtered['mrt_score'] = 0.0

    if mall_on:
        mall_base = score_distance(filtered['Mall_Nearest_Distance'], MALL_CAP)
        near500 = filtered.get('Mall_Within_500m', pd.Series(0, index=filtered.index)).fillna(0).clip(0, 3)
        near1k = filtered.get('Mall_Within_1km', pd.Series(0, index=filtered.index)).fillna(0).clip(0, 5)
        boost = 0.04 * (near500 / 3.0) + 0.03 * (near1k / 5.0)
        filtered['mall_score'] = np.clip(mall_base + boost, 0, 1)
    else:
        filtered['mall_score'] = 0.0

    if parents_on and parents_town:
        pa_parent = (df.loc[df['town'] == parents_town, 'planning_area'].mode().iloc[0]
                     if not df.loc[df['town'] == parents_town, 'planning_area'].dropna().empty
                     else None)
        same_town = (filtered['town'] == parents_town)
        same_pa = (filtered['planning_area'] == pa_parent) if pa_parent is not None else pd.Series(False, index=filtered.index)
        filtered['parents_score'] = np.where(same_town, 1.0, np.where(same_pa, 0.6, 0.0))
        filtered['parents_match'] = np.where(same_town, 'Same town',
                                             np.where(same_pa, 'Same planning area', 'Other'))
    else:
        filtered['parents_score'] = 0.0
        filtered['parents_match'] = '—'

    # Weights
    w_school = float(pri_weight or 0) if pri_on else 0.0
    w_mrt = float(mrt_weight or 0) if mrt_on else 0.0
    w_mall = float(mall_weight or 0) if mall_on else 0.0
    w_par = float(parents_weight or 0) if parents_on else 0.0
    total_w = w_school + w_mrt + w_mall + w_par

    if total_w == 0:
        filtered['total_score'] = 0.0
    else:
        filtered['total_score'] = (
            w_school * filtered['school_score'] +
            w_mrt * filtered['mrt_score'] +
            w_mall * filtered['mall_score'] +
            w_par * filtered['parents_score']
        ) / total_w

    # Sort and take top N
    filtered = filtered.sort_values('total_score', ascending=False)
    k = int(topn or 10)
    top_flats = filtered.head(k).copy()

    # Base columns (visible)
    columns = [
        {"name": "Address", "id": "address"},
        {"name": "Resale Price (SGD)", "id": "resale_price", "type": "numeric", "format": FormatTemplate.money(0)},
        {"name": "Storey Range", "id": "storey_range"},
        {"name": "Total Score", "id": "total_score", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)}
    ]

    # Dynamic columns linked to prioritisation (visible)
    extra_cols = []
    if pri_on:
        top_flats['pri_sch_nearest_distance'] = top_flats['pri_sch_nearest_distance'].round(0)
        extra_cols.append({"name": "Nearest Primary School (m)", "id": "pri_sch_nearest_distance", "type": "numeric",
                           "format": Format(precision=0, scheme=Scheme.fixed)})
    if mrt_on:
        top_flats['mrt_nearest_distance'] = top_flats['mrt_nearest_distance'].round(0)
        extra_cols.append({"name": "Nearest MRT (m)", "id": "mrt_nearest_distance", "type": "numeric",
                           "format": Format(precision=0, scheme=Scheme.fixed)})
    if mall_on:
        top_flats['Mall_Nearest_Distance'] = top_flats['Mall_Nearest_Distance'].round(0)
        extra_cols.append({"name": "Nearest Mall (m)", "id": "Mall_Nearest_Distance", "type": "numeric",
                           "format": Format(precision=0, scheme=Scheme.fixed)})
    if parents_on:
        extra_cols.append({"name": "Parents Match", "id": "parents_match"})

    # Prepare visible data columns
    show_cols = ['address', 'resale_price', 'storey_range', 'total_score'] + [c['id'] for c in extra_cols]
    for c in show_cols:
        if c not in top_flats.columns:
            top_flats[c] = np.nan

    # Add hidden columns needed for the map and info card
    map_cols = [
        'Latitude', 'Longitude',
        'flat_type', 'flat_model', 'floor_area_sqm', 'hdb_age',
        'mrt_name', 'mrt_nearest_distance', 'mrt_latitude', 'mrt_longitude', 'mrt_interchange', 'bus_interchange',
        'bus_stop_name', 'bus_stop_nearest_distance', 'bus_stop_latitude', 'bus_stop_longitude',
        'pri_sch_name', 'pri_sch_nearest_distance', 'pri_sch_affiliation', 'vacancy', 'pri_sch_latitude', 'pri_sch_longitude',
        'sec_sch_name', 'sec_sch_nearest_dist', 'cutoff_point', 'affiliation', 'sec_sch_latitude', 'sec_sch_longitude',
        'Mall_Nearest_Distance', 'Mall_Within_1km', 'Mall_Within_2km',
        'Hawker_Nearest_Distance', 'Hawker_Within_1km', 'Hawker_Within_2km'
    ]
    for c in map_cols:
        if c not in top_flats.columns:
            top_flats[c] = np.nan

    # Round score
    top_flats['total_score'] = top_flats['total_score'].round(3)

    # Build data dict including hidden map fields
    data = top_flats[show_cols + map_cols].to_dict('records')
    return data, columns + extra_cols

# ----------------------------
# CSV Download (exports current table)
# ----------------------------
@app.callback(
    Output('download-csv', 'data'),
    Input('download-btn', 'n_clicks'),
    State('results-table', 'data'),
    prevent_initial_call=True
)
def download_csv(n_clicks, table_data):
    if not table_data:
        return dash.no_update
    df_out = pd.DataFrame(table_data)
    return dcc.send_data_frame(df_out.to_csv, "hdb_recommendations.csv", index=False)

# ----------------------------
# Map + info card callback
# ----------------------------
@app.callback(
    [Output('reco-map', 'figure'),
     Output('map-info-card', 'children')],
    [Input('results-table', 'data'),
     Input('results-table', 'selected_rows'),
     Input('map-layers', 'value')]
)
def update_map(table_data, selected_rows, layer_values):
    # Default empty figure
    fig = go.Figure()
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # No data yet
    if not table_data:
        fig.update_layout(mapbox=dict(center={'lat': 1.3521, 'lon': 103.8198}, zoom=10))  # Singapore default
        return fig, html.Div("Select filters and click “Show Recommended Flats” to see map results.", style={'color':'#666'})

    df_map = pd.DataFrame(table_data)
    # Filter to rows with coordinates
    df_points = df_map.dropna(subset=['Latitude', 'Longitude']).copy()

    # All flats markers (small grey points)
    if not df_points.empty:
        fig.add_trace(go.Scattermapbox(
            lat=df_points['Latitude'],
            lon=df_points['Longitude'],
            mode='markers',
            marker=dict(size=9, color='rgba(140,140,140,0.75)', symbol='circle'),
            hoverinfo='text',
            name='Flats',
            text=[
                f"{row.get('address','')}<br>"
                f"Price: ${row.get('resale_price', ''):,.0f}<br>"
                f"{row.get('flat_type','')} • {row.get('floor_area_sqm','')} sqm • Age {row.get('hdb_age','')}"
                for _, row in df_points.iterrows()
            ],
            showlegend=False,
            hoverlabel=dict(bgcolor='white')
        ))

    # Selected flat
    info_card = html.Div("Select a row to view details.", style={'color':'#666'})
    sel_idx = (selected_rows or [None])[0]
    center = {'lat': 1.3521, 'lon': 103.8198}
    zoom = 11

    if sel_idx is not None and 0 <= sel_idx < len(df_map):
        sel = df_map.iloc[sel_idx]
        lat, lon = sel.get('Latitude'), sel.get('Longitude')
        if pd.notna(lat) and pd.notna(lon):
            center = {'lat': float(lat), 'lon': float(lon)}
            zoom = 15

            # Highlight marker
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode='markers',
                marker=dict(size=16, color='#007bff', symbol='marker'),
                hoverinfo='text',
                name='Selected Flat',
                text=[f"<b>{sel.get('address','')}</b><br>"
                      f"Price: ${sel.get('resale_price',0):,.0f}<br>"
                      f"{sel.get('flat_type','')} • {sel.get('floor_area_sqm','')} sqm • Age {sel.get('hdb_age','')} yrs<br>"
                      f"Storey: {sel.get('storey_range','')}"],
                showlegend=False
            ))

            # Amenity layers toggles
            wants = set(layer_values or [])
            # MRT
            if 'mrt' in wants and pd.notna(sel.get('mrt_latitude')) and pd.notna(sel.get('mrt_longitude')):
                mrt_label = f"{sel.get('mrt_name','MRT')} • {int(sel.get('mrt_nearest_distance',0))} m"
                flags = []
                if bool(sel.get('mrt_interchange')): flags.append("Interchange")
                if bool(sel.get('bus_interchange')): flags.append("Bus Int.")
                if flags: mrt_label += f" ({', '.join(flags)})"
                fig.add_trace(go.Scattermapbox(
                    lat=[sel.get('mrt_latitude')],
                    lon=[sel.get('mrt_longitude')],
                    mode='markers',
                    marker=dict(size=14, color='#2ecc71', symbol='diamond'),
                    name='MRT',
                    hoverinfo='text',
                    text=[mrt_label],
                    showlegend=False
                ))

            # Bus stop
            if 'bus' in wants and pd.notna(sel.get('bus_stop_latitude')) and pd.notna(sel.get('bus_stop_longitude')):
                bus_label = f"{sel.get('bus_stop_name','Bus Stop')} • {int(sel.get('bus_stop_nearest_distance',0))} m"
                fig.add_trace(go.Scattermapbox(
                    lat=[sel.get('bus_stop_latitude')],
                    lon=[sel.get('bus_stop_longitude')],
                    mode='markers',
                    marker=dict(size=12, color='#f1c40f', symbol='square'),
                    name='Bus',
                    hoverinfo='text',
                    text=[bus_label],
                    showlegend=False
                ))

            # Primary school
            if 'pri' in wants and pd.notna(sel.get('pri_sch_latitude')) and pd.notna(sel.get('pri_sch_longitude')):
                star = " ⭐" if bool(sel.get('pri_sch_affiliation')) else ""
                pri_label = f"{sel.get('pri_sch_name','Primary School')}{star} • {int(sel.get('pri_sch_nearest_distance',0))} m"
                vac = sel.get('vacancy')
                if pd.notna(vac):
                    pri_label += f" • Vacancies: {int(vac)}"
                fig.add_trace(go.Scattermapbox(
                    lat=[sel.get('pri_sch_latitude')],
                    lon=[sel.get('pri_sch_longitude')],
                    mode='markers',
                    marker=dict(size=14, color='#e67e22', symbol='star'),
                    name='Primary School',
                    hoverinfo='text',
                    text=[pri_label],
                    showlegend=False
                ))

            # Secondary school
            if 'sec' in wants and pd.notna(sel.get('sec_sch_latitude')) and pd.notna(sel.get('sec_sch_longitude')):
                flags = " (affiliated)" if bool(sel.get('affiliation')) else ""
                cutoff = sel.get('cutoff_point')
                cutoff_str = f" • Cutoff: {int(cutoff)}" if pd.notna(cutoff) else ""
                sec_label = f"{sel.get('sec_sch_name','Secondary School')}{flags}{cutoff_str} • {int(sel.get('sec_sch_nearest_dist',0))} m"
                fig.add_trace(go.Scattermapbox(
                    lat=[sel.get('sec_sch_latitude')],
                    lon=[sel.get('sec_sch_longitude')],
                    mode='markers',
                    marker=dict(size=14, color='#9b59b6', symbol='star'),
                    name='Secondary School',
                    hoverinfo='text',
                    text=[sec_label],
                    showlegend=False
                ))

            # Mini info card content (persistent under map)
            # Safely convert possibly NaN values to integers
            mall_1km = int(sel['Mall_Within_1km']) if pd.notna(sel['Mall_Within_1km']) else 0
            mall_2km = int(sel['Mall_Within_2km']) if pd.notna(sel['Mall_Within_2km']) else 0
            hawker_1km = int(sel['Hawker_Within_1km']) if pd.notna(sel['Hawker_Within_1km']) else 0
            hawker_2km = int(sel['Hawker_Within_2km']) if pd.notna(sel['Hawker_Within_2km']) else 0
            mall_dist = int(sel['Mall_Nearest_Distance']) if pd.notna(sel['Mall_Nearest_Distance']) else 0
            hawker_dist = int(sel['Hawker_Nearest_Distance']) if pd.notna(sel['Hawker_Nearest_Distance']) else 0

            info_card = html.Div([
                html.Div([
                    html.Span(sel.get('address', ''), style={'fontWeight': 'bold', 'fontSize': '15px'}),
                    html.Span(f"  •  {sel.get('flat_type','')} {sel.get('flat_model','')}", style={'color': '#666'}),
                ]),
                html.Div([
                    html.Span(f"Price: ${sel.get('resale_price',0):,.0f}", style={'marginRight': '8px'}),
                    html.Span(f"Size: {sel.get('floor_area_sqm','')} sqm", style={'marginRight': '8px'}),
                    html.Span(f"Age: {sel.get('hdb_age','')} yrs"),
                ], style={'marginTop': '4px'}),
                html.Div([
                    html.Span("Commute: ", style={'fontWeight': 'bold'}),
                    html.Span(f"MRT {sel.get('mrt_name','')} ({int(sel.get('mrt_nearest_distance',0))} m)"),
                    html.Span(" • "),
                    html.Span(f"Bus {sel.get('bus_stop_name','')} ({int(sel.get('bus_stop_nearest_distance',0))} m)")
                ], style={'marginTop': '6px'}),
                html.Div([
                    html.Span("Schools: ", style={'fontWeight': 'bold'}),
                    html.Span(f"Pri {sel.get('pri_sch_name','')} ({int(sel.get('pri_sch_nearest_distance',0))} m)"),
                    html.Span(" • "),
                    html.Span(f"Sec {sel.get('sec_sch_name','')} ({int(sel.get('sec_sch_nearest_dist',0))} m)")
                ], style={'marginTop': '4px'}),
                html.Div([
                    html.Span("Convenience: ", style={'fontWeight': 'bold'}),
                    html.Span(f"Mall {mall_dist} m; "),
                    html.Span(f"Hawker {hawker_dist} m; "),
                    html.Span(f"Malls within 1/2km: {mall_1km}/{mall_2km}; "),
                    html.Span(f"Hawkers within 1/2km: {hawker_1km}/{hawker_2km}")
                ], style={'marginTop': '4px', 'color': '#555'})
            ])


    # Center and zoom
    if sel_idx is None or not df_points.size:
        # Fallback: center on mean of visible points
        if not df_points.empty:
            center = {'lat': float(df_points['Latitude'].mean()), 'lon': float(df_points['Longitude'].mean())}
            zoom = 11

    fig.update_layout(
        mapbox=dict(center=center, zoom=zoom),
        uirevision='keep'  # prevents redraw on minor prop changes
    )
    return fig, info_card

# ----------------------------
# Analysis tab callbacks
# ----------------------------
@app.callback(
    Output('corr-heatmap', 'figure'),
    Input('data-range', 'value')
)
def update_corr_heatmap(data_scope):
    df_view = apply_data_scope(df, data_scope)
    cols = [c for c in analysis_features if c in df_view.columns]
    df_num = df_view[cols].apply(pd.to_numeric, errors='coerce')
    corr = df_num.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        aspect='auto'
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=24, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

@app.callback(
    Output('corr-summary', 'children'),
    Input('data-range', 'value')
)
def update_corr_summary(data_scope):
    df_view = apply_data_scope(df, data_scope)
    cols = [c for c in analysis_features if c in df_view.columns]
    if len(cols) < 2:
        return html.Div("Not enough data to compute a meaningful summary.")

    df_num = df_view[cols].apply(pd.to_numeric, errors='coerce')
    corr = df_num.corr().round(2)

    def pick(series, k=2):
        series = series.dropna()
        if series.empty:
            return [], []
        pos = series[series > 0].sort_values(ascending=False).head(k)
        neg = series[series < 0].sort_values(ascending=True).head(k)
        return list(pos.items()), list(neg.items())

    label_map = {
        'resale_price': 'resale price',
        'floor_area_sqm': 'floor area',
        'hdb_age': 'flat age',
        'pri_sch_nearest_distance': 'distance to primary school',
        'mrt_nearest_distance': 'distance to MRT',
        'Mall_Nearest_Distance': 'distance to mall',
        'bus_stop_nearest_distance': 'distance to bus stop',
        'Hawker_Nearest_Distance': 'distance to hawker centre',
        'Mall_Within_1km': 'malls within 1km',
        'Mall_Within_2km': 'malls within 2km',
        'Hawker_Within_1km': 'hawker centres within 1km',
        'Hawker_Within_2km': 'hawker centres within 2km',
    }

    def nice(name):
        return label_map.get(name, name.replace('_', ' '))

    bullets = []

    if 'resale_price' in corr.columns:
        s = corr['resale_price'].drop(labels=['resale_price'], errors='ignore')
        pos, neg = pick(s, 2)
        if pos:
            bullets.append(html.Li(
                "Prices tend to be higher when these increase: " +
                ", ".join([f"{nice(k)} ({v:+.2f})" for k, v in pos])
            ))
        if neg:
            bullets.append(html.Li(
                "Prices tend to be lower when these increase: " +
                ", ".join([f"{nice(k)} ({v:+.2f})" for k, v in neg])
            ))

    if 'floor_area_sqm' in corr.columns:
        s = corr['floor_area_sqm'].drop(labels=['floor_area_sqm'], errors='ignore')
        pos, neg = pick(s, 1)
        if pos or neg:
            bullets.append(html.Li(
                "Bigger homes trade-offs: " +
                ", ".join([f"align with higher {nice(k)} ({v:+.2f})" for k, v in pos] +
                          [f"align with lower {nice(k)} ({v:+.2f})" for k, v in neg])
            ))

    if 'hdb_age' in corr.columns:
        s = corr['hdb_age'].drop(labels=['hdb_age'], errors='ignore')
        pos, neg = pick(s, 1)
        if pos or neg:
            bullets.append(html.Li(
                "Older vs newer flats: " +
                ", ".join([f"older correlates with higher {nice(k)} ({v:+.2f})" for k, v in pos] +
                          [f"older correlates with lower {nice(k)} ({v:+.2f})" for k, v in neg])
            ))

    if bullets:
        intro = html.Div(
            "How to read this: positive numbers mean both move together; negative means they move in opposite directions.",
            style={'marginBottom': '6px'}
        )
        return html.Div([intro, html.Ul(bullets)])

    return html.Div("No strong patterns detected for the selected scope.")

@app.callback(
    Output('trend-filter-value', 'options'),
    Input('trend-filter-type', 'value')
)
def update_trend_filter_options(filter_type):
    if filter_type == 'town':
        values = sorted(df['town'].dropna().unique())
    else:
        values = sorted(df['flat_type'].dropna().unique())
    return [{'label': v, 'value': v} for v in values]

@app.callback(
    Output('price-trend-graph', 'figure'),
    [Input('trend-filter-type', 'value'),
     Input('trend-filter-value', 'value'),
     Input('trend-time-range', 'value'),
     Input('trend-rolling-toggle', 'value')]
)
def update_price_trend_graph(filter_type, filter_value, time_range, rolling_toggle):
    df_view = df.copy()
    df_view['Tranc_YearMonth'] = pd.to_datetime(df_view['Tranc_YearMonth'], errors='coerce')
    df_view = df_view.dropna(subset=['Tranc_YearMonth', 'resale_price'])

    # Filter by selected value
    if filter_value:
        df_view = df_view[df_view[filter_type] == filter_value]

    # Time range filter
    latest_date = df_view['Tranc_YearMonth'].max()
    if time_range == '1y':
        cutoff = latest_date - pd.DateOffset(years=1)
        df_view = df_view[df_view['Tranc_YearMonth'] >= cutoff]
    elif time_range == '2y':
        cutoff = latest_date - pd.DateOffset(years=2)
        df_view = df_view[df_view['Tranc_YearMonth'] >= cutoff]

    # Group by month
    df_monthly = df_view.groupby(df_view['Tranc_YearMonth'].dt.to_period('M')).agg(
        median_price=('resale_price', 'median')
    ).reset_index()
    df_monthly['Tranc_YearMonth'] = df_monthly['Tranc_YearMonth'].dt.to_timestamp()

    # Optional rolling average
    if 'smooth' in rolling_toggle:
        df_monthly['smoothed'] = df_monthly['median_price'].rolling(window=3, center=True).mean()

    # Plot
    fig = px.line(df_monthly, x='Tranc_YearMonth', y='median_price',
                  title=f"Median Resale Price Over Time ({filter_value})",
                  labels={'Tranc_YearMonth': 'Month', 'median_price': 'Median Price (SGD)'})

    if 'smooth' in rolling_toggle:
        fig.add_scatter(x=df_monthly['Tranc_YearMonth'], y=df_monthly['smoothed'],
                        mode='lines', name='3-month avg', line=dict(dash='dash', color='orange'))

    fig.update_layout(
    margin=dict(l=0, r=0, t=40, b=0),
    plot_bgcolor='rgba(0,0,0,0)',   # transparent plot area
    paper_bgcolor='rgba(0,0,0,0)'   # transparent outer area
)

    return fig


# ----------------------------
# Run
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
