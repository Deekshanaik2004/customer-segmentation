import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc

# Load and prepare data
df = pd.read_excel("Online Retail.xlsx")
df = df[df['InvoiceNo'].notnull()]
df = df[df['CustomerID'].notnull()]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalSum'] = df['Quantity'] * df['UnitPrice']

now = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (now - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalSum': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalSum': 'Monetary'
}).reset_index()

# Normalize
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = pca_components[:, 0]
rfm['PCA2'] = pca_components[:, 1]

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("🛍️ Customer Segmentation Dashboard", className="text-center my-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='color-dropdown',
                options=[
                    {'label': 'Cluster', 'value': 'Cluster'},
                    {'label': 'Recency', 'value': 'Recency'},
                    {'label': 'Frequency', 'value': 'Frequency'},
                    {'label': 'Monetary', 'value': 'Monetary'},
                ],
                value='Cluster',
                clearable=False
            )
        ], width=6),
    ], className="mb-3"),

    dcc.Graph(id='cluster-graph'),

    html.Div(id='cluster-summary', className="mt-4")
], fluid=True)

@app.callback(
    Output('cluster-graph', 'figure'),
    Output('cluster-summary', 'children'),
    Input('color-dropdown', 'value')
)
def update_graph(color_column):
    fig = px.scatter(
        rfm, x='PCA1', y='PCA2',
        color=rfm[color_column],
        hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary'],
        title=f"Customer Segmentation by {color_column}"
    )

    summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2).reset_index()
    return fig, html.Div([
        html.H5("📊 Cluster Summary (Mean RFM)"),
        dbc.Table.from_dataframe(summary, striped=True, bordered=True, hover=True)
    ])

if __name__ == "__main__":
    app.run(debug=True)
