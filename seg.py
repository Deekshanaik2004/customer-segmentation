import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px

# Step 1: Load data
df = pd.read_excel("Online Retail.xlsx")

# Step 2: Data Cleaning
df = df[df['InvoiceNo'].notnull()]
df = df[df['CustomerID'].notnull()]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ✅ Step 3: Create 'TotalSum' before RFM
df['TotalSum'] = df['Quantity'] * df['UnitPrice']

# Step 4: Build RFM Table
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

# Step 5: Normalize RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Step 6: KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(rfm_scaled)
rfm['Cluster'] = clusters

# Step 7: PCA for visualization
pca = PCA(n_components=2)
components = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = components[:, 0]
rfm['PCA2'] = components[:, 1]

# Step 8: Plot the clusters
fig = px.scatter(
    rfm, x='PCA1', y='PCA2', color='Cluster',
    hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary'],
    title="Customer Segments - Real Dataset"
)
fig.show()

# Step 9: Silhouette Score (cluster quality check)
score = silhouette_score(rfm_scaled, clusters)
print(f"Silhouette Score: {score:.2f}")
