# 🛍️ Customer Segmentation using RFM and Clustering

This project performs **customer segmentation** using RFM (Recency, Frequency, Monetary) analysis and **KMeans clustering**, with an interactive dashboard built using **Plotly Dash**.

## 📊 Objective

To identify distinct groups of customers based on purchasing behavior, enabling businesses to:

- Identify high-value or loyal customers
- Detect at-risk or inactive users
- Target specific groups with personalized marketing

---

## 📂 Dataset

- Dataset: [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
- Source: UCI Machine Learning Repository
- Format: Excel (`.xlsx`)
- Description: Transactions for a UK-based online retailer from 2010–2011

---

## ⚙️ Features Used — RFM Analysis

| Feature   | Meaning                                  |
|-----------|------------------------------------------|
| Recency   | Days since the last purchase             |
| Frequency | Total number of transactions             |
| Monetary  | Total money spent by the customer        |

---

## 🧠 Machine Learning Approach

- **Scaling**: StandardScaler
- **Clustering**: KMeans (4 clusters)
- **Validation**: Silhouette Score
- **Dimensionality Reduction**: PCA (2D for visualization)

---

## 💻 Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- Plotly, Dash
- Jupyter / VS Code

---

## 📈 Dashboard Preview

The dashboard allows:
- Interactive visualization of clusters using PCA
- Selection of Recency, Frequency, Monetary, or Cluster as the color
- Hover info for individual customers
- Summary table of each cluster's behavior

Run the dashboard with:

```bash
python app.py
