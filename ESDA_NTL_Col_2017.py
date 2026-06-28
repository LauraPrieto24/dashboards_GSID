import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import matplotlib.pyplot as plt
from libpysal.weights import KNN
from libpysal.weights.spatial_lag import lag_spatial
from esda.moran import Moran, Moran_Local
from splot.libpysal import plot_spatial_weights
import io
import base64

# Configuración de página de Streamlit para diseño ancho
st.set_page_config(page_title="Colombia Spatial Analysis Dashboard", layout="wide")

# --- Funciones de Caché para Optimizar la Velocidad de Carga ---
@st.cache_data
def load_and_process_data(csv_url, geojson_url):
    df = pd.read_csv(csv_url)
    gdf = gpd.read_file(geojson_url)

    year_col = 'year' if 'year' in df.columns else 'YEAR'
    val_col = 'mean' if 'mean' in df.columns else df.columns[-1]
    data_2017 = df[df[year_col] == 2017].copy()

    # ID Alignment
    geo_id_col, csv_id_col = 'ADM2_CODE', 'ADM2_CODE'
    gdf[geo_id_col] = gdf[geo_id_col].astype(str).str.zfill(5)
    data_2017[csv_id_col] = data_2017[csv_id_col].astype(str).str.zfill(5)

    # Merge
    merged = gdf.merge(data_2017, left_on=geo_id_col, right_on=csv_id_col, how='inner')
    merged = merged.set_geometry('geometry_x' if 'geometry_x' in merged.columns else 'geometry')
    if 'geometry_x' in merged.columns: 
        merged = merged.rename(columns={'geometry_x': 'geometry'}).set_geometry('geometry')

    # Transformation: ln(1 + mean)
    merged['ntl_log'] = np.log1p(merged[val_col])

    # Re-project to EPSG:3116 for precise centroids/spatial weights
    merged = merged.to_crs(epsg=3116)
    return merged

@st.cache_resource
def run_spatial_analysis(_merged):
    w = KNN.from_dataframe(_merged, k=5)
    w.transform = 'R'
    y = _merged['ntl_log'].values
    y_std = (y - y.mean()) / y.std()
    lag_y = lag_spatial(w, y_std)

    moran = Moran(y, w)
    li = Moran_Local(y, w)
    return w, y, y_std, lag_y, moran, li

# --- 1. Carga de Datos desde GitHub ---

csv_url = "NTL_VIIRS_like.csv"
geojson_url = "colombiaGeometry.geojson"

with st.spinner('Cargando y procesando capas geoespaciales desde GitHub...'):
    merged = load_and_process_data(csv_url, geojson_url)
    w, y, y_std, lag_y, moran, li = run_spatial_analysis(merged)

# LISA Quadrants
merged['lisa_cluster'] = 0
sig = 0.05
merged.loc[(li.p_sim <= sig) & (li.q == 1), 'lisa_cluster'] = 1
merged.loc[(li.p_sim <= sig) & (li.q == 3), 'lisa_cluster'] = 2
merged.loc[(li.p_sim <= sig) & (li.q == 2), 'lisa_cluster'] = 3
merged.loc[(li.p_sim <= sig) & (li.q == 4), 'lisa_cluster'] = 4

# Generate Static Weights Plot as Base64 for Dashboard Embedding
fig, ax = plt.subplots(figsize=(8, 10))
plot_spatial_weights(w, merged, ax=ax, node_kws=dict(color='red', markersize=3), edge_kws=dict(linewidth=0.3, color='gray'))
ax.set_title("KNN-5 Connectivity (Projected EPSG:3116)", fontsize=12)
ax.axis('off')
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
plt.close(fig)
weights_img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

# Extract Data for Frontend
merged_web = merged.to_crs(epsg=4326)
geojson_data = merged_web[['geometry', 'ADM2_NAME_x', 'lisa_cluster', 'ntl_log']].to_json()
scatter_data = [{'x': float(x), 'y': float(y_val), 'c': int(c)} for x, y_val, c in zip(y_std, lag_y, merged['lisa_cluster'])]
stats = {"moran_i": round(float(moran.I), 4), "p_value": round(float(moran.p_sim), 4), "z_score": round(float(moran.z_sim), 2), "n_obs": len(merged)}

# Generación del HTML usando strings planos y concatenación con '\n' (Inmune a fallos de indentación)
html_lines = [
    "<!DOCTYPE html><html><head>",
    "<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css' />",
    "<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>",
    "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>",
    "<style>",
    "body { font-family: 'Segoe UI', sans-serif; background: #f4f6f8; margin: 0; padding: 20px; }",
    ".header { background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #1a73e8; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }",
    ".kpi-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }",
    ".kpi-card { background: white; padding: 15px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }",
    ".kpi-val { font-size: 1.5rem; font-weight: bold; color: #1a73e8; }",
    ".main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }",
    ".viz-card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); display: flex; flex-direction: column; min-height: 500px; }",
    "#map { height: 100%; width: 100%; border-radius: 8px; flex-grow: 1; }",
    ".canvas-wrapper { position: relative; flex-grow: 1; min-height: 0; width: 100%; }",
    ".weights-img { width: 100%; height: 100%; object-fit: contain; border-radius: 8px; }",
    "h3 { margin-top: 0; color: #333; font-size: 1.1rem; border-bottom: 1px solid #eee; padding-bottom: 10px; }",
    ".note { font-size: 0.75rem; color: #777; font-style: italic; margin-top: 10px; }",
    "</style></head><body>",
    "<div class='header'>",
    "<h1>📊 Colombia ESDA Dashboard (2017)</h1>",
    "<p>Analysis: <b>ln(1 + mean NTL)</b> | Weights: <b>KNN (k=5)</b> | Projection: <b>EPSG:3116</b></p>",
    "</div>",
    "<div class='kpi-container'>",
    "<div class='kpi-card'>Moran's I <div class='kpi-val'>MORAN_I</div></div>",
    "<div class='kpi-card'>Z-Score <div class='kpi-val'>Z_SCORE</div></div>",
    "<div class='kpi-card'>P-Value <div class='kpi-val'>P_VALUE</div></div>",
    "<div class='kpi-card'>Municipalities <div class='kpi-val'>N_OBS</div></div>",
    "</div>",
    "<div class='main-grid'>",
    "<div class='viz-card'><h3>📍 LISA Cluster Map</h3><div id='map'></div></div>",
    "<div class='viz-card'><h3>🔗 Spatial Weights Connectivity</h3><img class='weights-img' src='data:image/png;base64,WEIGHTS_IMAGE' alt='Spatial Weights Plot'><p class='note'>Projected distance-based KNN connections.</p></div>",
    "<div class='viz-card'><h3>📦 Cluster Distribution</h3><div class='canvas-wrapper'><canvas id='distChart'></canvas></div></div>",
    "<div class='viz-card'><h3>📈 Moran Scatter Plot</h3><div class='canvas-wrapper'><canvas id='scatterPlot'></canvas></div><p class='note'>*Standardized Z-scores. Negative values indicate values below national mean.</p></div>",
    "</div><script>",
    "const geoData = GEOJSON_DATA;",
    "const scatter = SCATTER_DATA;",
    "const map = L.map('map').setView([4.5, -74], 5);",
    "L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png').addTo(map);",
    "const colors = ['#eeeeee', '#d73027', '#4575b4', '#91bfdb', '#fee090'];",
    "const labels = ['Not Significant', 'High-High', 'Low-Low', 'Low-High', 'High-Low'];",
    "L.geoJson(geoData, {",
    "style: f => ({fillColor: colors[f.properties.lisa_cluster], weight: 0.3, fillOpacity: 0.7, color: 'white'}),",
    "onEachFeature: (f, l) => l.bindPopup(`<b>${f.properties.ADM2_NAME_x}</b><br>ln(1+NTL): ${f.properties.ntl_log.toFixed(4)}<br>Cluster: ${labels[f.properties.lisa_cluster]}`)",
    "}).addTo(map);",
    "new Chart(document.getElementById('scatterPlot'), {",
    "type: 'scatter',",
    "data: { datasets: [{ label: 'Municipalities', data: scatter, backgroundColor: scatter.map(d => colors[d.c]) }] },",
    "options: { maintainAspectRatio: false, responsive: true, scales: { x: {title: {display:true, text:'Standardized ln(1+NTL)'}, grid: {color: '#ddd'}}, y: {title: {display:true, text:'Spatial Lag (Wz)'}, grid: {color: '#ddd'}} } }",
    "});",
    "const counts = [0,0,0,0,0]; scatter.forEach(d => counts[d.c]++);",
    "new Chart(document.getElementById('distChart'), {",
    "type: 'bar',",
    "data: { labels: labels, datasets: [{ data: counts, backgroundColor: colors }] },",
    "options: { maintainAspectRatio: false, responsive: true, plugins: { legend: {display:false} }, scales: { y: { beginAtZero: true } } }",
    "});",
    "</script></body></html>"
]

html_template = "\n".join(html_lines)

rendered = html_template.replace('MORAN_I', str(stats['moran_i'])) \
                        .replace('Z_SCORE', str(stats['z_score'])) \
                        .replace('P_VALUE', str(stats['p_value'])) \
                        .replace('N_OBS', str(stats['n_obs'])) \
                        .replace('GEOJSON_DATA', geojson_data) \
                        .replace('SCATTER_DATA', json.dumps(scatter_data)) \
                        .replace('WEIGHTS_IMAGE', weights_img_base64)

# Renderiza el Dashboard interactivo dentro de Streamlit
st.components.v1.html(rendered, height=1100, scrolling=True)
