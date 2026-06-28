import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from libpysal.weights import KNN
from libpysal.weights.spatial_lag import lag_spatial
from esda.moran import Moran, Moran_Local
from splot.libpysal import plot_spatial_weights

# Configuración de página
st.set_page_config(layout='wide', page_title='Colombia Spatial Analysis')

@st.cache_data
def load_and_process():
    # Nota: Asegúrate de que las rutas sean accesibles en tu entorno local
    df = pd.read_csv('/content/drive/MyDrive/GSID/First_AC_Mocoa_Col/Datasets/zonal_stats_colombia_municipalities.csv')
    gdf = gpd.read_file('https://raw.githubusercontent.com/LauraPrieto24/dashboards_GSID/refs/heads/main/NTL_VIIRS_like.csv')

    year_col = 'year' if 'year' in df.columns else 'YEAR'
    val_col = 'mean' if 'mean' in df.columns else df.columns[-1]
    data_2017 = df[df[year_col] == 2017].copy()

    geo_id_col, csv_id_col = 'ADM2_CODE', 'ADM2_CODE'
    gdf[geo_id_col] = gdf[geo_id_col].astype(str).str.zfill(5)
    data_2017[csv_id_col] = data_2017[csv_id_col].astype(str).str.zfill(5)

    merged = gdf.merge(data_2017, left_on=geo_id_col, right_on=csv_id_col, how='inner')
    if 'geometry_x' in merged.columns:
        merged = merged.rename(columns={'geometry_x': 'geometry'}).set_geometry('geometry')
    
    merged['ntl_log'] = np.log1p(merged[val_col])
    merged = merged.to_crs(epsg=3116)
    
    # Análisis Espacial
    w = KNN.from_dataframe(merged, k=5)
    w.transform = 'R'
    y = merged['ntl_log'].values
    moran = Moran(y, w)
    li = Moran_Local(y, w)
    
    y_std = (y - y.mean()) / y.std()
    lag_y = lag_spatial(w, y_std)

    merged['lisa_cluster'] = 0
    sig = 0.05
    merged.loc[(li.p_sim <= sig) & (li.q == 1), 'lisa_cluster'] = 1 # HH
    merged.loc[(li.p_sim <= sig) & (li.q == 3), 'lisa_cluster'] = 2 # LL
    merged.loc[(li.p_sim <= sig) & (li.q == 2), 'lisa_cluster'] = 3 # LH
    merged.loc[(li.p_sim <= sig) & (li.q == 4), 'lisa_cluster'] = 4 # HL
    
    return merged, w, moran, y_std, lag_y

mdf, w, moran, y_std, lag_y = load_and_process()

# --- UI --- 
st.title("📊 Colombia ESDA Dashboard (2017)")
st.markdown("**Variable:** ln(1 + mean NTL) | **Pesos:** KNN (k=5)")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Moran's I", round(moran.I, 4))
col2.metric("Z-Score", round(moran.z_sim, 2))
col3.metric("P-Value", round(moran.p_sim, 4))
col4.metric("Municipios", len(mdf))

# Layout Principal
row1_1, row1_2 = st.columns(2)

with row1_1:
    st.subheader("📍 LISA Cluster Map")
    mdf_web = mdf.to_crs(epsg=4326)
    fig_map = px.choropleth_mapbox(mdf_web, 
                                   geojson=mdf_web.geometry, 
                                   locations=mdf_web.index, 
                                   color='lisa_cluster', 
                                   color_discrete_map={0:'#eeeeee', 1:'#d73027', 2:'#4575b4', 3:'#91bfdb', 4:'#fee090'},
                                   mapbox_style="carto-positron",
                                   center={"lat": 4.5, "lon": -74}, zoom=4,
                                   hover_data=['ADM2_NAME_x', 'ntl_log'])
    st.plotly_chart(fig_map, use_container_width=True)

with row1_2:
    st.subheader("🔗 Spatial Weights Connectivity")
    fig_w, ax_w = plt.subplots()
    plot_spatial_weights(w, mdf, ax=ax_w, node_kws=dict(color='red', markersize=1), edge_kws=dict(linewidth=0.2, color='gray'))
    ax_w.axis('off')
    st.pyplot(fig_w)

row2_1, row2_2 = st.columns(2)

with row2_1:
    st.subheader("📦 Cluster Distribution")
    cluster_counts = mdf['lisa_cluster'].value_counts().sort_index()
    labels = ['Not Sig', 'High-High', 'Low-Low', 'Low-High', 'High-Low']
    fig_dist = px.bar(x=labels, y=cluster_counts.values, color=labels, 
                       color_discrete_sequence=['#eeeeee', '#d73027', '#4575b4', '#91bfdb', '#fee090'])
    st.plotly_chart(fig_dist, use_container_width=True)

with row2_2:
    st.subheader("📈 Moran Scatter Plot")
    scatter_df = pd.DataFrame({'Z': y_std, 'Wz': lag_y, 'Cluster': mdf['lisa_cluster'].astype(str)})
    fig_scatter = px.scatter(scatter_df, x='Z', y='Wz', color='Cluster', 
                             color_discrete_map={'0':'#eeeeee', '1':'#d73027', '2':'#4575b4', '3':'#91bfdb', '4':'#fee090'})
    st.plotly_chart(fig_scatter, use_container_width=True)
