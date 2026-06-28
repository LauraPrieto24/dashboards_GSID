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

# 1. Load & Process Data
# Rutas de GitHub en formato Raw para lectura directa
csv_url = "https://raw.githubusercontent.com/LauPrieto/First_AC_Mocoa_Col/main/Datasets/zonal_stats_colombia_municipalities.csv"
geojson_url = "https://raw.githubusercontent.com/LauPrieto/First_AC_Mocoa_Col/main/colombiaGeometry.geojson"

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
if 'geometry_x' in merged.columns: merged = merged.rename(columns={'geometry_x': 'geometry'}).set_geometry('geometry')

# Transformation: ln(1 + mean)
merged['ntl_log'] = np.log1p(merged[val_col])

# Re-project to EPSG:3116 for precise centroids/spatial weights
merged = merged.to_crs(epsg=3116)

# 2. Spatial Analysis
w = KNN.from_dataframe(merged, k=5)
w.transform = 'R'
y = merged['ntl_log'].values
y_std = (y - y.mean()) / y.std()
lag_y = lag_spatial(w, y_std)

moran = Moran(y, w)
li = Moran_Local(y, w)

# LISA Quadrants
merged['lisa_cluster'] = 0
sig = 0.05
merged.loc[(li.p_sim <= sig) & (li.q == 1), 'lisa_cluster'] = 1
merged.loc[(li.p_sim <= sig) & (li.q == 3), 'lisa_cluster'] = 2
merged.loc[(li.p_sim <= sig) & (li.q == 2), 'lisa_cluster'] = 3
merged.loc[(li.p_sim <= sig) & (li.q == 4), 'lisa_cluster'] = 4
