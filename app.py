import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np

st.set_page_config(
    page_title="RTK GeoVisualizer BMX",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ RTK GeoVisualizer — BMX")
st.caption("Telemetría GNSS RTK 10 Hz | Cali, Colombia")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Configuración")
    uploaded = st.file_uploader("Cargar archivo CSV", type=["csv"])
    use_demo = st.checkbox("Usar datos demo", value=True)
    st.divider()
    color_var = st.selectbox(
        "Colorear track por:",
        ["vel_kmh", "elevation_m", "acel_mag", "inc_mag"],
        format_func=lambda x: {
            "vel_kmh":     "Velocidad (km/h)",
            "elevation_m": "Altura (m)",
            "acel_mag":    "Aceleración (m/s²)",
            "inc_mag":     "Inclinación (°)"
        }[x]
    )
    map_style = st.selectbox(
        "Estilo de mapa:",
        ["Esri Satellite", "OpenStreetMap", "CartoDB Dark"]
    )

# ── Carga de datos ────────────────────────────────────────
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    # Renombrar columnas al estándar interno
    df = df.rename(columns={
        "latitude":   "lat",
        "longitude":  "lon",
        "speed_ms":   "vel_ms",
        "acc_x_ms2":  "acel_x",
        "acc_y_ms2":  "acel_y",
        "inc_x_deg":  "inc_x",
        "inc_y_deg":  "inc_y",
    })

    # Velocidad en km/h
    df["vel_kmh"] = df["vel_ms"] * 3.6

    # Magnitudes
    df["acel_mag"] = np.sqrt(df["acel_x"]**2 + df["acel_y"]**2)
    df["inc_mag"]  = np.sqrt(df["inc_x"]**2  + df["inc_y"]**2)

    # Etiqueta de tiempo legible
    df["time_str"] = df["timestamp_s"].apply(
        lambda s: f"{int(s//60):02d}:{s%60:05.2f}"
    )

    return df

if uploaded:
    df = load_csv(uploaded)
elif use_demo:
    df = load_csv("telemetria_bmx_10hz_final.csv")
else:
    st.info("Carga un CSV o activa los datos demo en el panel lateral.")
    st.stop()

# ── Métricas ──────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Puntos GPS",     f"{len(df)}")
c2.metric("Vel. máx",       f"{df['vel_kmh'].max():.1f} km/h")
c3.metric("Vel. promedio",  f"{df['vel_kmh'].mean():.1f} km/h")
c4.metric("Altitud máx",    f"{df['elevation_m'].max():.1f} m")
c5.metric("Acel. máx",      f"{df['acel_mag'].max():.2f} m/s²")

st.divider()

# ── Mapa satelital ────────────────────────────────────────
st.subheader("Mapa geoespacial")

center = [df["lat"].mean(), df["lon"].mean()]
m = folium.Map(location=center, zoom_start=17, control_scale=True)

tiles_map = {
    "Esri Satellite": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Esri WorldImagery"
    ),
    "OpenStreetMap": ("OpenStreetMap", "OpenStreetMap"),
    "CartoDB Dark": (
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        "CartoDB DarkMatter"
    ),
}
tile_url, tile_attr = tiles_map[map_style]
if tile_url != "OpenStreetMap":
    folium.TileLayer(tiles=tile_url, attr=tile_attr).add_to(m)

vals = df[color_var].values
vmin, vmax = vals.min(), vals.max()

def val_to_color(v):
    t = (v - vmin) / (vmax - vmin + 1e-9)
    stops = [
        (0.00, (24,  95, 165)),
        (0.33, (23, 158, 117)),
        (0.66, (239,159,  39)),
        (1.00, (226,  75,  74)),
    ]
    for i in range(len(stops)-1):
        t0, c0 = stops[i]
        t1, c1 = stops[i+1]
        if t <= t1:
            f  = (t - t0) / (t1 - t0)
            r  = int(c0[0] + f*(c1[0]-c0[0]))
            g  = int(c0[1] + f*(c1[1]-c0[1]))
            b  = int(c0[2] + f*(c1[2]-c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#E24B4A"

for i in range(len(df) - 1):
    p1 = [df["lat"].iloc[i],   df["lon"].iloc[i]]
    p2 = [df["lat"].iloc[i+1], df["lon"].iloc[i+1]]
    folium.PolyLine(
        [p1, p2], color=val_to_color(vals[i]), weight=5, opacity=0.9,
        tooltip=(f"T: {df['time_str'].iloc[i]} | "
                 f"Vel: {df['vel_kmh'].iloc[i]:.1f} km/h | "
                 f"Alt: {df['elevation_m'].iloc[i]:.1f} m | "
                 f"Acel: {df['acel_mag'].iloc[i]:.2f} m/s²")
    ).add_to(m)

folium.CircleMarker(
    [df["lat"].iloc[0], df["lon"].iloc[0]],
    radius=8, color="#22c55e", fill=True, fill_color="#22c55e",
    tooltip="Inicio"
).add_to(m)
folium.CircleMarker(
    [df["lat"].iloc[-1], df["lon"].iloc[-1]],
    radius=8, color="#ef4444", fill=True, fill_color="#ef4444",
    tooltip="Fin"
).add_to(m)

st_folium(m, width="100%", height=450)

st.divider()

# ── Gráficas ──────────────────────────────────────────────
st.subheader("Series temporales")

col_a, col_b = st.columns(2)

with col_a:
    fig_vel = px.line(df, x="time_str", y="vel_kmh",
                      title="Velocidad (km/h)",
                      color_discrete_sequence=["#378ADD"])
    fig_vel.update_layout(margin=dict(t=40,b=20), height=250,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Tiempo", yaxis_title="km/h")
    st.plotly_chart(fig_vel, use_container_width=True)

    fig_acel = px.line(df, x="time_str", y=["acel_x","acel_y"],
                       title="Aceleración X / Y (m/s²)",
                       color_discrete_sequence=["#EF9F27","#D4537E"])
    fig_acel.update_layout(margin=dict(t=40,b=20), height=250,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           xaxis_title="Tiempo", yaxis_title="m/s²")
    st.plotly_chart(fig_acel, use_container_width=True)

with col_b:
    fig_alt = px.area(df, x="time_str", y="elevation_m",
                      title="Altura (m)",
                      color_discrete_sequence=["#1D9E75"])
    fig_alt.update_layout(margin=dict(t=40,b=20), height=250,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Tiempo", yaxis_title="m")
    st.plotly_chart(fig_alt, use_container_width=True)

    fig_inc = px.line(df, x="time_str", y=["inc_x","inc_y"],
                      title="Inclinación X / Y (°)",
                      color_discrete_sequence=["#534AB7","#993C1D"])
    fig_inc.update_layout(margin=dict(t=40,b=20), height=250,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Tiempo", yaxis_title="°")
    st.plotly_chart(fig_inc, use_container_width=True)

st.divider()

# ── Tabla de datos ────────────────────────────────────────
with st.expander("Ver datos procesados"):
    st.dataframe(
        df[["time_str","lat","lon","elevation_m",
            "vel_kmh","acel_x","acel_y","inc_x","inc_y"]].rename(columns={
            "time_str":    "Tiempo",
            "lat":         "Latitud",
            "lon":         "Longitud",
            "elevation_m": "Altura (m)",
            "vel_kmh":     "Vel (km/h)",
            "acel_x":      "Acel X (m/s²)",
            "acel_y":      "Acel Y (m/s²)",
            "inc_x":       "Inc X (°)",
            "inc_y":       "Inc Y (°)",
        }),
        use_container_width=True
    )
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV procesado", csv_out,
                       "rtk_procesado.csv", "text/csv")