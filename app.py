import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import numpy as np
from shapely.geometry import Point, LineString

st.set_page_config(
    page_title="RTK GeoVisualizer BMX",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ RTK GeoVisualizer — BMX")
st.caption("Telemetría GNSS RTK 10 Hz | Cali, Colombia")

# ── Estado de sesión ──────────────────────────────────────
for key, val in {
    "geocercas":    [],
    "linea_inicio": None,
    "linea_fin":    None,
    "analizar":     False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Configuración")
    uploaded  = st.file_uploader("Cargar archivo CSV", type=["csv"])
    use_demo  = st.checkbox("Usar datos demo", value=True)
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

# ── Carga de datos ────────────────────────────────────────
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "latitude":  "lat",
        "longitude": "lon",
        "speed_ms":  "vel_ms",
        "acc_x_ms2": "acel_x",
        "acc_y_ms2": "acel_y",
        "inc_x_deg": "inc_x",
        "inc_y_deg": "inc_y",
    })
    df["vel_kmh"]  = df["vel_ms"] * 3.6
    df["acel_mag"] = np.sqrt(df["acel_x"]**2 + df["acel_y"]**2)
    df["inc_mag"]  = np.sqrt(df["inc_x"]**2  + df["inc_y"]**2)
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

# ── Métricas globales ─────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Puntos GPS",    f"{len(df)}")
c2.metric("Vel. máx",      f"{df['vel_kmh'].max():.1f} km/h")
c3.metric("Vel. promedio", f"{df['vel_kmh'].mean():.1f} km/h")
c4.metric("Altitud máx",   f"{df['elevation_m'].max():.1f} m")
c5.metric("Acel. máx",     f"{df['acel_mag'].max():.2f} m/s²")

st.divider()

# ── Helpers ───────────────────────────────────────────────
def linea_a_shapely(coords):
    return LineString([(c[0], c[1]) for c in coords])

def punto_mas_cercano_idx(df, linea):
    min_dist, idx_min = float("inf"), 0
    for i, row in df.iterrows():
        d = linea.distance(Point(row["lon"], row["lat"]))
        if d < min_dist:
            min_dist, idx_min = d, i
    return idx_min

def segmento_entre_lineas(df, linea_ini, linea_fin):
    idx_ini = punto_mas_cercano_idx(df, linea_ini)
    idx_fin = punto_mas_cercano_idx(df, linea_fin)
    if idx_ini > idx_fin:
        idx_ini, idx_fin = idx_fin, idx_ini
    return df.loc[idx_ini:idx_fin]

# ══════════════════════════════════════════════════════════
# SECCIÓN 1 — MAPA
# ══════════════════════════════════════════════════════════
st.subheader("🗺️ Mapa geoespacial y zonas")

if st.session_state.linea_inicio is None:
    st.info("**Paso 1:** Dibuja la línea de **inicio** de zona con la herramienta de línea.")
elif st.session_state.linea_fin is None:
    st.info("**Paso 2:** Dibuja la línea de **fin** de zona.")
else:
    st.success("Líneas capturadas. Asigna nombre y parámetros en el formulario de abajo.")

center = [df["lat"].mean(), df["lon"].mean()]
m = folium.Map(
    location=center,
    zoom_start=19,
    max_zoom=22,
    control_scale=True
)

# Capa satelital Google — máximo zoom 22
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google Satellite",
    name="Google Satellite",
    max_zoom=22,
    max_native_zoom=22,
).add_to(m)

# Track coloreado
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
            f = (t - t0) / (t1 - t0)
            r = int(c0[0] + f*(c1[0]-c0[0]))
            g = int(c0[1] + f*(c1[1]-c0[1]))
            b = int(c0[2] + f*(c1[2]-c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#E24B4A"

for i in range(len(df) - 1):
    folium.PolyLine(
        [[df["lat"].iloc[i],   df["lon"].iloc[i]],
         [df["lat"].iloc[i+1], df["lon"].iloc[i+1]]],
        color=val_to_color(vals[i]), weight=5, opacity=0.9,
        tooltip=(f"T: {df['time_str'].iloc[i]} | "
                 f"Vel: {df['vel_kmh'].iloc[i]:.1f} km/h | "
                 f"Alt: {df['elevation_m'].iloc[i]:.1f} m")
    ).add_to(m)

# Marcadores inicio / fin recorrido
folium.CircleMarker(
    [df["lat"].iloc[0], df["lon"].iloc[0]],
    radius=8, color="#22c55e", fill=True,
    fill_color="#22c55e", tooltip="Inicio recorrido"
).add_to(m)
folium.CircleMarker(
    [df["lat"].iloc[-1], df["lon"].iloc[-1]],
    radius=8, color="#ef4444", fill=True,
    fill_color="#ef4444", tooltip="Fin recorrido"
).add_to(m)

# Líneas en proceso de captura
if st.session_state.linea_inicio:
    folium.PolyLine(
        [[c[1], c[0]] for c in st.session_state.linea_inicio],
        color="#22c55e", weight=4, tooltip="Línea de inicio"
    ).add_to(m)

if st.session_state.linea_fin:
    folium.PolyLine(
        [[c[1], c[0]] for c in st.session_state.linea_fin],
        color="#ef4444", weight=4, tooltip="Línea de fin"
    ).add_to(m)

# Zonas guardadas
colors_gc = ["#FF6B6B","#4ECDC4","#FFE66D","#A8E6CF","#FF8B94","#B8B8FF"]
for idx, gc in enumerate(st.session_state.geocercas):
    color    = colors_gc[idx % len(colors_gc)]
    df_zona  = gc["df_zona"]
    if len(df_zona) > 1:
        folium.PolyLine(
            [[r["lat"], r["lon"]] for _, r in df_zona.iterrows()],
            color=color, weight=8, opacity=0.55,
            tooltip=gc["nombre"]
        ).add_to(m)
    for lc, label in [
        (gc["linea_inicio"], f"{gc['nombre']} — inicio"),
        (gc["linea_fin"],    f"{gc['nombre']} — fin"),
    ]:
        folium.PolyLine(
            [[c[1], c[0]] for c in lc],
            color=color, weight=3, dash_array="6", tooltip=label
        ).add_to(m)
    if len(df_zona) > 0:
        mid = df_zona.iloc[len(df_zona)//2]
        folium.Marker(
            [mid["lat"], mid["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:bold;'
                     f'color:{color};background:rgba(0,0,0,0.65);'
                     f'padding:2px 6px;border-radius:4px;white-space:nowrap;">'
                     f'{gc["nombre"]}</div>'
            )
        ).add_to(m)

# Herramienta de dibujo
Draw(
    export=False,
    draw_options={
        "polyline":     True,
        "polygon":      False,
        "rectangle":    False,
        "circle":       False,
        "marker":       False,
        "circlemarker": False,
    },
    edit_options={"edit": False}
).add_to(m)

map_data = st_folium(m, width="100%", height=500, key="main_map",
                     returned_objects=["last_active_drawing"])

# ── Capturar líneas — sin rerun automático ────────────────
if map_data and map_data.get("last_active_drawing"):
    geom = map_data["last_active_drawing"].get("geometry", {})
    if geom.get("type") == "LineString":
        coords = geom["coordinates"]
        if st.session_state.linea_inicio is None:
            st.session_state.linea_inicio = coords
        elif st.session_state.linea_fin is None:
            if coords != st.session_state.linea_inicio:
                st.session_state.linea_fin = coords

# ── Botón resetear ────────────────────────────────────────
if st.session_state.linea_inicio or st.session_state.linea_fin:
    if st.button("🔄 Resetear líneas"):
        st.session_state.linea_inicio = None
        st.session_state.linea_fin    = None
        st.rerun()

# ══════════════════════════════════════════════════════════
# SECCIÓN 2 — FORMULARIO DE ZONA
# ══════════════════════════════════════════════════════════
st.divider()
st.subheader("➕ Registrar nueva zona")

if (st.session_state.linea_inicio is not None and
        st.session_state.linea_fin is not None):

    linea_ini_sh = linea_a_shapely(st.session_state.linea_inicio)
    linea_fin_sh = linea_a_shapely(st.session_state.linea_fin)
    df_prev      = segmento_entre_lineas(df, linea_ini_sh, linea_fin_sh)

    st.success(f"Segmento detectado: **{len(df_prev)} puntos GPS** | "
               f"Vel: {df_prev['vel_kmh'].min():.1f} – "
               f"{df_prev['vel_kmh'].max():.1f} km/h")

    with st.form("form_zona", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            nombre = st.text_input("Nombre de la zona",
                                   placeholder="Ej: Recta inicial")
        with col2:
            tipo = st.selectbox("Tipo de zona",
                ["Recta", "Curva", "Salto", "Frenada", "Otra"])
        st.markdown("**Parámetros objetivo**")
        c1, c2, c3 = st.columns(3)
        with c1:
            vel_max_obj = st.number_input("Vel. máx objetivo (km/h)",
                                          0.0, 200.0, 40.0, 0.5)
        with c2:
            vel_min_obj = st.number_input("Vel. mín objetivo (km/h)",
                                          0.0, 200.0, 20.0, 0.5)
        with c3:
            vel_avg_obj = st.number_input("Vel. promedio objetivo (km/h)",
                                          0.0, 200.0, 30.0, 0.5)

        if st.form_submit_button("💾 Guardar zona") and nombre:
            st.session_state.geocercas.append({
                "nombre":       nombre,
                "tipo":         tipo,
                "linea_inicio": st.session_state.linea_inicio,
                "linea_fin":    st.session_state.linea_fin,
                "df_zona":      df_prev,
                "vel_max_obj":  vel_max_obj,
                "vel_min_obj":  vel_min_obj,
                "vel_avg_obj":  vel_avg_obj,
            })
            st.session_state.linea_inicio = None
            st.session_state.linea_fin    = None
            st.rerun()
else:
    if st.session_state.linea_inicio is None:
        st.warning("Dibuja la línea de inicio en el mapa.")
    else:
        st.warning("Ahora dibuja la línea de fin en el mapa.")

# ── Gestión de zonas ──────────────────────────────────────
if st.session_state.geocercas:
    with st.expander("🗑️ Gestionar zonas guardadas"):
        for idx, gc in enumerate(st.session_state.geocercas):
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{gc['nombre']}** — {gc['tipo']} "
                       f"({len(gc['df_zona'])} puntos)")
            if col2.button("Eliminar", key=f"del_{idx}"):
                st.session_state.geocercas.pop(idx)
                st.rerun()

# ══════════════════════════════════════════════════════════
# SECCIÓN 3 — BOTÓN ANALIZAR
# ══════════════════════════════════════════════════════════
st.divider()

if st.session_state.geocercas:
    if st.button("🔍 Analizar zonas", type="primary", use_container_width=True):
        st.session_state.analizar = True

if st.session_state.analizar and st.session_state.geocercas:
    st.subheader("📊 Análisis de rendimiento por zona")

    resultados = []
    for gc in st.session_state.geocercas:
        df_zona = gc["df_zona"]
        resultados.append({
            "Zona":          gc["nombre"],
            "Tipo":          gc["tipo"],
            "Puntos":        len(df_zona),
            "Vel máx real":  round(df_zona["vel_kmh"].max(), 1) if len(df_zona) else None,
            "Vel mín real":  round(df_zona["vel_kmh"].min(), 1) if len(df_zona) else None,
            "Vel prom real": round(df_zona["vel_kmh"].mean(), 1) if len(df_zona) else None,
            "Vel máx obj":   gc["vel_max_obj"],
            "Vel mín obj":   gc["vel_min_obj"],
            "Vel prom obj":  gc["vel_avg_obj"],
        })

    df_res = pd.DataFrame(resultados)

    # Tabla comparativa
    st.markdown("#### Tabla comparativa objetivo vs real")

    def highlight(row):
        styles = [""] * len(row)
        cols   = row.index.tolist()
        for real_col, obj_col in [
            ("Vel máx real",  "Vel máx obj"),
            ("Vel mín real",  "Vel mín obj"),
            ("Vel prom real", "Vel prom obj"),
        ]:
            if real_col in cols and obj_col in cols:
                real = row[real_col]
                obj  = row[obj_col]
                if real is not None and obj is not None:
                    diff  = abs(real - obj)
                    color = (
                        "background-color: #1a4a1a" if diff <= obj * 0.05
                        else "background-color: #4a3a0a" if diff <= obj * 0.15
                        else "background-color: #4a1a1a"
                    )
                    styles[cols.index(real_col)] = color
        return styles

    st.dataframe(
        df_res.style.apply(highlight, axis=1),
        use_container_width=True
    )
    st.caption("🟢 Dentro del 5% del objetivo  |  🟡 Dentro del 15%  |  🔴 Fuera del rango")

    # Gráficas
    st.markdown("#### Gráfica objetivo vs real por zona")
    zonas_ok = df_res[df_res["Puntos"] > 0]

    for metric, label in [
        ("Vel máx",  "Velocidad máxima (km/h)"),
        ("Vel mín",  "Velocidad mínima (km/h)"),
        ("Vel prom", "Velocidad promedio (km/h)"),
    ]:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Objetivo",
            x=zonas_ok["Zona"],
            y=zonas_ok[f"{metric} obj"],
            marker_color="#378ADD", opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            name="Real",
            x=zonas_ok["Zona"],
            y=zonas_ok[f"{metric} real"],
            marker_color="#EF9F27", opacity=0.9,
        ))
        fig.update_layout(
            title=label, barmode="group", height=280,
            margin=dict(t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════
# SECCIÓN 4 — SERIES TEMPORALES
# ══════════════════════════════════════════════════════════
st.subheader("📈 Series temporales globales")

col_a, col_b = st.columns(2)
with col_a:
    fig_vel = px.line(df, x="time_str", y="vel_kmh",
                      title="Velocidad (km/h)",
                      color_discrete_sequence=["#378ADD"])
    fig_vel.update_layout(margin=dict(t=40,b=20), height=250,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_vel, use_container_width=True)

    fig_acel = px.line(df, x="time_str", y=["acel_x","acel_y"],
                       title="Aceleración X / Y (m/s²)",
                       color_discrete_sequence=["#EF9F27","#D4537E"])
    fig_acel.update_layout(margin=dict(t=40,b=20), height=250,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_acel, use_container_width=True)

with col_b:
    fig_alt = px.area(df, x="time_str", y="elevation_m",
                      title="Altura (m)",
                      color_discrete_sequence=["#1D9E75"])
    fig_alt.update_layout(margin=dict(t=40,b=20), height=250,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_alt, use_container_width=True)

    fig_inc = px.line(df, x="time_str", y=["inc_x","inc_y"],
                      title="Inclinación X / Y (°)",
                      color_discrete_sequence=["#534AB7","#993C1D"])
    fig_inc.update_layout(margin=dict(t=40,b=20), height=250,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_inc, use_container_width=True)

st.divider()

with st.expander("Ver datos procesados"):
    st.dataframe(
        df[["time_str","lat","lon","elevation_m",
            "vel_kmh","acel_x","acel_y","inc_x","inc_y"]].rename(columns={
            "time_str":    "Tiempo",       "lat":         "Latitud",
            "lon":         "Longitud",     "elevation_m": "Altura (m)",
            "vel_kmh":     "Vel (km/h)",   "acel_x":      "Acel X (m/s²)",
            "acel_y":      "Acel Y (m/s²)","inc_x":       "Inc X (°)",
            "inc_y":       "Inc Y (°)",
        }),
        use_container_width=True
    )
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV procesado", csv_out,
                       "rtk_procesado.csv", "text/csv")