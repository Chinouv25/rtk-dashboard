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
    "recorridos":   {},
    "geocercas":    [],
    "linea_inicio": None,
    "linea_fin":    None,
    "analizar":     False,
    "last_drawing": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helpers tiempo ────────────────────────────────────────
def segundos_a_str(s):
    s = float(s)
    m = int(s // 60)
    seg = s % 60
    return f"{m:02d}:{seg:05.2f}"

def tiempo_track(df):
    return df["timestamp_s"].iloc[-1] - df["timestamp_s"].iloc[0]

def tiempo_zona(df_seg):
    if len(df_seg) < 2:
        return 0.0
    return df_seg["timestamp_s"].iloc[-1] - df_seg["timestamp_s"].iloc[0]

# ── Carga de datos ────────────────────────────────────────
@st.cache_data
def load_csv(file_content, filename):
    import io
    df = pd.read_csv(io.BytesIO(file_content))
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
    df["time_str"] = df["timestamp_s"].apply(segundos_a_str)
    df["recorrido"] = filename
    return df

# ── Colores por track ─────────────────────────────────────
TRACK_COLORS = ["#4ECDC4","#FF6B6B","#FFE66D","#A8E6CF","#B8B8FF","#FF8B94"]

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Recorridos")
    archivos = st.file_uploader(
        "Cargar CSV (múltiples)",
        type=["csv"],
        accept_multiple_files=True
    )
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
    unit_map  = {"vel_kmh":"km/h","elevation_m":"m","acel_mag":"m/s²","inc_mag":"°"}
    label_map = {"vel_kmh":"Vel","elevation_m":"Alt","acel_mag":"Acel","inc_mag":"Incl"}

# Procesar archivos cargados
if archivos:
    for f in archivos:
        if f.name not in st.session_state.recorridos:
            st.session_state.recorridos[f.name] = load_csv(f.read(), f.name)

# Lista recorridos en sidebar + eliminar
with st.sidebar:
    if st.session_state.recorridos:
        st.divider()
        st.markdown("**Recorridos cargados:**")
        to_delete = []
        for i, nombre in enumerate(list(st.session_state.recorridos.keys())):
            color = TRACK_COLORS[i % len(TRACK_COLORS)]
            col1, col2 = st.columns([3,1])
            col1.markdown(
                f'<span style="color:{color};">■</span> '
                f'{nombre.replace(".csv","")}',
                unsafe_allow_html=True
            )
            if col2.button("✕", key=f"del_rec_{nombre}"):
                to_delete.append(nombre)
        for nombre in to_delete:
            del st.session_state.recorridos[nombre]
            st.rerun()

# Sin recorridos — pantalla de bienvenida
if not st.session_state.recorridos:
    st.info("Carga uno o más archivos CSV en el panel lateral para comenzar.")
    st.stop()

recorridos_activos = st.session_state.recorridos

# ══════════════════════════════════════════════════════════
# MÉTRICAS SUPERIORES
# ══════════════════════════════════════════════════════════
st.subheader("📋 Resumen de recorridos")
cols_met = st.columns(len(recorridos_activos))
for i, (nombre, df_rec) in enumerate(recorridos_activos.items()):
    t_total = tiempo_track(df_rec)
    color   = TRACK_COLORS[i % len(TRACK_COLORS)]
    with cols_met[i]:
        st.markdown(
            f'<div style="border-left:4px solid {color};padding-left:10px;">'
            f'<b>{nombre.replace(".csv","")}</b></div>',
            unsafe_allow_html=True
        )
        st.metric("Vel. máx",      f"{df_rec['vel_kmh'].max():.2f} km/h")
        st.metric("Vel. promedio",  f"{df_rec['vel_kmh'].mean():.2f} km/h")
        st.metric("Tiempo total",   segundos_a_str(t_total))

st.divider()

# ── Helpers mapa ──────────────────────────────────────────
def val_to_color_rg(v, vmin, vmax):
    t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin + 1e-9)))
    stops = [(0.0,(220,50,50)),(0.5,(230,180,0)),(1.0,(50,200,80))]
    for i in range(len(stops)-1):
        t0,c0 = stops[i]; t1,c1 = stops[i+1]
        if t <= t1:
            f = (t-t0)/(t1-t0+1e-9)
            return "#{:02x}{:02x}{:02x}".format(
                int(c0[0]+f*(c1[0]-c0[0])),
                int(c0[1]+f*(c1[1]-c0[1])),
                int(c0[2]+f*(c1[2]-c0[2]))
            )
    return "#32c850"

def linea_a_shapely(coords):
    return LineString([(c[0], c[1]) for c in coords])

def punto_mas_cercano_idx(df, linea):
    min_dist, idx_min = float("inf"), df.index[0]
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
    return df.loc[idx_ini:idx_fin].copy()

# ══════════════════════════════════════════════════════════
# SECCIÓN 1 — MAPA
# ══════════════════════════════════════════════════════════
st.subheader("🗺️ Mapa geoespacial")

if st.session_state.linea_inicio is None:
    st.info("**Paso 1:** Dibuja la línea de **inicio** de zona.")
elif st.session_state.linea_fin is None:
    st.info("**Paso 2:** Dibuja la línea de **fin** de zona.")
else:
    st.success("Líneas capturadas. Completa el formulario abajo.")

all_lats = np.concatenate([d["lat"].values for d in recorridos_activos.values()])
all_lons = np.concatenate([d["lon"].values for d in recorridos_activos.values()])
center   = [float(np.mean(all_lats)), float(np.mean(all_lons))]

m = folium.Map(location=center, zoom_start=19, max_zoom=22, control_scale=True)
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google Satellite", name="Google Satellite",
    max_zoom=22, max_native_zoom=22,
).add_to(m)

var_label = label_map[color_var]
var_unit  = unit_map[color_var]

for rec_idx, (nombre, df_rec) in enumerate(recorridos_activos.items()):
    vals = df_rec[color_var].values
    vmin, vmax = vals.min(), vals.max()
    rec_label  = nombre.replace(".csv","")
    for i in range(len(df_rec)-1):
        v = vals[i]
        folium.PolyLine(
            [[df_rec["lat"].iloc[i],   df_rec["lon"].iloc[i]],
             [df_rec["lat"].iloc[i+1], df_rec["lon"].iloc[i+1]]],
            color=val_to_color_rg(v, vmin, vmax),
            weight=5, opacity=0.9,
            tooltip=f"<b>{rec_label}</b><br>"
                    f"T: {df_rec['time_str'].iloc[i]}<br>"
                    f"{var_label}: {v:.2f} {var_unit}"
        ).add_to(m)
    folium.CircleMarker(
        [df_rec["lat"].iloc[0], df_rec["lon"].iloc[0]],
        radius=7, color="#22c55e", fill=True,
        fill_color="#22c55e", tooltip=f"Inicio: {rec_label}"
    ).add_to(m)
    folium.CircleMarker(
        [df_rec["lat"].iloc[-1], df_rec["lon"].iloc[-1]],
        radius=7, color="#ef4444", fill=True,
        fill_color="#ef4444", tooltip=f"Fin: {rec_label}"
    ).add_to(m)

if st.session_state.linea_inicio:
    folium.PolyLine(
        [[c[1],c[0]] for c in st.session_state.linea_inicio],
        color="#22c55e", weight=4, tooltip="Línea inicio"
    ).add_to(m)
if st.session_state.linea_fin:
    folium.PolyLine(
        [[c[1],c[0]] for c in st.session_state.linea_fin],
        color="#ef4444", weight=4, tooltip="Línea fin"
    ).add_to(m)

colors_gc = ["#FF6B6B","#4ECDC4","#FFE66D","#A8E6CF","#FF8B94","#B8B8FF"]
for idx, gc in enumerate(st.session_state.geocercas):
    color  = colors_gc[idx % len(colors_gc)]
    df_ref = list(gc["segmentos"].values())[0]
    if len(df_ref) > 1:
        folium.PolyLine(
            [[r["lat"],r["lon"]] for _,r in df_ref.iterrows()],
            color=color, weight=9, opacity=0.4,
            tooltip=gc["nombre"]
        ).add_to(m)
    for lc, label in [
        (gc["linea_inicio"], f"{gc['nombre']} — inicio"),
        (gc["linea_fin"],    f"{gc['nombre']} — fin"),
    ]:
        folium.PolyLine(
            [[c[1],c[0]] for c in lc],
            color=color, weight=3, dash_array="6", tooltip=label
        ).add_to(m)
    if len(df_ref) > 0:
        mid = df_ref.iloc[len(df_ref)//2]
        folium.Marker(
            [mid["lat"], mid["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:bold;color:{color};'
                     f'background:rgba(0,0,0,0.65);padding:2px 6px;'
                     f'border-radius:4px;white-space:nowrap;">{gc["nombre"]}</div>'
            )
        ).add_to(m)

Draw(export=False, draw_options={
    "polyline":True,"polygon":False,"rectangle":False,
    "circle":False,"marker":False,"circlemarker":False
}, edit_options={"edit":False}).add_to(m)

map_data = st_folium(m, width="100%", height=500, key="main_map",
                     returned_objects=["last_active_drawing"])

# ── Captura de líneas ─────────────────────────────────────
if map_data and map_data.get("last_active_drawing"):
    geom = map_data["last_active_drawing"].get("geometry", {})
    if geom.get("type") == "LineString":
        coords     = geom["coordinates"]
        coords_str = str(coords)
        if coords_str != st.session_state.last_drawing:
            st.session_state.last_drawing = coords_str
            if st.session_state.linea_inicio is None:
                st.session_state.linea_inicio = coords
                st.rerun()
            elif st.session_state.linea_fin is None:
                st.session_state.linea_fin = coords
                st.rerun()

if st.session_state.linea_inicio or st.session_state.linea_fin:
    if st.button("🔄 Resetear líneas"):
        st.session_state.linea_inicio = None
        st.session_state.linea_fin    = None
        st.session_state.last_drawing = None
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
    segmentos_preview = {
        nombre: segmento_entre_lineas(df_rec, linea_ini_sh, linea_fin_sh)
        for nombre, df_rec in recorridos_activos.items()
    }
    total_pts    = sum(len(s) for s in segmentos_preview.values())
    tiempos_prev = {n: tiempo_zona(s) for n,s in segmentos_preview.items()}
    st.success(
        f"Segmento detectado en {len(segmentos_preview)} recorrido(s) — "
        f"{total_pts} puntos | Tiempos: " +
        ", ".join([f"{n.replace('.csv','')}: {segundos_a_str(t)}"
                   for n,t in tiempos_prev.items()])
    )

    with st.form("form_zona", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            nombre = st.text_input("Nombre de la zona",
                                   placeholder="Ej: Recta inicial")
        with col2:
            tipo = st.selectbox("Tipo de zona",
                ["Recta","Curva","Salto","Frenada","Otra"])
        st.markdown("**Parámetros objetivo**")
        c1, c2 = st.columns(2)
        with c1:
            vel_max_obj = st.number_input("Vel. máx objetivo (km/h)",
                                          0.0, 200.0, 40.0, 0.5)
        with c2:
            vel_min_obj = st.number_input("Vel. mín objetivo (km/h)",
                                          0.0, 200.0, 20.0, 0.5)
        if st.form_submit_button("💾 Guardar zona") and nombre:
            st.session_state.geocercas.append({
                "nombre":       nombre,
                "tipo":         tipo,
                "linea_inicio": st.session_state.linea_inicio,
                "linea_fin":    st.session_state.linea_fin,
                "segmentos":    segmentos_preview,
                "vel_max_obj":  vel_max_obj,
                "vel_min_obj":  vel_min_obj,
            })
            st.session_state.linea_inicio = None
            st.session_state.linea_fin    = None
            st.session_state.last_drawing = None
            st.rerun()
else:
    st.warning("Dibuja la línea de inicio en el mapa." if
               st.session_state.linea_inicio is None else
               "Ahora dibuja la línea de fin en el mapa.")

if st.session_state.geocercas:
    with st.expander("🗑️ Gestionar zonas guardadas"):
        for idx, gc in enumerate(st.session_state.geocercas):
            col1, col2 = st.columns([4,1])
            total = sum(len(s) for s in gc["segmentos"].values())
            col1.write(f"**{gc['nombre']}** — {gc['tipo']} ({total} pts)")
            if col2.button("Eliminar", key=f"del_gc_{idx}"):
                st.session_state.geocercas.pop(idx)
                st.rerun()

# ══════════════════════════════════════════════════════════
# SECCIÓN 3 — ANALIZAR
# ══════════════════════════════════════════════════════════
st.divider()

if st.session_state.geocercas:
    if st.button("🔍 Analizar zonas", type="primary", use_container_width=True):
        st.session_state.analizar = True

if st.session_state.analizar and st.session_state.geocercas:
    st.subheader("📊 Análisis por zona")

    for gc in st.session_state.geocercas:
        st.markdown(f"### {gc['nombre']} — {gc['tipo']}")
        st.caption(f"Objetivo → Vel máx: {gc['vel_max_obj']} km/h | "
                   f"Vel mín: {gc['vel_min_obj']} km/h")

        filas = []
        for rec_nombre, df_seg in gc["segmentos"].items():
            t_zona = tiempo_zona(df_seg)
            filas.append({
                "Recorrido":     rec_nombre.replace(".csv",""),
                "Tiempo zona":   segundos_a_str(t_zona),
                "Vel máx real":  round(df_seg["vel_kmh"].max(), 2) if len(df_seg) else "-",
                "Vel mín real":  round(df_seg["vel_kmh"].min(), 2) if len(df_seg) else "-",
                "Vel prom real": round(df_seg["vel_kmh"].mean(), 2) if len(df_seg) else "-",
                "Vel máx obj":   gc["vel_max_obj"],
                "Vel mín obj":   gc["vel_min_obj"],
                "_t_zona_s":     t_zona,
            })

        df_tabla = pd.DataFrame(filas)

        def highlight(row):
            styles = [""] * len(row)
            for col, obj in [
                ("Vel máx real", gc["vel_max_obj"]),
                ("Vel mín real", gc["vel_min_obj"]),
            ]:
                if col in row.index and row[col] != "-":
                    diff = abs(float(row[col]) - obj)
                    idx  = row.index.tolist().index(col)
                    styles[idx] = (
                        "background-color:#1a4a1a" if diff <= obj*0.05
                        else "background-color:#4a3a0a" if diff <= obj*0.15
                        else "background-color:#4a1a1a"
                    )
            return styles

        st.dataframe(
            df_tabla.style.apply(highlight, axis=1)
                          .hide(subset=["_t_zona_s"], axis="columns"),
            use_container_width=True
        )
        st.caption("🟢 ≤5% del objetivo  |  🟡 ≤15%  |  🔴 Fuera de rango")
        st.markdown("---")

    # ══════════════════════════════════════════════════════
    # BENCHMARK
    # ══════════════════════════════════════════════════════
    st.subheader("🏆 Benchmark entre recorridos")

    bench_rows = []
    for i, (nombre, df_rec) in enumerate(recorridos_activos.items()):
        t_total = tiempo_track(df_rec)
        bench_rows.append({
            "Recorrido":    nombre.replace(".csv",""),
            "nombre_raw":   nombre,
            "Vel máx":      round(df_rec["vel_kmh"].max(), 2),
            "Vel prom":     round(df_rec["vel_kmh"].mean(), 2),
            "Tiempo total": segundos_a_str(t_total),
            "_t_total_s":   t_total,
            "color":        TRACK_COLORS[i % len(TRACK_COLORS)],
        })

    df_bench = pd.DataFrame(bench_rows)
    ref_vel  = df_bench.loc[df_bench["Vel máx"].idxmax()]
    ref_time = df_bench.loc[df_bench["_t_total_s"].idxmin()]

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("#### Por velocidad máxima")
        st.success(f"🥇 Referencia: **{ref_vel['Recorrido']}** — {ref_vel['Vel máx']} km/h")

        filas_vel = []
        for _, row in df_bench.iterrows():
            diff = round(row["Vel máx"] - ref_vel["Vel máx"], 2)
            filas_vel.append({
                "Recorrido": row["Recorrido"],
                "Vel máx":   row["Vel máx"],
                "Δ vs ref":  f"{diff:+.2f} km/h",
                "_diff":     diff,
            })
        df_vel = pd.DataFrame(filas_vel)

        def color_diff_vel(row):
            styles = [""] * len(row)
            d   = row["_diff"]
            idx = row.index.tolist().index("Δ vs ref")
            styles[idx] = (
                "background-color:#1a4a1a" if d == 0
                else "background-color:#4a3a0a" if d >= -3
                else "background-color:#4a1a1a"
            )
            return styles

        st.dataframe(
            df_vel.style.apply(color_diff_vel, axis=1)
                        .hide(subset=["_diff"], axis="columns"),
            use_container_width=True
        )

        fig_vel_bench = go.Figure([
            go.Bar(name=row["Recorrido"], x=[row["Recorrido"]],
                   y=[row["Vel máx"]], marker_color=row["color"])
            for _, row in df_bench.iterrows()
        ])
        fig_vel_bench.update_layout(
            title="Velocidad máxima por recorrido", height=280,
            showlegend=False, margin=dict(t=40,b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="km/h"
        )
        st.plotly_chart(fig_vel_bench, use_container_width=True)

    with col_b2:
        st.markdown("#### Por tiempo total")
        st.success(f"🥇 Referencia: **{ref_time['Recorrido']}** — {ref_time['Tiempo total']}")

        filas_time = []
        for _, row in df_bench.iterrows():
            diff_s = round(row["_t_total_s"] - ref_time["_t_total_s"], 2)
            filas_time.append({
                "Recorrido":    row["Recorrido"],
                "Tiempo total": row["Tiempo total"],
                "Δ vs ref":     f"{diff_s:+.2f} s",
                "_diff_s":      diff_s,
            })
        df_time = pd.DataFrame(filas_time)

        def color_diff_time(row):
            styles = [""] * len(row)
            d   = row["_diff_s"]
            idx = row.index.tolist().index("Δ vs ref")
            styles[idx] = (
                "background-color:#1a4a1a" if d == 0
                else "background-color:#4a3a0a" if d <= 1
                else "background-color:#4a1a1a"
            )
            return styles

        st.dataframe(
            df_time.style.apply(color_diff_time, axis=1)
                         .hide(subset=["_diff_s"], axis="columns"),
            use_container_width=True
        )

        fig_time_bench = go.Figure([
            go.Bar(name=row["Recorrido"], x=[row["Recorrido"]],
                   y=[row["_t_total_s"]], marker_color=row["color"])
            for _, row in df_bench.iterrows()
        ])
        fig_time_bench.update_layout(
            title="Tiempo total por recorrido", height=280,
            showlegend=False, margin=dict(t=40,b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="segundos"
        )
        st.plotly_chart(fig_time_bench, use_container_width=True)

    # Benchmark por zonas
    if st.session_state.geocercas:
        st.markdown("#### Benchmark por zona")
        for gc in st.session_state.geocercas:
            st.markdown(f"**{gc['nombre']}**")
            tiempos_z  = {n: tiempo_zona(s) for n,s in gc["segmentos"].items()}
            ref_t_zona = min(tiempos_z.values()) if tiempos_z else 0
            vel_max_zona = max(
                (df_seg["vel_kmh"].max() for df_seg in gc["segmentos"].values()
                 if len(df_seg)), default=0
            )
            filas_z = []
            for rec_nombre, df_seg in gc["segmentos"].items():
                tz     = tiempos_z[rec_nombre]
                diff_z = round(tz - ref_t_zona, 2)
                diff_v = round(df_seg["vel_kmh"].max() - vel_max_zona, 2) \
                         if len(df_seg) else "-"
                filas_z.append({
                    "Recorrido":   rec_nombre.replace(".csv",""),
                    "Tiempo zona": segundos_a_str(tz),
                    "Δ tiempo":    f"{diff_z:+.2f} s",
                    "Vel máx":     round(df_seg["vel_kmh"].max(), 2) if len(df_seg) else "-",
                    "Vel prom":    round(df_seg["vel_kmh"].mean(), 2) if len(df_seg) else "-",
                    "Δ vel máx":   f"{diff_v:+.2f} km/h" if diff_v != "-" else "-",
                    "_diff_z":     diff_z,
                })
            df_zona_bench = pd.DataFrame(filas_z)

            def color_zona(row):
                styles = [""] * len(row)
                d   = row["_diff_z"]
                idx = row.index.tolist().index("Δ tiempo")
                styles[idx] = (
                    "background-color:#1a4a1a" if d == 0
                    else "background-color:#4a3a0a" if d <= 0.5
                    else "background-color:#4a1a1a"
                )
                return styles

            st.dataframe(
                df_zona_bench.style.apply(color_zona, axis=1)
                                   .hide(subset=["_diff_z"], axis="columns"),
                use_container_width=True
            )
            st.markdown("---")

st.divider()

# ══════════════════════════════════════════════════════════
# SECCIÓN 4 — SERIES TEMPORALES
# ══════════════════════════════════════════════════════════
st.subheader("📈 Series temporales")

rec_sel = st.selectbox(
    "Recorrido a graficar:",
    list(recorridos_activos.keys()),
    format_func=lambda x: x.replace(".csv",""),
    key="sel_graf"
)
df_graf = recorridos_activos[rec_sel]

col_a, col_b = st.columns(2)
with col_a:
    fig_vel = px.line(df_graf, x="time_str", y="vel_kmh",
                      title="Velocidad (km/h)",
                      color_discrete_sequence=["#378ADD"])
    fig_vel.update_layout(margin=dict(t=40,b=20), height=250,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_vel, use_container_width=True)

    fig_acel = px.line(df_graf, x="time_str", y=["acel_x","acel_y"],
                       title="Aceleración X / Y (m/s²)",
                       color_discrete_sequence=["#EF9F27","#D4537E"])
    fig_acel.update_layout(margin=dict(t=40,b=20), height=250,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_acel, use_container_width=True)

with col_b:
    fig_alt = px.area(df_graf, x="time_str", y="elevation_m",
                      title="Altura (m)",
                      color_discrete_sequence=["#1D9E75"])
    fig_alt.update_layout(margin=dict(t=40,b=20), height=250,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_alt, use_container_width=True)

    fig_inc = px.line(df_graf, x="time_str", y=["inc_x","inc_y"],
                      title="Inclinación X / Y (°)",
                      color_discrete_sequence=["#534AB7","#993C1D"])
    fig_inc.update_layout(margin=dict(t=40,b=20), height=250,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_inc, use_container_width=True)

st.divider()
with st.expander("Ver datos procesados"):
    st.dataframe(
        df_graf[["time_str","lat","lon","elevation_m",
                 "vel_kmh","acel_x","acel_y","inc_x","inc_y"]].rename(columns={
            "time_str":"Tiempo","lat":"Latitud","lon":"Longitud",
            "elevation_m":"Altura (m)","vel_kmh":"Vel (km/h)",
            "acel_x":"Acel X (m/s²)","acel_y":"Acel Y (m/s²)",
            "inc_x":"Inc X (°)","inc_y":"Inc Y (°)",
        }),
        use_container_width=True
    )
    csv_out = df_graf.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", csv_out, "rtk_procesado.csv", "text/csv")