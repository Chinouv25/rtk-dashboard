import streamlit as st
import pandas as pd
import plotly.express as px
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
    "escala": [0.0, 0.25, 0.5, 0.75, 1.0],
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helpers ───────────────────────────────────────────────
def segundos_a_str(s):
    s = float(s)
    return f"{int(s//60):02d}:{s%60:05.2f}"

def tiempo_track(df):
    return df["timestamp_s"].iloc[-1] - df["timestamp_s"].iloc[0]

def tiempo_zona(df_seg):
    if len(df_seg) < 2:
        return 0.0
    return df_seg["timestamp_s"].iloc[-1] - df_seg["timestamp_s"].iloc[0]

def fmt(v):
    try:
        return f"{float(v):.2f}"
    except:
        return str(v)

def load_csv(file_content, filename):
    import io
    df = pd.read_csv(io.BytesIO(file_content))
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "latitude":      "lat",
        "longitude":     "lon",
        "speed_ms":      "vel_ms",
        "acc_x_ms2":     "acel_x",
        "acc_y_ms2":     "acel_y",
        "inc_x_deg":     "inc_x",
        "inc_y_deg":     "inc_y",
        "jump_height_m": "jump_h",
    })
    df["vel_kmh"]  = (df["vel_ms"] * 3.6).round(2)
    df["acel_mag"] = np.sqrt(df["acel_x"]**2 + df["acel_y"]**2).round(2)
    df["inc_mag"]  = np.sqrt(df["inc_x"]**2  + df["inc_y"]**2).round(2)
    if "jump_h" not in df.columns:
        df["jump_h"] = 0.0
    df["jump_h"]   = df["jump_h"].round(2)
    df["time_str"] = df["timestamp_s"].apply(segundos_a_str)
    df["recorrido"] = filename
    return df

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

def val_to_color_custom(v, vmin, vmax, stops_rgb, stops_pos):
    t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin + 1e-9)))
    for i in range(len(stops_pos)-1):
        t0, t1 = stops_pos[i], stops_pos[i+1]
        if t <= t1:
            f  = (t - t0) / (t1 - t0 + 1e-9)
            c0 = stops_rgb[i]
            c1 = stops_rgb[i+1]
            return "#{:02x}{:02x}{:02x}".format(
                int(c0[0]+f*(c1[0]-c0[0])),
                int(c0[1]+f*(c1[1]-c0[1])),
                int(c0[2]+f*(c1[2]-c0[2])))
    return "#{:02x}{:02x}{:02x}".format(*stops_rgb[-1])

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

TRACK_COLORS = ["#4ECDC4","#FF6B6B","#FFE66D","#A8E6CF","#B8B8FF","#FF8B94"]

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Recorridos")
    archivos = st.file_uploader("Cargar CSV (múltiples)", type=["csv"],
                                accept_multiple_files=True)
    st.divider()
    color_var = st.selectbox(
        "Colorear track por:",
        ["vel_kmh","jump_h","acel_mag","inc_mag"],
        format_func=lambda x: {
            "vel_kmh":  "Velocidad (km/h)",
            "jump_h":   "Saltos (m)",
            "acel_mag": "Aceleración (m/s²)",
            "inc_mag":  "Inclinación (°)"
        }[x]
    )
    unit_map  = {"vel_kmh":"km/h","jump_h":"m",
                 "acel_mag":"m/s²","inc_mag":"°"}
    label_map = {"vel_kmh":"Vel","jump_h":"Salto",
                 "acel_mag":"Acel","inc_mag":"Incl"}

    st.divider()
    st.markdown("**Escala de colores (5 puntos)**")
    default_colors = ["#dc3232","#e07020","#e6b400","#80c820","#32c850"]
    stop_colors = []
    stop_vals   = []
    for k in range(5):
        cc, cv = st.columns([2,3])
        with cc:
            sc = st.color_picker(f"C{k+1}", default_colors[k],
                                 key=f"sc_{k}")
        with cv:
            pct = st.number_input(
                f"% pos {k+1}", 0, 100,
                int(st.session_state.escala[k]*100),
                5, key=f"sv_{k}",
                label_visibility="collapsed"
            )
            st.session_state.escala[k] = pct / 100.0
        stop_colors.append(sc)
        stop_vals.append(st.session_state.escala[k])

    stop_vals = sorted(stop_vals)
    stops_rgb = [hex_to_rgb(c) for c in stop_colors]
    stops_pos = stop_vals

# ── Procesar archivos sin cache ───────────────────────────
nombres_subidos = {f.name for f in archivos} if archivos else set()
for n in [n for n in st.session_state.recorridos if n not in nombres_subidos]:
    del st.session_state.recorridos[n]
if archivos:
    for f in archivos:
        if f.name not in st.session_state.recorridos:
            st.session_state.recorridos[f.name] = load_csv(f.read(), f.name)

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
                unsafe_allow_html=True)
            if col2.button("✕", key=f"del_rec_{nombre}"):
                to_delete.append(nombre)
        for nombre in to_delete:
            del st.session_state.recorridos[nombre]
            st.rerun()

if not st.session_state.recorridos:
    st.info("Carga uno o más archivos CSV en el panel lateral para comenzar.")
    st.stop()

recorridos_activos = st.session_state.recorridos

# ══════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════
st.subheader("📋 Resumen de recorridos")
cols_met = st.columns(len(recorridos_activos))
for i, (nombre, df_rec) in enumerate(recorridos_activos.items()):
    color = TRACK_COLORS[i % len(TRACK_COLORS)]
    with cols_met[i]:
        st.markdown(
            f'<div style="border-left:4px solid {color};padding-left:10px;">'
            f'<b>{nombre.replace(".csv","")}</b></div>',
            unsafe_allow_html=True)
        st.metric("Vel. máx",      fmt(df_rec["vel_kmh"].max()) + " km/h")
        st.metric("Vel. promedio",  fmt(df_rec["vel_kmh"].mean()) + " km/h")
        st.metric("Salto máx",      fmt(df_rec["jump_h"].max()) + " m")
        st.metric("Tiempo total",   segundos_a_str(tiempo_track(df_rec)))

st.divider()

# ══════════════════════════════════════════════════════════
# MAPA
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
    attr="Google Satellite", max_zoom=22, max_native_zoom=22,
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
            color=val_to_color_custom(v, vmin, vmax, stops_rgb, stops_pos),
            weight=5, opacity=0.9,
            tooltip=(
                f"<b>{rec_label}</b><br>"
                f"T: {df_rec['time_str'].iloc[i]}<br>"
                f"Vel: {df_rec['vel_kmh'].iloc[i]:.2f} km/h<br>"
                f"Salto: {df_rec['jump_h'].iloc[i]:.2f} m<br>"
                f"Acel: {df_rec['acel_mag'].iloc[i]:.2f} m/s²<br>"
                f"Incl: {df_rec['inc_mag'].iloc[i]:.2f} °"
            )
        ).add_to(m)
    folium.CircleMarker(
        [df_rec["lat"].iloc[0], df_rec["lon"].iloc[0]],
        radius=7, color="#22c55e", fill=True,
        fill_color="#22c55e", tooltip=f"<b>Inicio: {rec_label}</b>"
    ).add_to(m)
    folium.CircleMarker(
        [df_rec["lat"].iloc[-1], df_rec["lon"].iloc[-1]],
        radius=7, color="#ef4444", fill=True,
        fill_color="#ef4444", tooltip=f"<b>Fin: {rec_label}</b>"
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
                html=f'<div style="font-size:11px;font-weight:bold;'
                     f'color:{color};background:rgba(0,0,0,0.7);'
                     f'padding:2px 6px;border-radius:4px;'
                     f'white-space:nowrap;">{gc["nombre"]}</div>'
            )
        ).add_to(m)

# Leyenda
grad_stops = ", ".join([
    f"{stop_colors[i]} {int(stops_pos[i]*100)}%"
    for i in range(len(stop_colors))
])
legend_html = f"""
<div style="position:fixed;bottom:30px;right:10px;z-index:1000;
     background:rgba(0,0,0,0.75);padding:10px 14px;border-radius:8px;
     font-size:12px;color:white;min-width:150px;">
  <b>{var_label} ({var_unit})</b><br>
  <div style="display:flex;align-items:center;gap:6px;margin-top:6px;">
    <span style="font-size:10px;">Bajo</span>
    <div style="width:90px;height:10px;border-radius:4px;
         background:linear-gradient(to right,{grad_stops});"></div>
    <span style="font-size:10px;">Alto</span>
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

Draw(export=False, draw_options={
    "polyline":True,"polygon":False,"rectangle":False,
    "circle":False,"marker":False,"circlemarker":False
}, edit_options={"edit":False}).add_to(m)

map_data = st_folium(m, width="100%", height=500, key="main_map",
                     returned_objects=["last_active_drawing"])

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
# FORMULARIO NUEVA ZONA
# ══════════════════════════════════════════════════════════
st.divider()
st.subheader("➕ Registrar nueva zona")

if (st.session_state.linea_inicio is not None and
        st.session_state.linea_fin is not None):
    li = linea_a_shapely(st.session_state.linea_inicio)
    lf = linea_a_shapely(st.session_state.linea_fin)
    segs_prev = {n: segmento_entre_lineas(df, li, lf)
                 for n, df in recorridos_activos.items()}
    total_pts = sum(len(s) for s in segs_prev.values())
    t_prev    = {n: tiempo_zona(s) for n,s in segs_prev.items()}
    st.success(
        f"Segmento detectado — {total_pts} puntos | Tiempos: " +
        ", ".join([f"{n.replace('.csv','')}: {segundos_a_str(t)}"
                   for n,t in t_prev.items()])
    )
    with st.form("form_zona", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            nombre_zona = st.text_input("Nombre de la zona",
                                        placeholder="Ej: Recta inicial")
        with col2:
            tipo_zona = st.selectbox("Tipo",
                ["Recta","Curva","Salto","Frenada","Otra"])
        c1, c2 = st.columns(2)
        with c1:
            vel_max_obj = st.number_input("Vel. máx objetivo (km/h)",
                                          0.0, 200.0, 40.0, 0.5)
        with c2:
            vel_min_obj = st.number_input("Vel. mín objetivo (km/h)",
                                          0.0, 200.0, 20.0, 0.5)
        if st.form_submit_button("💾 Guardar zona") and nombre_zona:
            st.session_state.geocercas.append({
                "nombre":       nombre_zona,
                "tipo":         tipo_zona,
                "linea_inicio": st.session_state.linea_inicio,
                "linea_fin":    st.session_state.linea_fin,
                "segmentos":    segs_prev,
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
               "Ahora dibuja la línea de fin.")

if st.session_state.geocercas:
    with st.expander("⚙️ Gestionar y editar zonas"):
        for idx, gc in enumerate(st.session_state.geocercas):
            st.markdown(f"**{gc['nombre']}** — {gc['tipo']}")
            col1, col2, col3 = st.columns([2,2,1])
            with col1:
                nuevo_max = st.number_input(
                    "Vel. máx obj (km/h)", 0.0, 200.0,
                    float(gc["vel_max_obj"]), 0.5,
                    key=f"edit_max_{idx}")
            with col2:
                nuevo_min = st.number_input(
                    "Vel. mín obj (km/h)", 0.0, 200.0,
                    float(gc["vel_min_obj"]), 0.5,
                    key=f"edit_min_{idx}")
            with col3:
                st.write(""); st.write("")
                if st.button("💾", key=f"save_gc_{idx}"):
                    st.session_state.geocercas[idx]["vel_max_obj"] = nuevo_max
                    st.session_state.geocercas[idx]["vel_min_obj"] = nuevo_min
                    st.rerun()
            if st.button("🗑️ Eliminar", key=f"del_gc_{idx}"):
                st.session_state.geocercas.pop(idx)
                st.rerun()
            st.markdown("---")

# ══════════════════════════════════════════════════════════
# ANALIZAR
# ══════════════════════════════════════════════════════════
st.divider()

if st.session_state.geocercas:
    if st.button("🔍 Analizar zonas", type="primary", use_container_width=True):
        st.session_state.analizar = True

if st.session_state.analizar and st.session_state.geocercas:

    st.subheader("📊 Análisis por zona")
    for gc in st.session_state.geocercas:
        st.markdown(f"### {gc['nombre']} — {gc['tipo']}")
        st.caption(f"Objetivo → Vel máx: {gc['vel_max_obj']:.2f} km/h | "
                   f"Vel mín: {gc['vel_min_obj']:.2f} km/h")
        filas = []
        for rec_nombre, df_seg in gc["segmentos"].items():
            tz      = tiempo_zona(df_seg)
            vel_max = float(df_seg["vel_kmh"].max()) if len(df_seg) else -1
            filas.append({
                "Recorrido":     rec_nombre.replace(".csv",""),
                "Tiempo zona":   segundos_a_str(tz),
                "Vel máx real":  fmt(vel_max) if len(df_seg) else "-",
                "Vel mín real":  fmt(df_seg["vel_kmh"].min()) if len(df_seg) else "-",
                "Vel prom real": fmt(df_seg["vel_kmh"].mean()) if len(df_seg) else "-",
                "Salto máx":     fmt(df_seg["jump_h"].max()) if len(df_seg) else "-",
                "Vel máx obj":   fmt(gc["vel_max_obj"]),
                "Vel mín obj":   fmt(gc["vel_min_obj"]),
                "_vel_max_f":    vel_max,
            })
        df_tabla = pd.DataFrame(filas).sort_values("_vel_max_f", ascending=False)

        def highlight_zona(row):
            styles = [""] * len(row)
            for col, obj in [
                ("Vel máx real", gc["vel_max_obj"]),
                ("Vel mín real", gc["vel_min_obj"]),
            ]:
                if col in row.index and row[col] != "-":
                    try:
                        diff = abs(float(row[col]) - obj)
                        i    = row.index.tolist().index(col)
                        styles[i] = (
                            "background-color:#1a4a1a" if diff <= obj*0.05
                            else "background-color:#4a3a0a" if diff <= obj*0.15
                            else "background-color:#4a1a1a"
                        )
                    except:
                        pass
            return styles

        st.dataframe(
            df_tabla.drop(columns=["_vel_max_f"])
                    .style.apply(highlight_zona, axis=1)
                    .hide(axis="index"),
            use_container_width=True
        )
        st.caption("🟢 ≤5% objetivo  |  🟡 ≤15%  |  🔴 Fuera de rango")
        st.markdown("---")

    # ── Benchmark ─────────────────────────────────────────
    st.subheader("🏆 Benchmark entre recorridos")

    bench_rows = []
    for i, (nombre, df_rec) in enumerate(recorridos_activos.items()):
        bench_rows.append({
            "Recorrido":    nombre.replace(".csv",""),
            "Vel máx":      fmt(df_rec["vel_kmh"].max()),
            "Vel prom":     fmt(df_rec["vel_kmh"].mean()),
            "Salto máx":    fmt(df_rec["jump_h"].max()),
            "Tiempo total": segundos_a_str(tiempo_track(df_rec)),
            "_vel_f":       float(df_rec["vel_kmh"].max()),
            "_t_f":         tiempo_track(df_rec),
            "color":        TRACK_COLORS[i % len(TRACK_COLORS)],
        })
    df_bench   = pd.DataFrame(bench_rows)
    ref_vel_f  = df_bench["_vel_f"].max()
    ref_time_f = df_bench["_t_f"].min()
    ref_vel_r  = df_bench.loc[df_bench["_vel_f"].idxmax(), "Recorrido"]
    ref_time_r = df_bench.loc[df_bench["_t_f"].idxmin(), "Recorrido"]

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("#### Por velocidad máxima")
        st.success(f"🥇 **{ref_vel_r}** — {fmt(ref_vel_f)} km/h")
        filas_vel = []
        for _, row in df_bench.sort_values("_vel_f", ascending=False).iterrows():
            diff = float(row["_vel_f"]) - ref_vel_f
            filas_vel.append({
                "Recorrido": row["Recorrido"],
                "Vel máx":   row["Vel máx"],
                "Vel prom":  row["Vel prom"],
                "Salto máx": row["Salto máx"],
                "Δ vs ref":  f"{diff:+.2f} km/h",
            })
        df_vel = pd.DataFrame(filas_vel)

        def style_vel(row):
            styles = [""] * len(row)
            try:
                d = float(row["Δ vs ref"].replace("+","").replace(" km/h",""))
                i = row.index.tolist().index("Δ vs ref")
                styles[i] = (
                    "background-color:#1a4a1a" if d == 0
                    else "background-color:#4a3a0a" if d >= -3
                    else "background-color:#4a1a1a"
                )
            except:
                pass
            return styles

        st.dataframe(
            df_vel.style.apply(style_vel, axis=1).hide(axis="index"),
            use_container_width=True
        )

    with col_b2:
        st.markdown("#### Por tiempo total")
        st.success(f"🥇 **{ref_time_r}** — {segundos_a_str(ref_time_f)}")
        filas_time = []
        for _, row in df_bench.sort_values("_t_f", ascending=True).iterrows():
            diff_s = float(row["_t_f"]) - ref_time_f
            filas_time.append({
                "Recorrido":    row["Recorrido"],
                "Tiempo total": row["Tiempo total"],
                "Δ vs ref":     f"{diff_s:+.2f} s",
            })
        df_time = pd.DataFrame(filas_time)

        def style_time(row):
            styles = [""] * len(row)
            try:
                d = float(row["Δ vs ref"].replace("+","").replace(" s",""))
                i = row.index.tolist().index("Δ vs ref")
                styles[i] = (
                    "background-color:#1a4a1a" if d == 0
                    else "background-color:#4a3a0a" if d <= 1
                    else "background-color:#4a1a1a"
                )
            except:
                pass
            return styles

        st.dataframe(
            df_time.style.apply(style_time, axis=1).hide(axis="index"),
            use_container_width=True
        )

    # Benchmark por zonas
    st.markdown("#### Benchmark por zona")
    for gc in st.session_state.geocercas:
        st.markdown(f"**{gc['nombre']}**")
        tiempos_z    = {n: tiempo_zona(s) for n,s in gc["segmentos"].items()}
        ref_tz       = min(tiempos_z.values()) if tiempos_z else 0
        vel_max_zona = max(
            (df_seg["vel_kmh"].max() for df_seg in gc["segmentos"].values()
             if len(df_seg)), default=0
        )
        filas_z = []
        for rec_nombre, df_seg in gc["segmentos"].items():
            tz     = tiempos_z[rec_nombre]
            diff_z = tz - ref_tz
            diff_v = float(df_seg["vel_kmh"].max()) - vel_max_zona \
                     if len(df_seg) else None
            filas_z.append({
                "Recorrido":   rec_nombre.replace(".csv",""),
                "Tiempo zona": segundos_a_str(tz),
                "Δ tiempo":    f"{diff_z:+.2f} s",
                "Vel máx":     fmt(df_seg["vel_kmh"].max()) if len(df_seg) else "-",
                "Vel prom":    fmt(df_seg["vel_kmh"].mean()) if len(df_seg) else "-",
                "Salto máx":   fmt(df_seg["jump_h"].max()) if len(df_seg) else "-",
                "Δ vel máx":   f"{diff_v:+.2f} km/h" if diff_v is not None else "-",
                "_tz_f":       tz,
            })
        df_zb = pd.DataFrame(filas_z).sort_values("_tz_f", ascending=True)

        def style_zona(row):
            styles = [""] * len(row)
            try:
                d = float(row["Δ tiempo"].replace("+","").replace(" s",""))
                i = row.index.tolist().index("Δ tiempo")
                styles[i] = (
                    "background-color:#1a4a1a" if d == 0
                    else "background-color:#4a3a0a" if d <= 0.5
                    else "background-color:#4a1a1a"
                )
            except:
                pass
            return styles

        st.dataframe(
            df_zb.drop(columns=["_tz_f"])
                 .style.apply(style_zona, axis=1)
                 .hide(axis="index"),
            use_container_width=True
        )
        st.markdown("---")

st.divider()

# ══════════════════════════════════════════════════════════
# SERIES TEMPORALES
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
    for y, title, color in [
        ("vel_kmh",  "Velocidad (km/h)",   "#378ADD"),
        ("acel_mag", "Aceleración (m/s²)", "#EF9F27"),
    ]:
        fig = px.line(df_graf, x="time_str", y=y, title=title,
                      color_discrete_sequence=[color])
        fig.update_layout(margin=dict(t=40,b=20), height=250,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

with col_b:
    for y, title, color in [
        ("jump_h",  "Saltos (m)",       "#1D9E75"),
        ("inc_mag", "Inclinación (°)",  "#534AB7"),
    ]:
        fig = px.area(df_graf, x="time_str", y=y, title=title,
                      color_discrete_sequence=[color])
        fig.update_layout(margin=dict(t=40,b=20), height=250,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

st.divider()
with st.expander("Ver datos procesados"):
    st.dataframe(
        df_graf[["time_str","lat","lon","jump_h",
                 "vel_kmh","acel_x","acel_y","inc_x","inc_y"]].rename(columns={
            "time_str":"Tiempo","lat":"Latitud","lon":"Longitud",
            "jump_h":"Salto (m)","vel_kmh":"Vel (km/h)",
            "acel_x":"Acel X","acel_y":"Acel Y",
            "inc_x":"Inc X (°)","inc_y":"Inc Y (°)",
        }).style.hide(axis="index"),
        use_container_width=True
    )
    csv_out = df_graf.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", csv_out, "rtk_procesado.csv", "text/csv")