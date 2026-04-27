import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import numpy as np
from shapely.geometry import Point, LineString
import json, os, io
import gspread
from google.oauth2.service_account import Credentials
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import plotly.io as pio

st.set_page_config(
    page_title="RTK GeoVisualizer BMX",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ RTK GeoVisualizer — BMX")
st.caption("Telemetría GNSS RTK 10 Hz | Cali, Colombia")

# ══════════════════════════════════════════════════════════
# GOOGLE SHEETS
# ══════════════════════════════════════════════════════════
SHEET_ID = "1f1iaIJ7-F1Z372DgdbqAzpKn69Wq7hT_VPYd_6aWDms"
SCOPES   = ["https://www.googleapis.com/auth/spreadsheets",
             "https://www.googleapis.com/auth/drive"]

@st.cache_resource
def get_gsheet():
    try:
        if os.path.exists("credentials.json"):
            creds = Credentials.from_service_account_file(
                "credentials.json", scopes=SCOPES)
        else:
            info = {k: st.secrets["gcp_service_account"][k] for k in [
                "type","project_id","private_key_id","private_key",
                "client_email","client_id","auth_uri","token_uri",
                "auth_provider_x509_cert_url","client_x509_cert_url"]}
            info["universe_domain"] = st.secrets["gcp_service_account"].get(
                "universe_domain","googleapis.com")
            creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        gc = gspread.authorize(creds)
        return gc.open_by_key(SHEET_ID)
    except Exception as e:
        st.error(f"Error conectando Google Sheets: {e}")
        return None

def get_worksheet(sh, nombre):
    try:
        return sh.worksheet(nombre)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=nombre, rows=1000, cols=30)

def cargar_perfiles_sheet():
    sh = get_gsheet()
    if sh is None:
        return {}
    try:
        ws    = get_worksheet(sh, "perfiles")
        datos = ws.get_all_records()
        perfiles = {}
        for row in datos:
            nombre = row.get("nombre","")
            if nombre:
                perfiles[nombre] = {
                    "nombre":    nombre,
                    "edad":      row.get("edad",""),
                    "categoria": row.get("categoria",""),
                    "peso":      row.get("peso",""),
                    "sesiones":  json.loads(row.get("sesiones","{}"))
                }
        return perfiles
    except Exception as e:
        st.warning(f"No se pudieron cargar perfiles: {e}")
        return {}

def guardar_perfil_sheet(perfil):
    sh = get_gsheet()
    if sh is None:
        return False
    try:
        ws     = get_worksheet(sh, "perfiles")
        datos  = ws.get_all_records()
        nombres = [r.get("nombre","") for r in datos]
        fila   = [perfil["nombre"], perfil["edad"],
                  perfil["categoria"], perfil["peso"],
                  json.dumps(perfil["sesiones"])]
        if perfil["nombre"] in nombres:
            idx = nombres.index(perfil["nombre"]) + 2
            ws.update(f"A{idx}:E{idx}", [fila])
        else:
            if not datos:
                ws.append_row(["nombre","edad","categoria","peso","sesiones"])
            ws.append_row(fila)
        return True
    except Exception as e:
        st.error(f"Error guardando perfil: {e}")
        return False

def eliminar_perfil_sheet(nombre):
    sh = get_gsheet()
    if sh is None:
        return
    try:
        ws     = get_worksheet(sh, "perfiles")
        datos  = ws.get_all_records()
        nombres = [r.get("nombre","") for r in datos]
        if nombre in nombres:
            ws.delete_rows(nombres.index(nombre) + 2)
    except Exception as e:
        st.error(f"Error eliminando perfil: {e}")

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def segundos_a_str(s):
    try:
        s = float(s)
        return f"{int(s//60):02d}:{s%60:05.2f}"
    except:
        return "00:00.00"

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
    df = pd.read_csv(io.BytesIO(file_content))
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "latitude":"lat","longitude":"lon",
        "speed_ms":"vel_ms","acc_x_ms2":"acel_x",
        "acc_y_ms2":"acel_y","inc_x_deg":"inc_x",
        "inc_y_deg":"inc_y","jump_height_m":"jump_h",
    })
    df["vel_kmh"]  = (df["vel_ms"] * 3.6).round(2)
    df["acel_mag"] = np.sqrt(df["acel_x"]**2 + df["acel_y"]**2).round(2)
    df["inc_mag"]  = np.sqrt(df["inc_x"]**2  + df["inc_y"]**2).round(2)
    if "jump_h" not in df.columns:
        df["jump_h"] = 0.0
    df["jump_h"]   = df["jump_h"].round(2)
    if "timestamp_s" not in df.columns:
        df["timestamp_s"] = np.arange(len(df)) * 0.1
    df["timestamp_s"] = pd.to_numeric(df["timestamp_s"],
                                      errors="coerce").fillna(0)
    df["time_str"]  = df["timestamp_s"].apply(segundos_a_str)
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

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

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
                int(c0[2]+f*(c1[2]-c0[2])))
    return "#32c850"

def metricas_sesion(df, geocercas):
    met = {
        "vel_max":  round(float(df["vel_kmh"].max()), 2),
        "vel_prom": round(float(df["vel_kmh"].mean()), 2),
        "jump_max": round(float(df["jump_h"].max()), 2),
        "acel_max": round(float(df["acel_mag"].max()), 2),
        "inc_max":  round(float(df["inc_mag"].max()), 2),
        "t_total":  round(tiempo_track(df), 2),
        "zonas":    {}
    }
    for gc in geocercas:
        seg = gc["segmentos"].get(df["recorrido"].iloc[0], pd.DataFrame())
        if len(seg) > 0:
            met["zonas"][gc["nombre"]] = {
                "vel_max":  round(float(seg["vel_kmh"].max()), 2),
                "vel_min":  round(float(seg["vel_kmh"].min()), 2),
                "vel_prom": round(float(seg["vel_kmh"].mean()), 2),
                "jump_max": round(float(seg["jump_h"].max()), 2),
                "acel_max": round(float(seg["acel_mag"].max()), 2),
                "inc_max":  round(float(seg["inc_mag"].max()), 2),
                "t_zona":   round(tiempo_zona(seg), 2),
            }
    return met

# ── PDF generator ─────────────────────────────────────────
def generar_pdf_recorridos(recorridos_activos, geocercas, bench_rows):
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=1.5*cm, rightMargin=1.5*cm,
                               topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("RTK GeoVisualizer BMX — Reporte de Recorridos",
                            styles["Title"]))
    story.append(Spacer(1, 0.4*cm))

    # Resumen global
    story.append(Paragraph("Resumen de recorridos", styles["Heading2"]))
    hdr  = ["Recorrido","Vel máx (km/h)","Vel prom (km/h)",
            "Salto máx (m)","Tiempo total"]
    rows = [hdr] + [[
        r["Recorrido"], r["Vel máx"], r["Vel prom"],
        r["Salto máx"], r["Tiempo total"]
    ] for r in bench_rows]
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#185FA5")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTSIZE",   (0,0), (-1,-1), 8),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke, colors.white]),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Benchmark
    story.append(Paragraph("Benchmark", styles["Heading2"]))
    df_b = pd.DataFrame(bench_rows)
    if len(df_b) > 1:
        ref_v = float(df_b["_vel_f"].max())
        ref_t = float(df_b["_t_f"].min())
        hdr2  = ["Recorrido","Vel máx","Δ vel","Tiempo","Δ tiempo"]
        rows2 = [hdr2]
        for _, row in df_b.iterrows():
            dv = float(row["_vel_f"]) - ref_v
            dt = float(row["_t_f"])   - ref_t
            rows2.append([
                row["Recorrido"], row["Vel máx"],
                f"{dv:+.2f} km/h", row["Tiempo total"],
                f"{dt:+.2f} s"
            ])
        t2 = Table(rows2, repeatRows=1)
        t2.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#185FA5")),
            ("TEXTCOLOR", (0,0),(-1,0), colors.white),
            ("FONTSIZE",  (0,0),(-1,-1), 8),
            ("GRID",      (0,0),(-1,-1), 0.5, colors.grey),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
            ("ALIGN",     (0,0),(-1,-1), "CENTER"),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.5*cm))

    # Zonas
    if geocercas:
        story.append(Paragraph("Análisis por zona", styles["Heading2"]))
        for gc in geocercas:
            story.append(Paragraph(f"{gc['nombre']} — {gc['tipo']}",
                                   styles["Heading3"]))
            hdr3  = ["Recorrido","Tiempo zona","Vel máx","Vel mín",
                     "Vel prom","Salto máx","Obj máx","Obj mín"]
            rows3 = [hdr3]
            for rec_n, df_seg in gc["segmentos"].items():
                tz = tiempo_zona(df_seg)
                rows3.append([
                    rec_n.replace(".csv",""),
                    segundos_a_str(tz),
                    fmt(df_seg["vel_kmh"].max()) if len(df_seg) else "-",
                    fmt(df_seg["vel_kmh"].min()) if len(df_seg) else "-",
                    fmt(df_seg["vel_kmh"].mean()) if len(df_seg) else "-",
                    fmt(df_seg["jump_h"].max()) if len(df_seg) else "-",
                    fmt(gc["vel_max_obj"]),
                    fmt(gc["vel_min_obj"]),
                ])
            t3 = Table(rows3, repeatRows=1)
            t3.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#185FA5")),
                ("TEXTCOLOR", (0,0),(-1,0),colors.white),
                ("FONTSIZE",  (0,0),(-1,-1),7),
                ("GRID",      (0,0),(-1,-1),0.5,colors.grey),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),
                 [colors.whitesmoke,colors.white]),
                ("ALIGN",     (0,0),(-1,-1),"CENTER"),
            ]))
            story.append(t3)
            story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    buf.seek(0)
    return buf

def generar_pdf_perfil(perfil):
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=1.5*cm, rightMargin=1.5*cm,
                               topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph(
        f"Perfil Deportista — {perfil['nombre']}", styles["Title"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Edad: {perfil['edad']} años | Categoría: {perfil['categoria']} | "
        f"Peso: {perfil['peso']} kg", styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    if perfil["sesiones"]:
        story.append(Paragraph("Evolución semanal", styles["Heading2"]))
        semanas = sorted(perfil["sesiones"].keys())
        hdr = ["Semana","Vel máx","Vel prom","Salto máx","Acel máx","T total"]
        rows = [hdr]
        for s in semanas:
            ses = perfil["sesiones"][s]
            rows.append([
                s,
                f"{ses['vel_max']:.2f}",
                f"{ses['vel_prom']:.2f}",
                f"{ses['jump_max']:.2f}",
                f"{ses['acel_max']:.2f}",
                segundos_a_str(ses["t_total"]),
            ])
        t = Table(rows, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#185FA5")),
            ("TEXTCOLOR", (0,0),(-1,0),colors.white),
            ("FONTSIZE",  (0,0),(-1,-1),8),
            ("GRID",      (0,0),(-1,-1),0.5,colors.grey),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
            ("ALIGN",     (0,0),(-1,-1),"CENTER"),
        ]))
        story.append(t)

    doc.build(story)
    buf.seek(0)
    return buf

TRACK_COLORS = ["#4ECDC4","#FF6B6B","#FFE66D","#A8E6CF","#B8B8FF","#FF8B94"]
CATEGORIAS   = ["Junior","Elite","Master","Sub-23","Otra"]

# ══════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ══════════════════════════════════════════════════════════
for key, val in {
    "recorridos":    {},
    "tracks_vis":    {},
    "geocercas":     [],
    "linea_inicio":  None,
    "linea_fin":     None,
    "analizar":      False,
    "last_drawing":  None,
    "perfiles":      None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

if st.session_state.perfiles is None:
    st.session_state.perfiles = cargar_perfiles_sheet()

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🗺️ Análisis de recorridos", "👤 Perfiles de deportistas"])

# ══════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════
with tab1:

    with st.sidebar:
        st.header("Recorridos")
        archivos = st.file_uploader("Cargar CSV (múltiples)", type=["csv"],
                                    accept_multiple_files=True)
        st.divider()
        color_var = st.selectbox(
            "Colorear track por:",
            ["vel_kmh","jump_h","acel_mag","inc_mag"],
            format_func=lambda x: {
                "vel_kmh":"Velocidad (km/h)","jump_h":"Saltos (m)",
                "acel_mag":"Aceleración (m/s²)","inc_mag":"Inclinación (°)"
            }[x]
        )
        unit_map  = {"vel_kmh":"km/h","jump_h":"m",
                     "acel_mag":"m/s²","inc_mag":"°"}
        label_map = {"vel_kmh":"Vel","jump_h":"Salto",
                     "acel_mag":"Acel","inc_mag":"Incl"}

    # Procesar archivos
    nombres_subidos = {f.name for f in archivos} if archivos else set()
    for n in [n for n in st.session_state.recorridos
              if n not in nombres_subidos]:
        del st.session_state.recorridos[n]
        st.session_state.tracks_vis.pop(n, None)
    if archivos:
        for f in archivos:
            if f.name not in st.session_state.recorridos:
                st.session_state.recorridos[f.name] = load_csv(
                    f.read(), f.name)
                st.session_state.tracks_vis[f.name] = True

    if not st.session_state.recorridos:
        st.info("Carga uno o más archivos CSV en el panel lateral.")
    else:
        recorridos_activos = st.session_state.recorridos

        # ── Métricas ──────────────────────────────────────
        st.subheader("📋 Resumen de recorridos")
        cols_met = st.columns(len(recorridos_activos))
        for i, (nombre, df_rec) in enumerate(recorridos_activos.items()):
            color = TRACK_COLORS[i % len(TRACK_COLORS)]
            with cols_met[i]:
                st.markdown(
                    f'<div style="border-left:4px solid {color};'
                    f'padding-left:10px;"><b>'
                    f'{nombre.replace(".csv","")}</b></div>',
                    unsafe_allow_html=True)
                st.metric("Vel. máx",
                          fmt(df_rec["vel_kmh"].max()) + " km/h")
                st.metric("Vel. promedio",
                          fmt(df_rec["vel_kmh"].mean()) + " km/h")
                st.metric("Salto máx",
                          fmt(df_rec["jump_h"].max()) + " m")
                st.metric("Tiempo total",
                          segundos_a_str(tiempo_track(df_rec)))

        st.divider()

        # ── Control de visibilidad y reproducción ─────────
        st.subheader("🗺️ Mapa geoespacial")

        ctrl_cols = st.columns(len(recorridos_activos) + 1)
        for i, nombre in enumerate(list(recorridos_activos.keys())):
            color = TRACK_COLORS[i % len(TRACK_COLORS)]
            label = nombre.replace(".csv","")
            vis   = st.session_state.tracks_vis.get(nombre, True)
            with ctrl_cols[i]:
                if st.button(
                    f"{'👁️' if vis else '🙈'} {label[:12]}",
                    key=f"vis_{nombre}"
                ):
                    st.session_state.tracks_vis[nombre] = not vis
                    st.rerun()

        with ctrl_cols[-1]:
            reproducir_nombre = st.selectbox(
                "▶️ Reproducir",
                ["—"] + list(recorridos_activos.keys()),
                format_func=lambda x: "Seleccionar" if x=="—"
                else x.replace(".csv",""),
                key="sel_repro",
                label_visibility="collapsed"
            )

        # ── Mapa ──────────────────────────────────────────
        if st.session_state.linea_inicio is None:
            st.info("**Paso 1:** Dibuja la línea de **inicio** de zona.")
        elif st.session_state.linea_fin is None:
            st.info("**Paso 2:** Dibuja la línea de **fin** de zona.")
        else:
            st.success("Líneas capturadas. Completa el formulario abajo.")

        all_lats = np.concatenate(
            [d["lat"].values for d in recorridos_activos.values()])
        all_lons = np.concatenate(
            [d["lon"].values for d in recorridos_activos.values()])
        center = [float(np.mean(all_lats)), float(np.mean(all_lons))]

        m = folium.Map(location=center, zoom_start=19,
                       max_zoom=22, control_scale=True)
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite",
            max_zoom=22, max_native_zoom=22,
        ).add_to(m)

        var_label = label_map[color_var]
        var_unit  = unit_map[color_var]

        for rec_idx, (nombre, df_rec) in enumerate(
                recorridos_activos.items()):
            if not st.session_state.tracks_vis.get(nombre, True):
                continue
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
                fill_color="#22c55e",
                tooltip=f"<b>Inicio: {rec_label}</b>"
            ).add_to(m)
            folium.CircleMarker(
                [df_rec["lat"].iloc[-1], df_rec["lon"].iloc[-1]],
                radius=7, color="#ef4444", fill=True,
                fill_color="#ef4444",
                tooltip=f"<b>Fin: {rec_label}</b>"
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

        colors_gc = ["#FF6B6B","#4ECDC4","#FFE66D",
                     "#A8E6CF","#FF8B94","#B8B8FF"]
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
                             f'white-space:nowrap;">'
                             f'{gc["nombre"]}</div>'
                    )
                ).add_to(m)

        legend_html = f"""
        <div style="position:fixed;bottom:30px;right:10px;z-index:1000;
             background:rgba(0,0,0,0.75);padding:10px 14px;
             border-radius:8px;font-size:12px;color:white;min-width:150px;">
          <b>{var_label} ({var_unit})</b><br>
          <div style="display:flex;align-items:center;gap:6px;margin-top:6px;">
            <span style="font-size:10px;">Bajo</span>
            <div style="width:90px;height:10px;border-radius:4px;
                 background:linear-gradient(to right,#dc3232,#e6b400,
                 #32c850);"></div>
            <span style="font-size:10px;">Alto</span>
          </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        Draw(export=False, draw_options={
            "polyline":True,"polygon":False,"rectangle":False,
            "circle":False,"marker":False,"circlemarker":False
        }, edit_options={"edit":False}).add_to(m)

        map_data = st_folium(m, width="100%", height=500,
                             key="main_map",
                             returned_objects=["last_active_drawing"])

        if map_data and map_data.get("last_active_drawing"):
            geom = map_data["last_active_drawing"].get("geometry",{})
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

        # ── Reproducción animada ──────────────────────────
        if reproducir_nombre != "—" and \
                reproducir_nombre in recorridos_activos:
            st.divider()
            st.subheader(f"▶️ Reproducción — "
                         f"{reproducir_nombre.replace('.csv','')}")
            df_r  = recorridos_activos[reproducir_nombre]
            step  = max(1, len(df_r) // 200)
            df_r2 = df_r.iloc[::step].reset_index(drop=True)

            frames, steps_slider = [], []
            for k in range(1, len(df_r2)+1):
                sub = df_r2.iloc[:k]
                frames.append(go.Frame(
                    data=[
                        go.Scattermapbox(
                            lat=sub["lat"], lon=sub["lon"],
                            mode="lines",
                            line=dict(width=4, color="#378ADD"),
                            name="Track"
                        ),
                        go.Scattermapbox(
                            lat=[sub["lat"].iloc[-1]],
                            lon=[sub["lon"].iloc[-1]],
                            mode="markers",
                            marker=dict(size=12, color="#ef4444"),
                            name="Posición",
                            text=[
                                f"T: {sub['time_str'].iloc[-1]}<br>"
                                f"Vel: {sub['vel_kmh'].iloc[-1]:.2f} km/h"
                            ],
                            hoverinfo="text"
                        )
                    ],
                    name=str(k)
                ))
                steps_slider.append(dict(
                    args=[[str(k)],
                          {"frame":{"duration":50,"redraw":True},
                           "mode":"immediate"}],
                    label=sub["time_str"].iloc[-1],
                    method="animate"
                ))

            fig_anim = go.Figure(
                data=[
                    go.Scattermapbox(
                        lat=df_r2["lat"].iloc[:1],
                        lon=df_r2["lon"].iloc[:1],
                        mode="lines",
                        line=dict(width=4, color="#378ADD"),
                        name="Track"
                    ),
                    go.Scattermapbox(
                        lat=[df_r2["lat"].iloc[0]],
                        lon=[df_r2["lon"].iloc[0]],
                        mode="markers",
                        marker=dict(size=12, color="#ef4444"),
                        name="Posición"
                    )
                ],
                frames=frames
            )
            fig_anim.update_layout(
                mapbox=dict(
                    style="satellite",
                    accesstoken=None,
                    center=dict(lat=df_r2["lat"].mean(),
                                lon=df_r2["lon"].mean()),
                    zoom=17
                ),
                mapbox_style="white-bg",
                height=480,
                margin=dict(t=0,b=0,l=0,r=0),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    y=1.05, x=0,
                    xanchor="left",
                    buttons=[
                        dict(label="▶ Play",
                             method="animate",
                             args=[None,{"frame":{"duration":50,
                                                  "redraw":True},
                                         "fromcurrent":True}]),
                        dict(label="⏸ Pausa",
                             method="animate",
                             args=[[None],{"frame":{"duration":0},
                                           "mode":"immediate"}])
                    ]
                )],
                sliders=[dict(
                    steps=steps_slider,
                    active=0,
                    y=0, x=0, len=1.0,
                    currentvalue=dict(prefix="T: ", visible=True),
                    transition=dict(duration=0)
                )]
            )
            # Usar tile satelital de Google como fondo
            fig_anim.update_layout(
                mapbox=dict(
                    style="white-bg",
                    layers=[dict(
                        sourcetype="raster",
                        source=["https://mt1.google.com/vt/lyrs=s"
                                "&x={x}&y={y}&z={z}"],
                        below="traces"
                    )],
                    center=dict(lat=float(df_r2["lat"].mean()),
                                lon=float(df_r2["lon"].mean())),
                    zoom=17
                )
            )
            st.plotly_chart(fig_anim, use_container_width=True)

        # ── Formulario zona ───────────────────────────────
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
                    nombre_zona = st.text_input(
                        "Nombre de la zona",
                        placeholder="Ej: Recta inicial")
                with col2:
                    tipo_zona = st.selectbox("Tipo",
                        ["Recta","Curva","Salto","Frenada","Otra"])
                c1, c2 = st.columns(2)
                with c1:
                    vel_max_obj = st.number_input(
                        "Vel. máx objetivo (km/h)",
                        0.0, 200.0, 40.0, 0.5)
                with c2:
                    vel_min_obj = st.number_input(
                        "Vel. mín objetivo (km/h)",
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
                            "Vel. máx obj", 0.0, 200.0,
                            float(gc["vel_max_obj"]), 0.5,
                            key=f"edit_max_{idx}")
                    with col2:
                        nuevo_min = st.number_input(
                            "Vel. mín obj", 0.0, 200.0,
                            float(gc["vel_min_obj"]), 0.5,
                            key=f"edit_min_{idx}")
                    with col3:
                        st.write(""); st.write("")
                        if st.button("💾", key=f"save_gc_{idx}"):
                            st.session_state.geocercas[idx][
                                "vel_max_obj"] = nuevo_max
                            st.session_state.geocercas[idx][
                                "vel_min_obj"] = nuevo_min
                            st.rerun()
                    if st.button("🗑️ Eliminar", key=f"del_gc_{idx}"):
                        st.session_state.geocercas.pop(idx)
                        st.rerun()
                    st.markdown("---")

        # ── Analizar ──────────────────────────────────────
        st.divider()
        if st.session_state.geocercas:
            if st.button("🔍 Analizar zonas", type="primary",
                         use_container_width=True):
                st.session_state.analizar = True

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

        if st.session_state.analizar and st.session_state.geocercas:
            st.subheader("📊 Análisis por zona")
            for gc in st.session_state.geocercas:
                st.markdown(f"### {gc['nombre']} — {gc['tipo']}")
                st.caption(
                    f"Objetivo → Vel máx: {gc['vel_max_obj']:.2f} km/h | "
                    f"Vel mín: {gc['vel_min_obj']:.2f} km/h")
                filas = []
                for rec_n, df_seg in gc["segmentos"].items():
                    tz      = tiempo_zona(df_seg)
                    vel_max = float(df_seg["vel_kmh"].max()) \
                              if len(df_seg) else -1
                    filas.append({
                        "Recorrido":     rec_n.replace(".csv",""),
                        "Tiempo zona":   segundos_a_str(tz),
                        "Vel máx real":  fmt(vel_max) if len(df_seg) else "-",
                        "Vel mín real":  fmt(df_seg["vel_kmh"].min()) if len(df_seg) else "-",
                        "Vel prom real": fmt(df_seg["vel_kmh"].mean()) if len(df_seg) else "-",
                        "Salto máx":     fmt(df_seg["jump_h"].max()) if len(df_seg) else "-",
                        "Vel máx obj":   fmt(gc["vel_max_obj"]),
                        "Vel mín obj":   fmt(gc["vel_min_obj"]),
                        "_vel_max_f":    vel_max,
                    })
                df_tabla = pd.DataFrame(filas).sort_values(
                    "_vel_max_f", ascending=False)

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
                                    "background-color:#1a4a1a"
                                    if diff <= obj*0.05
                                    else "background-color:#4a3a0a"
                                    if diff <= obj*0.15
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
                st.caption("🟢 ≤5%  |  🟡 ≤15%  |  🔴 Fuera de rango")
                st.markdown("---")

            st.subheader("🏆 Benchmark")
            df_bench   = pd.DataFrame(bench_rows)
            ref_vel_f  = df_bench["_vel_f"].max()
            ref_time_f = df_bench["_t_f"].min()
            ref_vel_r  = df_bench.loc[df_bench["_vel_f"].idxmax(),
                                      "Recorrido"]
            ref_time_r = df_bench.loc[df_bench["_t_f"].idxmin(),
                                      "Recorrido"]

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("#### Por velocidad máxima")
                st.success(f"🥇 **{ref_vel_r}** — {fmt(ref_vel_f)} km/h")
                filas_vel = []
                for _, row in df_bench.sort_values(
                        "_vel_f", ascending=False).iterrows():
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
                        d = float(row["Δ vs ref"].replace(
                            "+","").replace(" km/h",""))
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
                    df_vel.style.apply(style_vel, axis=1)
                               .hide(axis="index"),
                    use_container_width=True)

            with col_b2:
                st.markdown("#### Por tiempo total")
                st.success(
                    f"🥇 **{ref_time_r}** — "
                    f"{segundos_a_str(ref_time_f)}")
                filas_time = []
                for _, row in df_bench.sort_values(
                        "_t_f", ascending=True).iterrows():
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
                        d = float(row["Δ vs ref"].replace(
                            "+","").replace(" s",""))
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
                    df_time.style.apply(style_time, axis=1)
                                 .hide(axis="index"),
                    use_container_width=True)

            st.markdown("#### Benchmark por zona")
            for gc in st.session_state.geocercas:
                st.markdown(f"**{gc['nombre']}**")
                tiempos_z    = {n: tiempo_zona(s)
                                for n,s in gc["segmentos"].items()}
                ref_tz       = min(tiempos_z.values()) if tiempos_z else 0
                vel_max_zona = max(
                    (df_seg["vel_kmh"].max()
                     for df_seg in gc["segmentos"].values()
                     if len(df_seg)), default=0)
                filas_z = []
                for rec_n, df_seg in gc["segmentos"].items():
                    tz     = tiempos_z[rec_n]
                    diff_z = tz - ref_tz
                    diff_v = float(df_seg["vel_kmh"].max()) - vel_max_zona \
                             if len(df_seg) else None
                    filas_z.append({
                        "Recorrido":   rec_n.replace(".csv",""),
                        "Tiempo zona": segundos_a_str(tz),
                        "Δ tiempo":    f"{diff_z:+.2f} s",
                        "Vel máx":     fmt(df_seg["vel_kmh"].max()) if len(df_seg) else "-",
                        "Vel prom":    fmt(df_seg["vel_kmh"].mean()) if len(df_seg) else "-",
                        "Salto máx":   fmt(df_seg["jump_h"].max()) if len(df_seg) else "-",
                        "Δ vel máx":   f"{diff_v:+.2f} km/h" if diff_v is not None else "-",
                        "_tz_f":       tz,
                    })
                df_zb = pd.DataFrame(filas_z).sort_values(
                    "_tz_f", ascending=True)
                def style_zona(row):
                    styles = [""] * len(row)
                    try:
                        d = float(row["Δ tiempo"].replace(
                            "+","").replace(" s",""))
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
                    use_container_width=True)
                st.markdown("---")

        # ── Descarga PDF recorridos ───────────────────────
        st.divider()
        st.subheader("📥 Descargar análisis")
        if st.button("📄 Generar PDF de recorridos",
                     use_container_width=True):
            pdf_buf = generar_pdf_recorridos(
                recorridos_activos,
                st.session_state.geocercas,
                bench_rows
            )
            st.download_button(
                "⬇️ Descargar PDF",
                pdf_buf,
                "reporte_recorridos.pdf",
                "application/pdf",
                use_container_width=True
            )

        st.divider()
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
                df_graf[["time_str","lat","lon","jump_h","vel_kmh",
                         "acel_x","acel_y","inc_x","inc_y"]].rename(columns={
                    "time_str":"Tiempo","lat":"Latitud","lon":"Longitud",
                    "jump_h":"Salto (m)","vel_kmh":"Vel (km/h)",
                    "acel_x":"Acel X","acel_y":"Acel Y",
                    "inc_x":"Inc X (°)","inc_y":"Inc Y (°)",
                }).style.hide(axis="index"),
                use_container_width=True
            )
            csv_out = df_graf.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV", csv_out,
                               "rtk_procesado.csv", "text/csv")

# ══════════════════════════════════════════════════════════
# TAB 2 — PERFILES
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("👤 Perfiles de deportistas")
    perfiles = st.session_state.perfiles

    with st.expander("➕ Crear nuevo perfil"):
        with st.form("form_perfil", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                p_nombre = st.text_input("Nombre completo",
                                         placeholder="Ej: Juan Pérez")
                p_edad   = st.number_input("Edad", 5, 80, 15)
            with c2:
                p_cat  = st.selectbox("Categoría", CATEGORIAS)
                p_peso = st.number_input("Peso (kg)", 20.0, 150.0, 60.0, 0.5)
            if st.form_submit_button("💾 Crear perfil") and p_nombre:
                if p_nombre in perfiles:
                    st.error("Ya existe un perfil con ese nombre.")
                else:
                    nuevo = {
                        "nombre":    p_nombre,
                        "edad":      p_edad,
                        "categoria": p_cat,
                        "peso":      p_peso,
                        "sesiones":  {}
                    }
                    perfiles[p_nombre] = nuevo
                    guardar_perfil_sheet(nuevo)
                    st.session_state.perfiles = perfiles
                    st.success(f"Perfil '{p_nombre}' creado.")
                    st.rerun()

    if not perfiles:
        st.info("No hay perfiles. Crea el primero arriba.")
    else:
        deportista_sel = st.selectbox(
            "Seleccionar deportista:",
            list(perfiles.keys()),
            key="sel_deportista"
        )
        p = perfiles[deportista_sel]

        col_i1,col_i2,col_i3,col_i4,col_i5 = st.columns(5)
        col_i1.metric("Nombre",    p["nombre"])
        col_i2.metric("Edad",      f"{p['edad']} años")
        col_i3.metric("Categoría", p["categoria"])
        col_i4.metric("Peso",      f"{p['peso']} kg")
        col_i5.metric("Sesiones",  len(p["sesiones"]))

        st.divider()

        with st.expander("📂 Cargar nueva sesión semanal"):
            with st.form("form_sesion", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    semana  = st.text_input("Identificador de semana",
                                            placeholder="Ej: Semana 1")
                    archivo = st.file_uploader("CSV del recorrido",
                                               type=["csv"])
                with c2:
                    nota = st.text_area(
                        "Notas de la sesión",
                        placeholder="Ej: Pista mojada...",
                        height=100)
                if st.form_submit_button("💾 Guardar sesión") \
                        and semana and archivo:
                    df_ses = load_csv(archivo.read(), archivo.name)
                    met    = metricas_sesion(
                        df_ses, st.session_state.geocercas)
                    met["nota"]    = nota
                    met["archivo"] = archivo.name
                    p["sesiones"][semana] = met
                    guardar_perfil_sheet(p)
                    st.session_state.perfiles[deportista_sel] = p
                    st.success(f"Sesión '{semana}' guardada.")
                    st.rerun()

        if p["sesiones"]:
            st.subheader(f"📈 Evolución de {deportista_sel}")
            semanas = sorted(p["sesiones"].keys())
            df_evol = pd.DataFrame([{
                "Semana":    s,
                "Vel máx":   p["sesiones"][s]["vel_max"],
                "Vel prom":  p["sesiones"][s]["vel_prom"],
                "Salto máx": p["sesiones"][s]["jump_max"],
                "Acel máx":  p["sesiones"][s]["acel_max"],
                "Incl máx":  p["sesiones"][s]["inc_max"],
                "T total":   p["sesiones"][s]["t_total"],
                "Nota":      p["sesiones"][s].get("nota",""),
            } for s in semanas])

            st.markdown("#### Tabla de evolución semanal")
            def color_evol(col):
                if col.name in ["Vel máx","Vel prom","Salto máx"]:
                    mn, mx = col.min(), col.max()
                    return [
                        f"background-color:"
                        f"{'#1a4a1a' if v==mx else '#4a1a1a' if v==mn else ''}"
                        for v in col
                    ]
                return [""] * len(col)
            st.dataframe(
                df_evol.style.apply(color_evol, axis=0)
                             .hide(axis="index")
                             .format({
                                 "Vel máx":"{:.2f}","Vel prom":"{:.2f}",
                                 "Salto máx":"{:.2f}","Acel máx":"{:.2f}",
                                 "Incl máx":"{:.2f}","T total":"{:.2f}",
                             }),
                use_container_width=True
            )
            st.caption("🟢 Mejor valor  |  🔴 Peor valor")

            st.markdown("#### Gráficas de evolución")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                for y, title, color in [
                    ("Vel máx",   "Velocidad máxima (km/h)",   "#378ADD"),
                    ("Salto máx", "Salto máximo (m)",           "#1D9E75"),
                ]:
                    fig = px.line(df_evol, x="Semana", y=y,
                                  title=title, markers=True,
                                  color_discrete_sequence=[color])
                    fig.update_layout(height=250, margin=dict(t=40,b=20),
                                      plot_bgcolor="rgba(0,0,0,0)",
                                      paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
            with col_g2:
                for y, title, color in [
                    ("Vel prom", "Velocidad promedio (km/h)", "#EF9F27"),
                    ("T total",  "Tiempo total (s)",           "#D4537E"),
                ]:
                    fig = px.line(df_evol, x="Semana", y=y,
                                  title=title, markers=True,
                                  color_discrete_sequence=[color])
                    fig.update_layout(height=250, margin=dict(t=40,b=20),
                                      plot_bgcolor="rgba(0,0,0,0)",
                                      paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

            # Evolución por zona
            zonas_disp = set()
            for ses in p["sesiones"].values():
                zonas_disp.update(ses.get("zonas",{}).keys())
            if zonas_disp:
                st.markdown("#### Evolución por zona")
                zona_sel = st.selectbox("Zona:", sorted(zonas_disp),
                                        key="zona_evol")
                df_ze = pd.DataFrame([{
                    "Semana":    s,
                    "Vel máx":   p["sesiones"][s]["zonas"].get(
                        zona_sel,{}).get("vel_max",None),
                    "Vel prom":  p["sesiones"][s]["zonas"].get(
                        zona_sel,{}).get("vel_prom",None),
                    "Salto máx": p["sesiones"][s]["zonas"].get(
                        zona_sel,{}).get("jump_max",None),
                    "T zona":    p["sesiones"][s]["zonas"].get(
                        zona_sel,{}).get("t_zona",None),
                } for s in semanas
                  if zona_sel in p["sesiones"][s].get("zonas",{})])

                if not df_ze.empty:
                    col_z1, col_z2 = st.columns(2)
                    with col_z1:
                        for y,title,color in [
                            ("Vel máx",f"Vel máx — {zona_sel}","#378ADD"),
                            ("T zona", f"Tiempo zona — {zona_sel}","#D4537E"),
                        ]:
                            fig = px.line(df_ze, x="Semana", y=y,
                                          title=title, markers=True,
                                          color_discrete_sequence=[color])
                            fig.update_layout(height=230,
                                              margin=dict(t=40,b=20),
                                              plot_bgcolor="rgba(0,0,0,0)",
                                              paper_bgcolor="rgba(0,0,0,0)")
                            st.plotly_chart(fig, use_container_width=True)
                    with col_z2:
                        for y,title,color in [
                            ("Vel prom",f"Vel prom — {zona_sel}","#EF9F27"),
                            ("Salto máx",f"Salto — {zona_sel}","#1D9E75"),
                        ]:
                            fig = px.line(df_ze, x="Semana", y=y,
                                          title=title, markers=True,
                                          color_discrete_sequence=[color])
                            fig.update_layout(height=230,
                                              margin=dict(t=40,b=20),
                                              plot_bgcolor="rgba(0,0,0,0)",
                                              paper_bgcolor="rgba(0,0,0,0)")
                            st.plotly_chart(fig, use_container_width=True)

            # Descarga PDF perfil
            st.divider()
            st.subheader("📥 Descargar perfil")
            if st.button("📄 Generar PDF del perfil",
                         use_container_width=True):
                pdf_buf = generar_pdf_perfil(p)
                st.download_button(
                    "⬇️ Descargar PDF",
                    pdf_buf,
                    f"perfil_{deportista_sel}.pdf",
                    "application/pdf",
                    use_container_width=True
                )

            st.divider()
            with st.expander("🗑️ Gestionar sesiones"):
                for semana in list(p["sesiones"].keys()):
                    ses = p["sesiones"][semana]
                    col1, col2 = st.columns([4,1])
                    col1.write(
                        f"**{semana}** — "
                        f"Vel máx: {ses['vel_max']:.2f} km/h | "
                        f"T: {segundos_a_str(ses['t_total'])} | "
                        f"{ses.get('nota','')[:40]}"
                    )
                    if col2.button("🗑️", key=f"del_ses_{semana}"):
                        del p["sesiones"][semana]
                        guardar_perfil_sheet(p)
                        st.session_state.perfiles[deportista_sel] = p
                        st.rerun()
        else:
            st.info("No hay sesiones. Carga la primera arriba.")

        st.divider()
        with st.expander("⚠️ Eliminar perfil"):
            if st.button(f"🗑️ Eliminar perfil de {deportista_sel}",
                         type="primary"):
                eliminar_perfil_sheet(deportista_sel)
                del st.session_state.perfiles[deportista_sel]
                st.rerun()