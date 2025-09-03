# app.py
# Demo de Dashboard CX (Satisfacción del Cliente) – basado en los tableros de la presentación
# Ejecuta: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import random
from pathlib import Path

st.set_page_config(
    page_title="Demo CX – Satisfacción del Cliente",
    page_icon="📊",
    layout="wide",
)

# =============== Utilidades ===============

# =============== Constantes ===============
UNIDADES = ["Consultoría", "Tecnología", "IA e Innovación", "Operaciones", "Comercial"]
PERSONAS = ["Director de proyecto", "Gerente", "Analista", "Usuario final", "Proveedor"]
MICRO_JOURNEYS = [
    "Retraso en recolección",
    "Momentos sin comunicación",
    "Datos difíciles de entender",
    "Respuestas poco amables",
]
STAGES = ["Descubrimiento", "Contratación", "Implementación", "Operación", "Cierre"]

MEJORAS = [
    "Comunicación oportuna",
    "Claridad de datos",
    "Tiempos de recolección",
    "Amabilidad del soporte",
    "Capacidad de respuesta",
    "Integración con sistemas",
    "Visibilidad de avances",
    "Capacitación",
]

FORTALEZAS = [
    "Equipo atento",
    "Rigor metodológico",
    "Flexibilidad",
    "Puntualidad",
    "Calidad de informes",
    "Soporte técnico",
    "Capacidad analítica",
]

RETRO_INTERNA = [
    "Realmente útil",
    "Mostrar el valor del trabajo",
    "Incluir ideas del equipo",
    "Herramienta para actuar",
    "Apoyo en formación/seguimiento",
    "Fácil de usar y entender",
]

# Logo (asumimos que el archivo existe en la misma carpeta)
LOGO_PATH = "logo.PNG"

def pesos(x):
    # Formateo COP abreviado
    if x >= 1_000_000_000:
        return f"${x/1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:,.0f}"

@st.cache_data(show_spinner=False)
def generar_datos(seed: int = 42, n_proyectos: int = 50, dias=180):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Proyectos
    proyectos = [f"P-{i:03d}" for i in range(1, n_proyectos + 1)]
    clientes = [f"Cliente {i:03d}" for i in range(1, n_proyectos + 1)]
    unidades = rng.choice(UNIDADES, size=n_proyectos)
    ticket_prom = rng.normal(180_000_000, 60_000_000, size=n_proyectos).clip(50_000_000, 400_000_000)
    freq_compra = rng.normal(3.1, 0.8, size=n_proyectos).clip(0.5, 6.0)
    lifespan = rng.normal(2.1, 0.6, size=n_proyectos).clip(0.5, 4.0)
    clientes_recurrentes = rng.choice([0, 1], size=n_proyectos, p=[0.5, 0.5])

    df_proyectos = pd.DataFrame({
        "cliente": clientes,
        "proyecto": proyectos,
        "unidad": unidades,
        "ticket_promedio": ticket_prom,
        "frecuencia_anual": freq_compra,
        "lifespan_anios": lifespan,
        "recurrente": clientes_recurrentes,
    })
    df_proyectos["CLV"] = (df_proyectos["ticket_promedio"]
                           * df_proyectos["frecuencia_anual"]
                           * df_proyectos["lifespan_anios"])

    # Encuestas
    n_enc = 3500
    hoy = datetime.today().date()
    fechas = [hoy - timedelta(days=int(x)) for x in rng.integers(0, dias, size=n_enc)]
    personas = rng.choice(PERSONAS, size=n_enc)
    micro = rng.choice(MICRO_JOURNEYS, size=n_enc)
    proyectos_muestra = rng.choice(proyectos, size=n_enc)
    # Mapear unidad desde proyecto
    mapa_unidad = dict(zip(df_proyectos["proyecto"], df_proyectos["unidad"]))
    unidades_enc = [mapa_unidad[p] for p in proyectos_muestra]
    # Probabilidades (levemente peores para ciertos micro-journeys)
    base_p = 0.65
    micro_penalty = {
        "Retraso en recolección": -0.08,
        "Momentos sin comunicación": -0.10,
        "Datos difíciles de entender": -0.06,
        "Respuestas poco amables": -0.12,
    }
    def p_ok(m):
        return float(np.clip(base_p + micro_penalty.get(m, 0), 0.35, 0.85))

    med1 = rng.choice([1, 0], size=n_enc, p=[0.63, 0.37])  # Med. 1 (Sí/No)
    med2 = rng.choice([1, 0], size=n_enc, p=[0.60, 0.40])
    med3 = rng.choice([1, 0], size=n_enc, p=[0.52, 0.48])
    med4 = rng.choice([1, 0], size=n_enc, p=[0.58, 0.42])

    # FCR (First Cycle/Contact Resolution) como binaria
    fcr = [rng.choice([1, 0], p=[p_ok(m), 1 - p_ok(m)]) for m in micro]
    # Lealtad (recurrente) aproximada
    lealtad = rng.choice([1, 0], size=n_enc, p=[0.5, 0.5])
    # Apego emocional (0-100)
    apego = np.clip(rng.normal(58, 12, size=n_enc), 10, 95)

    # Retrasos y SLA (verde <=10, amarillo 11-30, rojo >30)
    delay = np.clip(rng.normal(18, 16, size=n_enc), 0, 90)
    critico = (delay > 45) & (rng.random(n_enc) < 0.25)

    # Etapa actual y satisfacción por etapa
    stage = rng.choice(STAGES, size=n_enc)
    satisf = np.clip(rng.normal(3.7, 0.8, size=n_enc), 1, 5)
    # NPS (0-10) aproximado a partir de la satisfacción (1-5)
    nps_base = ((satisf - 1) / 4) * 10  # escala 1-5 -> 0-10
    nps = np.clip(np.round(nps_base + rng.normal(0, 1.2, size=n_enc)), 0, 10).astype(int)

    # Etiquetas de mejoras/fortalezas
    mejoras = rng.choice(MEJORAS, size=n_enc)
    fortalezas = rng.choice(FORTALEZAS, size=n_enc)

    df_enc = pd.DataFrame({
        "fecha": pd.to_datetime(fechas),
        "proyecto": proyectos_muestra,
        "unidad": unidades_enc,
        "persona": personas,
        "micro_journey": micro,
        "med1_si": med1,
        "med2_si": med2,
        "med3_si": med3,
        "med4_si": med4,
        "fcr": fcr,
        "lealtad": lealtad,
        "apego": apego,
        "delay_dias": delay,
        "incidente_critico": critico.astype(int),
        "etapa": stage,
        "satisfaccion": satisf,
        "nps": nps,
        "mejoras": mejoras,
        "fortalezas": fortalezas,
    })

    # Métricas de comunicaciones/cultura (por unidad y mes)
    fechas_mes = pd.date_range(end=hoy, periods=6, freq="MS")
    rows = []
    for u in UNIDADES:
        for fm in fechas_mes:
            rows.append({
                "unidad": u,
                "mes": fm,
                "coms_ejecutadas": np.clip(rng.normal(0.75, 0.08), 0.4, 0.98),
                "alcance": np.clip(rng.normal(0.68, 0.1), 0.3, 0.95),
                "recordacion": np.clip(rng.normal(0.55, 0.1), 0.2, 0.9),
                "interacciones": np.clip(rng.normal(0.40, 0.1), 0.1, 0.8),
                "capacitacion": np.clip(rng.normal(0.60, 0.1), 0.2, 0.95),
                "embajadores": np.clip(rng.normal(0.45, 0.1), 0.1, 0.8),
                "adopcion": np.clip(rng.normal(0.50, 0.1), 0.2, 0.9),
                "percepcion_interna": np.clip(rng.normal(0.55, 0.1), 0.2, 0.9),
            })
    df_com = pd.DataFrame(rows)

    # Retroalimentación interna (Likert 1-5)
    rows_r = []
    for u in UNIDADES:
        for item in RETRO_INTERNA:
            rows_r.append({
                "unidad": u,
                "aspecto": item,
                "promedio": np.clip(rng.normal(3.6, 0.6), 2, 4.8),
                "n": rng.integers(20, 120),
            })
    df_retro = pd.DataFrame(rows_r)

    return df_proyectos, df_enc, df_com, df_retro


# =============== Sidebar (Filtros globales) ===============

try:
    st.sidebar.image(LOGO_PATH, use_container_width=True)
except Exception:
    pass

st.sidebar.title("🎛️ Filtros")
seed = st.sidebar.number_input("Semilla (reproducibilidad)", min_value=0, value=42, step=1)
df_proy, df_enc, df_com, df_retro = generar_datos(seed=seed)

min_date, max_date = df_enc["fecha"].min().date(), df_enc["fecha"].max().date()
rango = st.sidebar.date_input(
    "Rango de fechas",
    value=(max(min_date, max_date - timedelta(days=90)), max_date),
    min_value=min_date,
    max_value=max_date,
)

unidades_sel = st.sidebar.multiselect("Unidad de negocio", UNIDADES, default=UNIDADES)
personas_sel = st.sidebar.multiselect("Customer persona", PERSONAS, default=PERSONAS)
micro_sel = st.sidebar.multiselect("Micro-journey", MICRO_JOURNEYS, default=MICRO_JOURNEYS)

proyectos_unidad = df_proy.query("unidad in @unidades_sel")["proyecto"].unique()
proyectos_sel = st.sidebar.multiselect("Proyecto", proyectos_unidad, default=list(proyectos_unidad)[:10])

st.sidebar.markdown("---")
st.sidebar.caption("Datos 100% ficticios para demo.")

# Aplicar filtros
if isinstance(rango, tuple):
    start_date, end_date = pd.to_datetime(rango[0]), pd.to_datetime(rango[1])
else:
    start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

mask = (
    (df_enc["fecha"] >= start_date)
    & (df_enc["fecha"] <= end_date)
    & (df_enc["unidad"].isin(unidades_sel))
    & (df_enc["persona"].isin(personas_sel))
    & (df_enc["micro_journey"].isin(micro_sel))
    & (df_enc["proyecto"].isin(proyectos_sel))
)
f_enc = df_enc.loc[mask].copy()
f_proy = df_proy[df_proy["proyecto"].isin(proyectos_sel) & df_proy["unidad"].isin(unidades_sel)].copy()
f_com = df_com[df_com["unidad"].isin(unidades_sel)].copy()
f_retro = df_retro[df_retro["unidad"].isin(unidades_sel)].copy()

# =============== Título ===============

try:
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image(LOGO_PATH, caption=None, use_container_width=True)
    with col_title:
        st.title("📊 Demo Dashboard de Experiencia del Cliente (CX)")
        st.caption("Incluye: KPIs financieros/CLV, FCR, SLAs, lealtad/apego, NPS, micro-journeys, CJM, trabajo de campo, comunicaciones/cultura y explorador de datos.")
    header_drawn = True
except Exception:
    header_drawn = False

if not header_drawn:
    st.title("📊 Demo Dashboard de Experiencia del Cliente (CX)")
    st.caption("Incluye: KPIs financieros/CLV, FCR, SLAs, lealtad/apego, NPS, micro-journeys, CJM, trabajo de campo, comunicaciones/cultura y explorador de datos.")

# =============== Tabs principales ===============
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Tablero 1 – Ejecutivo",
    "Tablero 2 – Percepción & Micro-journeys",
    "Tablero 3 – Customer Journey Map",
    "Tablero 4 – Trabajo de campo",
    "Tablero 5 – Cultura & Comunicaciones",
    "Tablero 6 – Benchmark por proyecto",
    "Explorador de datos",
])

# =============== Tablero 1: Ejecutivo ===============
with tab1:
    st.subheader("Resumen Ejecutivo")

    # KPIs
    total_clv = f_proy["CLV"].sum()
    n_proj = f_proy["proyecto"].nunique()
    fcr_rate = f_enc["fcr"].mean() if len(f_enc) else 0
    lealtad_rate = f_enc["lealtad"].mean() if len(f_enc) else 0
    apego_prom = f_enc["apego"].mean() if len(f_enc) else 0
    sin_novedades = (1 - (f_enc["incidente_critico"].mean() if len(f_enc) else 0))

    def calc_nps(series: pd.Series) -> float:
        if series is None or len(series) == 0:
            return 0.0
        promoters = (series >= 9).mean()
        detractors = (series <= 6).mean()
        return (promoters - detractors) * 100.0

    nps_val = calc_nps(f_enc["nps"]) if len(f_enc) else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("CLV Estimado", pesos(total_clv))
    c2.metric("Proyectos", n_proj)
    c3.metric("FCR", f"{fcr_rate*100:,.0f}%")
    c4.metric("Lealtad (clientes recurrentes)", f"{lealtad_rate*100:,.0f}%")
    c5.metric("Apego emocional", f"{apego_prom:,.0f} / 100")
    c6.metric("Proyectos sin novedades", f"{sin_novedades*100:,.0f}%")

    c7, c8, _, _, _, _ = st.columns(6)
    c7.metric("NPS", f"{nps_val:,.0f}")
    c8.metric("Proyectos sin novedades", f"{sin_novedades*100:,.0f}%")

    st.caption("CLV= Ticket Promedio × Frecuencia de compra × Duración (lifespan).")

    # Semáforo SLA (verde <=10, amarillo 11-30, rojo >30)
    if len(f_enc):
        sla = pd.cut(
            f_enc["delay_dias"],
            bins=[-0.01, 10, 30, 365],
            labels=["Verde (≤10 días)", "Amarillo (11–30)", "Rojo (>30)"]
        )
        sla_share = sla.value_counts(normalize=True).rename_axis("SLA").reset_index(name="pct")
        chart_sla = alt.Chart(sla_share).mark_bar().encode(
            x=alt.X("pct:Q", axis=alt.Axis(format="%"), title="Porcentaje"),
            y=alt.Y("SLA:N", sort=["Verde (≤10 días)", "Amarillo (11–30)", "Rojo (>30)"], title=None),
            tooltip=[alt.Tooltip("SLA:N"), alt.Tooltip("pct:Q", format=".0%")],
            color=alt.Color(
                "SLA:N",
                scale=alt.Scale(
                    domain=["Verde (≤10 días)", "Amarillo (11–30)", "Rojo (>30)"],
                    range=["#2ca02c", "#f1c40f", "#e74c3c"],
                ),
                legend=None,
            )
        ).properties(height=140)
        st.altair_chart(chart_sla, use_container_width=True)

        # Distribución NPS y NPS por unidad
        nps_bins = pd.Categorical(
            np.select(
                [f_enc["nps"] <= 6, f_enc["nps"].between(7, 8), f_enc["nps"] >= 9],
                ["Detractores (0-6)", "Pasivos (7-8)", "Promotores (9-10)"],
                default="Pasivos (7-8)"
            ),
            categories=["Detractores (0-6)", "Pasivos (7-8)", "Promotores (9-10)"]
        )
        # Barra 100% apilada para Detractores/Pasivos/Promotores
        dist_counts = pd.Series(nps_bins).value_counts().rename_axis("grupo").reset_index(name="n")
        dist_counts["barra"] = "Distribución NPS"
        dist_counts["pct"] = dist_counts["n"] / dist_counts["n"].sum()
        chart_nps_dist = alt.Chart(dist_counts).mark_bar().encode(
            x=alt.X("n:Q", stack="normalize", axis=alt.Axis(format="%"), title="Proporción"),
            y=alt.Y("barra:N", title=None),
            color=alt.Color(
                "grupo:N",
                scale=alt.Scale(
                    domain=["Detractores (0-6)", "Pasivos (7-8)", "Promotores (9-10)"],
                    range=["#e74c3c", "#95a5a6", "#2ecc71"],
                ),
                legend=alt.Legend(title=""),
            ),
            tooltip=["grupo", alt.Tooltip("n:Q", title="Conteo", format=","), alt.Tooltip("pct:Q", title="%", format=".0%")],
        ).properties(height=80)
        st.altair_chart(chart_nps_dist, use_container_width=True)

        nps_unidad = f_enc.groupby("unidad")["nps"].apply(lambda s: ((s>=9).mean() - (s<=6).mean())*100).reset_index(name="NPS")
        chart_nps_u = alt.Chart(nps_unidad).mark_bar().encode(
            x=alt.X("NPS:Q", title="NPS"),
            y=alt.Y("unidad:N", sort="-x", title=None),
            tooltip=["unidad", alt.Tooltip("NPS:Q", format=",.0f")]
        ).properties(height=180)
        st.altair_chart(chart_nps_u, use_container_width=True)
    else:
        st.info("No hay datos en el rango/segmento seleccionado.")

    # Ticket vs Frecuencia (dispersión) por proyecto
    if len(f_proy):
        chart_tf = alt.Chart(f_proy).mark_circle(size=100, opacity=0.7).encode(
            x=alt.X("ticket_promedio:Q", title="Ticket promedio (COP)"),
            y=alt.Y("frecuencia_anual:Q", title="Compras por año"),
            size=alt.Size("lifespan_anios:Q", title="Lifespan (años)"),
            color=alt.Color("unidad:N", legend=None),
            tooltip=[
                "proyecto", "unidad",
                alt.Tooltip("ticket_promedio:Q", title="Ticket", format=",.0f"),
                alt.Tooltip("frecuencia_anual:Q", title="Frecuencia", format=".2f"),
                alt.Tooltip("lifespan_anios:Q", title="Lifespan", format=".2f"),
                alt.Tooltip("CLV:Q", title="CLV", format=",.0f"),
            ],
        ).properties(height=280)
        st.altair_chart(chart_tf, use_container_width=True)

    # CLV por unidad (barra)
    if len(f_proy):
        clv_unidad = f_proy.groupby("unidad", as_index=False)["CLV"].sum().sort_values("CLV", ascending=False)
        chart_clv = alt.Chart(clv_unidad).mark_bar().encode(
            x=alt.X("CLV:Q", title="CLV total (COP)"),
            y=alt.Y("unidad:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("CLV:Q", format=",.0f"), "unidad:N"]
        ).properties(height=200)
        st.altair_chart(chart_clv, use_container_width=True)

# =============== Tablero 6: Benchmark por proyecto ===============
with tab6:
    st.subheader("Benchmark por proyecto")
    if len(f_enc):
        # Agregados por proyecto
        agg = f_enc.groupby("proyecto").agg(
            respuestas=("nps", "count"),
            nps=("nps", lambda s: ((s>=9).mean() - (s<=6).mean())*100),
            fcr=("fcr", "mean"),
            satisfaccion=("satisfaccion", "mean"),
            delay=("delay_dias", "mean"),
        ).reset_index()
        # Añadir cliente
        agg = agg.merge(df_proy[["proyecto", "cliente"]], on="proyecto", how="left")
        # Top/bottom por NPS
        top_n = agg.sort_values("nps", ascending=False).head(15)
        bottom_n = agg.sort_values("nps", ascending=True).head(15)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 15 NPS**")
            chart_top = alt.Chart(top_n).mark_bar().encode(
                x=alt.X("nps:Q", title="NPS", scale=alt.Scale(domain=[-100, 100])),
                y=alt.Y("cliente:N", sort="-x", title=None),
                color=alt.value("#2ecc71"),
                tooltip=["cliente", alt.Tooltip("nps:Q", format=",.0f"), "respuestas"]
            ).properties(height=360)
            st.altair_chart(chart_top, use_container_width=True)
        with c2:
            st.markdown("**Bottom 15 NPS**")
            chart_bot = alt.Chart(bottom_n).mark_bar().encode(
                x=alt.X("nps:Q", title="NPS", scale=alt.Scale(domain=[-100, 100])),
                y=alt.Y("cliente:N", sort="x", title=None),
                color=alt.value("#e74c3c"),
                tooltip=["cliente", alt.Tooltip("nps:Q", format=",.0f"), "respuestas"]
            ).properties(height=360)
            st.altair_chart(chart_bot, use_container_width=True)

        st.markdown("---")
        st.markdown("**Satisfacción vs. Retraso (por proyecto)**")
        scatter = alt.Chart(agg).mark_circle(opacity=0.7, size=100).encode(
            x=alt.X("satisfaccion:Q", title="Satisfacción (1–5)", scale=alt.Scale(domain=[1,5])),
            y=alt.Y("delay:Q", title="Retraso promedio (días)"),
            size=alt.Size("respuestas:Q", title="Respuestas"),
            color=alt.Color("nps:Q", title="NPS", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=["proyecto", alt.Tooltip("nps:Q", format=",.0f"), alt.Tooltip("fcr:Q", format=".0%"), alt.Tooltip("satisfaccion:Q", format=".2f"), alt.Tooltip("delay:Q", format=".1f"), "respuestas"]
        ).properties(height=360)
        st.altair_chart(scatter, use_container_width=True)

        st.caption("NPS = %Promotores (9–10) – %Detractores (0–6). Colores más verdes indican mejor NPS.")
    else:
        st.info("No hay datos disponibles con los filtros actuales.")

# =============== Tablero 2: Percepción & Micro-journeys ===============
with tab2:
    st.subheader("Percepción por Micro-journey y preguntas clave")

    mj = st.selectbox("Micro-journey", MICRO_JOURNEYS, index=0)
    sub = f_enc[f_enc["micro_journey"] == mj]
    cols = [
        ("med1_si", "Med. 1"),
        ("med2_si", "Med. 2"),
        ("med3_si", "Med. 3"),
        ("med4_si", "Med. 4"),
    ]
    if len(sub):
        cA, cB = st.columns([2, 3])

        # 4 tarjetas Sí/No con base
        with cA:
            st.markdown("**Resultados (Sí/No)**")
            cc1, cc2 = st.columns(2)
            for i, (col, nombre) in enumerate(cols):
                pct_si = sub[col].mean()
                base = len(sub)
                if i % 2 == 0:
                    container = cc1
                else:
                    container = cc2
                container.metric(f"{nombre} – Sí", f"{pct_si*100:,.0f}%", help=f"Base: {base} respuestas")

        # Barras por persona (sí)
        with cB:
            data_med = []
            for col, nombre in cols:
                tmp = sub.groupby("persona")[col].mean().rename("pct_si").reset_index()
                tmp["medida"] = nombre
                data_med.append(tmp)
            med_all = pd.concat(data_med, ignore_index=True)
            chart = alt.Chart(med_all).mark_bar().encode(
                x=alt.X("pct_si:Q", title="% Sí", axis=alt.Axis(format="%")),
                y=alt.Y("persona:N", sort="-x", title=None),
                color=alt.Color("medida:N", legend=alt.Legend(title="Medida")),
                tooltip=["persona", "medida", alt.Tooltip("pct_si:Q", format=".0%")],
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        colL, colR = st.columns(2)

        # ¿Qué procesos mejorar?
        with colL:
            st.markdown("**¿Qué procesos mejorar o replantear?**")
            mejoras_ct = sub["mejoras"].value_counts().reset_index()
            mejoras_ct.columns = ["proceso", "conteo"]
            chart_mej = alt.Chart(mejoras_ct).mark_bar().encode(
                x=alt.X("conteo:Q", title="Conteo"),
                y=alt.Y("proceso:N", sort="-x", title=None),
                tooltip=["proceso", "conteo"],
            ).properties(height=260)
            st.altair_chart(chart_mej, use_container_width=True)
            st.caption(f"Base: {len(sub)}")

        # ¿Fortalezas mostradas?
        with colR:
            st.markdown("**¿Qué fortalezas mostró el equipo?**")
            fort_ct = sub["fortalezas"].value_counts().reset_index()
            fort_ct.columns = ["fortaleza", "conteo"]
            chart_for = alt.Chart(fort_ct).mark_bar().encode(
                x=alt.X("conteo:Q", title="Conteo"),
                y=alt.Y("fortaleza:N", sort="-x", title=None),
                tooltip=["fortaleza", "conteo"],
            ).properties(height=260)
            st.altair_chart(chart_for, use_container_width=True)
            st.caption(f"Base: {len(sub)}")

    else:
        st.info("No hay respuestas para el micro-journey seleccionado con los filtros actuales.")

# =============== Tablero 3: Customer Journey Map (CJM) ===============
with tab3:
    st.subheader("Customer Journey Map (CJM)")

    if len(f_enc):
        # Heatmap promedio por persona y etapa
        heat = f_enc.groupby(["persona", "etapa"], as_index=False)["satisfaccion"].mean()
        chart_h = alt.Chart(heat).mark_rect().encode(
            x=alt.X("etapa:N", sort=STAGES, title=None),
            y=alt.Y("persona:N", title=None),
            color=alt.Color("satisfaccion:Q", scale=alt.Scale(scheme="yellowgreenblue"), title="Satisfacción"),
            tooltip=["persona", "etapa", alt.Tooltip("satisfaccion:Q", format=".2f")],
        ).properties(height=260)
        st.altair_chart(chart_h, use_container_width=True)

        # Línea por persona a través de las etapas
        linea = f_enc.groupby(["persona", "etapa"], as_index=False)["satisfaccion"].mean()
        chart_l = alt.Chart(linea).mark_line(point=True).encode(
            x=alt.X("etapa:N", sort=STAGES, title=None),
            y=alt.Y("satisfaccion:Q", scale=alt.Scale(domain=[1, 5]), title="Satisfacción (1–5)"),
            color=alt.Color("persona:N", legend=alt.Legend(title="Persona")),
            tooltip=["persona", "etapa", alt.Tooltip("satisfaccion:Q", format=".2f")],
        ).properties(height=260)
        st.altair_chart(chart_l, use_container_width=True)

        st.markdown("**Tabla de observaciones (muestra):**")
        muestra = f_enc[["fecha", "proyecto", "persona", "etapa", "micro_journey", "satisfaccion"]].sort_values("fecha", ascending=False).head(50)
        st.dataframe(muestra, use_container_width=True, hide_index=True)
    else:
        st.info("No hay datos disponibles con los filtros actuales.")

# =============== Tablero 4: Seguimiento al trabajo de campo ===============
with tab4:
    st.subheader("Seguimiento operativo / campo")

    if len(f_enc):
        # Incidentes críticos por proyecto (top)
        inc_proj = (f_enc.groupby("proyecto", as_index=False)["incidente_critico"].mean()
                    .sort_values("incidente_critico", ascending=False).head(20))
        chart_inc = alt.Chart(inc_proj).mark_bar().encode(
            x=alt.X("incidente_critico:Q", axis=alt.Axis(format="%"), title="% con incidente crítico"),
            y=alt.Y("proyecto:N", sort=alt.SortField(field="proyecto", order="ascending"), title=None),
            tooltip=["proyecto", alt.Tooltip("incidente_critico:Q", format=".0%")],
        ).properties(height=280)
        st.altair_chart(chart_inc, use_container_width=True)

        # Retraso promedio por unidad
        delay_u = f_enc.groupby("unidad", as_index=False)["delay_dias"].mean().sort_values("delay_dias", ascending=False)
        chart_delay = alt.Chart(delay_u).mark_bar().encode(
            x=alt.X("delay_dias:Q", title="Retraso promedio (días)"),
            y=alt.Y("unidad:N", sort="-x", title=None),
            tooltip=["unidad", alt.Tooltip("delay_dias:Q", format=".1f")],
            color=alt.Color("unidad:N", legend=None),
        ).properties(height=220)
        st.altair_chart(chart_delay, use_container_width=True)

        # Columnas separadas: Planificadas vs Realizadas
        rng_loc = np.random.default_rng(seed + 1)
        plan = rng_loc.integers(4, 12, size=len(proyectos_sel))
        done = (plan * np.clip(np.random.normal(0.78, 0.12, size=len(plan)), 0.3, 1.1)).round().astype(int)
        df_prog = pd.DataFrame({"proyecto": proyectos_sel, "planificadas": plan, "realizadas": done})
        order_proj = sorted(df_prog["proyecto"].tolist())
        df_plan = df_prog[["proyecto", "planificadas"]].rename(columns={"planificadas": "visitas"})
        df_done = df_prog[["proyecto", "realizadas"]].rename(columns={"realizadas": "visitas"})

        col_plan, col_done = st.columns(2)
        with col_plan:
            st.markdown("**Visitas planificadas**")
            chart_plan = alt.Chart(df_plan).mark_bar().encode(
                x=alt.X("visitas:Q", title="Visitas planificadas"),
                y=alt.Y("proyecto:N", sort=order_proj, title=None),
                color=alt.value("#1f77b4"),
                tooltip=["proyecto", alt.Tooltip("visitas:Q", title="Planificadas")],
            ).properties(height=320)
            st.altair_chart(chart_plan, use_container_width=True)
        with col_done:
            st.markdown("**Visitas realizadas**")
            chart_done = alt.Chart(df_done).mark_bar().encode(
                x=alt.X("visitas:Q", title="Visitas realizadas"),
                y=alt.Y("proyecto:N", sort=order_proj, title=None),
                color=alt.value("#aec7e8"),
                tooltip=["proyecto", alt.Tooltip("visitas:Q", title="Realizadas")],
            ).properties(height=320)
            st.altair_chart(chart_done, use_container_width=True)
    else:
        st.info("No hay datos disponibles con los filtros actuales.")

# =============== Tablero 5: Cultura & Comunicaciones ===============
with tab5:
    st.subheader("Comunicaciones y gestión del cambio")

    if len(f_com):
        # Tomar último mes por unidad para snapshot
        ult_mes = f_com["mes"].max()
        snap = f_com[f_com["mes"] == ult_mes]
        c1, c2, c3, c4 = st.columns(4)
        for i, metr in enumerate(["coms_ejecutadas", "alcance", "recordacion", "interacciones"]):
            val = snap[metr].mean() if len(snap) else 0
            col = [c1, c2, c3, c4][i]
            nombre = {
                "coms_ejecutadas": "Comunicaciones ejecutadas",
                "alcance": "Alcance",
                "recordacion": "Recordación del mensaje",
                "interacciones": "Interacciones recibidas",
            }[metr]
            col.metric(nombre, f"{val*100:,.0f}%")

        st.markdown("**Cultura CX (último mes)**")
        c5, c6, c7, c8 = st.columns(4)
        for i, metr in enumerate(["capacitacion", "embajadores", "adopcion", "percepcion_interna"]):
            val = snap[metr].mean() if len(snap) else 0
            col = [c5, c6, c7, c8][i]
            nombre = {
                "capacitacion": "Colaboradores capacitados",
                "embajadores": "Áreas con embajadores CX",
                "adopcion": "Índice de adopción",
                "percepcion_interna": "Percepción interna",
            }[metr]
            col.metric(nombre, f"{val*100:,.0f}%")

        st.markdown("---")
        st.markdown("**Retroalimentación interna al programa (Likert 1–5)**")
        if len(f_retro):
            chart_retro = alt.Chart(f_retro).mark_bar().encode(
                x=alt.X("promedio:Q", title="Promedio"),
                y=alt.Y("aspecto:N", sort="-x", title=None),
                color=alt.Color("unidad:N", legend=alt.Legend(title="Unidad")),
                tooltip=["unidad", "aspecto", alt.Tooltip("promedio:Q", format=".2f"), "n"],
            ).properties(height=280)
            st.altair_chart(chart_retro, use_container_width=True)
        else:
            st.info("Sin datos de retroalimentación con los filtros actuales.")
    else:
        st.info("No hay datos disponibles con los filtros actuales.")

# =============== Explorador de datos ===============
with tab7:
    st.subheader("Explorador de datos")
    st.caption("Filtra, ordena y descarga los registros base para auditoría o análisis adicional.")
    st.dataframe(
        f_enc.sort_values("fecha", ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )

    st.markdown("**Descargas**")
    colD1, colD2, colD3 = st.columns(3)
    colD1.download_button(
        "⬇️ Encuestas (CSV)",
        data=f_enc.to_csv(index=False).encode("utf-8"),
        file_name="encuestas_demo.csv",
        mime="text/csv",
    )
    colD2.download_button(
        "⬇️ Proyectos (CSV)",
        data=f_proy.to_csv(index=False).encode("utf-8"),
        file_name="proyectos_demo.csv",
        mime="text/csv",
    )
    # Construir tabla CJM agregada para descargar
    if len(f_enc):
        cjm = f_enc.groupby(["persona", "etapa"], as_index=False)["satisfaccion"].mean()
    else:
        cjm = pd.DataFrame(columns=["persona", "etapa", "satisfaccion"])
    colD3.download_button(
        "⬇️ CJM agregado (CSV)",
        data=cjm.to_csv(index=False).encode("utf-8"),
        file_name="cjm_agregado_demo.csv",
        mime="text/csv",
    )

# =============== Footer ===============
st.markdown("---")
st.caption("Demo ficticia de CX: incluye KPIs ejecutivos, micro-journeys, CJM, seguimiento de campo, comunicaciones/cultura, **NPS** y explorador de datos, con filtros globales y logo de marca.")
