"""
app.py — Interfaz Streamlit de la GAN con Quick Draw.

Proyecto: GAN con Quick Draw — Red Generadora vs. Red Discriminadora
Curso: Inteligencia Artificial Generativa (ISS9B) — Universidad Santo Tomás, Tunja
Autor: Diego Alejandro Urbano Pira (2386899)

Ejecutar en local:  streamlit run app.py
Las redes, las pérdidas y el entrenamiento están en gan_modelo.py.
"""
import inspect
import io
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

import gan_modelo as gm

st.set_page_config(page_title="GAN Quick Draw", page_icon="🎨", layout="wide")
alt.data_transformers.disable_max_rows()

# Paleta (la misma en todas las gráficas)
COLOR_G = "#2a78d6"      # azul       -> Generador / imágenes generadas
COLOR_D = "#eb6834"      # naranja    -> Discriminador
COLOR_REAL = "#1baf7a"   # verde agua -> imágenes reales
COLOR_ANTES = "#9a9993"  # gris       -> antes de entrenar
COLOR_DESP = "#6250d6"   # violeta    -> después de entrenar
COLOR_REF = "#8a8984"    # líneas de referencia

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LN2 = float(np.log(2))


# ============================================================
# UTILIDADES
# ============================================================
@st.cache_data(show_spinner=False, max_entries=4)
def obtener_datos(categoria, n, semilla):
    """Descarga la categoría (una sola vez) y devuelve (imagenes, total_disponible)."""
    ruta = gm.descargar_categoria(categoria)
    return gm.cargar_imagenes(ruta, n, semilla)


@st.cache_resource(show_spinner=False)
def redes_sin_entrenar(dim_latente, semilla):
    """Redes recién creadas, para mostrar su arquitectura antes de entrenar."""
    return gm.crear_redes(dim_latente, semilla, DISPOSITIVO)


def png_bytes(arreglo):
    buffer = io.BytesIO()
    Image.fromarray(arreglo).save(buffer, format="PNG")
    return buffer.getvalue()


def pesos_bytes(modelo):
    buffer = io.BytesIO()
    torch.save(modelo.state_dict(), buffer)
    return buffer.getvalue()


@torch.no_grad()
def generar(G, n, semilla):
    z = torch.randn(n, G.dim_latente, generator=torch.Generator().manual_seed(semilla)).to(DISPOSITIVO)
    G.eval()
    return G(z).cpu()


@torch.no_grad()
def probabilidad_real(D, imagenes):
    D.eval()
    return torch.sigmoid(D(imagenes.to(DISPOSITIVO))).squeeze(1).cpu().numpy()


def formato_tabla(df, columnas_entero=()):
    """Formato numérico legible para st.dataframe."""
    formatos = {c: "{:,.0f}" for c in columnas_entero}
    formatos.update({c: "{:.5f}" for c in df.select_dtypes("float").columns if c not in formatos})
    return df.style.format(formatos)


# ============================================================
# GRÁFICAS (Altair: interactivas, con tooltip al pasar el mouse)
# ============================================================
def _suavizar(valores, ventana):
    """Media móvil que conserva la longitud de la serie."""
    v = np.asarray(valores, dtype=float)
    if ventana <= 1 or len(v) == 0:
        return v
    acumulado = np.convolve(v, np.ones(ventana), mode="full")[: len(v)]
    return acumulado / np.minimum(np.arange(1, len(v) + 1), ventana)


def grafica_lineas(series, titulo, eje_x, eje_y, referencias=(), dominio_y=None, ventana=25, eje_x_tipo="Iteración"):
    """series: {nombre: (valores, color)}. Dibuja cada serie (suavizada) con línea guía al pasar el mouse."""
    partes = []
    for nombre, (valores, _) in series.items():
        suave = _suavizar(valores, ventana)
        paso = max(1, len(suave) // 600)                 # máximo ~600 puntos por serie
        idx = np.arange(0, len(suave), paso)
        x = idx if eje_x_tipo == "Iteración" else idx + 1
        partes.append(pd.DataFrame({"x": x, "Valor": suave[idx], "Serie": nombre}))
    df = pd.concat(partes, ignore_index=True)

    escala_color = alt.Scale(domain=list(series), range=[c for _, c in series.values()])
    escala_y = alt.Scale(domain=dominio_y) if dominio_y else alt.Scale(zero=False)
    base = alt.Chart(df).encode(
        x=alt.X("x:Q", title=eje_x),
        y=alt.Y("Valor:Q", title=eje_y, scale=escala_y),
        color=alt.Color("Serie:N", scale=escala_color, legend=alt.Legend(orient="top", title=None, labelLimit=420)),
    )
    cercano = alt.selection_point(nearest=True, on="pointerover", fields=["x"], empty=False)
    lineas = base.mark_line(strokeWidth=2)
    selector = alt.Chart(df).mark_point().encode(x="x:Q", opacity=alt.value(0)).add_params(cercano)
    puntos = base.mark_point(filled=True, size=50).encode(
        opacity=alt.when(cercano).then(alt.value(1)).otherwise(alt.value(0)),
        tooltip=[alt.Tooltip("Serie:N"), alt.Tooltip("x:Q", title=eje_x_tipo),
                 alt.Tooltip("Valor:Q", format=".4f")],
    )
    guia = alt.Chart(df).mark_rule(color=COLOR_REF).encode(x="x:Q").transform_filter(cercano)
    capas = [lineas, selector, puntos, guia]

    x_max = int(df["x"].max()) if len(df) else 1
    for valor, texto in referencias:
        ref = pd.DataFrame({"y": [valor], "texto": [texto]})
        capas.append(alt.Chart(ref).mark_rule(strokeDash=[5, 4], color=COLOR_REF).encode(y="y:Q"))
        capas.append(alt.Chart(ref).mark_text(align="right", dy=-7, fontSize=11, color=COLOR_REF)
                     .encode(y="y:Q", x=alt.datum(x_max), text="texto:N"))
    return alt.layer(*capas).properties(title=titulo, height=300)


def grafica_perdidas(historial):
    return grafica_lineas(
        {"Pérdida Discriminador L_D": (historial["perdida_D"], COLOR_D),
         "Pérdida Generador L_G": (historial["perdida_G"], COLOR_G)},
        "Funciones de pérdida durante el entrenamiento (media móvil)", "Iteración (lote)", "Pérdida (BCE)",
        referencias=[(2 * LN2, "equilibrio L_D = 2·ln2 ≈ 1,386"), (LN2, "equilibrio L_G = ln2 ≈ 0,693")],
    )


def grafica_salidas_d(historial):
    return grafica_lineas(
        {"D(x): prob. que D da a las REALES": (historial["D_x"], COLOR_REAL),
         "D(G(z)): prob. que D da a las FALSAS": (historial["D_G_z"], COLOR_G)},
        "Salidas del Discriminador (media móvil)", "Iteración (lote)", "Probabilidad de 'real'",
        referencias=[(0.5, "0,5 = D no distingue")], dominio_y=[0, 1],
    )


def grafica_componentes_d(historial):
    return grafica_lineas(
        {"−log D(x)  (error con reales)": (historial["perdida_D_real"], COLOR_REAL),
         "−log(1 − D(G(z)))  (error con falsas)": (historial["perdida_D_falsa"], COLOR_G)},
        "Las dos partes de la pérdida del Discriminador", "Iteración (lote)", "Pérdida (BCE)",
    )


def grafica_teoria():
    """Curvas de las funciones de pérdida en función de la salida de D."""
    p = np.linspace(0.005, 0.995, 300)
    df_d = pd.concat([
        pd.DataFrame({"p": p, "Pérdida": -np.log(p), "Curva": "Imagen REAL: −log D(x)"}),
        pd.DataFrame({"p": p, "Pérdida": -np.log(1 - p), "Curva": "Imagen FALSA: −log(1 − D(G(z)))"}),
    ])
    df_g = pd.concat([
        pd.DataFrame({"p": p, "Pérdida": np.log(1 - p), "Curva": "Minimax original: log(1 − D(G(z)))"}),
        pd.DataFrame({"p": p, "Pérdida": -np.log(p), "Curva": "Non-saturating: −log D(G(z))  ← usada"}),
    ])

    def _grafica(df, colores, titulo, eje_x, dominio_y):
        return alt.Chart(df).mark_line(strokeWidth=2.5, clip=True).encode(
            x=alt.X("p:Q", title=eje_x),
            y=alt.Y("Pérdida:Q", scale=alt.Scale(domain=dominio_y)),
            color=alt.Color("Curva:N", scale=alt.Scale(domain=list(df["Curva"].unique()), range=colores),
                            legend=alt.Legend(orient="top", title=None, columns=1, labelLimit=420)),
            tooltip=[alt.Tooltip("Curva:N"), alt.Tooltip("p:Q", title="salida de D", format=".3f"),
                     alt.Tooltip("Pérdida:Q", format=".3f")],
        ).properties(title=titulo, height=320)

    return (_grafica(df_d, [COLOR_REAL, COLOR_G], "Pérdida del DISCRIMINADOR",
                     "Salida de D (probabilidad de 'real')", [0, 5]),
            _grafica(df_g, [COLOR_ANTES, COLOR_G], "Pérdida del GENERADOR",
                     "D(G(z)): probabilidad que D da a una imagen falsa", [-5, 5]))


def grafica_histograma(grupos, titulo, eje_x, bins=50, rango=None):
    """grupos: {nombre: (valores, color)} superpuestos con transparencia."""
    todos = np.concatenate([np.asarray(v).ravel() for v, _ in grupos.values()])
    if rango is None:
        limite = float(np.percentile(np.abs(todos), 99.5)) or 1.0
        rango = (-limite, limite)
    bordes = np.linspace(rango[0], rango[1], bins + 1)
    partes = []
    for nombre, (valores, _) in grupos.items():
        cuenta, _ = np.histogram(np.asarray(valores).ravel(), bins=bordes)
        partes.append(pd.DataFrame({"inicio": bordes[:-1], "fin": bordes[1:], "Cantidad": cuenta, "Grupo": nombre}))
    df = pd.concat(partes, ignore_index=True)
    return alt.Chart(df).mark_bar(opacity=0.6).encode(
        x=alt.X("inicio:Q", title=eje_x, bin="binned"), x2="fin:Q",
        y=alt.Y("Cantidad:Q", stack=None),
        color=alt.Color("Grupo:N", scale=alt.Scale(domain=list(grupos), range=[c for _, c in grupos.values()]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Grupo:N"), alt.Tooltip("inicio:Q", format=".4f", title="desde"),
                 alt.Tooltip("fin:Q", format=".4f", title="hasta"), alt.Tooltip("Cantidad:Q")],
    ).properties(title=titulo, height=280)


# ============================================================
# BARRA LATERAL: CONFIGURACIÓN
# ============================================================
with st.sidebar:
    st.header("Configuración")
    opciones = list(gm.CATEGORIAS)
    elegida = st.selectbox("Categoría de Quick Draw", opciones, index=opciones.index("apple"),
                           format_func=lambda c: f"{gm.nombre_es(c)}  ({c})")
    otra = st.text_input("…u otra de las 345 categorías (en inglés)", placeholder="ej.: bicycle, guitar, owl")
    categoria = otra.strip().lower() or elegida

    num_imagenes = st.slider("Imágenes de entrenamiento", 2_000, 20_000, 8_000, step=1_000)
    epocas = st.slider("Épocas", 1, 40, 10)
    tam_lote = st.select_slider("Tamaño de lote", [32, 64, 128, 256], value=128)
    dim_latente = st.select_slider("Dimensión del ruido z", [16, 32, 64, 100, 128], value=100)
    with st.expander("Opciones avanzadas"):
        lr_g = st.number_input("Learning rate del Generador", 1e-5, 1e-2, 2e-4, step=1e-4, format="%.4f")
        lr_d = st.number_input("Learning rate del Discriminador", 1e-5, 1e-2, 2e-4, step=1e-4, format="%.4f")
        beta1 = st.number_input("Beta1 de Adam", 0.0, 0.99, 0.5, step=0.05)
        semilla = int(st.number_input("Semilla", 0, 10_000, 42, step=1))

    entrenar = st.button("Entrenar GAN", type="primary", icon=":material/play_arrow:", width="stretch")
    st.caption(f"Dispositivo: **{DISPOSITIVO.type.upper()}**. En CPU, 10 épocas con 8.000 imágenes "
               "tardan unos minutos.")

cfg = dict(categoria=categoria, num_imagenes=num_imagenes, epocas=epocas, tam_lote=tam_lote,
           dim_latente=dim_latente, lr_g=lr_g, lr_d=lr_d, beta1=beta1, semilla=semilla)
st.session_state.setdefault("semilla_generar", 0)


# ============================================================
# ENTRENAMIENTO EN VIVO
# ============================================================
def ejecutar_entrenamiento(cfg):
    cat, cat_es = cfg["categoria"], gm.nombre_es(cfg["categoria"])
    with st.status(f"Entrenando la GAN con la categoría '{cat}' ({cat_es})…", expanded=True) as estado:
        st.write("**1/3** · Descargando y preparando los dibujos de Quick Draw…")
        try:
            datos, total = obtener_datos(cat, cfg["num_imagenes"], cfg["semilla"])
        except Exception as error:
            estado.update(label="No se pudo descargar la categoría", state="error")
            st.error(f"No se encontró la categoría '{cat}' en Quick Draw o no hay conexión. Detalle: {error}")
            return
        st.write(f"{len(datos):,} dibujos listos (de {total:,} disponibles en la categoría).")

        st.write("**2/3** · Creando el Generador y el Discriminador…")
        G, D = gm.crear_redes(cfg["dim_latente"], cfg["semilla"], DISPOSITIVO)
        pesos_G_ini = {n: p.detach().cpu().clone() for n, p in G.named_parameters()}
        pesos_D_ini = {n: p.detach().cpu().clone() for n, p in D.named_parameters()}
        ruido_fijo = torch.randn(64, cfg["dim_latente"],
                                 generator=torch.Generator().manual_seed(cfg["semilla"])).to(DISPOSITIVO)

        st.write("**3/3** · Entrenando: en cada lote se actualiza primero D y después G.")
        barra = st.progress(0.0, text="Iniciando…")
        col1, col2 = st.columns(2)
        ph_perdidas, ph_salidas = col1.empty(), col2.empty()
        col3, col4 = st.columns([3, 2])
        ph_imagenes, ph_tabla = col3.empty(), col4.empty()
        inicio = time.time()

        def al_terminar_epoca(epoca, fila, imagenes, historial, resumen):
            barra.progress(epoca / cfg["epocas"],
                           text=f"Época {epoca}/{cfg['epocas']} · L_D = {fila['Pérdida D']:.3f} · "
                                f"L_G = {fila['Pérdida G']:.3f} · {time.time() - inicio:.0f} s")
            ph_perdidas.altair_chart(grafica_perdidas(historial), width="stretch")
            ph_salidas.altair_chart(grafica_salidas_d(historial), width="stretch")
            ph_imagenes.image(gm.mosaico(imagenes[:32]),
                              caption=f"Época {epoca}: dibujos generados (mismo ruido z en todas las épocas)")
            ph_tabla.dataframe(pd.DataFrame(resumen).set_index("Época")[
                ["Pérdida D", "Pérdida G", "D(x) reales", "D(G(z)) falsas"]].round(4), width="stretch")

        historial, resumen, imagenes_por_epoca = gm.entrenar_gan(
            G, D, torch.from_numpy(datos), epocas=cfg["epocas"], tam_lote=cfg["tam_lote"],
            lr_g=cfg["lr_g"], lr_d=cfg["lr_d"], beta1=cfg["beta1"], dispositivo=DISPOSITIVO,
            ruido_fijo=ruido_fijo, al_terminar_epoca=al_terminar_epoca)
        minutos = (time.time() - inicio) / 60

        st.session_state["resultado"] = dict(
            cfg=dict(cfg), G=G, D=D, pesos_G_ini=pesos_G_ini, pesos_D_ini=pesos_D_ini,
            historial=historial, resumen=pd.DataFrame(resumen).set_index("Época"),
            imagenes_por_epoca=imagenes_por_epoca, reales=torch.from_numpy(datos[:2000].copy()),
            total=total, minutos=minutos)
        estado.update(label=f"GAN entrenada · categoría '{cat}' ({cat_es}) · {cfg['epocas']} épocas · "
                            f"{minutos:.1f} min", state="complete", expanded=False)
    st.toast("Entrenamiento terminado. Revisa la pestaña «Resultado».", icon="✅")


# ============================================================
# ENCABEZADO
# ============================================================
st.title("GAN con Quick Draw: Generador vs. Discriminador")
st.caption("Inteligencia Artificial Generativa — Grupo ISS9B · Universidad Santo Tomás, Seccional Tunja · "
           "Diego Alejandro Urbano Pira (2386899)")

if entrenar:
    ejecutar_entrenamiento(cfg)

res = st.session_state.get("resultado")
if res is not None:
    G_vista, D_vista = res["G"], res["D"]
    cat_usada = res["cfg"]["categoria"]
else:
    G_vista, D_vista = redes_sin_entrenar(cfg["dim_latente"], cfg["semilla"])
    cat_usada = cfg["categoria"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Categoría", f"{gm.nombre_es(cat_usada)}", help=f"Nombre en Quick Draw: '{cat_usada}'", border=True)
m2.metric("Parámetros del Generador", f"{gm.total_parametros(G_vista):,}".replace(",", "."), border=True)
m3.metric("Parámetros del Discriminador", f"{gm.total_parametros(D_vista):,}".replace(",", "."), border=True)
m4.metric("Estado", "Entrenada" if res else "Sin entrenar",
          help=f"{res['cfg']['epocas']} épocas" if res else "Configura y pulsa «Entrenar GAN»", border=True)

if res is None:
    st.info("Elige una categoría en la barra lateral y pulsa **Entrenar GAN**. Mientras tanto puedes revisar "
            "la teoría, la arquitectura y los parámetros iniciales de las dos redes.", icon=":material/info:")

pestanas = st.tabs(["Teoría", "Datos", "Generador (G)", "Discriminador (D)", "Funciones de pérdida",
                    "Entrenamiento", "Resultado"])


# ============================================================
# PESTAÑA 1: TEORÍA
# ============================================================
with pestanas[0]:
    st.subheader("¿Qué es una GAN?")
    st.markdown(
        "Una **Red Generativa Adversaria** (Goodfellow et al., 2014) tiene **dos redes que compiten**: el "
        "**Generador (G)** crea imágenes falsas a partir de ruido y el **Discriminador (D)** intenta distinguirlas "
        "de las reales. Se entrenan por turnos: D aprende a detectar falsificaciones y G usa el error de D para "
        "mejorar. En el equilibrio, D ya no distingue y responde **≈ 0,5**.")
    dot = """
    digraph {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, fontcolor="#0b0b0b", color="#8a8984"];
      edge [color="#8a8984", fontname="Helvetica", fontsize=10, fontcolor="#8a8984"];
      z [label="Ruido z ~ N(0,1)\\n(__DIM__ valores)", fillcolor="#f0efec"];
      G [label="GENERADOR (G)\\n__PG__ parámetros", fillcolor="#d6e6f8", color="#2a78d6", penwidth=2];
      falsa [label="Imagen falsa G(z)\\n28 × 28", fillcolor="#f0efec"];
      real [label="Imagen real x\\n(Quick Draw: __CAT__)", fillcolor="#d4f0e5", color="#1baf7a"];
      D [label="DISCRIMINADOR (D)\\n__PD__ parámetros", fillcolor="#fbe0d5", color="#eb6834", penwidth=2];
      p [label="D(·) = P(real)\\n0 = falsa · 1 = real", shape=ellipse, fillcolor="#f0efec"];
      z -> G -> falsa -> D; real -> D; D -> p;
      p -> G [label="gradiente de L_G", style=dashed, color="#2a78d6", fontcolor="#2a78d6", constraint=false];
    }"""
    dot = (dot.replace("__DIM__", str(G_vista.dim_latente)).replace("__CAT__", cat_usada)
              .replace("__PG__", f"{gm.total_parametros(G_vista):,}".replace(",", "."))
              .replace("__PD__", f"{gm.total_parametros(D_vista):,}".replace(",", ".")))
    st.graphviz_chart(dot, width="stretch")

    st.markdown("""
| Red | Entrada | Salida | Objetivo |
|-----|---------|--------|----------|
| **Generador (G)** | Ruido aleatorio `z` | Imagen falsa 28×28 | Engañar a D |
| **Discriminador (D)** | Imagen real o falsa | Probabilidad de que sea real | Distinguir real de falso |

**Funciones de pérdida más conocidas** (curso de GAN de Google):

| Función | Discriminador | Generador | Comentario |
|---------|---------------|-----------|------------|
| **Minimax** (paper original) | maximiza `log D(x) + log(1 − D(G(z)))` | minimiza `log(1 − D(G(z)))` | G se estanca al inicio |
| **Minimax modificada** (*non-saturating*) | igual | maximiza `log D(G(z))` | **La usada en este proyecto** |
| **Wasserstein** (WGAN) | crítico: maximiza `D(x) − D(G(z))` | maximiza `D(G(z))` | Predeterminada en TF-GAN |

**Referencias:** [Curso de GAN — Google for Developers](https://developers.google.com/machine-learning/gan?hl=es-419) ·
[GAN Lab](https://poloclub.github.io/ganlab/) (visualización interactiva de una GAN en 2D) ·
[Quick Draw Dataset](https://quickdraw.withgoogle.com/data) ·
Radford et al. (2015), *DCGAN*.
""")


# ============================================================
# PESTAÑA 2: DATOS
# ============================================================
with pestanas[1]:
    st.subheader(f"Dataset Quick Draw — categoría: '{cat_usada}' ({gm.nombre_es(cat_usada)})")
    st.markdown(
        "Google publica cada una de las **345 categorías** como un archivo `.npy` (*numpy bitmap*): cada dibujo es "
        "una imagen de **28×28 píxeles** en escala de grises (0 = fondo, 255 = trazo). Antes de entrenar, los "
        "píxeles se pasan de `[0, 255]` a **`[-1, 1]`**, el mismo rango de la `Tanh` del Generador.\n\n"
        f"URL: `{gm.URL_BASE}{cat_usada.replace(' ', '%20')}.npy`")
    if res is not None:
        st.markdown(f"Se usaron **{len(res['reales']) if res['cfg']['num_imagenes'] <= 2000 else res['cfg']['num_imagenes']:,}** "
                    f"dibujos de **{res['total']:,}** disponibles.")
        st.image(gm.mosaico(res["reales"][:48], columnas=12), caption="Dibujos REALES usados en el entrenamiento")
    elif st.button(f"Ver dibujos reales de '{cat_usada}'", icon=":material/download:"):
        try:
            with st.spinner("Descargando de Quick Draw…"):
                datos, total = obtener_datos(cat_usada, cfg["num_imagenes"], cfg["semilla"])
            st.markdown(f"**{total:,}** dibujos disponibles en la categoría.")
            st.image(gm.mosaico(datos[:48], columnas=12), caption="Dibujos REALES de Quick Draw")
        except Exception as error:
            st.error(f"No se pudo descargar la categoría '{cat_usada}': {error}")


# ============================================================
# PESTAÑAS 3 y 4: GENERADOR Y DISCRIMINADOR
# ============================================================
def seccion_red(tipo):
    es_G = tipo == "G"
    modelo = G_vista if es_G else D_vista
    clase = gm.Generador if es_G else gm.Discriminador
    pesos_ini = (res["pesos_G_ini"] if es_G else res["pesos_D_ini"]) if res else None

    if es_G:
        st.markdown(
            f"El **Generador** convierte un vector de **{modelo.dim_latente} números aleatorios** en una imagen de "
            "**28×28**. Sigue la arquitectura **DCGAN**: una capa densa proyecta el ruido a 128 mapas de 7×7 y dos "
            "**convoluciones transpuestas** duplican el tamaño (7 → 14 → 28). La `Tanh` final deja los píxeles en [-1, 1].")
    else:
        st.markdown(
            "El **Discriminador** es un **clasificador binario**: recibe una imagen de 28×28 y entrega un *logit*. "
            "Dos **convoluciones con paso 2** reducen la imagen (28 → 14 → 7) y una capa densa da la respuesta. "
            "La **sigmoide** que convierte el logit en probabilidad `D(x)` se aplica dentro de la función de pérdida "
            "(`BCEWithLogitsLoss`), que es numéricamente más estable.")

    st.markdown(f"#### Dónde se crea la red (`gan_modelo.py`, clase `{clase.__name__}`)")
    st.code(inspect.getsource(clase), language="python")
    st.code(f"# Instancia que se entrena (gan_modelo.crear_redes):\n"
            f"{'G = Generador(dim_latente)' if es_G else 'D = Discriminador()'}.to(dispositivo)\n"
            f"{'G' if es_G else 'D'}.apply(inicializar_pesos)   # pesos ~ N(0; 0,02), como en DCGAN",
            language="python")

    st.markdown("#### Arquitectura capa por capa")
    if es_G:
        entrada = torch.randn(16, modelo.dim_latente, generator=torch.Generator().manual_seed(0)).to(DISPOSITIVO)
        nota = "Se pasaron 16 vectores de ruido por la red; las columnas «salida» muestran los valores que produce cada capa."
    else:
        if res is not None:
            entrada, nota = res["reales"][:16].to(DISPOSITIVO), "Se pasaron 16 dibujos reales por la red."
        else:
            entrada, nota = generar(G_vista, 16, 0).to(DISPOSITIVO), "Se pasaron 16 imágenes del Generador por la red."
    arq = gm.tabla_arquitectura(modelo, entrada)
    st.dataframe(formato_tabla(arq, ["Parámetros"]), hide_index=True, width="stretch")
    st.caption(f"{nota} Total de parámetros entrenables: **{gm.total_parametros(modelo):,}**.")

    st.markdown("#### Valores de los parámetros")
    momento = "Entrenados"
    if res is not None:
        momento = st.segmented_control("Mostrar valores", ["Entrenados", "Iniciales"], default="Entrenados",
                                       key=f"momento_{tipo}") or "Entrenados"
    if momento == "Iniciales" and pesos_ini is not None:
        copia = clase(modelo.dim_latente) if es_G else clase()
        copia.load_state_dict({**copia.state_dict(), **pesos_ini})
        tabla = gm.tabla_parametros(copia)
    else:
        tabla = gm.tabla_parametros(modelo)
        if res is None:
            st.caption("La red aún no está entrenada: estos son los valores iniciales (N(0; 0,02)).")
    st.dataframe(formato_tabla(tabla, ["N.º de valores"]), hide_index=True, width="stretch")

    nombres = [n for n, _ in modelo.named_parameters()]
    c1, c2 = st.columns([2, 3])
    with c1:
        elegido = st.selectbox("Ver un tensor de pesos número por número", nombres, key=f"tensor_{tipo}")
    tensor = dict(modelo.named_parameters())[elegido]
    with c2:
        forma = tuple(tensor.shape)
        st.markdown(f"**Forma:** `{forma}` · **{tensor.numel():,} valores**"
                    + (" · se muestra reorganizado en 2D (filas = filtros/neuronas de salida)" if len(forma) > 2 else "")
                    + " · primeras 16 × 16 posiciones")
    st.dataframe(gm.matriz_pesos(tensor).style.format("{:.4f}"), width="stretch")

    if res is not None:
        st.markdown("#### Antes vs. después de entrenar")
        st.dataframe(formato_tabla(gm.comparar_parametros(pesos_ini, modelo)), hide_index=True, width="stretch")
        st.altair_chart(grafica_histograma(
            {"Antes de entrenar": (pesos_ini[elegido].numpy(), COLOR_ANTES),
             "Después de entrenar": (tensor.detach().cpu().numpy(), COLOR_DESP)},
            f"Distribución de los valores de {elegido}", "Valor del peso"), width="stretch")

        if not es_G:
            filtros = modelo.caracteristicas.conv1.weight.detach().cpu()            # (64, 1, 4, 4)
            filtros = (filtros - filtros.min()) / (filtros.max() - filtros.min()) * 2 - 1
            imagen = gm.mosaico(-filtros, columnas=16, escala=12, separacion=1)
            st.image(imagen, caption="Los 64 filtros 4×4 aprendidos por la primera convolución de D "
                                     "(caracteristicas.conv1): detectores de trazos")


with pestanas[2]:
    st.subheader("Red neuronal Generadora (G)")
    seccion_red("G")

with pestanas[3]:
    st.subheader("Red neuronal Discriminadora (D)")
    seccion_red("D")


# ============================================================
# PESTAÑA 5: FUNCIONES DE PÉRDIDA
# ============================================================
with pestanas[4]:
    st.subheader("Funciones de pérdida")
    st.markdown("Se usa la **entropía cruzada binaria (BCE)**, donde *p* es la salida de D y *y* la etiqueta "
                "(1 = real, 0 = falso):")
    st.latex(r"\text{BCE}(p, y) = -\big[\,y \log p + (1-y)\log(1-p)\,\big]")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Discriminador:** debe decir 1 a las reales y 0 a las generadas.")
        st.latex(r"L_D = \underbrace{-\log D(x)}_{\text{BCE}(D(x),\,1)} \;\underbrace{-\,\log\big(1 - D(G(z))\big)}_{\text{BCE}(D(G(z)),\,0)}")
        st.caption("Minimizar L_D equivale a maximizar log D(x) + log(1 − D(G(z))), el objetivo minimax original.")
    with col2:
        st.markdown("**Generador** (*non-saturating*): quiere que D diga 1 a sus imágenes falsas.")
        st.latex(r"L_G = -\log D(G(z)) = \text{BCE}(D(G(z)),\,1)")
        st.caption("Valores en el equilibrio (D = 0,5): L_D = 2·ln 2 ≈ 1,386 y L_G = ln 2 ≈ 0,693.")

    g_d, g_g = grafica_teoria()
    col1, col2 = st.columns(2)
    col1.altair_chart(g_d, width="stretch")
    col2.altair_chart(g_g, width="stretch")
    st.caption("A la derecha: al inicio D rechaza con facilidad las falsas (D(G(z)) ≈ 0). Ahí la curva original "
               "log(1 − D(G(z))) es casi plana y G casi no recibe gradiente; la versión non-saturating tiene "
               "pendiente fuerte justo en esa zona. Por eso es la que se usa.")

    st.markdown("#### Valores de las pérdidas según la salida de D")
    salidas = np.array([0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01])
    st.dataframe(pd.DataFrame({
        "Salida de D (prob. de 'real')": salidas,
        "D con imagen REAL: −log(p)": -np.log(salidas),
        "D con imagen FALSA: −log(1−p)": -np.log(1 - salidas),
        "G minimax original: log(1−p)": np.log(1 - salidas),
        "G non-saturating: −log(p)": -np.log(salidas),
    }).style.format("{:.4f}"), hide_index=True, width="stretch")

    logit = torch.log(torch.tensor([0.9]) / torch.tensor([0.1]))
    st.markdown(
        f"**Comprobación con PyTorch** (D responde 0,9): `BCEWithLogitsLoss(logit, 1)` = "
        f"**{gm.criterio(logit, torch.ones(1)).item():.4f}** = −log(0,9) = {-np.log(0.9):.4f} · "
        f"`BCEWithLogitsLoss(logit, 0)` = **{gm.criterio(logit, torch.zeros(1)).item():.4f}** = −log(0,1) = {-np.log(0.1):.4f}")

    st.markdown("#### Dónde se definen en el código (`gan_modelo.py`)")
    st.code("criterio = nn.BCEWithLogitsLoss()\n\n" + inspect.getsource(gm.perdida_discriminador)
            + "\n" + inspect.getsource(gm.perdida_generador), language="python")

    if res is not None:
        st.markdown("#### Pérdidas durante el entrenamiento")
        st.altair_chart(grafica_perdidas(res["historial"]), width="stretch")
        st.altair_chart(grafica_componentes_d(res["historial"]), width="stretch")
        st.caption("En una GAN las pérdidas **no deben llegar a cero**: si L_D → 0, D ganó y G deja de aprender. "
                   "Lo sano es que oscilen de forma estable cerca de las referencias del equilibrio.")


# ============================================================
# PESTAÑA 6: ENTRENAMIENTO
# ============================================================
with pestanas[5]:
    st.subheader("Entrenamiento por turnos")
    st.markdown(
        "En cada lote de imágenes:\n"
        "1. **Paso del Discriminador:** ve dibujos reales (etiqueta 1) y generados (etiqueta 0). Los generados se "
        "pasan con `.detach()` para **no** modificar a G en este paso. Solo `opt_D` actualiza pesos.\n"
        "2. **Paso del Generador:** D evalúa los dibujos generados y G se actualiza para que D los crea reales. "
        "Solo `opt_G` actualiza pesos.\n\n"
        "Optimizador: **Adam** (uno por red), β₁ = 0,5 como recomienda DCGAN.")
    with st.expander("Ver el código del ciclo de entrenamiento"):
        st.code(inspect.getsource(gm.entrenar_gan), language="python")

    if res is None:
        st.info("Aún no hay entrenamiento. Pulsa **Entrenar GAN** en la barra lateral.", icon=":material/info:")
    else:
        c = res["cfg"]
        st.markdown(f"**Configuración usada:** categoría `{c['categoria']}` · {c['num_imagenes']:,} imágenes · "
                    f"{c['epocas']} épocas · lote {c['tam_lote']} · z de {c['dim_latente']} · "
                    f"lr G {c['lr_g']:.4f} · lr D {c['lr_d']:.4f} · β₁ {c['beta1']} · "
                    f"tiempo {res['minutos']:.1f} min")
        st.markdown("#### Valores por época")
        st.dataframe(formato_tabla(res["resumen"]), width="stretch")
        st.altair_chart(grafica_salidas_d(res["historial"]), width="stretch")
        st.altair_chart(grafica_lineas(
            {"Generador": (res["resumen"]["Norma pesos G"].to_numpy(), COLOR_G),
             "Discriminador": (res["resumen"]["Norma pesos D"].to_numpy(), COLOR_D)},
            "Norma L2 de todos los parámetros de cada red", "Época", "‖pesos‖₂", ventana=1, eje_x_tipo="Época"),
            width="stretch")

        st.markdown("#### Evolución del Generador (mismo ruido z en cada época)")
        lista_epocas = list(res["imagenes_por_epoca"])
        if len(lista_epocas) > 1:
            ep = st.select_slider("Época", lista_epocas, value=lista_epocas[-1])
        else:
            ep = lista_epocas[0]
        st.image(gm.mosaico(res["imagenes_por_epoca"][ep][:32]), caption=f"Época {ep}")
        hitos = sorted({lista_epocas[0], *lista_epocas[:: max(1, len(lista_epocas) // 5)], lista_epocas[-1]})
        filas = torch.cat([res["imagenes_por_epoca"][e][:10] for e in hitos])
        st.image(gm.mosaico(filas, columnas=10),
                 caption="Cada fila es una época: " + ", ".join(str(e) for e in hitos))


# ============================================================
# PESTAÑA 7: RESULTADO
# ============================================================
with pestanas[6]:
    G_res, D_res, cat_res, origen = None, None, None, ""
    if res is not None:
        G_res, D_res, cat_res, origen = res["G"], res["D"], res["cfg"]["categoria"], "entrenada en esta sesión"
    elif "G_subido" in st.session_state:
        G_res, cat_res = st.session_state["G_subido"]
        origen = "generador cargado desde archivo"

    if G_res is None:
        st.info("Entrena la GAN desde la barra lateral, o carga un generador ya entrenado "
                "(por ejemplo, el `.pt` que guarda el notebook de Colab).", icon=":material/info:")
        archivo = st.file_uploader("Cargar generador entrenado (.pt)", type=["pt"])
        if archivo is not None:
            try:
                estado_pesos = torch.load(io.BytesIO(archivo.getvalue()), map_location="cpu")
                G_cargado = gm.Generador(estado_pesos["proyeccion.fc.weight"].shape[1])
                G_cargado.load_state_dict(estado_pesos)
                G_cargado.to(DISPOSITIVO).eval()
                cat_archivo = archivo.name.removesuffix(".pt").removeprefix("generador_").replace("_", " ")
                st.session_state["G_subido"] = (G_cargado, cat_archivo)
                st.rerun()
            except Exception as error:
                st.error(f"El archivo no corresponde a este Generador: {error}")
    else:
        st.markdown(f"## Categoría usada: **{gm.nombre_es(cat_res)}** (`{cat_res}`)")
        st.caption(f"Resultado de la GAN ({origen}).")

        cabecera = st.columns([4, 1])
        if cabecera[1].button("Generar otros", icon=":material/refresh:", width="stretch"):
            st.session_state["semilla_generar"] += 1
        generadas = generar(G_res, 64, 1000 + st.session_state["semilla_generar"])

        if res is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Dibujos REALES (Quick Draw)**")
                st.image(gm.mosaico(res["reales"][:64]))
            with col2:
                st.markdown(f"**Dibujos GENERADOS por la GAN** (época {res['cfg']['epocas']})")
                st.image(gm.mosaico(generadas))
            u = res["resumen"].iloc[-1]
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Pérdida D (última época)", f"{u['Pérdida D']:.4f}", border=True)
            v2.metric("Pérdida G (última época)", f"{u['Pérdida G']:.4f}", border=True)
            v3.metric("D(x) reales", f"{u['D(x) reales']:.3f}", border=True)
            v4.metric("D(G(z)) generadas", f"{u['D(G(z)) falsas']:.3f}", border=True)
        else:
            st.markdown("**Dibujos GENERADOS**")
            st.image(gm.mosaico(generadas))
            if st.button("Quitar generador cargado"):
                del st.session_state["G_subido"]
                st.rerun()

        if D_res is not None:
            st.markdown("#### ¿Qué opina el Discriminador?")
            p_reales = probabilidad_real(D_res, res["reales"][:1000])
            p_falsas = probabilidad_real(D_res, generar(G_res, 1000, 7))
            for etiqueta, imgs, probs in [("Real", res["reales"][:8], p_reales[:8]),
                                          ("Generada", generadas[:8], probabilidad_real(D_res, generadas[:8]))]:
                columnas = st.columns(8)
                for i, col in enumerate(columnas):
                    col.image(gm.mosaico(imgs[i:i + 1], columnas=1, escala=4), caption=f"{etiqueta} · D = {probs[i]:.2f}")
            st.altair_chart(grafica_histograma(
                {"Imágenes REALES": (p_reales, COLOR_REAL), "Imágenes GENERADAS": (p_falsas, COLOR_G)},
                "Salidas de D en 1.000 reales y 1.000 generadas", "D(·) = probabilidad de 'real'",
                bins=40, rango=(0, 1)), width="stretch")
            st.caption(f"D clasifica como reales el **{np.mean(p_reales > 0.5) * 100:.1f} %** de las reales y "
                       f"detecta como falsas el **{np.mean(p_falsas <= 0.5) * 100:.1f} %** de las generadas.")

        st.markdown("#### Espacio latente: interpolación entre dos vectores de ruido")
        with torch.no_grad():
            gen = torch.Generator().manual_seed(2000 + st.session_state["semilla_generar"])
            filas_interp = []
            for _ in range(4):
                z1, z2 = torch.randn(2, G_res.dim_latente, generator=gen)
                alfas = torch.linspace(0, 1, 10).unsqueeze(1)
                filas_interp.append(G_res((1 - alfas) * z1 + alfas * z2).cpu() if DISPOSITIVO.type == "cpu"
                                    else G_res(((1 - alfas) * z1 + alfas * z2).to(DISPOSITIVO)).cpu())
        st.image(gm.mosaico(torch.cat(filas_interp), columnas=10),
                 caption="z = (1 − α)·z₁ + α·z₂: transiciones suaves indican que G aprendió una representación continua")

        st.markdown("#### Descargas")
        d1, d2, d3, d4 = st.columns(4)
        nombre = cat_res.replace(" ", "_")
        d1.download_button("Imagen del resultado (.png)", png_bytes(gm.mosaico(generadas)),
                           f"resultado_gan_{nombre}.png", "image/png", on_click="ignore", width="stretch")
        d2.download_button("Generador (.pt)", pesos_bytes(G_res), f"generador_{nombre}.pt",
                           on_click="ignore", width="stretch")
        if res is not None:
            d3.download_button("Discriminador (.pt)", pesos_bytes(D_res), f"discriminador_{nombre}.pt",
                               on_click="ignore", width="stretch")
            d4.download_button("Valores por época (.csv)", res["resumen"].to_csv().encode("utf-8"),
                               f"valores_por_epoca_{nombre}.csv", "text/csv", on_click="ignore", width="stretch")
