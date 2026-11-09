"""
gan_modelo.py — Núcleo de la GAN: datos, redes, funciones de pérdida y entrenamiento.

Proyecto: GAN con Quick Draw — Red Generadora vs. Red Discriminadora
Curso: Inteligencia Artificial Generativa (ISS9B) — Universidad Santo Tomás, Tunja

Este archivo NO depende de Streamlit: la interfaz (app.py) solo lo importa y lo usa.
Los nombres de las capas son los mismos del notebook GAN_QuickDraw.ipynb, por lo que
un generador entrenado en Colab se puede cargar en la app.
"""
import os
import time
import urllib.parse
import urllib.request
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONSTANTES
# ============================================================
TAM_IMAGEN = 28        # Quick Draw entrega imágenes de 28 x 28
CANALES = 1            # escala de grises
URL_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

# Categorías sugeridas (nombre en inglés del dataset -> nombre en español)
CATEGORIAS = {
    "apple": "manzana", "cat": "gato", "dog": "perro", "star": "estrella", "fish": "pez",
    "house": "casa", "tree": "árbol", "smiley face": "carita feliz", "face": "cara",
    "car": "carro", "bicycle": "bicicleta", "flower": "flor", "sun": "sol", "moon": "luna",
    "butterfly": "mariposa", "airplane": "avión", "donut": "dona", "eye": "ojo",
    "umbrella": "sombrilla", "clock": "reloj", "cup": "taza", "banana": "banano",
}


def nombre_es(categoria):
    return CATEGORIAS.get(categoria, categoria)


# ============================================================
# 1. DATOS: QUICK DRAW
# ============================================================
def descargar_categoria(categoria, carpeta="datos"):
    """Descarga <categoria>.npy de Quick Draw (si no existe) y devuelve la ruta local."""
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"{categoria}.npy")
    if not os.path.exists(ruta):
        url = URL_BASE + urllib.parse.quote(categoria) + ".npy"
        temporal = ruta + ".part"
        urllib.request.urlretrieve(url, temporal)   # se descarga a un temporal para no dejar archivos a medias
        os.replace(temporal, ruta)
    return ruta


def cargar_imagenes(ruta, n, semilla=42):
    """
    Lee n dibujos al azar del .npy (cada uno: 784 valores 0..255) y los deja listos para la red:
    forma (n, 1, 28, 28) y rango [-1, 1] (el mismo de la Tanh del Generador).
    Devuelve (imagenes, total_disponible).
    """
    dibujos = np.load(ruta, mmap_mode="r")                       # no carga todo el archivo a memoria
    total = dibujos.shape[0]
    rng = np.random.default_rng(semilla)
    indices = np.sort(rng.choice(total, size=min(n, total), replace=False))
    x = np.asarray(dibujos[indices], dtype=np.float32)
    x = x.reshape(-1, CANALES, TAM_IMAGEN, TAM_IMAGEN)            # (n, 1, 28, 28)
    x = x / 127.5 - 1.0                                           # [0, 255] -> [-1, 1]
    return x, total


# ============================================================
# 2. >>> AQUÍ SE CREA LA RED NEURONAL GENERADORA (G) <<<
# ============================================================
class Generador(nn.Module):
    """
    Red Generadora: recibe ruido z (dim_latente valores) y devuelve una imagen 1 x 28 x 28.
    Su meta es producir dibujos que el Discriminador confunda con dibujos reales.
    """

    def __init__(self, dim_latente=100):
        super().__init__()
        self.dim_latente = dim_latente

        # ---- Bloque 1: PROYECCIÓN del ruido a un mapa de características pequeño ----
        self.proyeccion = nn.Sequential(OrderedDict([
            # Capa densa: dim_latente entradas -> 128*7*7 = 6272 neuronas
            ("fc",    nn.Linear(dim_latente, 128 * 7 * 7)),
            # Normalización por lotes: estabiliza el entrenamiento
            ("bn0",   nn.BatchNorm1d(128 * 7 * 7)),
            # Activación ReLU: deja pasar solo valores positivos
            ("relu0", nn.ReLU()),
        ]))

        # ---- Bloque 2: DECONVOLUCIONES que agrandan la imagen 7x7 -> 14x14 -> 28x28 ----
        self.deconvolucion = nn.Sequential(OrderedDict([
            # 128 mapas de 7x7 -> 64 mapas de 14x14 (kernel 4x4, paso 2 duplica el tamaño)
            ("deconv1", nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)),
            ("bn1",     nn.BatchNorm2d(64)),
            ("relu1",   nn.ReLU()),
            # 64 mapas de 14x14 -> 1 imagen de 28x28
            ("deconv2", nn.ConvTranspose2d(64, CANALES, kernel_size=4, stride=2, padding=1)),
            # Tanh deja los píxeles en [-1, 1], igual que las imágenes reales normalizadas
            ("tanh",    nn.Tanh()),
        ]))

    def forward(self, z):
        h = self.proyeccion(z)              # (lote, dim_latente) -> (lote, 6272)
        h = h.view(-1, 128, 7, 7)           # (lote, 6272) -> (lote, 128, 7, 7)
        return self.deconvolucion(h)        # (lote, 128, 7, 7) -> (lote, 1, 28, 28)


# ============================================================
# 3. >>> AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA (D) <<<
# ============================================================
class Discriminador(nn.Module):
    """
    Red Discriminadora: recibe una imagen 1 x 28 x 28 y devuelve un logit.
    sigmoid(logit) = probabilidad de que la imagen sea REAL (1) y no FALSA (0).
    """

    def __init__(self):
        super().__init__()

        # ---- Bloque 1: EXTRACCIÓN DE CARACTERÍSTICAS con convoluciones 28x28 -> 14x14 -> 7x7 ----
        self.caracteristicas = nn.Sequential(OrderedDict([
            # 1 canal -> 64 filtros de 4x4; paso 2 reduce la imagen a la mitad (28 -> 14)
            ("conv1",  nn.Conv2d(CANALES, 64, kernel_size=4, stride=2, padding=1)),
            # LeakyReLU: como ReLU pero deja pasar un 20 % de los negativos (evita neuronas "muertas")
            ("lrelu1", nn.LeakyReLU(0.2)),
            # Dropout: apaga 30 % de neuronas al azar para que D no se vuelva demasiado fuerte
            ("drop1",  nn.Dropout(0.3)),
            # 64 -> 128 filtros (14 -> 7)
            ("conv2",  nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            ("bn2",    nn.BatchNorm2d(128)),
            ("lrelu2", nn.LeakyReLU(0.2)),
            ("drop2",  nn.Dropout(0.3)),
        ]))

        # ---- Bloque 2: CLASIFICADOR (real o falso) ----
        self.clasificador = nn.Sequential(OrderedDict([
            ("aplanar", nn.Flatten()),                 # (128, 7, 7) -> 6272
            ("fc",      nn.Linear(128 * 7 * 7, 1)),    # 6272 -> 1 logit
        ]))

    def forward(self, imagen):
        h = self.caracteristicas(imagen)    # (lote, 1, 28, 28) -> (lote, 128, 7, 7)
        return self.clasificador(h)         # (lote, 128, 7, 7) -> (lote, 1)


def inicializar_pesos(modulo):
    """Inicialización DCGAN: pesos ~ N(0, 0.02); BatchNorm ~ N(1, 0.02); sesgos = 0."""
    if isinstance(modulo, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
        if modulo.bias is not None:
            nn.init.zeros_(modulo.bias)
    elif isinstance(modulo, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(modulo.weight, mean=1.0, std=0.02)
        nn.init.zeros_(modulo.bias)


def crear_redes(dim_latente=100, semilla=42, dispositivo="cpu"):
    """Crea e inicializa las dos redes de la GAN."""
    torch.manual_seed(semilla)
    G = Generador(dim_latente).to(dispositivo)     # <<< INSTANCIA DE LA RED GENERADORA
    D = Discriminador().to(dispositivo)            # <<< INSTANCIA DE LA RED DISCRIMINADORA
    G.apply(inicializar_pesos)
    D.apply(inicializar_pesos)
    return G, D


# ============================================================
# 4. >>> AQUÍ SE DEFINEN LAS FUNCIONES DE PÉRDIDA <<<
# ============================================================
# BCEWithLogitsLoss = sigmoide + entropía cruzada binaria en un solo paso (numéricamente estable)
criterio = nn.BCEWithLogitsLoss()


def perdida_discriminador(logits_reales, logits_falsos):
    """L_D = BCE(D(x), 1) + BCE(D(G(z)), 0) = -log D(x) - log(1 - D(G(z)))"""
    etiquetas_reales = torch.ones_like(logits_reales)    # y = 1 -> "es real"
    etiquetas_falsas = torch.zeros_like(logits_falsos)   # y = 0 -> "es falsa"
    perdida_real = criterio(logits_reales, etiquetas_reales)    # -log D(x)
    perdida_falsa = criterio(logits_falsos, etiquetas_falsas)   # -log(1 - D(G(z)))
    return perdida_real + perdida_falsa, perdida_real, perdida_falsa


def perdida_generador(logits_falsos):
    """L_G = BCE(D(G(z)), 1) = -log D(G(z))   (versión non-saturating)"""
    # Se usa la etiqueta 1 ("real") para las imágenes falsas:
    # así G recibe gradiente para que D las clasifique como reales.
    return criterio(logits_falsos, torch.ones_like(logits_falsos))


# ============================================================
# 5. ENTRENAMIENTO
# ============================================================
def norma_parametros(modelo):
    """Norma L2 de todos los pesos de la red."""
    return torch.sqrt(sum((p.detach() ** 2).sum() for p in modelo.parameters())).item()


def entrenar_gan(G, D, datos, *, epocas, tam_lote, lr_g, lr_d, beta1, dispositivo,
                 ruido_fijo, al_terminar_epoca=None):
    """
    Entrena la GAN alternando un paso de D y un paso de G por lote.
    Llama a al_terminar_epoca(epoca, fila, imagenes, historial, resumen) al final de cada época.
    Devuelve (historial por iteración, resumen por época, imágenes de ruido_fijo por época).
    """
    cargador = DataLoader(TensorDataset(torch.as_tensor(datos)), batch_size=tam_lote,
                          shuffle=True, drop_last=True)
    # Un optimizador por red: cada uno actualiza SOLO los parámetros de su red
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, 0.999))

    historial = {"perdida_D": [], "perdida_G": [], "perdida_D_real": [],
                 "perdida_D_falsa": [], "D_x": [], "D_G_z": []}
    resumen, imagenes_por_epoca = [], {}
    dim_latente = G.dim_latente

    for epoca in range(1, epocas + 1):
        G.train()
        D.train()
        t0 = time.time()

        for (reales,) in cargador:
            reales = reales.to(dispositivo)
            b = reales.size(0)

            # ---------------- PASO 1: ENTRENAR EL DISCRIMINADOR ----------------
            z = torch.randn(b, dim_latente, device=dispositivo)
            falsas = G(z)
            logits_reales = D(reales)
            logits_falsos = D(falsas.detach())          # detach: G NO se actualiza en este paso
            loss_D, loss_D_real, loss_D_falsa = perdida_discriminador(logits_reales, logits_falsos)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()                                # actualiza SOLO los pesos de D

            # ---------------- PASO 2: ENTRENAR EL GENERADOR ----------------
            loss_G = perdida_generador(D(falsas))       # el gradiente fluye a través de D hasta G
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()                                # actualiza SOLO los pesos de G

            # ---------------- REGISTRO DE VALORES ----------------
            historial["perdida_D"].append(loss_D.item())
            historial["perdida_G"].append(loss_G.item())
            historial["perdida_D_real"].append(loss_D_real.item())
            historial["perdida_D_falsa"].append(loss_D_falsa.item())
            historial["D_x"].append(torch.sigmoid(logits_reales).mean().item())
            historial["D_G_z"].append(torch.sigmoid(logits_falsos).mean().item())

        n_it = len(cargador)
        fila = {"Época": epoca,
                "Pérdida D": float(np.mean(historial["perdida_D"][-n_it:])),
                "Pérdida G": float(np.mean(historial["perdida_G"][-n_it:])),
                "D(x) reales": float(np.mean(historial["D_x"][-n_it:])),
                "D(G(z)) falsas": float(np.mean(historial["D_G_z"][-n_it:])),
                "Norma pesos G": norma_parametros(G),
                "Norma pesos D": norma_parametros(D),
                "Tiempo (s)": time.time() - t0}
        resumen.append(fila)

        G.eval()
        with torch.no_grad():
            imagenes_por_epoca[epoca] = G(ruido_fijo).cpu()
        G.train()

        if al_terminar_epoca is not None:
            al_terminar_epoca(epoca, fila, imagenes_por_epoca[epoca], historial, resumen)

    G.eval()
    D.eval()
    return historial, resumen, imagenes_por_epoca


# ============================================================
# 6. HERRAMIENTAS PARA VER LOS VALORES DE LAS REDES
# ============================================================
def total_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def tabla_parametros(modelo):
    """Una fila por tensor de parámetros: forma, cantidad y estadísticas de sus valores."""
    filas = []
    for nombre, p in modelo.named_parameters():
        v = p.detach().float().cpu().flatten()
        filas.append({
            "Parámetro": nombre,
            "Forma (shape)": str(tuple(p.shape)),
            "N.º de valores": v.numel(),
            "Media": v.mean().item(),
            "Desv. estándar": v.std().item() if v.numel() > 1 else 0.0,
            "Mínimo": v.min().item(),
            "Máximo": v.max().item(),
            "Primeros 4 valores": ", ".join(f"{x:.4f}" for x in v[:4].tolist()),
        })
    return pd.DataFrame(filas)


def comparar_parametros(pesos_iniciales, modelo):
    """Compara los valores de cada tensor antes y después de entrenar."""
    filas = []
    for nombre, p in modelo.named_parameters():
        antes = pesos_iniciales[nombre].float().flatten()
        despues = p.detach().cpu().float().flatten()
        filas.append({
            "Parámetro": nombre,
            "Media antes": antes.mean().item(),
            "Media después": despues.mean().item(),
            "Desv. antes": antes.std().item() if antes.numel() > 1 else 0.0,
            "Desv. después": despues.std().item() if despues.numel() > 1 else 0.0,
            "Cambio medio |Δ|": (despues - antes).abs().mean().item(),
        })
    return pd.DataFrame(filas)


@torch.no_grad()
def tabla_arquitectura(modelo, entrada):
    """
    Pasa 'entrada' por la red y registra, capa por capa: operación, forma de salida,
    número de parámetros y estadísticas de los valores que salen de esa capa.
    """
    filas = [{"Capa": "entrada", "Operación": "ruido z ~ N(0, 1)" if isinstance(modelo, Generador) else "imagen",
              "Forma de salida": str(tuple(entrada.shape[1:])), "Parámetros": 0,
              "Media salida": entrada.mean().item(), "Mín salida": entrada.min().item(),
              "Máx salida": entrada.max().item()}]
    ganchos = []
    for nombre, capa in modelo.named_modules():
        if len(list(capa.children())) == 0:                      # solo capas "hoja"
            def gancho(m, _entrada, salida, nombre=nombre):
                filas.append({"Capa": nombre, "Operación": repr(m),
                              "Forma de salida": str(tuple(salida.shape[1:])),
                              "Parámetros": sum(p.numel() for p in m.parameters()),
                              "Media salida": salida.mean().item(),
                              "Mín salida": salida.min().item(), "Máx salida": salida.max().item()})
            ganchos.append(capa.register_forward_hook(gancho))
    estaba_entrenando = modelo.training
    modelo.eval()
    try:
        salida = modelo(entrada)
    finally:
        for g in ganchos:
            g.remove()
        modelo.train(estaba_entrenando)

    if isinstance(modelo, Generador):                            # el reshape no es una capa: se agrega a mano
        pos = next(i for i, f in enumerate(filas) if f["Capa"] == "proyeccion.relu0") + 1
        filas.insert(pos, {"Capa": "view (reshape)", "Operación": "h.view(-1, 128, 7, 7)",
                           "Forma de salida": "(128, 7, 7)", "Parámetros": 0,
                           "Media salida": filas[pos - 1]["Media salida"],
                           "Mín salida": filas[pos - 1]["Mín salida"],
                           "Máx salida": filas[pos - 1]["Máx salida"]})
    else:
        p = torch.sigmoid(salida)
        filas.append({"Capa": "sigmoid(logit)", "Operación": "probabilidad D(x) (dentro de la pérdida)",
                      "Forma de salida": str(tuple(p.shape[1:])), "Parámetros": 0,
                      "Media salida": p.mean().item(), "Mín salida": p.min().item(),
                      "Máx salida": p.max().item()})
    return pd.DataFrame(filas)


def matriz_pesos(tensor, max_filas=16, max_columnas=16):
    """Devuelve una parte del tensor de pesos como tabla 2D para verla número por número."""
    t = tensor.detach().cpu().float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    elif t.dim() > 2:
        t = t.reshape(t.shape[0], -1)
    t = t[:max_filas, :max_columnas]
    return pd.DataFrame(t.numpy(), columns=[f"c{j}" for j in range(t.shape[1])],
                        index=[f"f{i}" for i in range(t.shape[0])])


def mosaico(imagenes, columnas=8, escala=3, separacion=2):
    """
    Une imágenes (N, 1, H, W) en [-1, 1] en una sola imagen uint8 (trazo negro sobre blanco).
    Sirve para dibujos de 28x28 y también para filtros (por ejemplo 4x4).
    'escala' amplía cada píxel para que se vea nítido en pantalla.
    """
    x = imagenes.detach().cpu().float().numpy() if torch.is_tensor(imagenes) else np.asarray(imagenes, dtype=np.float32)
    if x.ndim == 4:                     # (N, 1, H, W)
        x = x[:, 0]
    elif x.ndim == 2:                   # (N, 784): dibujos aplanados
        x = x.reshape(-1, TAM_IMAGEN, TAM_IMAGEN)
    alto_img, ancho_img = x.shape[-2:]
    x = 1.0 - (np.clip(x, -1, 1) + 1.0) / 2.0
    n = len(x)
    columnas = min(columnas, n)
    filas = int(np.ceil(n / columnas))
    alto = filas * alto_img + (filas + 1) * separacion
    ancho = columnas * ancho_img + (columnas + 1) * separacion
    lienzo = np.full((alto, ancho), 0.82, dtype=np.float32)
    for i, img in enumerate(x):
        f, c = divmod(i, columnas)
        y0 = separacion + f * (alto_img + separacion)
        x0 = separacion + c * (ancho_img + separacion)
        lienzo[y0:y0 + alto_img, x0:x0 + ancho_img] = img
    lienzo = np.kron(lienzo, np.ones((escala, escala), dtype=np.float32))
    return (lienzo * 255).astype(np.uint8)
