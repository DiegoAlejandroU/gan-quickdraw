# GAN con Quick Draw — Generador vs. Discriminador (Streamlit)

Aplicación web interactiva que entrena una **Red Generativa Adversaria (GAN)** tipo DCGAN para dibujar objetos de una **categoría del dataset Quick Draw de Google** y explica, paso a paso:

- **Dónde se crea** la Red Generadora y la Red Discriminadora (código comentado visible en la app).
- Los **parámetros y valores** de cada red: capas, formas, cantidad de pesos, estadísticas, valores número por número y comparación antes/después de entrenar.
- Las **funciones de pérdida**: fórmulas, curvas teóricas, comprobación numérica y su evolución durante el entrenamiento.
- El **resultado de la GAN**, indicando la **categoría usada**, la opinión del discriminador y la interpolación en el espacio latente.

**Curso:** Inteligencia Artificial Generativa — Grupo ISS9B · Universidad Santo Tomás, Seccional Tunja
**Autor:** Diego Alejandro Urbano Pira (2386899)

---

## Estructura

| Archivo | Contenido |
|---------|-----------|
| `app.py` | Interfaz Streamlit (pestañas: Teoría, Datos, Generador, Discriminador, Funciones de pérdida, Entrenamiento, Resultado) |
| `gan_modelo.py` | Núcleo: descarga de Quick Draw, **clases `Generador` y `Discriminador`**, **funciones de pérdida**, ciclo de entrenamiento y utilidades para ver los valores |
| `requirements.txt` | Dependencias (PyTorch en versión solo CPU) |
| `.streamlit/config.toml` | Tema y límite de carga de archivos |

Las capas tienen los mismos nombres que en el notebook `GAN_QuickDraw.ipynb`, así que un generador entrenado en Colab (`generador_<categoria>.pt`) se puede cargar en la pestaña **Resultado**.

## Ejecutar en local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`. En la barra lateral se elige la categoría, el número de imágenes y las épocas, y se pulsa **Entrenar GAN**.

## Desplegar en Streamlit Community Cloud (gratis)

1. Crear un repositorio en GitHub (por ejemplo `gan-quickdraw`) y subir **todos** los archivos de esta carpeta, incluida la carpeta `.streamlit/`.
2. Entrar a <https://share.streamlit.io> e iniciar sesión con la cuenta de GitHub.
3. Crear una nueva app desde el repositorio: rama `main`, archivo principal `app.py`.
4. En la configuración avanzada, elegir **Python 3.12** (recomendado).
5. Desplegar. La primera vez tarda unos minutos porque instala PyTorch. Al terminar queda una URL pública del tipo `https://<nombre>.streamlit.app`.

## Tiempos de entrenamiento orientativos

Streamlit Community Cloud no tiene GPU. Con la configuración por defecto (8.000 imágenes, 10 épocas, lote 128) el entrenamiento en CPU tarda unos pocos minutos y ya produce dibujos reconocibles. Para resultados de mejor calidad conviene entrenar en Colab con GPU (notebook) y cargar aquí el `.pt` del generador.

## Referencias

- Curso de GAN — Google for Developers: <https://developers.google.com/machine-learning/gan?hl=es-419>
- GAN Lab: <https://poloclub.github.io/ganlab/>
- Quick Draw Dataset: <https://quickdraw.withgoogle.com/data> (licencia CC BY 4.0)
- Radford, Metz y Chintala (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*.
