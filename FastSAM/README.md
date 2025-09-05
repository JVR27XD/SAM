# FastSAM – Evaluación por *prompting* (Colab)

---

## 🧭 Resumen

Este bloque hace tres cosas:

1. Instala la versión correcta del programa base (Ultralytics 8.0.120) y desactiva funciones extra que no se necesitan (como ClearML o W&B), para que no den errores.
2. Carga de manera segura los archivos del modelo (.pt), que son los pesos preentrenado del modelo.
3. Prepara las imágenes y las etiquetas: ajusta todas al mismo tamaño (1024×1024), aplica una normalización estándar y genera las instrucciones (prompts) que se usan para guiar la segmentación en los cuatro modos (punto, caja, combinación y multipunto).

---

## ✅ Características clave

* **4 modos de *prompt***: `Punto`, `Caja`, `Combo` (punto+caja) y `Multi` (varios puntos).
* **Preprocesado consistente**: *letterbox* a 1024×1024 y normalización ImageNet.
* **Gestión de pesos**: descarga de `FastSAM-s.pt` y `FastSAM-x.pt` y construcción de modelo con cargador seguro.
* **Métricas**: IoU, Dice, Precisión, Recall, tiempo medio por imagen y VRAM media usada.
* **Visualización**: paneles con imagen, GT, predicción binaria y *overlay*.

---

## 🔧 Requisitos

* **Google Colab** (CPU o, preferiblemente, GPU).
* **PyTorch 2.6+** (entorno Colab reciente).
* **Google Drive** montado.
* **Dataset con anotaciones tipo LabelMe** (imágenes + JSON).

---

## 📁 Estructura de datos esperada

Coloca tu dataset en Drive, por ejemplo:

```
/content/drive/MyDrive/Colab Notebooks/SolDef_AI/Labeled/
  ├─ *.jpg / *.png         # imágenes
  ├─ *.json                # anotaciones LabelMe
  └─ generated_masks/      # (se genera automáticamente)
```

El script recorrerá los **JSON** para crear máscaras binarias en `generated_masks/`.

---

## 🚀 Instalación rápida (en Colab)

```bash
# 1) Dependencias
pip -q uninstall -y ultralytics
pip -q install "ultralytics==8.0.120" gdown shapely pycocotools opencv-python-headless
pip -q install --no-deps git+https://github.com/CASIA-IVA-Lab/FastSAM.git
```

---

## 🎛️ Montar Drive y descargar pesos

```python
from google.colab import drive
drive.mount('/content/drive')

# Los pesos se descargan automáticamente (vía gdown) a /content/weights:
# - FastSAM-x.pt (id: 1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv)
# - FastSAM-s.pt (id: 10XmSj6mmpmRb8NhXbtiuO9cTTBwR_9SV)
```

---

## ▶️ Uso básico

Ejemplo mínimo para evaluar una variante en un modo de *prompt*:

```python
summary, per_image = evaluate_fastsam(
    ckpt_path=fastsam_x_path,   # o fastsam_s_path
    mode="BOX",                # POINT | BOX | COMBO | MULTI
    n_multi=5,
    device="cuda",             # "cpu" si no hay GPU
    imgsz=1024,
    conf=0.4,
    iou=0.9,
    max_images=None             # o un entero para limitar
)
print(summary)  # {IoU, Dice, Prec, Rec, Images, Time, VRAM(MB)}
```

Para evaluar **todas** las combinaciones (s/x × 4 modos), el bloque incluye un bucle que imprime un **resumen final**.

---

## 👀 Visualización

Genera paneles con imagen, GT, predicción y *overlay* para una variante y un modo:

```python
visualize_masks_separado(
    model_path=fastsam_s_path,  # o fastsam_x_path
    mode="POINT",              # POINT | BOX | COMBO | MULTI
    n_samples=6,
    start_idx=0,
    device="cuda",
    imgsz=1024,
    conf=0.4,
    iou=0.9,
    save_dir=None             
)
```

El notebook también incluye una figura comparativa **FastSAM-s vs FastSAM-x** por filas de *prompt*.



## 📊 Resultados de ejemplo (mostrado en el notebook)

Valores medios por combinación **modelo × modo**.  

| Modelo   | Modo  | IoU   | Dice  | Precisión | Recall | Tiempo (s/imagen) | VRAM (MB) |
|----------|-------|-------|-------|-----------|--------|-------------------|-----------|
| FastSAM-x | Punto | 0.5915 | 0.6792 | 0.5976    | 0.9884 | 0.435             | 762.1     |
| FastSAM-x | Caja  | 0.7178 | 0.8055 | 0.7260    | 0.9853 | 0.374             | 1059.1    |
| FastSAM-x | Combo | 0.7178 | 0.8055 | 0.7260    | 0.9853 | 0.380             | 759.4     |
| FastSAM-x | Multi | 0.5725 | 0.6623 | 0.5768    | 0.9931 | 0.421             | 1163.1    |
| FastSAM-s | Punto | 0.4791 | 0.5872 | 0.5396    | 0.8311 | 0.135             | 1341.6    |
| FastSAM-s | Caja  | 0.5778 | 0.6935 | 0.6836    | 0.8174 | 0.104             | 1493.1    |
| FastSAM-s | Combo | 0.5778 | 0.6935 | 0.6836    | 0.8174 | 0.105             | 1646.0    |
| FastSAM-s | Multi | 0.5071 | 0.6217 | 0.5357    | 0.9336 | 0.139             | 1797.3    |


---

## 📜 Referencias
- [CASIA-IVA-Lab – FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)  
- [Ultralytics – Ultralytics](https://github.com/ultralytics/ultralytics)


---

