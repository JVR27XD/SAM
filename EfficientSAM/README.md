# EfficientSAM – Evaluación por *prompting* (Colab)

Bloque de notebook que prepara y ejecuta la evaluación de **EfficientSAM** en cuatro modalidades de *prompting*: **Punto / Caja / Caja + Punto / Multipunto**.

---

## 🧭 Resumen

Este bloque hace tres cosas:

1. **Clona el repositorio EfficientSAM y carga pesos (`efficient_sam_vits.pt` o `efficient_sam_vitt.pt`)** listos para inferencia en Colab.
2. **Genera máscaras GT desde anotaciones LabelMe (`.json`)** y valida su binariedad/consistencia.
3. **Evalúa EfficientSAM** con preprocesado (*letterbox* a 1024 y normalización ImageNet), proyectando *prompts* desde la GT para los cuatro modos y calculando métricas.

---

## ✅ Características clave

* **4 modos de *prompt***: `Punto`, `Caja`, `punto+caja` y `MULTI` (varios puntos).
* **Preprocesado consistente**: *letterbox* a 1024×1024 y normalización ImageNet.
* **Gestión de pesos**: uso de `efficient_sam_vit.pt`.
* **Métricas**: IoU, Dice, Precisión, Recall, tiempo medio por imagen y VRAM media usada.
* **Visualización**: paneles con imagen, GT, predicción binaria.

---

## 🔧 Requisitos

* **Google Colab** (CPU o, preferiblemente, GPU).
* **PyTorch 2.6+** (entorno Colab reciente).
* **Google Drive** montado (para tu dataset).

---

## 📁 Estructura de datos esperada

Coloca dataset en Drive, por ejemplo:

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
# 1) Clonar repo
git clone https://github.com/yformer/EfficientSAM.git
cd EfficientSAM

# 2) (Opcional) Descomprimir pesos si los tienes zipeados en weights/
# unzip weights/efficient_sam_vits.pt.zip -d weights/
```

---

## 🎛️ Montar Drive y preparar datos

```python
from google.colab import drive
drive.mount('/content/drive')

# Generar máscaras desde LabelMe y obtener listas de rutas
from adapter_efficientsam import generate_masks_from_labelme
labeled_dir = "/content/drive/MyDrive/Colab Notebooks/SolDef_AI/Labeled"
image_paths, mask_paths = generate_masks_from_labelme(labeled_dir)
```

---

## ▶️ Uso básico

Carga el modelo y evalúa EfficientSAM en los 4 modos:

```python
from adapter_efficientsam import load_efficientsam, evaluate_efficientsam

model, device = load_efficientsam(model_size="small", repo_dir="/content/EfficientSAM")

summary = evaluate_efficientsam(
    image_paths=image_paths,
    mask_paths=mask_paths,
    model=model,
    device=device,
    imgsz=1024,
    thr=0.5,
    max_images=None  # o un entero para limitar
)

# Resumen por modo y promedios globales
for k, v in summary.items():
    print(k, "=>", v)
```

---

## 👀 Visualización

Genera paneles con imagen, GT y predicción por modo en unas muestras:

```python
from adapter_efficientsam import visualize_samples

visualize_samples(
    image_paths=image_paths,
    mask_paths=mask_paths,
    model=model,
    device=device,
    n_samples=3,
    imgsz=1024,
    thr=0.5
)
```

---

## 🧪 Resultados (ejemplo notebook)

Valores medios por modalidad **(EfficientSAM)** sobre 428 imágenes (dataset SolDef\_AI). Pueden variar según hardware, datos y umbrales (`thr`).

| Modo   | IoU    | Dice   | Precisión | Recall | Tiempo (s/imagen) | VRAM (MB) |
|------- | ------ | ------ | --------- | ------ | ----------------- | --------- |
| POINT  | 0.1357 | 0.1959 | 0.5398    | 0.2056 | 1.339             | 1092.01   |
| BOX    | 0.7276 | 0.8143 | 0.7398    | 0.9807 | 1.339             | 1092.01   |
| COMBO  | 0.7283 | 0.8143 | 0.7389    | 0.9825 | 1.339             | 1092.01   |
| MULTI  | 0.6994 | 0.7970 | 0.7586    | 0.9308 | 1.339             | 1092.01   |

> **Nota**: El tiempo/VRAM mostrados son promedios globales del bucle; por simplicidad se repiten en la tabla.


## 📜 Referencias
- [yformer — EfficientSAM](https://github.com/yformer/EfficientSAM)
