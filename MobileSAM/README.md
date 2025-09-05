El bloque de notebook prepara y ejecuta la evaluación de **MobileSAM** en cuatro modalidades de *prompting*: **Punto / Caja / Caja+punto / Multipuntos** .

---

## 🧭 Resumen

Este bloque hace tres cosas:

1. **Clona e instala MobileSAM** y carga el peso `mobile_sam.pt`.
2. **Genera máscaras GT desde anotaciones LabelMe (`.json`)** y valida su binariedad/consistencia.
3. **Evalúa MobileSAM** con preprocesado (letterbox **centrado** a 1024 y **normalización SAM** 0–255 con medias/desvios de SAM), proyectando *prompts* desde la GT para los cuatro modos y calculando métricas.

---

## ✅ Características clave

* **4 modos de *prompt***: `punto`, `caja`, `caja+punto` y `multipunto` (varios puntos).
* **Preprocesado consistente**: *letterbox centrado* a 1024×1024 y **normalización SAM** (en 0–255).
* **Métricas**: IoU, Dice, Precisión, Recall, tiempo medio por imagen y VRAM media usada.
* **Visualización**: paneles con imagen+prompt, GT y predicción binaria por modo.

---

## 🔧 Requisitos

* **Google Colab** (CPU o, preferiblemente, GPU).
* **PyTorch reciente** (Colab estándar).
* **Google Drive** montado (para tu dataset).
* **Repo MobileSAM clonado e instalado** (editable): `git clone ... && pip install -e /content/MobileSAM`.

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
# 1) Clonar e instalar MobileSAM
git clone https://github.com/ChaoningZhang/MobileSAM.git
pip -q install -e /content/MobileSAM
```

> Asegúrate de tener el checkpoint en:
>
> `/content/MobileSAM/weights/mobile_sam.pt`

---

## 🎛️ Montar Drive y preparar datos

```python
from google.colab import drive
drive.mount('/content/drive')

from adapter_mobilesam import generate_masks_from_labelme
labeled_dir = "/content/drive/MyDrive/Colab Notebooks/SolDef_AI/Labeled"
image_paths, mask_paths = generate_masks_from_labelme(labeled_dir)
```

---

## ▶️ Uso básico

Carga el modelo y evalúa MobileSAM en los 4 modos:

```python
from adapter_mobilesam import load_mobilesam, evaluate_mobilesam, visualize_samples

mobile_sam, device = load_mobilesam(weights_path="/content/MobileSAM/weights/mobile_sam.pt")

summary = evaluate_mobilesam(
    image_paths=image_paths,
    mask_paths=mask_paths,
    model=mobile_sam,
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

Genera paneles con **imagen+prompt**, **GT** y **predicción** por modo en unas muestras:

```python
visualize_samples(
    image_paths=image_paths,
    mask_paths=mask_paths,
    model=mobile_sam,
    device=device,
    n_samples=3,
    imgsz=1024,
    thr=0.5
)
```

---

## 🧪 Resultados (ejemplo del notebook)

Valores medios por modalidad **(MobileSAM vit_t)** sobre 428 imágenes (dataset SolDef_AI). Pueden variar según hardware, datos y umbrales (`thr`).

| Modo       | IoU    | Dice   | Precisión | Recall | Tiempo (s/imagen) | VRAM (MB) |
|----------- | ------ | ------ | --------- | ------ | ----------------- | --------- |
| Punto      | 0.2385 | 0.3060 | 0.5562    | 0.4487 | 0.0678            | 107.35    |
| Caja       | 0.7259 | 0.8124 | 0.7364    | 0.9807 | 0.0091            | 107.35    |
| Caja+Punto | 0.7252 | 0.8119 | 0.7361    | 0.9817 | 0.0103            | 107.36    |
| Multipunto | 0.6825 | 0.7816 | 0.7308    | 0.9399 | 0.0104            | 107.38    |

> **Nota**: Los tiempos/VRAM son promedios de todo el bucle.

---

## 🩹 Consejos y problemas comunes

* **Pesos**: confirma que existe `weights/mobile_sam.pt` dentro del repo clonado.
* **LabelMe**: si hay `shape_type="rectangle"`, se convierte a polígono (4 vértices). Se avisa si una máscara queda vacía/llena.
* **Prompts**: `multipunto` usa un muestreo de 5 puntos por defecto (semilla fija, ajustable).
* **Normalización**: MobileSAM trabaja con **[0,255] → normalización SAM** (`MEAN=[123.675,116.28,103.53]`, `STD=[58.395,57.12,57.375]`).

---


## 📜 Referencias
- [ChaoningZhang — MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

