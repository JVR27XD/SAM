# TinySAM – Evaluación por *prompting* (Colab)

Bloque de notebook que prepara y ejecuta la evaluación de **TinySAM** en cuatro modalidades de *prompting*: **Punto / Caja / Caja+punto / Multipunto**.

---

## 🧭 Resumen

Este bloque hace tres cosas:

1. **Clona TinySAM** y prepara carpetas de trabajo (`weights/`, `outputs/`).
2. **Carga un checkpoint `.pth`/`.pt` de forma segura**, construyendo el modelo con `build_sam_vit_t` y dejándolo en modo *eval* en GPU/CPU.
3. **Evalúa TinySAM con letterboxing** (1024×1024), prompts generados desde la GT (punto, caja, combo, multipunto), **normalización ImageNet**, métricas, tiempos/VRAM y visualizaciones.

---

## ✅ Características clave

* **4 modos de *prompt***: `Punto`, `Caja`, `punto+caja` y `MULTI` (varios puntos dentro de la GT).
* **Preprocesado consistente**: *letterbox centrado* a 1024×1024 y normalización **ImageNet** (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).
* **Métricas**: IoU, Dice, Precisión, Recall + **tiempo** medio por imagen y **VRAM** media usada.
* **Visualización**: paneles con imagen+prompts, GT binaria y predicción (blanco/negro) por modo.

---

## 🔧 Requisitos

* **Google Colab** (CPU o, preferiblemente, GPU).
* **PyTorch** reciente (Colab estándar).
* **Google Drive** montado (para tu dataset y checkpoint).
* **Repo TinySAM clonado** (desde GitHub).

---

## 📁 Estructura de datos esperada

Colocar dataset en Drive, por ejemplo:

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
# 1) Clonar TinySAM y preparar carpetas
git clone https://github.com/xinghaochen/TinySAM.git
cd TinySAM
mkdir -p weights outputs
```

---

## 🎛️ Cargar checkpoint y preparar datos

```python
from google.colab import drive
drive.mount('/content/drive')

# Ruta a tu checkpoint (ajusta según Drive)
CKPT_TINYSAM = "/content/drive/MyDrive/Colab Notebooks/TFG/TinySAM/tinysam_42.3.pth"

# Verificación rápida
import os
if not os.path.exists(CKPT_TINYSAM):
    raise FileNotFoundError(f"No existe el checkpoint: {CKPT_TINYSAM}")
else:
    print("✅ Checkpoint encontrado.")

# Generar máscaras desde LabelMe y obtener listas de rutas
# (usa el bloque del notebook con `scale_and_clamp_polygon`)
labeled_dir = "/content/drive/MyDrive/Colab Notebooks/SolDef_AI/Labeled"
# -> produce image_paths, mask_paths y guarda PNGs en generated_masks/
```

---

## ▶️ Uso básico

Evalúa TinySAM en los 4 modos con letterboxing a 1024:

```python
# Cargar TinySAM
import torch
from tinysam.build_sam import build_sam_vit_t
from tinysam.modeling.sam import Sam

def load_tinysam_model(ckpt_path, device="cuda"):
    with torch.serialization.safe_globals([Sam]):
        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_sam_vit_t(checkpoint=None)
    model.load_state_dict(ckpt_data, strict=False)
    model.to(device).eval()
    return model

model_tinysam = load_tinysam_model(CKPT_TINYSAM, device="cuda")

# Ejecutar evaluación (usa el helper del notebook `eval_tinysam_on_dataset`)
results, summary = eval_tinysam_on_dataset(
    image_paths, mask_paths,
    ckpt=CKPT_TINYSAM,
    prompt_types=("POINT","BOX","COMBO","MULTI"),
    n_multi_points=5,
    threshold=0.5
)
print(summary)
```

---

## 👀 Visualización

Muestras con **imagen+prompts**, **GT** y **predicción** (blanco/negro):

```python
# Usa el helper del notebook `visualize_tinysam_examples`
visualize_tinysam_examples(
    model_tinysam, image_paths, mask_paths,
    n_examples=8, n_multi_points=5, thr=0.5,
    save_dir=None, clean=False, pred_viz="mask"
)
```

---

## 🧪 Resultados (notebook)

Valores medios por modalidad (**TinySAM**, dataset SolDef_AI, 428 imágenes). Pueden variar según hardware, datos y umbrales.

| Modo  | IoU    | Dice   | Precisión | Recall | Tiempo (s/imagen) | VRAM (MB) |
|------ | ------ | ------ | --------- | ------ | ----------------- | --------- |
| Punto | 0.0762 | 0.1187 | 0.5525    | 0.0782 | 0.0941            | 356.34    |
| Caja   | 0.6399 | 0.7571 | 0.7521    | 0.8355 | 0.0743            | 356.34    |
| Caja+punto | 0.6301 | 0.7504 | 0.7512    | 0.8098 | 0.0749            | 356.34    |
| Multipunto | 0.5383 | 0.6509 | 0.8398    | 0.5922 | 0.0747            | 356.34    |

> **Nota**: el preprocesado aplica **letterbox centrado 1024** y normalización **ImageNet**; los *prompts* se proyectan coherentemente al espacio 1024×1024.

---

## 🩹 Consejos y problemas comunes

* **Checkpoint**: comprueba la ruta en Drive y su tamaño para evitar rutas rotas.
* **Normalización**: TinySAM aquí usa **ImageNet**; si cambias a otra normalización, tus métricas pueden variar.
* **Prompts**: `MULTI` toma (por defecto) 5 puntos dentro de la GT; ajusta `n_multi_points` si quieres más densidad.
* **GT y letterbox**: recuerda proyectar la GT con el **mismo** `scale`, `pad_x`, `pad_y` del *letterbox*.
* **Velocidad/VRAM**: se miden con `time()` y `torch.cuda.max_memory_allocated()` para cada imagen.

---

## 📜 Crédititos

- [xinghaochen — TinySAM (GitHub)](https://github.com/xinghaochen/TinySAM)  
- [TinySAM — arXiv:2312.13789](https://arxiv.org/abs/2312.13789)
