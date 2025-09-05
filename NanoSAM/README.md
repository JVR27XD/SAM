# NanoSAM – Evaluación por *prompting* con ONNX Runtime (Colab)

Preparar y evaluar **NanoSAM (encoder+decoder ONNX)** en cuatro modalidades de *prompting*: **Punto / Caja / Caja+Punto / Multipunto**, además de una evaluación**por número de puntos K**.

---

## 🧭 Resumen

Este bloque hace cuatro cosas:

1. **Clona `nanosam`** y ajusta el `PYTHONPATH` en Colab.
2. **Prepara dependencias estables** (NumPy 1.26.4, OpenCV 4.8.1, ONNX Runtime 1.17.3) **sin TensorRT**.
3. **Carga sesiones ONNX**: `resnet18_image_encoder.onnx` (encoder) y `mobile_sam_mask_decoder.onnx` (decoder) con `CPUExecutionProvider`.
4. **Evalúa NanoSAM**: preprocesa a 1024×1024 con **normalización ImageNet**, proyecta *prompts* desde la GT, decodifica máscaras, calcula métricas (IoU/Dice/Prec/Rec), tiempos y visualizaciones. Incluye **barrido K** (1,2,3,5,10 puntos).

---

## ✅ Características clave

- **4 modos de *prompt***: `punto`, `caja`, `caja+punto`, `multipunto`.
- **Preprocesado**:  a **1024×1024** + **normalización ImageNet**.
- **Inferencia ONNX**: encoder y decoder, inputs/outputs verificados.
- **Métricas**: IoU, Dice, Precisión, Recall, Tiempo (s/imagen). 
- **Visualización**: imagen+prompt, GT (1024), predicción (1024).

---

## 🔧 Requisitos

- **Google Colab** (CPU o GPU, aunque aquí usamos ORT-CPU).
- **ONNX Runtime 1.17+**, **OpenCV 4.8+**, **NumPy 1.26.x**.
- **Google Drive** montado (dataset + modelos ONNX).
- **Pesos ONNX**:
  - `resnet18_image_encoder.onnx`
  - `mobile_sam_mask_decoder.onnx`

---

## 📁 Estructura de datos esperada

Colocar dataset en Drive, por ejemplo:

```
/content/drive/MyDrive/Colab Notebooks/SolDef_AI/Labeled/
  ├─ *.jpg / *.png         # imágenes
  ├─ *.json                # anotaciones LabelMe
  └─ generated_masks/      # (se genera automáticamente)
```

El script creará máscaras binarias en `generated_masks/` a partir de los **JSON**.

---

## 🚀 Instalación rápida (en Colab)

```bash
# 1) Clonar NanoSAM
git clone https://github.com/NVIDIA-AI-IOT/nanosam.git /content/nanosam

# 2) Dependencias (pila estable sin TensorRT)
pip -q uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip -q install "numpy==1.26.4" "opencv-python==4.8.1.78" "onnxruntime==1.17.3" pillow matplotlib
```

> Si tu entorno trae paquetes conflictivos (numba/cuml/spacy, etc.), desinstálalos antes para evitar choques de dependencias.

---

## 🧩 Colocar los modelos ONNX y crear sesiones

Coloca los `.onnx` en el Drive (ajusta `data_dir` a tu ruta):

```python
import onnxruntime as ort, os

data_dir = "/content/drive/MyDrive/Colab Notebooks/TFG/NanoSam"
ENCODER_ONNX = os.path.join(data_dir, "resnet18_image_encoder.onnx")
DECODER_ONNX = os.path.join(data_dir, "mobile_sam_mask_decoder.onnx")

providers = ["CPUExecutionProvider"]
enc_sess = ort.InferenceSession(ENCODER_ONNX, providers=providers)
dec_sess = ort.InferenceSession(DECODER_ONNX, providers=providers)
```

**Entradas/salidas esperadas** (pueden variar por export):  
- Encoder: `image` → `(1,3,1024,1024)` → `image_embeddings` `(1,256,64,64)`  
- Decoder: `image_embeddings`, `point_coords (1,N,2)`, `point_labels (1,N)`, `mask_input (1,1,256,256)`, `has_mask_input (1)` → `low_res_masks (1,4,256,256)`, `iou_predictions (1,4)`

> Nota: **`point_labels` en float32** para algunos exports (como en este notebook).

---

## 🎛️ Montar Drive y generar máscaras

```python
from google.colab import drive
drive.mount('/content/drive')

# Recorre JSON LabelMe → genera PNGs binarios 0/255 en generated_masks/
# Devuelve: image_paths, mask_paths (listas alineadas)
# (usa las funciones del notebook: scale_polygon/clamp_points + dibujo)
```

---

## ▶️ Uso básico (inferencia + métricas)

```python
# 1) Preprocesado imagen/máscara a 1024 con normalización ImageNet
input_tensor, gt_1024 = preprocess_to_1024(image_pil, mask_pil)  # (1,3,1024,1024), (1024,1024)

# 2) Encoder ONNX → embeddings (1,256,64,64)
embeddings = encode_image_onnx(enc_sess, input_tensor)

# 3) Construir prompts (en 1024×1024) y decodificar
iou_pred, lowres_masks = run_decoder_points(dec_sess, embeddings, pts_1024)

# 4) Seleccionar mejor máscara por IoU predicho y reescalar
pred_mask_1024, pred_mask_orig = pick_and_upscale(lowres_masks, iou_pred, (orig_h, orig_w), return_both=True)

# 5) Métricas (NumPy)
iou = iou_np(gt_1024, pred_mask_1024)
dice = dice_coef_np(gt_1024, pred_mask_1024)
prec = precision_np(gt_1024, pred_mask_1024)
rec  = recall_np(gt_1024, pred_mask_1024)
```

**Modos de prompt** implementados en el notebook:
- `punto`: centroide de GT.
- `caja`: 4 esquinas del bbox + centro (5 puntos positivos).
- `caja+punto`: igual que `punto` (incluye punto en el centro).
- `multipunto`: K puntos muestreados dentro de la GT.

---

## 🔬 Barrido de K puntos

Incluye un experimento para **K = 1,2,3,5,10**, con promedios (IoU/Dice/Prec/Rec/Tiempo) por K.  

Tendencia observada: **mejoras de IoU/Dice al aumentar K**, con tiempos casi constantes en ORT-CPU.

---

## 👀 Visualización

El notebook genera para cada modo: **imagen + prompt**, **GT (1024)** y **predicción (1024)**, y guarda PNGs en `/content/resultados_pred/`.

---

## 🧪 Resultados (Del notebook)

Valores medios por modalidad (**NanoSAM ONNX**, dataset SolDef_AI, 428 imágenes). Pueden variar según hardware/datos.

| Método     | IoU   | Dice  | Precisión | Recall | Tiempo (s/imagen) |
|----------- |------:|------:|----------:|------:|------------------:|
| Punto      | 0.2912 | 0.3671 | 0.6132 | 0.4142 | 0.0965 |
| Caja        | 0.2562 | 0.3804 | 0.2578 | 0.9177 | 0.0976 |
| Caja+punto  | 0.2562 | 0.3804 | 0.2578 | 0.9177 | 0.0980 |
| Multipunto | 0.6689 | 0.7721 | 0.7318 | 0.9186 | 0.0991 |

** K (resumen)**: para `K = 1,2,3,5,10` se observa aumento progresivo hasta ~**IoU ≈ 0.70 / Dice ≈ 0.80** para `K=10`, con **≈0.10 s/img**.

| K\_points |   N | IoU\_mean | IoU\_std | Dice\_mean | Dice\_std | Prec\_mean | Prec\_std | Rec\_mean | Rec\_std | Time\_mean\_s | FPS\_mean |
| --------: | --: | --------: | -------: | ---------: | --------: | ---------: | --------: | --------: | -------: | ------------: | --------: |
|         1 | 428 |    0.5132 |   0.2953 |     0.6293 |    0.2554 |     0.7566 |    0.2790 |    0.7199 |   0.3338 |        0.0946 |   10.5748 |
|         2 | 428 |    0.5886 |   0.2757 |     0.7028 |    0.2226 |     0.7237 |    0.2861 |    0.8412 |   0.2473 |        0.0955 |   10.4748 |
|         3 | 428 |    0.6434 |   0.2651 |     0.7502 |    0.2048 |     0.7453 |    0.2764 |    0.8777 |   0.1982 |        0.0954 |   10.4842 |
|         5 | 428 |    0.6587 |   0.2511 |     0.7658 |    0.1890 |     0.7336 |    0.2761 |    0.9071 |   0.1463 |        0.0965 |   10.3634 |
|        10 | 428 |    0.7043 |   0.2523 |     0.7992 |    0.1851 |     0.7372 |    0.2690 |    0.9540 |   0.0718 |        0.1000 |    9.9966 |
---

## 🩹 Consejos y problemas comunes

- **Rendimiento**: con ORT-CPU el tiempo por imagen ronda ~0.1 s. Con GPUExecutionProvider los tiempos pueden bajar.


---

## 📜 Créditos

- [NVIDIA-AI-IOT — NanoSAM (GitHub)](https://github.com/NVIDIA-AI-IOT/nanosam)







