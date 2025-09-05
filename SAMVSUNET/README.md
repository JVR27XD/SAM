Montaje de Drive, Split 80/20, Entrenamiento U‑Net y Evaluación con SAM

Este README documenta **todo el flujo en Google Colab** para:
1) Montar Google Drive, 2) **organizar** un dataset **.jpg + .json** al estilo *train/val* (80/20), 3) **entrenar** una U‑Net (ResNet34), 4) **evaluar** U‑Net y **SAM** (caja/punto prompts y modo automático), y 5) generar visualizaciones.  


---

## 0) Requisitos rápidos (Colab)

- Aconsejado: sesión con **GPU** en Colab.
- El dataset etiquetado está en una carpeta con **pares**: `imagen.jpg` + `imagen.json` (formato **LabelMe** con `shapes` y polígonos/rectángulos).
- SAM  se usará la variante **ViT-B** (`sam_vit_b_01ec64.pth`).

---

## 1) Montar Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
# → Mounted at /content/drive
```

---

## 2) Organización del dataset (split 80/20)

**Objetivo:** Buscar pares `*.jpg` + `*.json` en `Labeled/`, barajar, dividir en 80%/20% y **copiar** a la estructura:
```
SolDef_AI_Organized_80_20/
  ├─ train/
  │   ├─ img/
  │   └─ masks/
  └─ val/
      ├─ img/
      └─ masks/
```

**Script empleado (resumen):**
```python
import os, glob, random, shutil
from tqdm import tqdm

source_labeled_dir = "/content/drive/MyDrive/Colab Notebooks/SolDef_AI/Labeled/"
target_base_dir   = "/content/drive/MyDrive/Colab Notebooks/SolDef_AI_Organized_80_20"
json_mask_extension = ".json"
img_extension       = ".jpg"
train_split_ratio   = 0.8
random_seed = 42

all_json_files = glob.glob(os.path.join(source_labeled_dir, f"*{json_mask_extension}"))
labeled_basenames = []
for json_path in all_json_files:
    base = os.path.basename(json_path).replace(json_mask_extension, "")
    img_path = os.path.join(source_labeled_dir, base + img_extension)
    if os.path.exists(img_path): labeled_basenames.append(base)

random.seed(random_seed); random.shuffle(labeled_basenames)
split_idx = int(train_split_ratio * len(labeled_basenames))
train_basenames = labeled_basenames[:split_idx]
val_basenames   = labeled_basenames[split_idx:]

# Crear carpetas
for p in ["train/img","train/masks","val/img","val/masks"]:
    os.makedirs(os.path.join(target_base_dir, p), exist_ok=True)

# Copiar
for base in tqdm(train_basenames, desc="Train Copy"):
    shutil.copy2(os.path.join(source_labeled_dir, base + img_extension),
                 os.path.join(target_base_dir, "train/img", base + img_extension))
    shutil.copy2(os.path.join(source_labeled_dir, base + json_mask_extension),
                 os.path.join(target_base_dir, "train/masks", base + json_mask_extension))

for base in tqdm(val_basenames, desc="Valid Copy"):
    shutil.copy2(os.path.join(source_labeled_dir, base + img_extension),
                 os.path.join(target_base_dir, "val/img", base + img_extension))
    shutil.copy2(os.path.join(source_labeled_dir, base + json_mask_extension),
                 os.path.join(target_base_dir, "val/masks", base + json_mask_extension))

print("✅ ¡Dataset organizado!")
```

**Salida real:**
- Pares hallados: **428** → **342** train / **86** val.  
- Tiempo de copiado ≈ 4 min (aprox. en Colab).

> **Tip:** si las imágenes no son `.jpg`, ajusta `img_extension`. Para máscaras, el código espera **`.json` (LabelMe)**.

---

## 3) Instalación de dependencias

```bash
!pip install git+https://github.com/facebookresearch/segment-anything.git
!pip install Pillow
!pip install segmentation-models-pytorch supervisely albumentations pandas tqdm matplotlib torchinfo
```


## 4) Configuración / Hiperparámetros

```python
import os, torch

DRIVE_MYDRIVE_PATH = "/content/drive/MyDrive/"
COLAB_NOTEBOOKS_FOLDER_PATH = os.path.join(DRIVE_MYDRIVE_PATH, "Colab Notebooks")
MODEL_SAVE_PATH = os.path.join(COLAB_NOTEBOOKS_FOLDER_PATH, "modelos_guardados/")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Datasets
DATASET_PATH_TASK2 = os.path.join(COLAB_NOTEBOOKS_FOLDER_PATH, "SolDef_AI_Organized_80_20/")
TASK = "Task2"
base_path = DATASET_PATH_TASK2

# Clases (Task2)
CLASS_MAPPING = {
    "good": 1,
    "exc_solder": 2,
    "poor_solder": 3,
    "spike": 4
}
num_classes = len(CLASS_MAPPING) + 1  # 0=fondo

# Modelo/entrenamiento
BACKBONE = "resnet34"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Early Stopping / Scheduler
EARLY_STOPPING_PATIENCE = 15
METRIC_TO_MONITOR = "valid_iou"
BEST_MODEL_FILENAME = f"unet_{BACKBONE}_{TASK.lower()}_best_epoch_v1.pth"
full_model_save_path = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_FILENAME)
```

**Rutas train/val:**
```
{base_path}/train/img
{base_path}/train/masks
{base_path}/val/img
{base_path}/val/masks
```

---

## 5) Dataset personalizado (LabelMe .json → máscara multiclase)

- Lee imágenes (`.png/.jpg/...`) y máscaras **LabelMe** (`.json`).
- Convierte cada `shape` (`polygon`/`rectangle`) a máscara 2D con ID de clase según `CLASS_MAPPING`.
- Devuelve tensores listos para **CrossEntropy** (`mask` como `long`, 0=fondo).

> Asegúrate de que en cada `json` las entradas tienen `label`, `points` y `shape_type` en `["polygon","rectangle"]`.

---

## 6) Transforms y DataLoaders

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

train_aug = A.Compose([A.HorizontalFlip(p=0.5),
                       A.VerticalFlip(p=0.5),
                       A.RandomRotate90(p=0.5)])
preproc = A.Compose([A.Resize(384, 384),
                     A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                     ToTensorV2()])

train_dataset = PcbDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, CLASS_MAPPING, augmentation=train_aug, preprocessing=preproc)
valid_dataset = PcbDataset(VALID_IMG_DIR, VALID_MASK_DIR, CLASS_MAPPING, augmentation=None, preprocessing=preproc)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=12, pin_memory=(DEVICE=='cuda'))
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=12, pin_memory=(DEVICE=='cuda'))
```

---

## 7) Modelo, pérdida y optimización

```python
import segmentation_models_pytorch as smp
import torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = smp.Unet(encoder_name=BACKBONE, encoder_weights="imagenet", in_channels=3, classes=num_classes).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)
```

---

## 8) Entrenamiento con Early Stopping + métricas manuales

- Se calcula **IoU** y **F1/Dice** macro de forma **manual** (ignorando `0=fondo`).  
- Early Stopping con `patience=15` en `valid_iou`.  
- Se guarda el **mejor checkpoint** en `modelos_guardados/`.

**Resumen:**
- Mejor **valid_iou**: **0.5125** (época **66**).  
- Mejor **valid_f1 (Dice)**: **0.6677**.  
- Tiempo total entrenamiento ≈ **49.9 min**.

---

## 9) Evaluación final U‑Net (manual)

Se recorre `eval_loader` (val) y se acumulan *intersections*, *pred_sums*, *gt_sums*, *unions* por clase.

**Resultados:**

- **IoU (macro)**: **0.5125**
- **F1/Dice (macro)**: **0.6677**

| Class ID | Class Name  | IoU   | F1/Dice |
|---------:|-------------|:-----:|:-------:|
| 1 | good        | 0.5934 | 0.7448 |
| 2 | exc_solder  | 0.6534 | 0.7904 |
| 3 | poor_solder | 0.3163 | 0.4806 |
| 4 | spike       | 0.4871 | 0.6551 |

**Tiempos U‑Net:** ~**20.1 ms/img** (inferencia pura) y **4.76 s** total de evaluación (86 imágenes).

---

## 10) Configuración de SAM (ViT‑B) y *predictor*

- Comprueba/descarga `sam_vit_b_01ec64.pth` (si no existe).
- Carga en **GPU** si está disponible (cae a CPU si falla).

```python
from segment_anything import sam_model_registry, SamPredictor
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"

sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE).eval()
predictor = SamPredictor(sam_model)
```

---

## 11) SAM (zero‑shot) con **bounding boxes** desde la GT

- Función `get_bboxes_from_mask(mask_gt, class_mapping)` detecta componentes conectados por clase y crea bboxes `[xmin, ymin, xmax, ymax]`.
- Para cada caja: `predictor.predict(box=bbox, multimask_output=False)`.
- Se arma una máscara **multiclase** con las predicciones de SAM y se computan **IoU/F1** manuales.

**Resultados**

- **IoU (macro)**: **0.8111**
- **F1/Dice (macro)**: **0.8922**

| Class ID | Class Name  | IoU   | F1/Dice |
|---------:|-------------|:-----:|:-------:|
| 1 | good        | 0.9217 | 0.9593 |
| 2 | exc_solder  | 0.8580 | 0.9236 |
| 3 | poor_solder | 0.6506 | 0.7883 |
| 4 | spike       | 0.8142 | 0.8976 |

**Tiempos SAM (bbox):** ~**357 ms/img** (set_image + predict(s)).

> **Nota:** este “zero‑shot” usa **cajas de la GT** como *prompts*; por lo que tiene un rendimiento superior con prompts perfectos.

---

## 12) SAM Automático (*SamAutomaticMaskGenerator*)

- Sin *prompts*. Devuelve una lista de máscaras con score/area/bbox… por imagen.
- En la sesión (86 imágenes): **~6m 54s** total (≈ **4.8 s/img**).

```python
from segment_anything import SamAutomaticMaskGenerator
mask_generator = SamAutomaticMaskGenerator(model=sam_model)
masks = mask_generator.generate(img_np_rgb)  # img_np_rgb: RGB, uint8, HxWx3
```

---

## 13) Prompts por **puntos** (centroides)

`get_centroids_from_mask(mask_gt, class_mapping)` encuentra **centroides** por componente (por clase) y genera *point prompts* (`point_coords`) para SAM:

```python
masks_sam, scores, logits = predictor.predict(
    point_coords=np.array([[cx, cy]]),
    point_labels=np.array([1]),
    multimask_output=False
)
```

Se puede construir una máscara multiclase igual que con bboxes.

---

## 14) Visualizaciones comparativas (6 paneles)

Para algunas imágenes (*val*), muestra:
1) **Original**, 2) **SAM automático**, 3) **GT**, 4) **U‑Net**, 5) **SAM con bbox** (GT), 6) **SAM con puntos** (GT).

Pautas:
- Reconstruye la imagen a **RGB uint8** desde el tensor normalizado (invierte *normalize*).
- Para SAM **automático**, usa un colormap estable (e.g., `tab20`) y compón alfa.
- Asegura `predictor.set_image(img_bgr)` antes de cada lote de *prompts*.

> Si quieres, puedo dejarte una celda lista que guarde rejillas `.png` a Drive (pide “**dump visualizaciones**”).

---

## 15) Estructura final resultante

```
Colab Notebooks/
├─ SolDef_AI_Organized_80_20/
│  ├─ train/
│  │  ├─ img/    (342 .jpg)
│  │  └─ masks/  (342 .json)
│  └─ val/
│     ├─ img/    (86 .jpg)
│     └─ masks/  (86 .json)
└─ modelos_guardados/
   └─ unet_resnet34_task2_best_epoch_v1.pth
```

---

## 16) Conjuntos de clases

| ID | Nombre       |
|---:|--------------|
| 0  | background   |
| 1  | good         |
| 2  | exc_solder   |
| 3  | poor_solder  |
| 4  | spike        |




## 17) Referencias

- **SAM**: https://github.com/facebookresearch/segment-anything  
- Ronneberger, O., Fischer, P., & Brox, T. — **U-Net: Convolutional Networks for Biomedical Image Segmentation** (arXiv:1505.04597). https://arxiv.org/abs/1505.04597  
- milesial — **Pytorch-UNet** (GitHub). https://github.com/milesial/Pytorch-UNet


---

### Checklist de uso rápido

1. Monta Drive.  
2. Ejecuta **split 80/20** y revisa conteos.  
3. Instala dependencias.  
4. Configura `CLASS_MAPPING`, paths y *transforms*.  
5. Entrena (espera a **best checkpoint**).  
6. Evalúa U‑Net.  
7. Prueba SAM (bbox/point) y SAM automático.  
8. (Opcional) Genera visualizaciones.
