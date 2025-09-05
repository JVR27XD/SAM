# Fine‑tuning de Segment Anything (SAM ViT‑B) en Google Colab (PCB Defects)

Notebook para preparar datos con **LabelMe**, entrenar **solo el decodificador de máscaras** de SAM (congelando el resto), validar, comparar *pre‑trained vs fine‑tuned* con varios *prompts* y latencia/VRAM. Probado en Colab con GPU.

---

## 0) Requisitos

- **Google Colab** con GPU (recomendado).
- Estructura en Drive:
  ```text
  /content/drive/MyDrive/Colab Notebooks/SolDef_AI/
  ├─ Dataset/          # imágenes RGB (JPG/PNG…)
  ├─ Labeled/          # .json de LabelMe con polígonos/rectángulos
  └─ checkpoints/      # se crean automáticamente
  ```
- Los JSON de **LabelMe** deben tener `imagePath`, `imageWidth`, `imageHeight` y `shapes` con `points`.

> **Nota:** El *dataset* se reescala a **1024×1024** para entrenamiento/validación.

---

## 1) Montar Drive y rutas

```python
from google.colab import drive
drive.mount('/content/drive')

RUTA_PROYECTO = "/content/drive/MyDrive/Colab Notebooks/SolDef_AI"
LABELED_DIR   = f"{RUTA_PROYECTO}/Labeled"
CHECKPOINTS_DIR = f"{RUTA_PROYECTO}/checkpoints"

import os
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
```

---

## 2) Clonar SAM (Meta) y descargar pesos ViT‑B

```python
import os, urllib.request

# Clonar una sola vez
if not os.path.isdir("/content/segment-anything"):
    !git clone -q https://github.com/facebookresearch/segment-anything.git /content/segment-anything
%cd /content/segment-anything

# Pesos ViT-B
import os
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
checkpoint_path = os.path.join(CHECKPOINTS_DIR, "sam_vit_b.pth")
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if not os.path.exists(checkpoint_path):
    print("⬇️ Descargando SAM ViT-B…")
    urllib.request.urlretrieve(url, checkpoint_path)
else:
    print(f"✅ Checkpoint ya existe en: {checkpoint_path}")
```

---

## 3) Cargar SAM (ViT‑B)

```python
import torch
from segment_anything import sam_model_registry

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = sam_model_registry['vit_b'](checkpoint=checkpoint_path).to(DEVICE)
sam.train()  # lo pondremos en modo train para ajustar SOLO el decoder
print("Parámetros totales:", sum(p.numel() for p in sam.parameters()))
```

---

## 4) Máscaras desde LabelMe y *Dataset* 1024×1024

Incluye soporte para `polygon`, `rectangle`, `polyline/linestrip`. Convierte a **máscara binaria** (0/1).

```python
# --- helper principal ---
def mask_from_json(json_path, original_size, target_size=(1024,1024), return_bool=True):
    import json, numpy as np
    from PIL import Image, ImageDraw
    with open(json_path,"r") as f:
        data = json.load(f)
    ann_w = int(data.get("imageWidth",  original_size[0]))
    ann_h = int(data.get("imageHeight", original_size[1]))
    tgt_w, tgt_h = map(int, target_size)
    sx, sy = tgt_w/float(ann_w), tgt_h/float(ann_h)
    mask = Image.new("L",(tgt_w,tgt_h),0); draw = ImageDraw.Draw(mask)

    def _scale(points):
        out = []
        for x,y in points:
            xi = int(round(max(0, min(tgt_w-1, x*sx))))
            yi = int(round(max(0, min(tgt_h-1, y*sy))))
            out.append((xi,yi))
        return out

    for shape in data.get("shapes", []):
        pts = shape.get("points", [])
        if not pts: continue
        st = str(shape.get("shape_type","polygon")).lower()
        if st=="rectangle" and len(pts)==2:
            (x0,y0),(x1,y1) = pts
            pts = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
        if st in ("polyline","linestrip") and len(pts)>=3:
            pts = pts + [pts[0]]
        pts = _scale(pts)
        if len(pts)>=3: draw.polygon(pts, outline=255, fill=255)

    m = (np.array(mask)>127).astype("uint8") if return_bool else mask
    return m
```

```python
# --- Dataset torch ---
import os, glob, torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, InterpolationMode as IM
import torchvision.transforms.functional as TF
from PIL import Image

def _find_images(img_dir, exts=(".jpg",".jpeg",".png",".JPG",".JPEG",".PNG")):
    paths = []
    for e in exts: paths += glob.glob(os.path.join(img_dir,"**",f"*{e}"), recursive=True)
    return sorted(paths)

class PCBDefectDatasetFixed(Dataset):
    def __init__(self, img_dir, json_dir, size=(1024,1024)):
        self.img_paths = [
            p for p in _find_images(img_dir)
            if os.path.exists(os.path.join(json_dir, os.path.splitext(os.path.basename(p))[0] + ".json"))
        ]
        self.json_dir = json_dir
        self.size = (int(size[0]), int(size[1]))

    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        p = self.img_paths[idx]; base = os.path.splitext(os.path.basename(p))[0]
        jp = os.path.join(self.json_dir, base + ".json")
        image = Image.open(p).convert("RGB"); W,H = image.size
        m_bin = mask_from_json(jp, (W,H), self.size, return_bool=True)
        image_rs = TF.resize(image, self.size, interpolation=IM.BILINEAR)
        image_t = ToTensor()(image_rs)                 # (3,1024,1024) en [0,1]
        mask_t  = torch.from_numpy(m_bin).float()[None]# (1,1024,1024) en {0,1}
        return image_t, mask_t
```

**Comprobación rápida**
```python
IMG_DIR  = f"{RUTA_PROYECTO}/Dataset"
JSON_DIR = f"{RUTA_PROYECTO}/Labeled"
dataset  = PCBDefectDatasetFixed(IMG_DIR, JSON_DIR, size=(1024,1024))
print("Total ejemplos:", len(dataset))
img, m = dataset[0]; print(img.shape, m.shape, m.unique())
```

---

## 5) *Dataloaders* y *split* train/val

```python
from torch.utils.data import DataLoader, random_split
N = len(dataset); n_val = int(round(N*0.2)); n_train = N - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=2, pin_memory=True,  persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=2, pin_memory=True,  persistent_workers=True)
print(f"Total {N} | Train {len(train_ds)} | Val {len(val_ds)}")
```

---

## 6) Entrenar **solo el *mask decoder***

La función:
- Congela todo excepto `sam.mask_decoder`.
- *Prompt* de entrenamiento: **un punto positivo (centroide)** por muestra.
- Guarda: mejor *checkpoint* (`sam_decoder_best_*.pth`), *checkpoints* periódicos y CSV con métricas.
- Traza curvas (PNG).

```python
# train_sam_decoder(sam, train_loader, ..., eval_fn=evaluate_sam_predictor)
# evaluate_sam_predictor usa el pipeline oficial (ResizeLongestSide + padding) vía SamPredictor.
```

**Lanzar entrenamiento (ejemplo):**
```python
df_metrics = train_sam_decoder(
    sam,
    train_loader,
    epochs=20,
    lr=1e-4,
    device=DEVICE,
    save_dir="/content/drive/MyDrive/TFG/checkpoints",
    val_loader=val_loader,
    eval_fn=evaluate_sam_predictor,   # validación con SamPredictor
    best_by="dice",
    save_every=5,
    use_amp=True,
    grad_clip=1.0,
)
```

**Archivos generados:**
- `.../checkpoints/sam_decoder_best_dice.pth`
- `.../checkpoints/sam_decoder_epoch{N}.pth`
- `.../checkpoints/sam_decoder_metrics.csv`
- `.../checkpoints/sam_decoder_training_curves.png`

---

## 7) Evaluar **SAM preentrenado** (sin FT)

```python
results_pre = evaluate_sam_pretrained(
    sam, val_loader, device=DEVICE, imagenet_norm=False, thr=0.5
)
print(results_pre)  # IoU / Dice / Prec / Rec
```

---

## 8) Comparativa por *prompt* (PRE vs FT)

Prompts incluidos:
- `point_centroid` → **punto centroide (1+)**
- `points_posneg_5_5` → **puntos (5+ / 5−)**
- `box_tight` → **caja ajustada**
- `box_loose10` → **caja +10% margen**
- `box_plus_point` → **caja + punto**

```python
# PRE
from segment_anything import sam_model_registry
sam_pre = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to(DEVICE)
df_pre  = eval_sam_with_prompts(sam_pre, val_loader, device=DEVICE, thr=0.5, visualizar=True, max_vis=3)

# FT (cargar solo decoder ajustado)
sam_ft = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to(DEVICE)
sam_ft.mask_decoder.load_state_dict(torch.load("/content/drive/MyDrive/TFG/checkpoints/sam_decoder_best_dice.pth", map_location=DEVICE))
df_ft  = eval_sam_with_prompts(sam_ft,  val_loader, device=DEVICE, thr=0.5, visualizar=True, max_vis=3)

# Merge y gráfica (barras por Dice)
```

---

## 9)  **Latencia/VRAM** con `SamPredictor`

- Mide tiempo de `set_image(...)` + `predict(...)` y pico de VRAM por *prompt*.
- Devuelve tabla con media/STD y comparación PRE vs FT.

```python
df_rt_pre, sum_rt_pre = profile_model_over_loader(sam_pre, val_loader, device=DEVICE,
                                                  prompts_to_test=("box_plus_point","point_centroid"),
                                                  max_imgs=60)
df_rt_ft,  sum_rt_ft  = profile_model_over_loader(sam_ft,  val_loader, device=DEVICE,
                                                  prompts_to_test=("box_plus_point","point_centroid"),
                                                  max_imgs=60)
rt_comp = sum_rt_pre.merge(sum_rt_ft, on="prompt", suffixes=("_PRE","_FT"))
print(rt_comp)
```

---

## 10) Resultados de **referencia** (ejemplo real)

> **Estos números variarán** según tu GPU/datos/semilla. Se muestran para verificar orden de magnitud.

- **Entrenamiento (mejor val Dice)**: `0.9706` (época 20). *Checkpoint*: `sam_decoder_best_dice.pth`.
- **SAM preentrenado (point prompt)** en *val*: IoU **0.9247** | Dice **0.9594** | Prec **0.9532** | Rec **0.9679** (N=86).

**Comparativa por prompt (val):**

| Prompt             | Dice PRE | Dice FT |
|--------------------|---------:|--------:|
| punto centroide    | 0.2843   | **0.9594** |
| caja + punto       | 0.8312   | **0.9246** |
| caja ajustada      | 0.8378   | **0.8597** |
| puntos 5+/5−       | 0.7922   | **0.8512** |
| caja +10% margen   | 0.7991   | **0.8445** |

**Latencia/VRAM (60 imágenes):**

| Prompt         | PRE ms (±) | FT ms (±) | Speed‑up | VRAM (MB) aprox. |
|----------------|------------:|----------:|---------:|------------------:|
| caja + punto   | 462.7 (21.3) | 456.1 (17.9) | 1.01× | ~3828 |
| punto centroide| 482.6 (29.1) | 471.6 (20.3) | 1.02× | ~3828 |

---


### Referencias

- **SAM**: https://github.com/facebookresearch/segment-anything  

