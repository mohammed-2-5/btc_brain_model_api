# main.py  ─────────────────────────────────────────────────────────────
"""
FastAPI backend:
  • /predict      → 6 PNG slices (2-D مع الماسك)
  • /plot_preview → صورة PNG فيها 4 لوحات nilearn
  • /tumor_mesh   → ملف GLB ثلاثى الأبعاد ملوَّن
كل المخرجات تُحفظ تحت static/  وتُقدَّم كرابط HTTP ثابت.
"""
import os, re, uuid, tempfile, pathlib, logging
from typing import List

import cv2, numpy as np, nibabel as nib
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

# ───── backend بلا شاشة لـ matplotlib / nilearn ─────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nilearn.plotting as nlplt
import nilearn.image     as nlimg

from skimage import measure                        # marching-cubes
import trimesh                                     # glTF / GLB I/O

logging.basicConfig(level=logging.INFO)

# ───────────────────── (A) مجلدات ثابتة
BASE   = pathlib.Path(__file__).resolve().parent
STATIC = BASE / "static"        ; STATIC.mkdir(exist_ok=True)
SLICE_DIR = STATIC / "slices"   ; SLICE_DIR.mkdir(exist_ok=True)
PLOT_DIR  = STATIC / "plots"    ; PLOT_DIR.mkdir(exist_ok=True)
MESH_DIR  = STATIC / "models"   ; MESH_DIR.mkdir(exist_ok=True)

# ───────────────────── (B) تطبيق FastAPI
app = FastAPI(title="3-D MRI Brain-Segmentation API (slices + preview + mesh)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# ───────────────────── (C) نموذج /predict (اختياري)
IMG_SIZE, VOLUME_SLICES, VOLUME_START_AT = 128, 6, 60
MODEL_PATH = BASE / "models" / "3D_MRI_Brain_tumor_segmentation.h5"
try:
    from tensorflow.keras.models import load_model
    model = load_model(str(MODEL_PATH))
    logging.info("✔︎ Segmentation model loaded")
except Exception as ex:                               # Fallback
    logging.warning("⚠︎ Using Dummy model (%s)", ex)
    class _Dummy:
        def predict(self, x, **kw): return np.zeros((*x.shape[:-1], 4), np.float32)
    model = _Dummy()

# ───────────────────── (D) دوال مساعدة
def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip() or "file"

def _predict_slices(flair: np.ndarray, t1ce: np.ndarray) -> np.ndarray:
    x = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2), np.float32)
    for j in range(VOLUME_SLICES):
        idx = VOLUME_START_AT + j
        x[j, :, :, 0] = cv2.resize(flair[:, :, idx], (IMG_SIZE, IMG_SIZE))
        x[j, :, :, 1] = cv2.resize(t1ce [:, :, idx], (IMG_SIZE, IMG_SIZE))
    x /= np.max(x) or 1.0
    return model.predict(x, verbose=0)                # (6,128,128,4)

def _overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(rgb); red[..., 2] = 255
    rgb[mask > .5] = 0.5*rgb[mask>.5] + 0.5*red[mask>.5]
    return rgb.astype(np.uint8)

def _save_plot(fig) -> str:
    p = PLOT_DIR / f"{uuid.uuid4().hex}.png"
    fig.savefig(p, bbox_inches="tight", dpi=200); plt.close(fig)
    return f"/static/plots/{p.name}"

# ───────────────────── (E) ‎/predict  (كما هو)
@app.post("/predict")
async def predict(nifti_flair: UploadFile = File(...),
                  nifti_t1ce : UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as td:
        p1 = pathlib.Path(td)/"f.nii"; p1.write_bytes(await nifti_flair.read())
        p2 = pathlib.Path(td)/"t.nii"; p2.write_bytes(await nifti_t1ce.read())
        try:
            flair, t1ce = nib.load(p1).get_fdata(), nib.load(p2).get_fdata()
        except Exception as e:
            raise HTTPException(400, f"NIfTI error: {e}")

    preds  = _predict_slices(flair, t1ce)
    run_id = uuid.uuid4().hex
    outdir = SLICE_DIR / run_id ; outdir.mkdir()
    titles = ["all classes", "NECROTIC/CORE predicted",
              "EDEMA predicted", "ENHANCING predicted",
              "slice 4", "slice 5"]
    result: List[dict] = []
    for i in range(VOLUME_SLICES):
        gray = cv2.resize(flair[:, :, VOLUME_START_AT+i], (IMG_SIZE, IMG_SIZE))
        mask = preds[i,:,:,1:4].max(-1)
        ui   = titles[i] if i < len(titles) else f"slice {i}"
        fn   = f"{_safe(ui)}.png"
        cv2.imwrite(str(outdir/fn), _overlay(gray, mask))
        result.append({"title": ui,
                       "filename": fn,
                       "url": f"/static/slices/{run_id}/{fn}"})
    return JSONResponse({"run_id": run_id, "slices": result})

# ───────────────────── (F) ‎/plot_preview  (كما هو)
@app.post("/plot_preview")
async def plot_preview(flair: UploadFile = File(...), seg: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as td:
        p_f = pathlib.Path(td)/"f.nii"; p_f.write_bytes(await flair.read())
        p_s = pathlib.Path(td)/"s.nii"; p_s.write_bytes(await seg.read())
        try:
            niimg  = nlimg.load_img(p_f)
            nimask = nlimg.load_img(p_s)
        except Exception as e:
            raise HTTPException(400, f"NIfTI error: {e}")

    fig, ax = plt.subplots(4, 1, figsize=(20, 28))
    nlplt.plot_anat(niimg ,  axes=ax[0], title="plot_anat")
    nlplt.plot_epi (niimg ,  axes=ax[1], title="plot_epi")
    nlplt.plot_img (niimg ,  axes=ax[2], title="plot_img")
    nlplt.plot_roi (nimask, bg_img=niimg, axes=ax[3],
                    cmap="Paired", title="ROI overlay")

    return JSONResponse({"url": _save_plot(fig)})

# ───────────────────── (G) ‎/tumor_mesh  (ملوَّن) ──────────────────
@app.post("/tumor_mesh")
async def tumor_mesh(
    seg: UploadFile = File(..., description="segmentation mask (.nii /.nii.gz)"),
    iso_level: float = Query(0.5, gt=0.0, lt=1.0)
):
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td)/"mask.nii"; p.write_bytes(await seg.read())
        try:
            vol = nib.load(p).get_fdata()
        except Exception as e:
            raise HTTPException(400, f"NIfTI error: {e}")

    if vol.max() == 0:
        raise HTTPException(422, "Mask is empty – nothing to export")

    # marching-cubes
    verts, faces, _n, _v = measure.marching_cubes(vol, level=iso_level,
                                                  allow_degenerate=False)

    # تحديد لون لكل وجه حسب label (يؤخذ عند إحداثيات مركز الوجه)
    face_centroids = verts[faces].mean(1)
    # ترتيب المحاور: z,y,x  ↦  indices صحيحة داخل المصفوفة
    cz, cy, cx = np.round(face_centroids).astype(int).T
    cz = np.clip(cz, 0, vol.shape[0]-1)
    cy = np.clip(cy, 0, vol.shape[1]-1)
    cx = np.clip(cx, 0, vol.shape[2]-1)
    labels = vol[cz, cy, cx].astype(int)

    label2rgba = {
        1: (255,   0,   0, 255),   # NCR / NET
        2: (  0, 255,   0, 255),   # ED
        3: (255, 255,   0, 255),   # ET
    }
    face_colors = np.array([label2rgba.get(l, (200,200,200,255))
                            for l in labels], dtype=np.uint8)

    mesh = trimesh.Trimesh(
        verts, faces,
        process=False,
        visual=trimesh.visual.ColorVisuals(face_colors=face_colors))

    out = MESH_DIR / f"{uuid.uuid4().hex}.glb"
    mesh.export(out)
    return JSONResponse({"url": f"/static/models/{out.name}"})


@app.post("/brain_mesh")
async def brain_mesh(
    flair : UploadFile = File(..., description="FLAIR volume (.nii/.nii.gz)"),
    seg   : UploadFile = File(..., description="Segmentation mask (.nii/.nii.gz)"),
    iso   : float      = Query(0.12 , ge=0.02 , le=0.9 , description="iso-threshold for brain surface"),
    tumor_iso : float  = Query(0.5  , ge=0.1  , le=1.0 , description="iso for tumour marching-cubes")
):
    """
    يُرجع glTF (GLB) يحتوى:

      1. قشرة كاملة للدماغ (رمادى فاتح)
      2. شبكة الورم ملونة (أحمر شفاف) داخل نفس الملف
    """
    with tempfile.TemporaryDirectory() as td:
        p_f = pathlib.Path(td)/"f.nii"; p_f.write_bytes(await flair.read())
        p_s = pathlib.Path(td)/"s.nii"; p_s.write_bytes(await seg.read())
        try:
            vol_img  = nib.load(p_f).get_fdata()
            mask_img = nib.load(p_s).get_fdata().astype(bool)
        except Exception as e:
            raise HTTPException(400, f"NIfTI error: {e}")

    # ── 1) قشرة الدماغ ───────────────────────────────────────────
    # (اختيار أبسط) threshold نسبى من أقصى قيمة
    brain_thresh = iso * vol_img.max()
    verts_b, faces_b, *_ = measure.marching_cubes(vol_img, level=brain_thresh)

    brain_mesh = trimesh.Trimesh(
        verts_b, faces_b,
        process=False,
        visual=trimesh.visual.ColorVisuals(face_colors=np.tile([200,200,200,255],
                                                               (faces_b.shape[0],1)))
    )

    # ── 2) شبكة الورم ────────────────────────────────────────────
    if mask_img.max() == 0:
        raise HTTPException(422, "Mask is empty – nothing to export")

    verts_t, faces_t, *_ = measure.marching_cubes(mask_img.astype(float),
                                                  level=tumor_iso)

    tumour_mesh = trimesh.Trimesh(
        verts_t, faces_t,
        process=False,
        visual=trimesh.visual.ColorVisuals(face_colors=np.tile([255,0,0,180],
                                                               (faces_t.shape[0],1)))
    )

    # ── 3) دمج المشهد + حفظ GLB ─────────────────────────────────
    scene = trimesh.Scene()
    scene.add_geometry(brain_mesh, node_name="brain")
    scene.add_geometry(tumour_mesh, node_name="tumour")

    out = MESH_DIR / f"{uuid.uuid4().hex}.glb"
    scene.export(out)                # one GLB – طبقتان ملونتان

    return JSONResponse({"url": f"/static/models/{out.name}"})
# ───────────────────── Local run ─────────────────────
if __name__ == "__main__":
    import uvicorn, sys
    uvicorn.run("main:app", host="0.0.0.0",
                port=(int(sys.argv[1]) if len(sys.argv) > 1 else 8000),
                reload=True)
