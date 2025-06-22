
# 🧠 FastAPI Backend – Brain MRI Segmentation

This FastAPI backend serves endpoints for brain tumor segmentation, image visualization, and 3D mesh generation from MRI scans in NIfTI format.

---

## ⚙️ Features

- `/predict`: Returns 6 PNG slices from MRI scan with overlaid tumor predictions.
- `/plot_preview`: Generates a 4-panel anatomical image using Nilearn.
- `/tumor_mesh`: Produces a color-coded 3D tumor mesh (GLB format).
- `/brain_mesh`: Generates a combined 3D brain and tumor mesh (GLB format).

All output files are served under the `static/` directory.

---

## 📦 Requirements

Install Python 3.8 or higher, then:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
```

### Required Packages
Create a `requirements.txt` or install manually:

```bash
pip install fastapi uvicorn python-multipart numpy opencv-python nibabel nilearn matplotlib trimesh scikit-image tensorflow
```

---

## 📁 Project Structure

```
.
├── main.py               # Main FastAPI app
├── models/               # Pre-trained segmentation model (.h5)
├── static/               # Output PNGs, plots, GLB files
│   ├── slices/
│   ├── plots/
│   └── models/
```

Ensure this folder structure exists (created automatically by the script if missing).

---

## 🔍 Model

Place your trained Keras model in:

```
models/3D_MRI_Brain_tumor_segmentation.h5
```

If missing, a dummy model will be used that returns empty predictions.

---

## 🚀 How to Run the Server

Run the API using `uvicorn`:

```bash
uvicorn main:app --reload
```

### Or run directly with Python

```bash
python main.py
```

### Optional: Specify port

```bash
python main.py 8080
```

---

## 🧪 Sample Usage

### 1. Predict slices

```bash
curl -X POST http://127.0.0.1:8000/predict   -F "nifti_flair=@flair.nii.gz"   -F "nifti_t1ce=@t1ce.nii.gz"
```

### 2. Plot preview

```bash
curl -X POST http://127.0.0.1:8000/plot_preview   -F "flair=@flair.nii.gz"   -F "seg=@seg.nii.gz"
```

### 3. Tumor mesh

```bash
curl -X POST http://127.0.0.1:8000/tumor_mesh   -F "seg=@seg.nii.gz"   -F "iso_level=0.5"
```

### 4. Brain mesh

```bash
curl -X POST http://127.0.0.1:8000/brain_mesh   -F "flair=@flair.nii.gz"   -F "seg=@seg.nii.gz"   -F "iso=0.12"   -F "tumor_iso=0.5"
```

---

## 🌐 Output Access

Files are saved and accessible under:

```
/static/slices/
/static/plots/
/static/models/
```

Example:
```
http://127.0.0.1:8000/static/slices/<run_id>/<file>.png
```

---

## 📝 Notes

- All NIfTI files must be 3D `.nii` or `.nii.gz` format.
- You can use tools like 3D Slicer or FSL to generate or view segmentation files.
- Works seamlessly with the Flutter frontend (see separate `main.dart`).

---

## 📧 Contact

For support, feature requests, or bug reports, please raise an issue on the GitHub repo.
