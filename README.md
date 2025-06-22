🧠 Brain MRI Segmentation API (FastAPI)
A powerful FastAPI backend for processing 3D MRI brain scans. It:

Segments brain tumors from FLAIR and T1ce NIfTI images.

Visualizes image slices and segmentation overlays.

Exports tumor meshes and brain surfaces as .glb 3D models.

📦 Features
/predict: Returns 6 slice images with predicted tumor masks.

/plot_preview: Generates a PNG plot with nilearn overlays.

/tumor_mesh: Exports tumor regions as 3D GLB models.

/brain_mesh: Exports a complete brain + tumor mesh (GLB).

All static outputs are served via /static/.

🔧 Installation
1. Clone the project
bash
Copy
Edit
git clone https://github.com/your-username/brain-mri-fastapi.git
cd brain-mri-fastapi
2. Create a virtual environment
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
If you don't have requirements.txt, create one with:

txt
Copy
Edit
fastapi
uvicorn
python-multipart
numpy
opencv-python
nibabel
nilearn
matplotlib
trimesh
scikit-image
tensorflow  # Optional: if you want to use the trained model
🧪 Running the App Locally
Make sure your model is in the path:

Copy
Edit
models/3D_MRI_Brain_tumor_segmentation.h5
Then start the API:

bash
Copy
Edit
python main.py
Or specify a custom port:

bash
Copy
Edit
python main.py 8080
📂 Folder Structure
javascript
Copy
Edit
main.py
models/                   → Trained model file (.h5)
static/
  ├── slices/             → Output images from `/predict`
  ├── plots/              → Output plots from `/plot_preview`
  └── models/             → Output .glb files from `/tumor_mesh`, `/brain_mesh`
🧠 Sample Request
Predict Tumor Slices
bash
Copy
Edit
curl -X POST http://localhost:8000/predict \
  -F "nifti_flair=@/path/to/flair.nii.gz" \
  -F "nifti_t1ce=@/path/to/t1ce.nii.gz"
🌐 CORS
All origins and methods are allowed:

python
Copy
Edit
allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
📎 Notes
Make sure uploaded NIfTI files are 3D .nii or .nii.gz.

Tumor mesh colors: red, green, yellow by label.

Brain mesh is light gray; tumors are semi-transparent red in GLB.

✅ To-Do / Future Improvements
Add Swagger UI docs for visual testing.

Enable GPU-based TensorFlow optimization.

Add user auth for protected endpoints (if needed).

