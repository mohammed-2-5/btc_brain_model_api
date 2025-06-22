
import base64, json, pathlib, requests

# ---------- (A) اجلب استجابة الـ API ----------
# أسهل طريقة: حمِّل ملفّين الـ NIfTI واطلب الـ API
files = {
 "nifti_flair": open("BraTS20_Training_004_flair.nii.gz", "rb"),
 "nifti_t1ce":  open("BraTS20_Training_004_t1ce.nii.gz", "rb"),
}
resp = requests.post(
 "http://127.0.0.1:8000/predict",
 files=files,
 params={"start_slice": 60, "nslices": 100},
)
resp.raise_for_status()          # يتأكد أنه 200 OK
data = resp.json()               # ← نفس resp_json

# ---------- (B) فكّ التشفير واحفظ الصور ----------
out_dir = pathlib.Path("slices")
out_dir.mkdir(exist_ok=True)

for idx, b64str in enumerate(data["images"]):
 (out_dir / f"slice_{idx}.png").write_bytes(base64.b64decode(b64str))

print("✓ حُفظت الصور داخل", out_dir.resolve())
