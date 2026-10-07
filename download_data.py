import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

DATA_DIR = Path("DyBench")

# Download everything from Hugging Face (skips files already downloaded)
snapshot_download("ddz16/DyBench", repo_type="dataset", local_dir=DATA_DIR)

# Unzip each video zip into its own folder, e.g. DyBench/video/event_sequence/
for zip_path in (DATA_DIR / "video").glob("*.zip"):
    out_dir = zip_path.with_suffix("")
    if out_dir.exists():
        print(f"Already unzipped: {zip_path.name}")
        continue
    print(f"Unzipping {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)

print("Done. Data is in", DATA_DIR.resolve())

