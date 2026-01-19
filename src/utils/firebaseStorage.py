import os
import tempfile
from src.config.firebaseConfig import get_bucket

def descargar_csv_firebase(cedula: str, tipo: str) -> str | None:
    bucket = get_bucket()

    nombre_archivo = f"{cedula}/{cedula}_{tipo}.csv"

    # carpeta temporal multiplataforma
    tmp_dir = tempfile.gettempdir()
    local_path = os.path.join(tmp_dir, f"{cedula}_{tipo}.csv")

    blob = bucket.blob(nombre_archivo)

    if not blob.exists():
        print(f"No existe en Firebase: {nombre_archivo}")
        return None

    blob.download_to_filename(local_path)
    print(f"Archivo descargado: {local_path}")

    return local_path
