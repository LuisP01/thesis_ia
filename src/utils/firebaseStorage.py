import os
import tempfile
from src.config.firebaseConfig import get_bucket

import logging

logger = logging.getLogger(__name__)

def descargar_csv_firebase(cedula: str, tipo: str) -> str | None:
    bucket = get_bucket()

    nombre_archivo = f"{cedula}/{cedula}_{tipo}.csv"

    tmp_dir = tempfile.gettempdir()
    local_path = os.path.join(tmp_dir, f"{cedula}_{tipo}.csv")

    blob = bucket.blob(nombre_archivo)

    if not blob.exists():
        logger.warning(f"No existe en Firebase: {nombre_archivo}")
        return None

    blob.download_to_filename(local_path)
    logger.info(f"Archivo descargado: {local_path}")

    return local_path
