import logging
from dotenv import load_dotenv
load_dotenv()

from src.config.loggin_config import setup_logging
from src.ml.mlmodel import ejecutar_sistema_completo
from src.services.getUsers import obtener_usuarios
import json

TIPOS = ["agua", "luz"]

def main():
    setup_logging()

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("INICIANDO JOB DE PREDICCIÓN MASIVA")
    logger.info("=" * 60)

    usuarios = obtener_usuarios(limit=3)

    logger.info(f"Usuarios obtenidos: {usuarios}")
    logger.info(f"Total ejecuciones esperadas: {len(usuarios) * len(TIPOS)}")

    for user in usuarios:
        cedula = user["username"]
        id = user["id"]

        agua_data = json.loads(user["agua"]) if user["agua"] else None
        luz_data = json.loads(user["luz"]) if user["luz"] else None

        for tipo in TIPOS:
            logger.info(f"Ejecutando ML | Usuario: {cedula} | Tipo: {tipo}")

            try:
                ejecutar_sistema_completo(
                    id,
                    cedula,
                    tipo,
                    agua_data,
                    luz_data
                )
            except Exception as e:
                logger.error(
                    f"Error con usuario {cedula} ({tipo}): {e}",
                    exc_info=True
                )
                continue

    logger.info("=" * 60)
    logger.info("JOB FINALIZADO CORRECTAMENTE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()