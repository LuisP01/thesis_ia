from dotenv import load_dotenv
load_dotenv()

from src.ml.mlmodel import ejecutar_sistema_completo
from src.services.getUsers import obtener_usuarios
import json

TIPOS = ["agua", "luz"]

def main():
    print("=" * 60)
    print("INICIANDO JOB DE PREDICCIÓN MASIVA")
    print("=" * 60)

    usuarios = obtener_usuarios(limit=3)

    print(f"Usuarios obtenidos: {usuarios}")
    print(f"Total ejecuciones esperadas: {len(usuarios) * len(TIPOS)}")

    for user in usuarios:
        cedula = user["username"]
        id = user["id"]
        # Parsear JSON si existe
        agua_data = json.loads(user["agua"]) if user["agua"] else None
        luz_data = json.loads(user["luz"]) if user["luz"] else None

        for tipo in TIPOS:
            print("\n" + "-" * 60)
            print(f"Ejecutando ML | Usuario: {cedula} | Tipo: {tipo}")
            print("-" * 60)

            try:
                ejecutar_sistema_completo(
                    id,
                    cedula,
                    tipo,
                    agua_data,
                    luz_data
                )
            except Exception as e:
                print(f"Error con usuario {cedula} ({tipo}): {e}")
                continue

    print("\n" + "=" * 60)
    print("JOB FINALIZADO CORRECTAMENTE")
    print("=" * 60)

if __name__ == "__main__":
    main()
