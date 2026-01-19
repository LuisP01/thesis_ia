
from src.config.dbConfig import get_postgres_connection
from src.queries.queries import OBTAIN_USERS


def obtener_usuarios(limit=3):
    conn = get_postgres_connection()
    cur = conn.cursor()
    
    cur.execute(OBTAIN_USERS, (limit,))
    
    usuarios = [row[0] for row in cur.fetchall()]
    
    cur.close()
    conn.close()
    
    return usuarios
