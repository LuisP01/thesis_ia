
from src.config.dbConfig import get_postgres_connection
from src.queries.queries import OBTAIN_USERS


def obtener_usuarios(limit=3):
    conn = get_postgres_connection()
    cur = conn.cursor()
    
    cur.execute(OBTAIN_USERS, (limit,))
    
    rows = cur.fetchall()

    usuarios = [
        {   
            "id": row[0],
            "username": row[1],
            "agua": row[2],
            "luz": row[3]
        } for row in rows
    ]
    
    cur.close()
    conn.close()
    
    return usuarios
