import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
import os
from typing import Generator


load_dotenv()

def get_db() -> Generator:
    connection = None
    try:
        connection = get_postgres_connection()
        yield connection
    finally:
        if connection:
            connection.close()

def get_postgres_connection():
    DB_HOST = os.getenv("db_host")
    DB_USER = os.getenv("db_user")
    DB_PASSWORD = os.getenv("db_password")
    DB_DATABASE = os.getenv("db_database")
    DB_PORT = os.getenv("db_port", 5432) 

    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_DATABASE,
            port=DB_PORT
        )
        
        if connection:
            print("Conectado a la base de datos PostgreSQL")
            return connection
            
    except Error as e:
        print("Error al conectar con PostgreSQL:", e)
        return None
