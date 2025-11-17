import os
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Konfigurasi database
DB_HOST = os.getenv('DB_HOST', '192.168.11.5')
DB_USER = os.getenv('DB_USER', 'rsds_db')
DB_PASS = os.getenv('DB_PASS', 'rsdsD4t4b4s3')
DB_NAME = os.getenv('DB_NAME', 'rsds_db')
DB_PORT = int(os.getenv('DB_PORT', 3306))

# Konfigurasi lainnya
DATA_PATH = os.path.join(os.getcwd(), 'data')
OUTPUT_PATH = os.path.join(os.getcwd(), 'output')
LOG_PATH = os.path.join(os.getcwd(), 'logs')

def get_database_url():
    return f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"