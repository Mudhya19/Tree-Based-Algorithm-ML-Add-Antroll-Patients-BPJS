import pandas as pd
from sqlalchemy import create_engine
from config.config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self):
        self.config = Config()
        self.engine = None
        self.connection = None
    
    def connect(self):
        """Create database engine and establish connection"""
        try:
            database_url = self.config.get_database_url()
            self.engine = create_engine(database_url)
            logger.info("Database engine created successfully")
            return self.engine
        except Exception as e:
            logger.error(f"Error creating database engine: {e}")
            raise
    
    def execute_query(self, query, params=None):
        """Execute a SQL query and return results as DataFrame"""
        try:
            if not self.engine:
                self.connect()
            
            df = pd.read_sql(query, self.engine, params=params)
            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def execute_non_query(self, query, params=None):
        """Execute a SQL query that doesn't return results (INSERT, UPDATE, DELETE)"""
        try:
            if not self.engine:
                self.connect()
            
            with self.engine.connect() as conn:
                conn.execute(query, params)
                conn.commit()
            logger.info("Non-query executed successfully")
        except Exception as e:
            logger.error(f"Error executing non-query: {e}")
            raise
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")

# Specific query methods for BPJS Antrol data
def get_bpjs_antrol_data(start_date, end_date):
    """Get BPJS Antrol patient data for specified date range"""
    query = """
    SELECT
        rp.no_rawat,
        rp.tgl_registrasi,
        rp.jam_reg,
        rp.kd_dokter,
        d.nm_dokter,
        rp.no_rkm_medis,
        pas.nm_pasien,
        rp.kd_poli,
        p.nm_poli,
        rp.status_lanjut,
        rp.kd_pj,
        pj.png_jawab,
        mar.tanggal_periksa,
        mar.nomor_kartu,
        mar.nomor_referensi,
        mar.kodebooking,
        mar.jenis_kunjungan,
        mar.status_kirim,
        mar.keterangan,
        bs.USER
    FROM
        reg_periksa rp
        JOIN mlite_antrian_referensi mar ON rp.no_rkm_medis = mar.no_rkm_medis
        JOIN poliklinik p ON rp.kd_poli = p.kd_poli
        JOIN dokter d ON rp.kd_dokter = d.kd_dokter
        JOIN penjab pj ON rp.kd_pj = pj.kd_pj
        JOIN pasien pas ON rp.no_rkm_medis = pas.no_rkm_medis
        JOIN bridging_sep bs ON rp.no_rawat = bs.no_rawat
    WHERE
        rp.tgl_registrasi BETWEEN %s AND %s
        AND mar.tanggal_periksa BETWEEN %s AND %s
        AND rp.kd_poli NOT IN ('IGDK', 'HDL', 'BBL', 'IRM', '006', 'U0016')
        AND rp.status_lanjut NOT IN ('Ranap')
    ORDER BY
        rp.no_rawat;
    """
    params = (start_date, end_date, start_date, end_date)
    db = DatabaseConnection()
    return db.execute_query(query, params)

def get_mlite_query_logs(start_date, end_date):
    """Get mlite query logs for specified date range"""
    query = """
    SELECT
        *
    FROM
        mlite_query_logs
    WHERE
        DATE(created_at) BETWEEN %s AND %s
    ORDER BY
        created_at DESC;
    """
    params = (start_date, end_date)
    db = DatabaseConnection()
    return db.execute_query(query, params)

# Example usage
if __name__ == "__main__":
    db = DatabaseConnection()
    # Example: db.execute_query("SELECT * FROM your_table")
    
    # Example of using specific query methods:
    # results = get_bpjs_antrol_data('2023-01-01', '2023-01-31')
    # logs = get_mlite_query_logs('2023-01-01', '2023-01-31')