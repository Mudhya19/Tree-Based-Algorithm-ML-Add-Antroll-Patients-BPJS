import os
import pandas as pd
from config.config import Config
from database.database_connection import get_bpjs_antrol_data

def test_csv_connection():
    """Test the connection to the CSV file in the database folder"""
    print("Testing CSV connection...")
    
    # Check if CSV file exists
    csv_file_path = os.path.join('database', "bpjs antrol.csv")
    print(f"Checking if CSV file exists at: {csv_file_path}")
    
    if os.path.exists(csv_file_path):
        print("[OK] CSV file exists")
        
        # Try to load a small sample of the CSV file
        try:
            # Load just the first few rows to test
            df_sample = pd.read_csv(csv_file_path, nrows=5)
            print(f"[OK] Successfully loaded sample data. Shape: {df_sample.shape}")
            print(f"Columns in the CSV: {list(df_sample.columns)}")
            
            # Load full CSV file to test complete connection
            print("Loading full CSV file...")
            df = pd.read_csv(csv_file_path)
            print(f"[OK] Successfully loaded full CSV file. Shape: {df.shape}")
            
            # Check for 'tgl_registrasi' column which is used in the filtering
            if 'tgl_registrasi' in df.columns:
                print("[OK] 'tgl_registrasi' column exists for date filtering")
            else:
                print("[WARN] 'tgl_registrasi' column not found")
                
            print("CSV connection test completed successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading CSV file: {e}")
            return False
    else:
        print(f"[ERROR] CSV file does not exist at {csv_file_path}")
        return False

def test_load_data_from_db_fallback():
    """Test the load_data_from_db_fallback function from the notebook"""
    print("\nTesting load_data_from_db_fallback function...")
    
    def load_data_from_csv(file_path):
        # Load patient data from CSV file
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}, shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            raise

    def load_data_from_db_fallback():
        # Load patient data from database with CSV fallback
        config = Config()
        
        # Try to load from database first
        try:
            print("Attempting to load data from database...")
            from datetime import datetime, timedelta
            # Set date range for 2025, from January to June
            start_date = '2025-01-01'
            end_date = '2025-06-30'
            
            df = get_bpjs_antrol_data(start_date, end_date)
            print(f"Successfully loaded {len(df)} records from database")
            return df
        except Exception as e:
            print(f"Database connection failed: {e}")
            print("Loading data from CSV fallback...")
            
            # Fallback to CSV file
            csv_file_path = os.path.join('database', "bpjs antrol.csv")
            if os.path.exists(csv_file_path):
                df = load_data_from_csv(csv_file_path)
                print("[OK] Loaded data from CSV fallback successfully")
                # Filter data for the required date range (2025-01-01 to 2025-06-30)
                if 'tgl_registrasi' in df.columns:
                    df['tgl_registrasi'] = pd.to_datetime(df['tgl_registrasi'], format='mixed', dayfirst=True, errors='coerce')
                    df = df[(df['tgl_registrasi'] >= '2025-01-01') & (df['tgl_registrasi'] <= '2025-06-30')]
                    print(f"Filtered CSV data for date range 2025-01-01 to 2025-06-30, shape: {df.shape}")
                return df
            else:
                print(f"[ERROR] CSV file not found at {csv_file_path}")
                raise FileNotFoundError(f"Neither database connection nor CSV file available")

    try:
        df = load_data_from_db_fallback()
        print(f"[OK] Data loading test completed successfully. Final dataset shape: {df.shape}")
        return True
    except Exception as e:
        print(f"[ERROR] Data loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting CSV connection tests...\n")
    
    # Test basic CSV connection
    csv_test_result = test_csv_connection()
    
    # Test the full load function
    load_test_result = test_load_data_from_db_fallback()
    
    print(f"\nTest Results:")
    print(f"CSV Connection Test: {'PASSED' if csv_test_result else 'FAILED'}")
    print(f"Load Function Test: {'PASSED' if load_test_result else 'FAILED'}")
    
    if csv_test_result and load_test_result:
        print("\n[OK] All tests passed! The notebook should be able to connect to the CSV file successfully.")
    else:
        print("\n[ERROR] Some tests failed. Please check the error messages above.")