#!/bin/bash

# BPJS Antrol Patients Analysis Project Setup Script
# This script will set up the complete project environment, structure, and dependencies

set -e # Exit immediately if a command exits with a non-zero status

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} BPJS Antrol Patients Analysis ${NC}"
    echo -e "${BLUE}      Project Setup Script      ${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_step() {
    echo -e "${YELLOW}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}>>> $1${NC}"
}

print_error() {
    echo -e "${RED}>>> $1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
setup_project() {
    print_header
    echo ""
    
    # Check if running on Windows (including Git Bash, WSL, or Command Prompt)
    # Use alternative method to detect Windows since OSTYPE might not be reliable in all environments
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == *win* || "$OSTYPE" == *mingw* ]] || command -v python.exe >/dev/null 2>&1 || [ -d "/c/Windows" ]; then
        print_step "Detected Windows OS"
        # For Windows, we'll use Python from PATH
        PYTHON_CMD="python"
        PIP_CMD="pip"
    else
        PYTHON_CMD="python3"
        PIP_CMD="pip3"
    fi
    
    # Check if Python is installed
    if ! command_exists $PYTHON_CMD && ! command_exists "${PYTHON_CMD}.exe"; then
        print_error "Python is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Update PYTHON_CMD to include .exe extension if needed
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == *win* || "$OSTYPE" == *mingw* ]] || command -v python.exe >/dev/null 2>&1 || [ -d "/c/Windows" ]; then
        if ! command_exists $PYTHON_CMD && command_exists "${PYTHON_CMD}.exe"; then
            PYTHON_CMD="${PYTHON_CMD}.exe"
        fi
    fi
    
    # Check Python version
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == *win* || "$OSTYPE" == *mingw* ]] || command -v python.exe >/dev/null 2>&1 || [ -d "/c/Windows" ]; then
        # For Windows environments, use python.exe if python command is not found
        if ! command_exists $PYTHON_CMD && command_exists "${PYTHON_CMD}.exe"; then
            PYTHON_VERSION=$($PYTHON_CMD.exe --version 2>&1 | cut -d' ' -f2)
        else
            PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
        fi
    else
        PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    fi
    echo "Python version: $PYTHON_VERSION"
    
    # Check if version is 3.8 or higher
    if [[ $(printf '%s\n' "3.8" "$PYTHON_VERSION" | sort -V | head -n1) == "3.8" ]] || [[ "$PYTHON_VERSION" == "3.8" ]]; then
        print_success "Python version is compatible"
    else
        print_error "Python version must be 3.8 or higher. Current version: $PYTHON_VERSION"
        exit 1
    fi
    
    # Check if pip is installed
    if ! command_exists $PIP_CMD && ! command_exists "${PIP_CMD}.exe" && ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
        print_error "pip is not installed. Please install pip."
        exit 1
    fi
    
    print_step "Creating project directory structure..."
    
    # Create main project directories
    dirs=(
        ".venv"
        ".vscode"
        "config"
        "data"
        "database"
        "docs"
        "image"
        "notebooks"
        "output"
        "src"
        "test"
        "logs"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_step "Directory already exists: $dir"
        fi
    done
    
    print_step "Creating configuration files..."
    
    # Create .env.example
    if [ ! -f ".env.example" ]; then
        cat > .env.example << EOF
# Database Configuration
DB_HOST=192.168.11.5
DB_PORT=3306
DB_NAME=rsds_db
DB_USER=rsds_db
DB_PASS=rsdsD4t4b4s3

# File Paths
DATA_PATH=./data
OUTPUT_PATH=./output
LOG_PATH=./logs
EOF
        print_success "Created .env.example"
    else
        print_step ".env.example already exists"
    fi
    
    # Create .env if it doesn't exist (copy from .env.example)
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_success "Created .env from .env.example"
        echo "Please update .env with your actual credentials"
    else
        print_step ".env already exists"
    fi
    
    # Create config/__init__.py
    if [ ! -f "config/__init__.py" ]; then
        touch config/__init__.py
        print_success "Created config/__init__.py"
    fi
    
    # Create config/config.py
    if [ ! -f "config/config.py" ]; then
        cat > config/config.py << 'EOF'
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
EOF
        print_success "Created config/config.py"
    fi
    
    # Create database/__init__.py
    if [ ! -f "database/__init__.py" ]; then
        touch database/__init__.py
        print_success "Created database/__init__.py"
    fi
    
    # Create database/database_connection.py
    if [ ! -f "database/database_connection.py" ]; then
        cat > database/database_connection.py << 'EOF'
import pandas as pd
from sqlalchemy import create_engine
from config.config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self):
        self.config = Config
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
EOF
        print_success "Created database/database_connection.py"
    fi
    
    # Create bpjs_add_antrol.py
    if [ ! -f "bpjs_add_antrol.py" ]; then
        cat > bpjs_add_antrol.py << 'EOF'
"""
BPJS Antrol Patients Analysis
This script performs machine learning analysis on BPJS antrol patients data
using tree-based algorithms to predict patient outcomes or patterns.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from database.database_connection import DatabaseConnection
from config.config import Config
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_csv(file_path):
    """Load patient data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def load_data_from_db(query, params=None):
    """Load patient data from database"""
    try:
        db = DatabaseConnection()
        df = db.execute_query(query, params)
        logger.info(f"Data loaded successfully from database, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        raise

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Encode categorical variables if any
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    
    logger.info("Data preprocessing completed")
    return df

def train_tree_models(X_train, y_train):
    """Train tree-based models"""
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate trained models"""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        logger.info(f"{name} evaluation completed")
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
    
    return results

def save_results(results, output_path):
    """Save model results to output directory"""
    os.makedirs(output_path, exist_ok=True)
    
    for model_name, result in results.items():
        # Save classification report
        report_df = pd.DataFrame(result['classification_report']).transpose()
        report_df.to_csv(os.path.join(output_path, f"{model_name}_report.csv"))
        
        # Save confusion matrix
        cm_df = pd.DataFrame(result['confusion_matrix'])
        cm_df.to_csv(os.path.join(output_path, f"{model_name}_confusion_matrix.csv"))
    
    logger.info(f"Results saved to {output_path}")

def main():
    """Main function to run the BPJS antrol analysis"""
    config = Config()
    
    # Load data - you can choose to load from CSV or database
    csv_file_path = os.path.join(config.DATA_PATH, "bpjs antrol.csv")  # Adjust filename as needed
    
    if os.path.exists(csv_file_path):
        df = load_data_from_csv(csv_file_path)
    else:
        # Example query to load data from database
        # You can customize this query based on your needs
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        df = get_bpjs_antrol_data(start_date, end_date)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Assuming the target column is named 'target' - adjust as needed
    if 'target' in df.columns:
        X = df.drop('target', axis=1)
        y = df['target']
    else:
        # If no target column exists, you might want to define one based on your analysis
        # For now, using the last column as target (common in some datasets)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = train_tree_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Save results
    save_results(results, config.OUTPUT_PATH)
    
    logger.info("BPJS antrol analysis completed successfully!")

if __name__ == "__main__":
    main()
EOF
        print_success "Created bpjs_add_antrol.py"
    fi
    
    # Create requirements.txt
    if [ ! -f "requirements.txt" ]; then
        cat > requirements.txt << EOF
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
sqlalchemy>=1.4.0
pymysql>=1.0.0
python-dotenv>=0.19.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0
notebook>=6.4.0
EOF
        print_success "Created requirements.txt"
    fi
    
    # Create setup.py
    if [ ! -f "setup.py" ]; then
        cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="bpjs-antrol-analysis",
    version="0.1.0",
    description="BPJS Antrol Patients Analysis using Tree-Based Algorithms",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "sqlalchemy>=1.4.0",
        "pymysql>=1.0.0",
        "python-dotenv>=0.19.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)
EOF
        print_success "Created setup.py"
    fi
    
    # Create .gitignore
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local
.env.dev

# OS
.DS_Store
Thumbs.db

# Data
data/*.csv
data/*.xlsx
data/*.json

# Output
output/*.csv
output/*.json
output/*.png
output/*.jpg
output/*.pdf

# Logs
logs/*.log
*.log

# Jupyter
.ipynb_checkpoints
EOF
        print_success "Created .gitignore"
    fi
    
    # Create VSCode settings
    if [ ! -f ".vscode/settings.json" ]; then
        mkdir -p .vscode
        cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--line-length",
        "88"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
EOF
        print_success "Created .vscode/settings.json"
    fi
    
    # Create VSCode launch.json for debugging
    if [ ! -f ".vscode/launch.json" ]; then
        cat > .vscode/launch.json << EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: BPJS Antrol Analysis",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/bpjs_add_antrol.py",
            "console": "integratedTerminal",
            "python": "${workspaceFolder}/.venv/bin/python",
            "envFile": "${workspaceFolder}/.env"
        }
    ]
}
EOF
        print_success "Created .vscode/launch.json"
    fi
    
    print_step "Setting up Python virtual environment..."
    
    # Create virtual environment
    if [ ! -d ".venv" ]; then
        $PYTHON_CMD -m venv .venv
        print_success "Created virtual environment in .venv"
    else
        print_step "Virtual environment already exists"
    fi
    
    # Activate virtual environment and install packages
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == *win* || "$OSTYPE" == *mingw* ]] || command -v python.exe >/dev/null 2>&1 || [ -d "/c/Windows" ]; then
            # Windows
            if [ -f ".venv/Scripts/activate" ]; then
                source .venv/Scripts/activate
            else
                print_error "Windows virtual environment activation script not found"
                exit 1
            fi
        else
            # Linux/Mac
            if [ -f ".venv/bin/activate" ]; then
                source .venv/bin/activate
            else
                print_error "Linux/Mac virtual environment activation script not found"
                exit 1
            fi
        fi
    
    # Upgrade pip
    $PYTHON_CMD -m pip install --upgrade pip
    
    # Install packages from requirements.txt
    if [ -f "requirements.txt" ]; then
        $PYTHON_CMD -m pip install -r requirements.txt
        print_success "Installed packages from requirements.txt"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_step "Creating initial data structure documentation..."
    
    # Create initial documentation
    if [ ! -f "docs/structure.md" ]; then
        cat > docs/structure.md << EOF
# Project Structure

## Directories
- **config/**: Configuration files for database and application settings
- **data/**: Raw data files (CSV, Excel, etc.)
- **database/**: Database connection and query functions
- **docs/**: Project documentation
- **image/**: Images and visualizations
- **notebooks/**: Jupyter notebooks for exploratory analysis
- **output/**: Analysis results and model outputs
- **src/**: Additional source code modules
- **test/**: Unit tests
- **logs/**: Application logs

## Files
- **.env**: Environment variables (not committed to version control)
- **.env.example**: Example environment variables file
- **bpjs_add_antrol.py**: Main analysis script
- **requirements.txt**: Python dependencies
- **nano.sh**: Setup script
EOF
        print_success "Created docs/structure.md"
    fi
    
    if [ ! -f "docs/database_schema.md" ]; then
        cat > docs/database_schema.md << EOF
# Database Schema Information

## BPJS Antrol Tables

### reg_periksa
- no_rawat: Registration number
- tgl_registrasi: Registration date
- jam_reg: Registration time
- kd_dokter: Doctor code
- no_rkm_medis: Medical record number
- kd_poli: Polyclinic code
- status_lanjut: Continuation status
- kd_pj: Payment method code

### mlite_antrian_referensi
- tanggal_periksa: Examination date
- nomor_kartu: Card number
- nomor_referensi: Reference number
- kodebooking: Booking code
- jenis_kunjungan: Visit type
- status_kirim: Send status
- keterangan: Notes

### Other related tables
- poliklinik: Polyclinic information
- dokter: Doctor information
- penjab: Payment method information
- pasien: Patient information
- bridging_sep: SEP bridging information
EOF
        print_success "Created docs/database_schema.md"
    fi
    
    print_step "Creating example notebook..."
    
    # Create example Jupyter notebook
    if [ ! -f "notebooks/example_analysis.ipynb" ]; then
        cat > notebooks/example_analysis.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# BPJS Antrol Data Analysis Example\n",
    "\n",
    "This notebook demonstrates how to load and analyze BPJS antrol data using the project's database connection functions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Import required libraries\n",
    "import sys\n",
    "import os\n",
    "sys.path.append(os.path.abspath('.'))\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from datetime import datetime, timedelta\n",
    "\n",
    "# Import project modules\n",
    "from database.database_connection import get_bpjs_antrol_data, get_mlite_query_logs\n",
    "from config.config import DATA_PATH, OUTPUT_PATH"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load BPJS Antrol data for the last 30 days\n",
    "end_date = datetime.now().strftime('%Y-%m-%d')\n",
    "start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')\n",
    "\n",
    "print(f\"Loading data from {start_date} to {end_date}\")\n",
    "df = get_bpjs_antrol_data(start_date, end_date)\n",
    "print(f\"Loaded {len(df)} records\")\n",
    "print(f\"Columns: {list(df.columns)}\")"
   ]
  },
 {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Display first few rows\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Basic statistics\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Check for missing values\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Analysis example: Patient distribution by polyclinic\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "poly_count = df['nm_poli'].value_counts().head(10)\n",
    "sns.barplot(x=poly_count.values, y=poly_count.index)\n",
    "plt.title('Top 10 Polyclinics by Patient Count')\n",
    "plt.xlabel('Number of Patients')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
        print_success "Created notebooks/example_analysis.ipynb"
    fi
    
    print_success "Project setup completed successfully!"
    echo ""
    print_step "Next steps:"
    echo "1. Update .env with your actual database credentials"
    echo "2. Run 'source .venv/bin/activate' (Linux/Mac) or '.venv\\Scripts\\activate' (Windows) to activate the virtual environment"
    echo "3. Run 'python bpjs_add_antrol.py' to execute the main analysis script"
    echo "4. Check the output/ directory for results"
    echo ""
    print_success "Setup script execution finished!"
}

# Run the setup
setup_project