# Tree-Based Algorithm ML Add Antroll Patients BPJS

## Overview
This project focuses on analyzing BPJS antrol (appointment) patients data using tree-based machine learning algorithms. The goal is to predict patient outcomes, identify patterns in patient appointments, and provide insights for better healthcare resource management.

## Features
- Complete project structure for ML analysis
- Database connection to hospital management system
- Tree-based algorithms implementation (Decision Tree, Random Forest)
- Automated setup script
- Configuration management
- Data preprocessing capabilities

## Project Structure
```
Tree Based Algorithm ML Add Antroll Patients BPJS/
├── .env.example              # Contoh file konfigurasi environment
├── .venv/                   # Virtual environment (akan dibuat saat setup)
├── .vscode/                 # Konfigurasi VSCode
├── config/                  # File konfigurasi aplikasi
│   ├── __init__.py
│   └── config.py            # Konfigurasi database dan path
├── data/                    # Folder untuk data CSV
├── database/                # File-file terkait database
│   ├── __init__.py
│   └── database_connection.py # Fungsi koneksi dan query database
├── docs/                    # Dokumentasi proyek
├── image/                   # Gambar-gambar pendukung
├── notebooks/               # Jupyter notebooks
├── output/                  # Hasil analisis disimpan di sini
├── src/                     # Source code tambahan
├── test/                    # File-file pengujian
├── bpjs_add_antrol.py       # File utama untuk analisis
├── nano.sh                  # Script setup proyek
├── requirements.txt         # Daftar dependensi Python
└── setup.py                 # File setup untuk package
```

## Requirements
- Python 3.8 or higher
- MySQL database access
- Required Python packages (see requirements.txt)

## Setup
1. Clone this repository
2. Run the setup script:
   ```bash
   bash nano.sh
   ```
3. Update the `.env` file with your database credentials
4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
5. Run the main analysis script:
   ```bash
   python bpjs_add_antrol.py
   ```

## Database Configuration
The project is configured to connect to a MySQL database with the following default settings:
- Host: 192.168.11.5
- User: rsds_db
- Password: rsdsD4t4b4s3
- Database: rsds_db
- Port: 3306

These can be modified in the `.env` file.

## Analysis Queries
The project includes specific queries for:
- BPJS Antrol patient data
- mlite query logs

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License
This project is licensed under the MIT License.
