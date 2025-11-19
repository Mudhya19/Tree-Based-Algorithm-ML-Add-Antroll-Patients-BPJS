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
    ├── .gitignore               # File untuk mengabaikan file-file tertentu
    ├── bpjs_add_antrol.py       # File utama untuk analisis
    ├── nano.sh                  # Script setup proyek
    ├── README.md                # Dokumentasi proyek
    ├── requirements.txt         # Daftar dependensi Python
    └── setup.py                 # File setup untuk package
    ├── config/                  # File konfigurasi aplikasi
    │   ├── __init__.py
    │   └── config.py            # Konfigurasi database dan path
    ├── database/                # File-file terkait database
    │   ├── __init__.py
    │   ├── bpjs antrol.csv      # File data BPJS antrol dalam format CSV
    │   └── database_connection.py # Fungsi koneksi dan query database
    ├── docs/                    # Dokumentasi proyek
    │   ├── database_schema.md   # Informasi skema database
    │   ├── NOTES NOTEBOOK.md    # Catatan dan dokumentasi tambahan
    │   └── structure.md         # Dokumentasi struktur proyek
    ├── image/                   # Gambar-gambar hasil visualisasi EDA
    │   ├── advanced_eda_correlation.png
    │   ├── advanced_eda_distribution_nomor_kartu.png
    │   ├── advanced_eda_scatter_nomor_kartu_vs_jam_reg_hour.png
    │   ├── advanced_eda_success_by_day.png
    │   ├── advanced_eda_success_by_month.png
    │   ├── advanced_eda_success_by_poli.png
    │   ├── advanced_eda_success_failure_by_hour.png
    │   ├── advanced_eda_time_diff.png
    │   ├── advanced_eda_timeline.png
    │   ├── advanced_eda_visualizations.png
    │   ├── eda_bulan_registrasi.png
    │   ├── eda_hari_registrasi.png
    │   ├── eda_jenis_kunjungan.png
    │   ├── eda_nm_poli.png
    │   ├── eda_png_jawab.png
    │   ├── eda_status_kirim.png
    │   ├── eda_status_lanjut.png
    │   ├── eda_status_vs_jenis.png
    │   ├── eda_status_vs_poli.png
    │   └── eda_visualizations.png
    ├── logs/                    # Log aplikasi
    ├── notebooks/               # Jupyter notebooks
    │   ├── bpjs_add_antrol.ipynb # Notebook untuk analisis BPJS antrol
    │   └── example_analysis.ipynb # Contoh analisis data
    ├── output/                  # Model dan hasil analisis disimpan di sini
    │   ├── decision_tree_model.pkl
    │   ├── gradient_boosting_model.pkl
    │   ├── label_encoders.pkl
    │   ├── random_forest_model.pkl
    │   ├── scaler.pkl
    │   └── target_encoder.pkl
    ├── src/                     # Source code tambahan (jika ada)
    └── test/                    # File-file pengujian (jika ada)
    ```
    
    ## Key Features
    1. **Understanding Business**: Pemahaman kebutuhan bisnis dan tujuan analisis
    2. **Data Understanding**: Eksplorasi dan pemahaman struktur data
    3. **Data Preparation/Wrangling**: Persiapan dan pengolahan data
    4. **Data Cleaning**: Pembersihan data dari ketidakkonsistenan
    5. **Explanatory Data Analysis (EDA Deskriptif)**: Analisis deskriptif awal
    6. **Exploratory Data Analysis (EDA Mendalam)**: Eksplorasi data secara mendalam
    7. **Data Preprocessing**: Pra-pemrosesan data untuk modeling
    8. **Training Modeling**: Pelatihan model machine learning
    9. **Evaluation Modeling**: Evaluasi performa model
    10. **Save Model**: Penyimpanan model terbaik
    11. **Insight & Conclusion**: Penarikan kesimpulan dan rekomendasi
    
    ## Analysis Focus
    Proyek ini fokus pada analisis pola keberhasilan dan kegagalan pendaftaran pasien BPJS melalui dua kanal yaitu Anjungan Pendaftaran Mandiri (APM) dan aplikasi Mobile JKN. Melalui pendekatan analisis log dan metode clustering dalam penerapan machine learning dengan menggunakan Tree-Based Algorithm (Decision Tree, Random Forest dan Gradient Boosting), bertujuan untuk mengidentifikasi faktor-faktor penyebab dan memberikan rekomendasi peningkatan efektivitas pelayanan digital rumah sakit.
    
    
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
    
    ## Additional Components
    - **Documentation**: Detailed documentation in the `docs/` directory
    - **Jupyter Notebooks**: Example analysis notebook in `notebooks/` directory
    - **VSCode Configuration**: Pre-configured settings for VSCode development
    - **Virtual Environment**: Isolated Python environment for dependencies
    - **Logging**: Application logs stored in `logs/` directory
    
    ## Contributing
    1. Fork the repository
    2. Create a feature branch
    3. Make your changes
    4. Submit a pull request
    
    ## License
    This project is licensed under the MIT License.
