"""
BPJS Antrol Patients Analysis
This script performs machine learning analysis on BPJS antrol patients data
using tree-based algorithms to predict patient outcomes or patterns.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from database.database_connection import DatabaseConnection, get_bpjs_antrol_data
from config.config import Config
import os
import logging
import joblib
from datetime import datetime, timedelta

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

def load_data_from_db_fallback():
    """Load patient data from database with CSV fallback"""
    config = Config()
    
    # Try to load from database first
    try:
        logger.info("Attempting to load data from database...")
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        df = get_bpjs_antrol_data(start_date, end_date)
        logger.info(f"Successfully loaded {len(df)} records from database")
        return df
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
        logger.info("Loading data from CSV fallback...")
        
        # Fallback to CSV file
        csv_file_path = os.path.join('database', "bpjs antrol.csv")
        if os.path.exists(csv_file_path):
            df = load_data_from_csv(csv_file_path)
            logger.info("Loaded data from CSV fallback successfully")
            return df
        else:
            logger.error(f"CSV file not found at {csv_file_path}")
            raise FileNotFoundError(f"Neither database connection nor CSV file available")

def create_features(df):
    """Create additional features from the raw data"""
    df = df.copy()
    
    # Convert date columns to datetime if they exist
    date_columns = ['tgl_registrasi', 'tanggal_periksa']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create additional features
    if 'tgl_registrasi' in df.columns:
        df['hari_registrasi'] = df['tgl_registrasi'].dt.day_name()
        df['bulan_registrasi'] = df['tgl_registrasi'].dt.month
        df['tahun_registrasi'] = df['tgl_registrasi'].dt.year
    
    if 'tanggal_periksa' in df.columns:
        df['hari_periksa'] = df['tanggal_periksa'].dt.day_name()
        df['bulan_periksa'] = df['tanggal_periksa'].dt.month
        df['tahun_periksa'] = df['tanggal_periksa'].dt.year

    # Calculate difference between registration and examination dates
    if 'tgl_registrasi' in df.columns and 'tanggal_periksa' in df.columns:
        df['hari_antara_reg_periksa'] = (df['tanggal_periksa'] - df['tgl_registrasi']).dt.days

    return df

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    logger.info("Starting data preprocessing...")
    
    # Create features
    df = create_features(df)
    
    # Identify categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Categorical columns: {categorical_columns}")
    logger.info(f"Numerical columns: {numerical_columns}")
    
    # Handle missing values
    for col in numerical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    # Prepare features for modeling
    # Select relevant features for modeling
    feature_columns = []
    
    # Add numerical features
    feature_columns.extend([col for col in numerical_columns if col not in ['tahun_registrasi', 'tahun_periksa']])
    
    # Add categorical features that are relevant
    relevant_categorical = ['nm_poli', 'png_jawab', 'hari_registrasi', 'status_lanjut', 'jenis_kunjungan']
    for col in relevant_categorical:
        if col in df.columns:
            feature_columns.append(col)
    
    logger.info(f"Selected {len(feature_columns)} features for modeling: {feature_columns}")
    
    # Prepare X (features) and y (target)
    X = df[feature_columns].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if pd.api.types.is_object_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Create target variable - for this example, we'll predict payment method (png_jawab)
    # If png_jawab exists in the data, use it as target, otherwise create a default
    if 'png_jawab' in df.columns:
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(df['png_jawab'].astype(str))
        logger.info(f"Created target variable from 'png_jawab' with {len(target_encoder.classes_)} classes: {target_encoder.classes_}")
    else:
        # Create a default binary target if png_jawab is not available
        y = np.random.randint(0, 2, size=len(df))
        target_encoder = None
        logger.info("Created random binary target variable")
    
    logger.info(f"Final feature matrix shape: {X.shape}")
    logger.info(f"Target vector shape: {y.shape}")
    
    return X, y, label_encoders, target_encoder

def train_tree_models(X_train, y_train):
    """Train tree-based models as specified in the requirements"""
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"{name} training completed")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate trained models using appropriate metrics for classification"""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate multiple metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        logger.info(f"{name} evaluation completed")
        print(f"\n{name} Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"\nClassification Report:")
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
        
        # Save summary metrics
        metrics_df = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [result['accuracy']],
            'Precision': [result['precision']],
            'Recall': [result['recall']],
            'F1_Score': [result['f1_score']]
        })
        metrics_df.to_csv(os.path.join(output_path, f"{model_name}_metrics.csv"), index=False)
    
    logger.info(f"Results saved to {output_path}")

def save_model(model, model_name, label_encoders, target_encoder, scaler=None):
    """Save the trained model and preprocessors"""
    output_path = Config().OUTPUT_PATH
    os.makedirs(output_path, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_path, f"{model_name.lower().replace(' ', '_')}_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save label encoders
    encoders_path = os.path.join(output_path, "label_encoders.pkl")
    joblib.dump(label_encoders, encoders_path)
    logger.info(f"Saved label encoders to {encoders_path}")
    
    # Save target encoder if it exists
    if target_encoder is not None:
        target_encoder_path = os.path.join(output_path, "target_encoder.pkl")
        joblib.dump(target_encoder, target_encoder_path)
        logger.info(f"Saved target encoder to {target_encoder_path}")
    
    # Save scaler if provided
    if scaler is not None:
        scaler_path = os.path.join(output_path, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

def main():
    """Main function to run the BPJS antrol analysis"""
    logger.info("Starting BPJS antrol analysis...")
    
    # Load data with database fallback to CSV
    try:
        df = load_data_from_db_fallback()
    except FileNotFoundError as e:
        logger.error(f"Data loading failed: {e}")
        return
    
    # Preprocess data
    X, y, label_encoders, target_encoder = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = train_tree_models(X_train_scaled, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test_scaled, y_test)
    
    # Find the best model based on F1 score
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_model = models[best_model_name]
    
    logger.info(f"Best model: {best_model_name} with F1-Score: {results[best_model_name]['f1_score']:.4f}")
    
    # Save the best model
    save_model(best_model, best_model_name, label_encoders, target_encoder, scaler)
    
    # Save all results
    config = Config()
    save_results(results, config.OUTPUT_PATH)
    
    logger.info("BPJS antrol analysis completed successfully!")
    print(f"\nBest performing model: {best_model_name}")
    print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")

if __name__ == "__main__":
    main()