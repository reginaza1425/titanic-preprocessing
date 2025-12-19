import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    # 1. Tentukan Path secara Dinamis (Agar jalan di Laptop & GitHub Actions)
    # Base dir adalah folder tempat script ini berada (folder 'preprocessing')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path Input (Naik satu level ke folder 'namadataset_raw')
    raw_data_path = os.path.join(base_dir, '..', 'titanic_raw', 'Titanic-Dataset.csv')
    
    # Path Output (Simpan di 'namadataset_preprocessing')
    output_dir = os.path.join(base_dir, 'titanic_preprocessing')
    output_path = os.path.join(output_dir, 'titanic_clean.csv')

    os.makedirs(output_dir, exist_ok=True)

    # 2. Proses Data (Sesuai Notebook Eksperimen_MSML.ipynb Anda)
    print(f"Loading data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    # --- Mulai Logika Preprocessing dari Notebook ---
    # (Salin logika cleaning Anda di sini: Drop Cabin, Imputasi Age, OneHot Encoding, dll)
    # Contoh singkat:
    df.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True, errors='ignore')
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # ... dst ...
    # ------------------------------------------------
    
    # 3. Simpan Hasil
    df.to_csv(output_path, index=False)
    print(f"Success! Data processed and saved to: {output_path}")

if __name__ == "__main__":
    load_and_preprocess_data()