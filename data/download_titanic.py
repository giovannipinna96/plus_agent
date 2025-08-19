"""Script to download the Titanic dataset."""

import pandas as pd
import os

def download_titanic_dataset():
    """Download the Titanic dataset from seaborn."""
    try:
        import seaborn as sns
        # Load the titanic dataset from seaborn
        titanic = sns.load_dataset('titanic')
        
        # Save to CSV
        output_path = os.path.join(os.path.dirname(__file__), 'titanic.csv')
        titanic.to_csv(output_path, index=False)
        
        print(f"Titanic dataset downloaded and saved to: {output_path}")
        print(f"Dataset shape: {titanic.shape}")
        print("\nColumns:", list(titanic.columns))
        print("\nFirst few rows:")
        print(titanic.head())
        
        return output_path
        
    except ImportError:
        print("Seaborn not installed. Creating a sample dataset...")
        
        # Create a sample Titanic-like dataset
        import numpy as np
        np.random.seed(42)
        
        n_samples = 891
        sample_data = {
            'survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
            'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
            'sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'age': np.random.normal(29.7, 14.5, n_samples),
            'sibsp': np.random.choice([0, 1, 2, 3, 4, 5, 8], n_samples, p=[0.68, 0.23, 0.07, 0.02, 0.002, 0.001, 0.001]),
            'parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.006, 0.004, 0.001, 0.001]),
            'fare': np.random.lognormal(mean=3.0, sigma=1.0, size=n_samples),
            'embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72]),
            'class': np.random.choice(['First', 'Second', 'Third'], n_samples, p=[0.24, 0.21, 0.55]),
            'who': np.random.choice(['man', 'woman', 'child'], n_samples, p=[0.57, 0.31, 0.12]),
            'adult_male': np.random.choice([True, False], n_samples, p=[0.57, 0.43]),
            'deck': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', None], n_samples, p=[0.02, 0.05, 0.07, 0.04, 0.04, 0.02, 0.001, 0.859]),
            'embark_town': np.random.choice(['Cherbourg', 'Queenstown', 'Southampton'], n_samples, p=[0.19, 0.09, 0.72]),
            'alive': np.random.choice(['yes', 'no'], n_samples, p=[0.38, 0.62])
        }
        
        # Add some NaN values to age column
        age_mask = np.random.choice([True, False], n_samples, p=[0.8, 0.2])
        sample_data['age'][~age_mask] = np.nan
        
        df = pd.DataFrame(sample_data)
        
        # Save to CSV
        output_path = os.path.join(os.path.dirname(__file__), 'titanic.csv')
        df.to_csv(output_path, index=False)
        
        print(f"Sample Titanic dataset created and saved to: {output_path}")
        print(f"Dataset shape: {df.shape}")
        print("\nColumns:", list(df.columns))
        
        return output_path

if __name__ == "__main__":
    download_titanic_dataset()