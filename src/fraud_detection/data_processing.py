import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data(path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    return pd.read_csv(path)

def create_pipeline() -> Pipeline:
    """Creates a preprocessing pipeline for the transaction data."""
    # The 'Time' and 'Amount' columns need scaling.
    # The 'V' columns are already scaled in this specific dataset.
    return Pipeline([
        ('scaler', StandardScaler())
    ])

def split_data(df: pd.DataFrame, target: str, test_size: float, random_state: int):
    """Splits data into features and target, then into training and testing sets."""
    X = df.drop(columns=[target])
    y = df[target]
    
    # Use stratify to maintain the same class distribution in train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test