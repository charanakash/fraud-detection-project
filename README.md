Credit Card Fraud Detection Project - Quickstart
A machine learning project to train and compare models for detecting credit card fraud.

Setup
First, clone the repository. Then, run these commands from the project directory:

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Running the Project
Start MLflow UI
Open a new terminal, activate the environment, and run:

mlflow ui

View the dashboard at http://127.0.0.1:5000

Run Training
In your original terminal, run the training script:

python -m src.fraud_detection.train
