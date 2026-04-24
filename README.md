House Price Prediction with DVC & Flask
This project is an end-to-end Machine Learning application designed to predict house prices. It leverages DVC (Data Version Control) to manage data lineage and ML pipelines, ensuring reproducibility, and uses Flask to provide a user-friendly web interface for real-time predictions.

🚀 Project Overview
The goal of this project is to build a robust regression model that estimates property values based on various features (e.g., area, number of bedrooms, location). By integrating DVC, the project follows a modular "pipeline" approach where each stage (data ingestion, preprocessing, training, evaluation) is version-controlled and reproducible.

✨ Key Features
Data Versioning: Track changes in datasets and models using DVC without bloating the Git repository.

Automated Pipeline: Reproduce the entire workflow with a single command (dvc repro).

Web Interface: A responsive Flask application where users can input house details and get instant price estimates.

Experiment Tracking: Efficiently manage different model versions and hyperparameters.

🛠️ Tech Stack
Language: Python

ML Libraries: Scikit-Learn, Pandas, NumPy

MLOps: DVC (Data Version Control)

Backend: Flask (Web Framework)

Frontend: HTML/CSS (Jinja2 Templates)

📁 Project Structure
Plaintext
├── data/               # Raw and processed datasets (tracked by DVC)
├── models/             # Trained model artifacts (.pkl or .joblib)
├── notebooks/          # Jupyter notebooks for EDA and prototyping
├── src/                # Source code for the ML pipeline
│   ├── stage_01_load.py
│   ├── stage_02_split.py
│   └── stage_03_train.py
├── templates/          # HTML files for the Flask app
├── static/             # CSS and images
├── app.py              # Flask entry point
├── dvc.yaml            # DVC pipeline definition
├── params.yaml         # Hyperparameters and configurations
└── requirements.txt    # Project dependencies
⚙️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/alybinmurtaza/houseprice-dvc-flask.git
cd houseprice-dvc-flask
Create a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Pull the data (if DVC remote is configured):

Bash
dvc pull
🏃 Usage
Running the ML Pipeline
To run the end-to-end data processing and training pipeline:

Bash
dvc repro
Running the Web App
To launch the Flask server locally:

Bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000/.

📊 Model Evaluation
After running the pipeline, you can check the metrics (e.g., RMSE, R²) using:

Bash
dvc metrics show
🤝 Contributing
Contributions are welcome! If you'd like to improve the model accuracy or add new features, please fork the repo and create a pull request.
