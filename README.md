# House Price Prediction (DVC + Flask)

This repository contains an end-to-end Machine Learning pipeline for predicting house prices. It integrates **DVC (Data Version Control)** for pipeline management and **Flask** for serving the model via a web interface.

## 🚀 Features
* **Modular Pipeline:** Managed via `dvc.yaml` for reproducible experiments.
* **Data Versioning:** Tracks large datasets and model weights without cluttering Git.
* **Web Deployment:** A lightweight Flask API to get real-time predictions.
* **Config Driven:** All hyperparameters and file paths are managed in `params.yaml`.

---

## 🛠️ Project Structure
text
├── data/               # Data directory (tracked by DVC)
├── models/             # Saved model artifacts
├── src/                # Source code
│   ├── stage_01_prepare.py
│   ├── stage_02_train.py
│   └── stage_03_evaluate.py
├── templates/          # Flask HTML templates
├── app.py              # Flask application entry point
├── dvc.yaml            # DVC pipeline stages
├── params.yaml         # Parameters for training
└── requirements.txt    # Python dependencies    


⚙️ Getting Started
1. Environment Setup
Clone the repository and create a virtual environment:

Bash
git clone [https://github.com/alybinmurtaza/houseprice-dvc-flask.git](https://github.com/alybinmurtaza/houseprice-dvc-flask.git)
cd houseprice-dvc-flask
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Reproduce the Pipeline
If you have DVC installed, you can trigger the entire workflow (data prep to evaluation) with:

Bash
dvc repro
3. Run the Web App
Start the Flask server to view the UI:

Bash
python app.py
Visit http://127.0.0.1:5000 in your browser.

📊 MLOps Workflow
Data Ingestion: Raw data is pulled and versioned.

Training: Scikit-learn models are trained based on params.yaml.

Evaluation: Metrics are logged to track model performance.

Deployment: The best model is loaded by Flask for inference.

🤝 Contributing
Feel free to open an issue or submit a pull request if you want to improve the model or the UI.

Author: Ali Murtaza


---

### Why it might have looked "unformatted" before:
* **Missing File Extension:** Ensure your file ends in `.md`.
* **Preview Mode:** If you are using an editor like VS Code, press `Ctrl+Shift+V` to see the rendered version.
* **GitHub View:** Once you commit and push `README.md` to the root of your GitHub.
