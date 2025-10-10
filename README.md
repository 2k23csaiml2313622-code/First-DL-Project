Bank Churn Prediction: Deep Learning ANN Classification & Streamlit Web App

📝 Description
This project aims to predict customer churn in a bank using a Deep Learning model built with an Artificial Neural Network (ANN). The prediction model is deployed with a user-friendly web interface using Streamlit, allowing users to input customer parameters and receive an instant churn prediction.

🌟 Key Features
Deep Learning Model: Utilizes a multi-layered Artificial Neural Network (ANN) for highly accurate binary classification (Churn: Yes/No).

Data Preprocessing: Includes steps like handling categorical variables (One-Hot Encoding), feature scaling (Standardization/Normalization), and managing data imbalance (if applicable).

Interactive Web App: A clean, intuitive front-end built with Streamlit for real-time predictions.

Data Source: Trained and tested on a publically available Bank Customer Churn Dataset from Kaggle.

Environment Management: Uses a Python Virtual Environment (venv) for isolated and reproducible dependency management.

⚙️ Technologies Used-

Deep Learning-	TensorFlow / Keras:	Building and training the ANN model.
Data Science-	Pandas, NumPy:	Data manipulation and numerical operations.
Web Framework-	Streamlit:	Creating the interactive front-end.
Model Persistence-	Pickle / H5:	Saving and loading the trained model and scaler object.
Environment-	Python (3.x), venv:	Project coding and dependency isolation.

Export to Sheets
📂 Project Structure
A concise overview of the directory structure:

Bank_Churn_Prediction/
├── data/
│   └── Churn_Modelling.csv         # The raw dataset
├── model/
│   ├── ann_model.h5                # Saved Keras model file
│   └── scaler.pkl                  # Saved StandardScaler/MinMaxScaler object
├── app/
│   └── app.py                      # Streamlit application script
├── notebooks/
│   └── Churn_Prediction_EDA_ANN.ipynb # Jupyter Notebook for model development
├── requirements.txt                # List of project dependencies
└── README.md
🚀 Getting Started
Follow these steps to set up and run the project locally.

1. Clone the Repository
Bash

git clone https://github.com/2k23csaiml2313622-code/First-DL-Project.git
cd 2k23csaiml2313622-code/First-DL-Project
2. Create and Activate Virtual Environment
It is highly recommended to use a virtual environment.

Bash

# Create the venv
python -m venv venv

# Activate the venv (on Windows)
.\venv\Scripts\activate

# Activate the venv (on macOS/Linux)
source venv/bin/activate
3. Install Dependencies
Install all required packages using the requirements.txt file.

Bash

pip install -r requirements.txt
4. Run the Streamlit Application
Navigate to the app directory (or wherever your main Streamlit file is) and run the app.

Bash

# Assuming your Streamlit file is named app.py and is in the main directory
streamlit run app.py
This command will open the web application in your default browser, usually at http://localhost:8501.

📊 Model Performance
Briefly present the key performance metrics of your final ANN model.

Accuracy-	86.20%
Precision-	78.50%
Recall-	68.90%
F1-Score-	73.40%
Model Architecture-	Input Layer (11 features), Hidden Layers (64, 32 nodes, ReLU), Output Layer (1 node, Sigmoid)

🤝 Contributing
Contributions are welcome! If you have suggestions for improving the model performance or the Streamlit interface, feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.
