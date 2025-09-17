# Breast-Cancer-Prediction-App

An interactive Breast Cancer Prediction Web App built with Python, Streamlit, and Machine Learning.
This project uses a trained ML model to predict whether a breast tumor is Malignant (Cancerous ❌) or Benign (Non-Cancerous ✅) based on 30 medical features.

📌 Project Overview

Breast cancer is one of the most common and life-threatening diseases among women worldwide.
This project demonstrates how Machine Learning can be applied in healthcare by building an interactive prediction system using Streamlit.
The app allows users to input medical feature values through sliders and predicts whether the tumor is Malignant (Cancerous) or Benign (Non-Cancerous).

⚙️ Key Concepts

1.Machine Learning Model 🤖
-A trained classifier (likely Logistic Regression / Random Forest / SVM).
-Predicts binary outcome: Malignant (0) or Benign (1).

2.Feature Scaling 📏
-StandardScaler is used to normalize values so the model can make accurate predictions.

3.Streamlit Framework 🎨
-Provides a web-based UI for user interaction.
-Sliders let users input values for 30 different tumor features.

4.Pickle Serialization 📦
-Model, scaler, and feature names are saved as .pkl files.
-Ensures reproducibility and fast loading without retraining.


🔑 Algorithm & Workflow

1.Data Preprocessing
-Collected breast cancer dataset (e.g., from Sklearn Breast Cancer Dataset).
-Cleaned & normalized data using StandardScaler.

2.Model Training
-Applied classification algorithms (Logistic Regression / Random Forest).
-Split dataset into training and testing.
-Saved trained model using pickle.

3.App Development
-Loaded cancer_model.pkl, scaler.pkl, and feature_names.pkl.
-Created 30 sliders (each representing a feature like radius, texture, perimeter, etc.).
-Scaled input values before prediction.

🛠 Technologies Used
-Python 🐍
-Streamlit 🎨
-Scikit-learn 🤖
-Numpy 🔢
-Pickle 📦

🧭 Process Flow
-Dataset Collection 📂
-Preprocessing & Scaling ⚖️
-Model Training 🤖
-Saving Trained Model 📦
-Streamlit Frontend Development 🎨
-Deployment 🚀

✅ Conclusion

This project shows how AI & Machine Learning can support healthcare by providing early predictions of breast cancer.
Although this tool is not a substitute for professional diagnosis, it demonstrates the potential of data-driven decision support systems.
✨ "Early detection saves lives – technology makes it possible."
