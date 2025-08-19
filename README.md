# **StatMLfromScratch: Statistical Models from Scratch on Iris Dataset**

StatMLfromScratch is a machine learning project focused on building, evaluating, and deploying statistical models entirely **from scratch** without using scikit-learn’s built-in classifiers.  
It demonstrates how machine learning fundamentals can be implemented from first principles, including data exploration, preprocessing, model implementation, evaluation with custom-built metrics, and deployment with Streamlit.

---

## **Demo**

- 🎥 [View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_machinelearning-fromscratch-svm-activity-7363676141431246849-UxQv?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)  
- 🌐 [Try the App Live on Streamlit](https://statmlfromscratch-9rwvc4skrejihrbf4xcmem.streamlit.app/)

![App Demo](https://github.com/rawan-alwadiya/StatMLfromScratch/blob/main/StatMLfromScratch.png)

---

## **Project Overview**

The workflow includes:  
- **Data visualization and exploration**  
- **Data preprocessing (standardization)**  
- Implementation of **four core statistical models from scratch**  
- Evaluation using **custom-built metrics**  
- Deployment of the chosen model via a **Streamlit web application**

---

## **Objective**

Develop and deploy machine learning models built from first principles to classify flower species in the **Iris dataset**, showcasing a full pipeline of exploratory analysis, model building, evaluation, and deployment.

---

## **Dataset**

- **Source**: [Iris Dataset (UCI Repository)](https://archive.ics.uci.edu/ml/datasets/iris)  
- **Samples**: 150  
- **Features**: 4 numerical features (sepal length, sepal width, petal length, petal width)  
- **Target**: 3 flower species (Setosa, Versicolor, Virginica)

---

## **Project Workflow**

- **Exploration & Visualization (EDA)**: Class distributions, feature relationships  
- **Preprocessing**: Standardization of features  
- **Modeling**: Implemented models entirely from scratch  
  - Logistic Regression (OvR)  
  - Linear SVM (OvR)  
  - Polynomial Kernel SVM (OvR)  
  - RBF Kernel SVM (OvR)  
- **Evaluation Metrics (from scratch)**:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  
  - Classification Report  
- **Deployment**: Interactive Streamlit app for real-time species classification

---

## **Performance Results**

**Chosen Model – RBF Kernel SVM (OvR):**  
- **Accuracy**: 96.67%  
- **Precision**: 96.97%  
- **Recall**: 96.67%  
- **F1-score**: 96.66%  

Both the **Polynomial Kernel SVM** and **RBF Kernel SVM** achieved top performance, but the RBF Kernel SVM was selected for deployment due to its robust performance across varying decision boundaries.

---

## **Project Links**

- **GitHub Repository**: [StatMLfromScratch](https://github.com/rawan-alwadiya/StatMLfromScratch)  
- **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/rawanalwadeya/statmlfromscratch-statistical-models-from-scratch?scriptVersionId=256902526)  
- **Live Streamlit App**: [Try it Now](https://statmlfromscratch-9rwvc4skrejihrbf4xcmem.streamlit.app/)

---

## **Tech Stack**

**Languages & Libraries**:  
- Python, Pandas, NumPy  
- Matplotlib, Seaborn  
- Streamlit (Deployment)  

**Techniques**:  
- Logistic Regression (OvR) from scratch  
- SVM (Linear, Polynomial, RBF) from scratch  
- One-vs-Rest (OvR) multiclass classification  
- Custom Evaluation Metrics  
- Streamlit Deployment
