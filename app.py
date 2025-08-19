import streamlit as st
import numpy as np
import joblib

class RBFSVM:
    def __init__(self, C=1.0, gamma=0.5, lr=0.001, n_iter=1000):
        self.C = C
        self.gamma = gamma
        self.lr = lr
        self.n_iter = n_iter
        self.alpha = None
        self.X = None
        self.y = None
        self.b = 0

    def rbf_kernel(self, X1, X2):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * sq_dists)

    def fit(self, X, y):
        m, n = X.shape
        y = np.where(y == 1, 1, -1)
        self.X = X
        self.y = y
        self.alpha = np.zeros(m)

        K = self.rbf_kernel(X, X)

        for _ in range(self.n_iter):
            for i in range(m):
                margin = np.sum(self.alpha * self.y * K[:, i]) + self.b
                if self.y[i] * margin < 1:
                    self.alpha[i] += self.lr * (1 - self.y[i] * margin)
                    self.alpha[i] = min(max(self.alpha[i], 0), self.C)
                    self.b += self.lr * self.y[i]

    def decision_function(self, X):
        K = self.rbf_kernel(X, self.X)
        return np.dot(K, self.alpha * self.y) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


class RBFSVM_OVR:
    def __init__(self, C=1.0, gamma=0.5, lr=0.001, n_iter=1000):
        self.models = []
        self.classes_ = []
        self.C = C
        self.gamma = gamma
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            model = RBFSVM(C=self.C, gamma=self.gamma, lr=self.lr, n_iter=self.n_iter)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        decision_scores = np.array([model.decision_function(X) for model in self.models])
        return self.classes_[np.argmax(decision_scores, axis=0)]



scaler = joblib.load("scaler.pkl")
model = joblib.load("RBF_SVM_model.pkl")


species_dict = {0: "Setosa 🌸", 1: "Versicolor 🌼", 2: "Virginica 🌺"}


st.set_page_config(page_title="Flower Species Classifier", page_icon="🌹", layout="centered")


st.markdown("<h1 style='text-align: center;'>🌹 Flower Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Predict Setosa, Versicolor, or Virginica from flower features.</h3>", unsafe_allow_html=True)

st.subheader("Enter Flower Measurements")

sepal_length = st.slider("Sepal Length (cm)", min_value=4.3, max_value=7.9, value=5.8, step=0.1)
sepal_width  = st.slider("Sepal Width (cm)",  min_value=2.0, max_value=4.4, value=3.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=6.9, value=4.3, step=0.1)
petal_width  = st.slider("Petal Width (cm)",  min_value=0.1, max_value=2.5, value=1.3, step=0.1)


features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
features_scaled = scaler.transform(features)


if st.button("🔍 Predict Species"):
    prediction = model.predict(features_scaled)[0]
    species_name = species_dict[prediction]

    st.markdown("---")
    st.subheader("🌟 Prediction Result")
    st.success(f"The predicted species is: **{species_name}**")
