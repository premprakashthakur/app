
import streamlit as st
import pickle
import numpy as np

# Load model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Fare", 0.0, 600.0, 30.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

sex = 1 if sex == "Male" else 0
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

if st.button("Predict"):
    result = model.predict(input_data)[0]
    st.success("🎉 Survived!" if result == 1 else "❌ Did not survive.")
