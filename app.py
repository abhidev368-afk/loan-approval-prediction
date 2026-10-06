import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💳")


@st.cache_resource
def load_model():
  return joblib.load("loan_model.pkl")


model = load_model()

st.title("💳 Loan Approval Prediction System")
st.write(
    "Provide the applicant's financial and personal details below to check"
    " eligibility."
)

with st.form("loan_form"):
  col1, col2 = st.columns(2)

  with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

  with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input(
        "Coapplicant Income", min_value=0, value=0
    )
    loan_amount = st.number_input(
        "Loan Amount (in thousands)", min_value=0, value=150
    )
    loan_amount_term = st.number_input(
        "Loan Amount Term (in days)", min_value=0, value=360
    )

  credit_history = st.selectbox(
      "Credit History", [1.0, 0.0], help="1.0: Good, 0.0: Bad"
  )
  property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

  submit = st.form_submit_button("Predict")

if submit:
  g_val = 1 if gender == "Male" else 0
  m_val = 1 if married == "Yes" else 0
  edu_val = 0 if education == "Graduate" else 1
  emp_val = 1 if self_employed == "Yes" else 0
  dep_val = int(dependents.replace("+", "")) if dependents != "3+" else 3
  prop_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
  prop_val = prop_map[property_area]

  features = np.array([[
      g_val,
      m_val,
      dep_val,
      edu_val,
      emp_val,
      applicant_income,
      coapplicant_income,
      loan_amount,
      loan_amount_term,
      credit_history,
      prop_val,
  ]])

  prediction = model.predict(features)

  if prediction[0] == 1:
    st.success("Congratulations! The Loan is Approved.")
  else:
    st.error("Sorry, the Loan is Rejected.")