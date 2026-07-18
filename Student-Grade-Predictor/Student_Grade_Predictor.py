import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Academic Performance Analytics", page_icon="🎓", layout="wide")

st.title("🎓 Student Grade Predictor & Performance Analytics")
st.markdown("Predict final student grades using **Supervised Linear Regression** pipelines.")

DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "student_grades.csv")


def generate_synthetic_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    np.random.seed(42)
    num_students = 250
    attendance = np.random.uniform(60, 100, num_students)
    assignment_scores = attendance * 0.8 + np.random.normal(5, 7, num_students)
    assignment_scores = np.clip(assignment_scores, 40, 100)
    
    midterm = (assignment_scores * 0.7 + attendance * 0.3) + np.random.normal(0, 5, num_students)
    midterm = np.clip(midterm, 40, 100)
    final_grade = (assignment_scores * 0.4 + midterm * 0.4 + attendance * 0.2) + np.random.normal(0, 3, num_students)
    final_grade = np.clip(final_grade, 40, 100)
    
    df = pd.DataFrame({
        'Student_ID': [f"STU{i:03d}" for i in range(1, num_students + 1)],
        'Attendance_Pct': np.round(attendance, 1),
        'Assignment_Avg': np.round(assignment_scores, 1),
        'Midterm_Grade': np.round(midterm, 1),
        'Final_Grade': np.round(final_grade, 1)
    })
    
    df.to_csv(DATA_PATH, index=False)

if not os.path.exists(DATA_PATH):
    generate_synthetic_data()
df = pd.read_csv(DATA_PATH)
@st.cache_resource
def train_regression_model():
    X = df[['Attendance_Pct', 'Assignment_Avg', 'Midterm_Grade']]
    y = df['Final_Grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return model, mae, r2

model, model_mae, model_r2 = train_regression_model()
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Model Performance Metrics")
    
    m_col1, m_col2 = st.columns(2)
    m_col1.metric(label="R-Squared Score (Fit)", value=f"{model_r2:.2f}")
    m_col2.metric(label="Mean Absolute Error", value=f"{model_mae:.2f}%")
    
    st.caption(" **R-Squared** shows how much variance our model accounts for. **MAE** represents the average deviation between predictions and actual targets.")

    st.subheader(" Input Current Student Parameters")
    in_attendance = st.slider("Class Attendance Percentage:", 0.0, 100.0, 85.0, 0.5)
    in_assignments = st.slider("Assignment Average Score:", 0.0, 100.0, 78.0, 0.5)
    in_midterm = st.slider("Midterm Exam Grade Score:", 0.0, 100.0, 75.0, 0.5)
    input_features = np.array([[in_attendance, in_assignments, in_midterm]])
    predicted_grade = model.predict(input_features)[0]
    predicted_grade = np.clip(predicted_grade, 0, 100)

with col2:
    st.subheader(" Prediction Output Matrix")
    if predicted_grade >= 85:
        bg_color, status = "#D4EDDA", "Excellent (Targeting Grade A)"
    elif predicted_grade >= 60:
        bg_color, status = "#CCE5FF", "Clear (Targeting Grade B/C)"
    else:
        bg_color, status = "#F8D7DA", "Critical Risk Profile (Needs Intervention)"
        
    st.markdown(
        f"""
        <div style="background-color:{bg_color}; padding:25px; border-radius:10px; border: 1px solid rgba(0,0,0,0.1)">
            <h4 style="margin:0; color:#333;">Predicted Final Grade Score</h4>
            <h1 style="margin:10px 0; font-size:48px; color:#111;">{predicted_grade:.1f}%</h1>
            <p style="margin:0; font-weight:bold; color:#555;">Status: {status}</p>
        </div>
        """, 
        unsafe_allow_html=True  
    )
    
    st.subheader("📉 Historical Training Records Summary")
    st.dataframe(df.head(7), use_container_width=True)