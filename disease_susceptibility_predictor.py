import streamlit as st
import pandas as pd
import hashlib
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import re
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Disease Susceptibility Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database initialization
def init_db():
    """Initialize SQLite databases for users and predictions"""
    # Users database
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            date_of_birth DATE,
            gender TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    
    # Predictions database
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            patient_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            blood_pressure_systolic INTEGER,
            blood_pressure_diastolic INTEGER,
            cholesterol_total REAL,
            cholesterol_hdl REAL,
            cholesterol_ldl REAL,
            blood_sugar_fasting REAL,
            blood_sugar_random REAL,
            hba1c REAL,
            bmi REAL,
            smoking_status TEXT,
            alcohol_consumption TEXT,
            exercise_frequency TEXT,
            family_history TEXT,
            predicted_diseases TEXT,
            risk_scores TEXT,
            recommendations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Authentication functions
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hash_password(password) == hashed

def create_user(username, email, password, full_name, dob, gender):
    """Create new user account"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    try:
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, date_of_birth, gender)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, email, password_hash, full_name, dob, gender))
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError as e:
        conn.close()
        if "username" in str(e):
            return False, "Username already exists!"
        elif "email" in str(e):
            return False, "Email already registered!"
        else:
            return False, "Registration failed!"

def authenticate_user(username, password):
    """Authenticate user login"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, password_hash, full_name, email FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    
    if user and verify_password(password, user[1]):
        # Update last login
        cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.now(), user[0]))
        conn.commit()
        conn.close()
        return True, {"id": user[0], "username": username, "full_name": user[2], "email": user[3]}
    
    conn.close()
    return False, None

# Disease prediction logic
def calculate_disease_risk(data):
    """Calculate disease susceptibility based on medical data"""
    risks = {}
    recommendations = []
    
    # Cardiovascular Disease Risk
    cv_risk = 0
    if data['age'] > 45:
        cv_risk += 20
    if data['gender'].lower() == 'male':
        cv_risk += 10
    if data['blood_pressure_systolic'] > 140 or data['blood_pressure_diastolic'] > 90:
        cv_risk += 25
        recommendations.append("Monitor blood pressure regularly and consider medication if advised by doctor")
    if data['cholesterol_total'] > 240:
        cv_risk += 20
        recommendations.append("Follow a low-cholesterol diet and consider statin therapy")
    if data['cholesterol_hdl'] < 40:
        cv_risk += 15
    if data['bmi'] > 30:
        cv_risk += 15
        recommendations.append("Weight management through diet and exercise")
    if data['smoking_status'] in ['Current smoker', 'Heavy smoker']:
        cv_risk += 30
        recommendations.append("Quit smoking immediately - seek professional help if needed")
    if 'heart disease' in data.get('family_history', '').lower():
        cv_risk += 20
    
    risks['Cardiovascular Disease'] = min(cv_risk, 100)
    
    # Diabetes Risk
    diabetes_risk = 0
    if data['age'] > 45:
        diabetes_risk += 15
    if data['bmi'] > 25:
        diabetes_risk += 20
    if data['blood_sugar_fasting'] > 126:
        diabetes_risk += 40
        recommendations.append("Immediate consultation with endocrinologist required")
    elif data['blood_sugar_fasting'] > 100:
        diabetes_risk += 25
        recommendations.append("Monitor blood sugar levels and follow diabetic-friendly diet")
    if data.get('hba1c', 0) > 6.5:
        diabetes_risk += 35
    if 'diabetes' in data.get('family_history', '').lower():
        diabetes_risk += 25
    if data['exercise_frequency'] in ['Never', 'Rarely']:
        diabetes_risk += 15
        recommendations.append("Increase physical activity to at least 150 minutes per week")
    
    risks['Type 2 Diabetes'] = min(diabetes_risk, 100)
    
    # Hypertension Risk
    hypertension_risk = 0
    if data['blood_pressure_systolic'] > 120 or data['blood_pressure_diastolic'] > 80:
        hypertension_risk += 30
    if data['age'] > 50:
        hypertension_risk += 20
    if data['bmi'] > 25:
        hypertension_risk += 15
    if data['alcohol_consumption'] in ['Heavy drinker', 'Daily']:
        hypertension_risk += 20
        recommendations.append("Reduce alcohol consumption")
    if 'hypertension' in data.get('family_history', '').lower():
        hypertension_risk += 25
    
    risks['Hypertension'] = min(hypertension_risk, 100)
    
    # Metabolic Syndrome Risk
    metabolic_risk = 0
    if data['bmi'] > 30:
        metabolic_risk += 25
    if data['blood_sugar_fasting'] > 100:
        metabolic_risk += 20
    if data['cholesterol_hdl'] < 50:
        metabolic_risk += 15
    if data['blood_pressure_systolic'] > 130:
        metabolic_risk += 20
    
    risks['Metabolic Syndrome'] = min(metabolic_risk, 100)
    
    # Stroke Risk
    stroke_risk = 0
    if data['age'] > 55:
        stroke_risk += 20
    if data['blood_pressure_systolic'] > 140:
        stroke_risk += 25
    if data['smoking_status'] in ['Current smoker', 'Heavy smoker']:
        stroke_risk += 25
    if 'stroke' in data.get('family_history', '').lower():
        stroke_risk += 30
    
    risks['Stroke'] = min(stroke_risk, 100)
    
    # General recommendations
    if not recommendations:
        recommendations.append("Maintain current healthy lifestyle")
    recommendations.append("Regular health check-ups every 6-12 months")
    recommendations.append("Follow a balanced diet rich in fruits and vegetables")
    
    return risks, recommendations

def save_prediction(user_id, patient_data, risks, recommendations):
    """Save prediction to database"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # Convert risks and recommendations to strings
    risks_str = str(risks)
    recommendations_str = "; ".join(recommendations)
    predicted_diseases = ", ".join([f"{disease}: {score}%" for disease, score in risks.items()])
    
    cursor.execute('''
        INSERT INTO predictions (
            user_id, patient_name, age, gender, blood_pressure_systolic, blood_pressure_diastolic,
            cholesterol_total, cholesterol_hdl, cholesterol_ldl, blood_sugar_fasting, 
            blood_sugar_random, hba1c, bmi, smoking_status, alcohol_consumption, 
            exercise_frequency, family_history, predicted_diseases, risk_scores, recommendations
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, patient_data['patient_name'], patient_data['age'], patient_data['gender'],
        patient_data['blood_pressure_systolic'], patient_data['blood_pressure_diastolic'],
        patient_data['cholesterol_total'], patient_data['cholesterol_hdl'], patient_data['cholesterol_ldl'],
        patient_data['blood_sugar_fasting'], patient_data['blood_sugar_random'], patient_data.get('hba1c', 0),
        patient_data['bmi'], patient_data['smoking_status'], patient_data['alcohol_consumption'],
        patient_data['exercise_frequency'], patient_data['family_history'], predicted_diseases, risks_str, recommendations_str
    ))
    
    conn.commit()
    conn.close()

def get_user_predictions(user_id):
    """Get all predictions for a user"""
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql_query('''
        SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC
    ''', conn, params=(user_id,))
    conn.close()
    return df

# Streamlit app
def main():
    init_db()
    
    # Session state management
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    
    if not st.session_state.logged_in:
        auth_page()
    else:
        main_app()

def auth_page():
    """Authentication page with login and registration"""
    st.title("🏥 Disease Susceptibility Predictor")
    st.markdown("### Predict Your Health Risks Based on Medical Reports")
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", type="primary", use_container_width=True):
                if username and password:
                    success, user_info = authenticate_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_info = user_info
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password!")
                else:
                    st.error("Please enter both username and password!")
        
        with col2:
            if st.button("Demo Login", use_container_width=True):
                # Create demo user if not exists
                demo_success, _ = create_user("demo", "demo@example.com", "demo123", "Demo User", "1990-01-01", "Other")
                success, user_info = authenticate_user("demo", "demo123")
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_info = user_info
                    st.success("Demo login successful!")
                    st.rerun()
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("registration_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                reg_username = st.text_input("Username")
                reg_email = st.text_input("Email")
                reg_password = st.text_input("Password", type="password")
                reg_confirm_password = st.text_input("Confirm Password", type="password")
            
            with col2:
                reg_full_name = st.text_input("Full Name")
                reg_dob = st.date_input("Date of Birth", value=datetime(1990, 1, 1))
                reg_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            if st.form_submit_button("Create Account", type="primary"):
                if not all([reg_username, reg_email, reg_password, reg_full_name]):
                    st.error("Please fill in all required fields!")
                elif reg_password != reg_confirm_password:
                    st.error("Passwords do not match!")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters long!")
                elif not re.match(r'^[^@]+@[^@]+\.[^@]+$', reg_email):
                    st.error("Please enter a valid email address!")
                else:
                    success, message = create_user(reg_username, reg_email, reg_password, 
                                                 reg_full_name, reg_dob, reg_gender)
                    if success:
                        st.success(message)
                        st.info("Please login with your new account!")
                    else:
                        st.error(message)

def main_app():
    """Main application after login"""
    # Sidebar
    st.sidebar.title(f"Welcome, {st.session_state.user_info['full_name']}!")
    
    page = st.sidebar.selectbox("Choose a page:", [
        "🔬 Disease Prediction", 
        "📊 My Predictions History", 
        "📈 Health Analytics",
        "ℹ️ About Disease Risks"
    ])
    
    if st.sidebar.button("Logout", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.user_info = None
        st.rerun()
    
    if page == "🔬 Disease Prediction":
        prediction_page()
    elif page == "📊 My Predictions History":
        history_page()
    elif page == "📈 Health Analytics":
        analytics_page()
    else:
        info_page()

def prediction_page():
    """Disease prediction page"""
    st.title("🔬 Disease Susceptibility Prediction")
    st.markdown("### Enter Patient Information and Medical Reports")
    st.markdown("---")
    
    with st.form("prediction_form"):
        # Patient Information
        st.subheader("👤 Patient Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            patient_name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
        
        with col3:
            smoking_status = st.selectbox("Smoking Status", 
                ["Never smoked", "Former smoker", "Current smoker", "Heavy smoker"])
            alcohol_consumption = st.selectbox("Alcohol Consumption", 
                ["Never", "Occasionally", "Weekly", "Daily", "Heavy drinker"])
        
        # Medical Reports
        st.subheader("🩺 Blood Pressure")
        col1, col2 = st.columns(2)
        with col1:
            bp_systolic = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120)
        with col2:
            bp_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=150, value=80)
        
        st.subheader("🩸 Cholesterol Levels (mg/dL)")
        col1, col2, col3 = st.columns(3)
        with col1:
            cholesterol_total = st.number_input("Total Cholesterol", min_value=100, max_value=400, value=200)
        with col2:
            cholesterol_hdl = st.number_input("HDL (Good)", min_value=20, max_value=100, value=50)
        with col3:
            cholesterol_ldl = st.number_input("LDL (Bad)", min_value=50, max_value=300, value=100)
        
        st.subheader("🍯 Blood Sugar Levels")
        col1, col2, col3 = st.columns(3)
        with col1:
            blood_sugar_fasting = st.number_input("Fasting Glucose (mg/dL)", min_value=60, max_value=400, value=90)
        with col2:
            blood_sugar_random = st.number_input("Random Glucose (mg/dL)", min_value=70, max_value=500, value=120)
        with col3:
            hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=5.5, step=0.1)
        
        st.subheader("🏃 Lifestyle & Family History")
        col1, col2 = st.columns(2)
        with col1:
            exercise_frequency = st.selectbox("Exercise Frequency", 
                ["Daily", "4-6 times/week", "2-3 times/week", "Once a week", "Rarely", "Never"])
        with col2:
            family_history = st.text_area("Family History of Diseases", 
                placeholder="e.g., diabetes, heart disease, hypertension, stroke...")
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Disease Susceptibility", type="primary")
        
        if submitted:
            if not patient_name:
                st.error("Please enter patient name!")
            else:
                # Prepare data
                patient_data = {
                    'patient_name': patient_name,
                    'age': age,
                    'gender': gender,
                    'bmi': bmi,
                    'blood_pressure_systolic': bp_systolic,
                    'blood_pressure_diastolic': bp_diastolic,
                    'cholesterol_total': cholesterol_total,
                    'cholesterol_hdl': cholesterol_hdl,
                    'cholesterol_ldl': cholesterol_ldl,
                    'blood_sugar_fasting': blood_sugar_fasting,
                    'blood_sugar_random': blood_sugar_random,
                    'hba1c': hba1c,
                    'smoking_status': smoking_status,
                    'alcohol_consumption': alcohol_consumption,
                    'exercise_frequency': exercise_frequency,
                    'family_history': family_history
                }
                
                # Calculate risks
                risks, recommendations = calculate_disease_risk(patient_data)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Disease Susceptibility Results")
                
                # Risk visualization
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(risks.keys()),
                        y=list(risks.values()),
                        marker_color=['red' if v >= 70 else 'orange' if v >= 40 else 'green' for v in risks.values()],
                        text=[f"{v}%" for v in risks.values()],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    title="Disease Risk Assessment",
                    xaxis_title="Diseases",
                    yaxis_title="Risk Percentage (%)",
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk levels
                st.subheader("🎯 Risk Levels")
                for disease, risk in risks.items():
                    if risk >= 70:
                        st.error(f"🔴 **{disease}**: {risk}% - HIGH RISK")
                    elif risk >= 40:
                        st.warning(f"🟠 **{disease}**: {risk}% - MODERATE RISK")
                    else:
                        st.success(f"🟢 **{disease}**: {risk}% - LOW RISK")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                # Save prediction
                save_prediction(st.session_state.user_info['id'], patient_data, risks, recommendations)
                st.success("✅ Prediction saved to your history!")

def history_page():
    """User's prediction history"""
    st.title("📊 My Prediction History")
    st.markdown("---")
    
    df = get_user_predictions(st.session_state.user_info['id'])
    
    if df.empty:
        st.info("No predictions found. Make your first prediction!")
    else:
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(df))
        with col2:
            avg_age = df['age'].mean()
            st.metric("Average Patient Age", f"{avg_age:.0f}")
        with col3:
            latest_date = df['created_at'].iloc[0].split()[0]
            st.metric("Latest Prediction", latest_date)
        with col4:
            unique_patients = df['patient_name'].nunique()
            st.metric("Unique Patients", unique_patients)
        
        st.markdown("---")
        
        # Display predictions
        for _, row in df.iterrows():
            with st.expander(f"🔬 {row['patient_name']} - {row['created_at'].split()[0]}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Patient Info:**")
                    st.write(f"Age: {row['age']}, Gender: {row['gender']}")
                    st.write(f"BMI: {row['bmi']}")
                    st.write(f"BP: {row['blood_pressure_systolic']}/{row['blood_pressure_diastolic']}")
                    
                with col2:
                    st.write("**Predicted Diseases:**")
                    st.write(row['predicted_diseases'])
                
                st.write("**Recommendations:**")
                recommendations = row['recommendations'].split('; ')
                for rec in recommendations:
                    st.write(f"• {rec}")

def analytics_page():
    """Health analytics dashboard"""
    st.title("📈 Health Analytics Dashboard")
    st.markdown("---")
    
    df = get_user_predictions(st.session_state.user_info['id'])
    
    if df.empty:
        st.info("No data available for analytics. Make some predictions first!")
    else:
        # Trends over time
        st.subheader("📅 Prediction Trends")
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        daily_predictions = df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(daily_predictions, x='date', y='count', title="Predictions Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Age distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 Age Distribution")
            fig = px.histogram(df, x='age', bins=10, title="Patient Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚖️ Gender Distribution")
            gender_counts = df['gender'].value_counts()
            fig = px.pie(values=gender_counts.values, names=gender_counts.index, title="Gender Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # BMI analysis
        st.subheader("📊 BMI Analysis")
        fig = px.scatter(df, x='age', y='bmi', color='gender', title="BMI vs Age by Gender")
        st.plotly_chart(fig, use_container_width=True)

def info_page():
    """Information about disease risks"""
    st.title("ℹ️ About Disease Risks")
    st.markdown("---")
    
    st.subheader("🏥 How Our Prediction Works")
    st.markdown("""
    Our disease susceptibility prediction model analyzes various health parameters to assess your risk for common diseases:
    
    **Factors Considered:**
    - Age and Gender
    - Blood Pressure readings
    - Cholesterol levels (Total, HDL, LDL)
    - Blood Sugar levels (Fasting, Random, HbA1c)
    - BMI and lifestyle factors
    - Smoking and alcohol consumption
    - Exercise frequency
    - Family medical history
    """)
    
    st.subheader("🎯 Diseases We Assess")
    
    disease_info = {
        "Cardiovascular Disease": {
            "description": "Heart and blood vessel diseases including heart attacks and strokes",
            "risk_factors": "High blood pressure, high cholesterol, smoking, obesity, family history",
            "prevention": "Regular exercise, healthy diet, quit smoking, manage stress"
        },
        "Type 2 Diabetes": {
            "description": "A condition where blood sugar levels are consistently high",
            "risk_factors": "Obesity, sedentary lifestyle, family history, age over 45",
            "prevention": "Maintain healthy weight, regular physical activity, balanced diet"
        },
        "Hypertension": {
            "description": "High blood pressure that can lead to serious health complications",
            "risk_factors": "Age, obesity, excessive salt intake, lack of exercise, stress",
            "prevention": "Low-sodium diet, regular exercise, maintain healthy weight, limit alcohol"
        },
        "Metabolic Syndrome": {
            "description": "A cluster of conditions that increase risk of heart disease and diabetes",
            "risk_factors": "Abdominal obesity, insulin resistance, high blood pressure",
            "prevention": "Weight management, regular exercise, healthy eating habits"
        },
        "Stroke": {
            "description": "Interrupted blood flow to the brain causing cell damage",
            "risk_factors": "High blood pressure, smoking, diabetes, heart disease, age",
            "prevention": "Control blood pressure, quit smoking, manage diabetes, stay active"
        }
    }
    
    for disease, info in disease_info.items():
        with st.expander(f"🔍 {disease}"):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Risk Factors:** {info['risk_factors']}")
            st.write(f"**Prevention:** {info['prevention']}")
    
    st.subheader("⚠️ Important Disclaimer")
    st.warning("""
    **This tool is for educational purposes only and should not replace professional medical advice.**
    
    - Our predictions are based on statistical models and general risk factors
    - Individual health conditions may vary significantly
    - Always consult with healthcare professionals for accurate diagnosis
    - Regular medical check-ups are essential regardless of risk scores
    - If you have concerning symptoms, seek immediate medical attention
    """)
    
    st.subheader("🩺 When to Consult a Doctor")
    st.error("""
    **Seek immediate medical attention if you experience:**
    - Chest pain or discomfort
    - Shortness of breath
    - Sudden severe headache
    - Vision changes
    - Numbness or weakness
    - Persistent high blood sugar readings
    - Blood pressure readings consistently above 180/120
    """)

if __name__ == "__main__":
    main()