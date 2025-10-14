import streamlit as st
import numpy as np
import joblib as joblib

st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="📝"
)

st.title('Lung Cancer Prediction')
st.markdown('- Model: Gradient Boosting Classifier')
st.markdown('- Library: scikit-learn')


@st.cache_resource
def load_model():
    return joblib.load("cancer_model.pkl")

model = load_model()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input(label = "Age",
                                min_value=14,
                                max_value=90,
                                step=1,
                                placeholder="Place your age...")
    
    height = st.number_input(label = "Height",
                                min_value=100.0,
                                max_value=250.0,
                                step=0.1,
                                placeholder="Place your height...")
    
    weight = st.number_input(label = "Weight",
                                min_value=30.0,
                                max_value=300.0,
                                step=0.1,
                                placeholder="Place your Weight...")
    
    smoking = st.selectbox("Amount of cigarettes",
                            ("Not smoking","less than 1 per week", "1-2 per week", "1-5 per day", "6-10 per day", "11-20 per day", "21-30 per day", "more than 30 per day"),
                            key=(1))
    
    passive_smoking = st.selectbox("Amount of passive smoke exposure you experience",
                            ("No Exposure", "Rare Exposure", "Occasional Exposure", "Low Exposure", "Moderate Exposure", "High Exposure", "Very High Exposure", "Extreme Exposure"),
                            key=(2))
    
    air_pollution = st.selectbox("Where do you live",
                            ("Rural, little industrial or vehicular activity", "Suburban, little industrial activity", "Small town, some pollution", "Urban, vehicular and some industries", "Large city, significant vehicles and some industries", "Large city, high vehicles and many industries", "Industrial, many factories and heavy vehicles", "Heavy pollution from various sources"),
                            key=(3))
    
    occupation = st.selectbox("Occupation",
                            ("Office Worker", "Retail Worker", "Teacher/Professor", "Public Transport Worker", "Doctor/Healthcare Worker", "Construction Worker", "Manufacturing Industry Worker", "Mining Worker"),
                            key=(4))
    
with col2: 
    genetic_risk = st.selectbox("Genetic risk in your family",
                            ("No family history", "Lung cancer in distant relatives", "Lung cancer in one close relative", "Lung cancer in one middle-aged close relative", "Lung cancer in multiple close relatives or one young close relative", "Lung cancer in several middle-aged close relatives or multiple young close relatives", "Lung cancer in many close relatives, including young ones"),
                            key=(5))
            
    coughing_of_blood = st.selectbox("Coughing of Blood",
                                        ["None",
                                        "Very mild or rare",
                                        "Mild, few times/month",
                                        "Moderate, few times/week",
                                        "Frequent, may need attention",
                                        "Frequent, may indicate serious issue",
                                        "Severe, needs immediate attention",
                                        "Very severe, critical condition",
                                        "Life-threatening"],
                                        key=6)
    
    shortness_of_breath= st.selectbox("Shortness of Breath",
                                        ["None",
                                            "During strenuous activity",
                                            "During light activity",
                                            "Periodic daily activities",
                                            "Frequent, may need attention",
                                            "May need care",
                                            "Severe, needs immediate attention",
                                            "Life-threatening, needs immediate attention",
                                            "May need assistance"],
                                        key=7)

    chest_pain = st.selectbox("Chest Pain",
                                ["None",
                                "Mild occasional pain",
                                "Mild pain several times/month",
                                "Moderate, several times/week",
                                "Frequent, may need attention",
                                "May indicate issue",
                                "Severe, needs immediate attention",
                                "Life-threatening, needs immediate attention",
                                "Needs immediate attention"],
                                key=8)


    weight_loss = st.selectbox("Weight Loss Category",
                                ["< 2 kg, several weeks/months",
                                    "2-4 kg, several weeks/months",
                                    "3-5 kg, several weeks/months",
                                    "5-7 kg, several weeks/months",
                                    "7-10 kg, several months",
                                    "> 10-15 kg, several months",
                                    "> 15-20 kg, several months",
                                    "> 20 kg, several months"])

    clubbing_of_finger_nails = st.selectbox("Clubbing of Finger Nails",
                                                ["None",
                                                "Mild, possibly due to other factors",
                                                "Mild, not significant",
                                                "Moderate, within normal limits",
                                                "Requires attention",
                                                "Suspicious, may indicate issue",
                                                "Requires medical attention",
                                                "Needs immediate medical attention",
                                                "Life-threatening"],
                                                key=10)

    

def bmi(height, weight):
    height = height/100
    bmi = float(weight / (height * height))
    if bmi < 16.0:
        bmi2 = 1
    elif bmi >=16.0 and bmi < 17.0:
        bmi2 = 2
    elif bmi >= 17.0 and bmi < 18.5:
        bmi2 = 3
    elif bmi >= 18.5 and bmi < 25.0:
        bmi2 = 4
    elif bmi >= 25.0 and bmi < 30.0:
        bmi2 = 5
    elif bmi >= 30.0 and bmi < 35.0:
        bmi2 = 6
    else:
        bmi2 = 7
    
    return bmi2

def smoking_status(selected):
    if selected == 'Not smoking':
        return 1
    elif selected == "less than 1 per week":
        return 2
    elif selected == "1-2 per week":
        return 3
    elif selected == "1-5 per day":
        return 4
    elif selected == "6-10 per day":
        return 5
    elif selected == "11-20 per day":
        return 6
    elif selected == "21-30 per day":
        return 7
    else:
        return 8

def passive_smoking_status(selected):
    if selected == "No Exposure":
        return 1
    elif selected == "Rare Exposure":
        return 2
    elif selected == "Occasional Exposure":
        return 3
    elif selected == "Low Exposure":
        return 4
    elif selected == "Moderate Exposure":
        return 5
    elif selected == "High Exposure":
        return 6
    elif selected == "Very High Exposure":
        return 7
    else:
        return 8
    
def pollution(selected):
    if selected == "Rural, little industrial or vehicular activity":
        return 1
    elif selected == "Suburban, little industrial activity":
        return 2
    elif selected == "Small town, some pollution":
        return 3
    elif selected == "Urban, vehicular and some industries":
        return 4
    elif selected == "Large city, significant vehicles and some industries":
        return 5
    elif selected == "Large city, high vehicles and many industries":
        return 6
    elif selected == "Industrial, many factories and heavy vehicles":
        return 7
    else:
        return 8

def occupation_status(selected):
    if selected == "Office Worker":
        return 1
    elif selected == "Retail Worker":
        return 2
    elif selected == "Teacher/Professor":
        return 3
    elif selected == "Public Transport Worker":
        return 4
    elif selected == "Doctor/Healthcare Worker":
        return 5
    elif selected == "Construction Worker":
        return 6
    elif selected == "Manufacturing Industry Worker":
        return 7
    else:
        return 8
    
def genetic_risk_status(selected):
    if selected == "No family history":
        return 1
    elif selected == "Lung cancer in distant relatives":
        return 2
    elif selected == "Lung cancer in one close relative":
        return 3
    elif selected == "Lung cancer in one middle-aged close relative":
        return 4
    elif selected == "Lung cancer in multiple close relatives or one young close relative":
        return 5
    elif selected == "Lung cancer in several middle-aged close relatives or multiple young close relatives":
        return 6
    else:
        return 7
    
def get_category_value(category):
    if category == "None":
        return 1
    elif category == "Only during strenuous activity":
        return 2
    elif category == "During light activity":
        return 3
    elif category == "Periodic daily activities":
        return 4
    elif category == "Frequent, may need attention":
        return 5
    elif category == "During light activity, may need care":
        return 6
    elif category == "Frequent and severe, needs immediate attention":
        return 7
    elif category == "Severe, life-threatening, needs immediate attention":
        return 8
    else:
        return 9
    
def categorize_coughing_of_blood(category):
    if category == "None":
        return 0
    elif category == "Very mild or rare":
        return 1
    elif category == "Mild, few times/month":
        return 2
    elif category == "Moderate, few times/week":
        return 3
    elif category == "Frequent, may need attention":
        return 4
    elif category == "Frequent, may indicate serious issue":
        return 5
    elif category == "Severe, needs immediate attention":
        return 6
    elif category == "Very severe, critical condition":
        return 7
    elif category == "Life-threatening":
        return 8

def categorize_shortness_of_breath(category):
    if category == "None":
        return 0
    elif category == "During strenuous activity":
        return 1
    elif category == "During light activity":
        return 2
    elif category == "Periodic daily activities":
        return 3
    elif category == "Frequent, may need attention":
        return 4
    elif category == "May need care":
        return 5
    elif category == "Severe, needs immediate attention":
        return 6
    elif category == "Life-threatening, needs immediate attention":
        return 7
    elif category == "May need assistance":
        return 8

def categorize_chest_pain(category):
    if category == "None":
        return 0
    elif category == "Mild occasional pain":
        return 1
    elif category == "Mild pain several times/month":
        return 2
    elif category == "Moderate, several times/week":
        return 3
    elif category == "Frequent, may need attention":
        return 4
    elif category == "May indicate issue":
        return 5
    elif category == "Severe, needs immediate attention":
        return 6
    elif category == "Life-threatening, needs immediate attention":
        return 7
    elif category == "Needs immediate attention":
        return 8

def categorize_weight_loss(category):
    if "< 2 kg" in category:
        return 1
    elif "2-4 kg" in category:
        return 2
    elif "3-5 kg" in category:
        return 3
    elif "5-7 kg" in category:
        return 4
    elif "7-10 kg" in category:
        return 5
    elif "> 10-15 kg" in category:
        return 6
    elif "> 15-20 kg" in category:
        return 7
    elif "> 20 kg" in category:
        return 8

def categorize_clubbing_of_finger_nails(category):
    if category == "None":
        return 0
    elif category == "Mild, possibly due to other factors":
        return 1
    elif category == "Mild, not significant":
        return 2
    elif category == "Moderate, within normal limits":
        return 3
    elif category == "Requires attention":
        return 4
    elif category == "Suspicious, may indicate issue":
        return 5
    elif category == "Requires medical attention":
        return 6
    elif category == "Needs immediate medical attention":
        return 7
    elif category == "Life-threatening":
        return 8

if st.button("Submit"):
    Obesity = bmi(height, weight)
    Smoking = smoking_status(smoking)
    Passive_smoking = passive_smoking_status(passive_smoking)
    Air_pollution = pollution(air_pollution)
    Occupation = occupation_status(occupation)
    Genetic_risk = genetic_risk_status(genetic_risk)
    Coughing_of_blood = categorize_coughing_of_blood(coughing_of_blood)
    Shortness_of_breath = categorize_shortness_of_breath(shortness_of_breath)
    Chest_pain = categorize_chest_pain(chest_pain)
    Weight_loss = categorize_weight_loss(weight_loss)
    Clubbing_of_finger_nails = categorize_clubbing_of_finger_nails(clubbing_of_finger_nails)
    
    input_data = [[age, Obesity, Smoking, Passive_smoking, Air_pollution, Occupation, Genetic_risk, Coughing_of_blood, Shortness_of_breath, Chest_pain, Weight_loss, Clubbing_of_finger_nails]] 
    
    input_array = np.array(input_data)
    
    prediksi = model.predict(input_data)
    
    if prediksi == 0:
        st.subheader("Low Risk")
        st.write('Indicates a lower likelihood of developing lung cancer, often associated with factors such as younger age, non-smoking status, and minimal exposure to environmental hazards.')
    elif prediksi == 1:
        st.subheader("Medium Risk")
        st.write('Suggests a moderate probability of developing the disease, potentially influenced by a combination of factors like age, smoking history, and genetic predisposition. ')
    else:
        st.subheader("High Risk")
        st.write('Signifies a significantly elevated chance of developing lung cancer, commonly observed in older individuals with a history of heavy smoking, genetic mutations, or prolonged exposure to carcinogens.')