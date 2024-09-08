import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import google.generativeai as gemini

gemini.configure(api_key="AIzaSyASUFBrNl_EsBuo8QD2_1HDGZXlcVAiG_o")

# Load the dataset to get feature names and initialize scaler
stroke_data = pd.read_csv("HEART_MODEL/Stroke/stroke_dataset.csv")
X = stroke_data.drop(columns="stroke", axis=1)

# Re-initialize label encoders for categorical columns
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Apply Label Encoding to categorical columns to match training script
le = LabelEncoder()
for col in categorical_columns:
    X[col] = le.fit_transform(X[col])

# Initialize and fit the scaler
scaler = StandardScaler()
scaler.fit(X)  # Fit scaler on the original data

# Load the trained model from the pickle file
with open('HEART_MODEL/Stroke/stroke_prediction.pkl', 'rb') as model_file:
    ensemble_model = pickle.load(model_file)

print("Model loaded successfully.")

# Function to predict stroke risk and provide risk percentage
def predict_stroke(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_df = pd.DataFrame([input_data_as_numpy_array], columns=X.columns)  # Create DataFrame with feature names
    input_data_scaled = scaler.transform(input_data_df)  # Scale input data
    
    # Predict using the ensemble model
    prediction = ensemble_model.predict(input_data_scaled)[0]
    
    if prediction == 1:
        # Get the probability of the positive class (stroke)
        probability = ensemble_model.predict_proba(input_data_scaled)[0][1]
        risk_percentage = probability * 100
        risk = "High risk of stroke"
        disease_type = "Stroke"
    else:
        # If no stroke, set risk percentage to 0
        risk_percentage = 0
        risk = "Low risk of stroke"
        disease_type = "No Stroke"

    return risk, risk_percentage, disease_type

# Function to generate a prevention report based on risk and disease
def generate_prevention_report(risk, disease, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
        -Purpose of the Report: Clearly state why this report is being generated, including its relevance to the individual’s health.
        -Overview of Health & Wellness: Briefly describe the importance of understanding and managing health risks, with a focus on proactive wellness and disease prevention.
        -Personalized Context: Include the user's specific details such as age, gender, and any relevant medical history that can be linked to the risk factor and disease.
    
    2. **Risk Description**
        -Detailed Explanation of Risk: Describe the identified risk factor in detail, including how it impacts the body and its potential consequences if left unaddressed.
        -Associated Conditions: Mention any other health conditions commonly associated with this risk factor.
        -Prevalence and Statistics: Provide some general statistics or prevalence rates to contextualize the risk (e.g., how common it is in the general population or specific age groups).
    
    3. **Stage of Risk**
        -Risk Level Analysis: Provide a more granular breakdown of the risk stages (e.g., low, medium, high), explaining what each stage means in terms of potential health outcomes.
        -Progression: Discuss how the risk may progress over time if not managed, and what signs to watch for that indicate worsening or improvement.
    
    4. **Risk Assessment**
        -Impact on Health: Explore how this specific risk factor might affect various aspects of health (e.g., cardiovascular, metabolic, etc.).
        -Modifiable vs. Non-Modifiable Risks: Distinguish between risks that can be changed (e.g., lifestyle factors) and those that cannot (e.g., genetic predisposition).
        -Comparative Risk: Compare the individual's risk to average levels in the general population or among peers.
        
    5. **Findings**
        -In-Depth Health Observations: Summarize the key findings from the assessment, explaining any critical areas of concern.
        -Diagnostic Insights: Provide insights into how the disease was identified, including the symptoms, biomarkers, or other diagnostic criteria used.
        -Data Interpretation: Offer a more detailed interpretation of the user's health data, explaining what specific values or results indicate.
    
    6. **Recommendations**
        -Personalized Action Plan: Suggest specific, actionable steps the individual can take to mitigate the risk or manage the disease (e.g., dietary changes, exercise plans, medical treatments).
        -Lifestyle Modifications: Tailor suggestions to the individual’s lifestyle, providing practical tips for integrating these changes.
        -Monitoring and Follow-up: Recommend how the user should monitor their health and when to seek follow-up care.
        
    7. **Way Forward**
        -Next Steps: Provide a clear path forward, including short-term and long-term goals for managing the identified risk or disease.
        -Preventive Measures: Highlight preventive strategies to avoid worsening the condition or preventing its recurrence.
        -Health Resources: Suggest additional resources, such as apps, websites, or support groups, that could help the individual manage their health.
        
    8. **Conclusion**
        -Summary of Key Points: Recap the most important points from the report, focusing on what the individual should remember and prioritize.
        -Encouragement: Offer positive reinforcement and encouragement for taking proactive steps toward better health.
    
    9. **Contact Information**
        -Professional Guidance: Include information on how to get in touch with healthcare providers for more personalized advice or follow-up.
        -Support Services: List any available support services, such as nutritionists, fitness coaches, or mental health professionals, that could assist in managing the risk.
    
    10. **References**
        -Scientific Sources: Provide references to the scientific literature or authoritative health guidelines that support the information and recommendations given in the report.
        -Further Reading: Suggest articles, books, or other educational materials for the individual to learn more about their condition and how to manage it.

    **Details:**
    Risk: {risk:.2f}%
    Disease: {disease}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """

    try:
        response = gemini.generate_text(
            prompt=prompt,
            temperature=0.5,
            max_output_tokens=1000
        )
        
        report = response.result if hasattr(response, 'result') else None
        
        if not report:
            print("The response from the API did not contain a result.")
        
        return report
    except Exception as e:
        print(f"An error occurred: {e}")

# Get user input for each variable
gender = int(input("Enter gender (1 = male, 0 = female): "))
age = int(input("Enter age: "))
hypertension = int(input("Do you have hypertension? (yes = 1, no = 0): "))
heart_disease = int(input("Do you have heart disease? (yes = 1, no = 0): "))
ever_married = int(input("Have you ever been married? (yes = 1, no = 0): "))
work_type = int(input("Enter work type (Private = 0, Self-employed = 1, Govt_job = 2, children = 3): "))
Residence_type = int(input("Enter residence type (Urban = 1, Rural = 0): "))
avg_glucose_level = float(input("Enter average glucose level: "))
bmi = float(input("Enter BMI: "))
smoking_status = int(input("Enter smoking status (never smoked = 0, smokes = 1, formerly smoked = 2): "))

# Combine user input into a single tuple
input_data = (gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status)

# Predict and display results
risk, risk_percentage, disease_type = predict_stroke(input_data)

# Display results
print(f"Risk: {risk}")
print(f"Risk Percentage: {risk_percentage:.2f}%")
print(f"Problem: {disease_type}")

# Generate the wellness report only if there is a stroke prediction and the risk percentage is greater than 0
if disease_type == "Stroke" and risk_percentage > 0:
    report = generate_prevention_report(
        risk=risk_percentage,
        disease=disease_type,
        age=age
    )

    if report:
        print("\nGenerated Wellness Report:")
        print(report)
    else:
        print("Failed to generate a report. Please check the API response and try again.")
else:
    print("No report generated as the risk is low or there is no stroke prediction.")
