import streamlit as st
import google.generativeai as genai
import os
import re
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Google Gemini API and LangChain model
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)
llm_resto = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GOOGLE_API_KEY)

# Define the prompt template
prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'allergics', 'foodtype'],
    template="Based on the following personal information, please generate recommendations:\n"
             "- Age: {age}\n"
             "- Gender: {gender}\n"
             "- Weight: {weight} kg\n"
             "- Height: {height} feet\n"
             "- Diet preference (veg/non-veg): {veg_or_nonveg}\n"
             "- Health condition: {disease}\n"
             "- Region: {region}\n"
             "- Allergies: {allergics}\n"
             "- Preferred food type: {foodtype}\n\n"
             "Please provide:\n"
             "1. Six restaurant recommendations suitable for this individual.\n"
             "2. Six breakfast options aligned with the individual's preferences and health needs.\n"
             "3. Five dinner suggestions suitable for their diet and health conditions.\n"
             "4. Six workout suggestions appropriate for their health and fitness level.\n\n"
             "Format your response exactly as follows:\n\n"
             "Restaurants:\n- Restaurant 1\n- Restaurant 2\n...\n\n"
             "Breakfast:\n- Breakfast 1\n- Breakfast 2\n...\n\n"
             "Dinner:\n- Dinner 1\n- Dinner 2\n...\n\n"
             "Workouts:\n- Workout 1\n- Workout 2\n..."
)

# Create LangChain with the prompt template
chain_resto = prompt_template_resto | llm_resto

# Streamlit App
st.title("Diet Recommendation System")

# Input Form
with st.form(key='user_input_form'):
    age = st.number_input("Age", min_value=1, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", min_value=1, step=1)
    height = st.number_input("Height (feet)", min_value=1, step=1)
    veg_or_nonveg = st.selectbox("Diet Preference", ["Veg", "Non-Veg"])
    disease = st.text_input("Health Condition")
    region = st.text_input("Region")
    allergics = st.text_input("Allergies")
    foodtype = st.text_input("Preferred Food Type")
    
    # Submit button
    submit_button = st.form_submit_button(label="Get Recommendations")

# Process input and display results
if submit_button:
    input_data = {
        'age': age,
        'gender': gender.lower(),
        'weight': weight,
        'height': height,
        'veg_or_nonveg': veg_or_nonveg.lower(),
        'disease': disease,
        'region': region,
        'allergics': allergics,
        'foodtype': foodtype
    }

    # Get recommendations from LangChain
    results = chain_resto.invoke(input_data)
    results_str = str(results)
    cleaned_results = re.sub(r'\s*\n\s*', '\n', results_str).strip()

    # Display the recommendations
    content_start = results_str.find('*Restaurants:*')
    content_end = results_str.find('additional_kwargs')
    if content_start != -1 and content_end != -1:
        content = results_str[content_start:content_end].strip()
        st.subheader("Diet Recommendations")
        st.write(content.replace('\\n', '\n'))  # Replace '\n' for better formatting
    else:
        st.write("No recommendations found.")