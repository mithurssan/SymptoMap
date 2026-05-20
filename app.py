import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel

st.set_page_config(
    page_title="SymptoMap",
    page_icon="",
    layout="centered"
)

#splitting based on urgency
urgent ={ "Heart Attack", "Stroke", "Appendicitis", "Pneumonia", "Kidney Stones", "Shingles"}

routine = { "Hypertension", "Type 2 Diabetes", "Hypothyroidism", "Arthritis", "Asthma", "Depression", "Anxiety Disorder", "GERD", "IBS", "Anaemia", "Gout", "Psoriasis", "Eczema", "Migraine"}

self_care = {"Common Cold", "Acne", "Tension Headache", "Allergic Rhinitis", "Conjunctivitis", "Bronchitis", "Sinusitis", "Influenza", "Chickenpox", "Tonsillitis", "Ear Infection", "Food Poisoning", "Gastroenteritis", "UTI", "COVID-19"}

def get_referral(disease):
    if disease in urgent:
        return "Urgent GP / A&E Referral Recommended", "red"
    elif disease in routine:
        return "Routine GP Appointment Recommended", "orange"
    else:
        return "Self-Care / Pharmacist Advice Recommended", "green"
    

#Model
class BioClinicalBERT(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.fc(cls_output)

#Preprocessing
def clean_text(text):
    text = text.lower()
    
    cleaned_chars = []
    for char in text:
        if char.isalpha() or char == " ":
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ") #replace punctuation and numbers with a space
    
    text = "".join(cleaned_chars)
    text = " ".join(text.split()) # making sure its single spaced
    
    return text

#Loading required assets
@st.cache_resource
def load_assets():
    model_name= "emilyalsentzer/Bio_ClinicalBERT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("label_encoder.pickle", "rb") as f:
        le = pickle.load(f)
        
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    
    model = BioClinicalBERT(
        model_name=model_name,
        num_classes = len(le.classes_),
        dropout=0.3
    ).to(device)
    
    model.load_state_dict(torch.load("bert_best.pt", map_location=device))
    model.eval()

    return model, tokeniser, le, device

#Preserve state to prevent re-render by streamlit
if "results" not in st.session_state:
    st.session_state.results = None
if "referral_msg" not in st.session_state:
    st.session_state.referral_msg = None
if "colour" not in st.session_state:
    st.session_state.colour = None
if "symptom_input_saved" not in st.session_state:
    st.session_state.symptom_input_saved = None

#Prediction
def predict(text, model, tokeniser, le, device, top_k=3):
    cleaned=clean_text(text)
    tokenise = tokeniser(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    input_ids = tokenise["input_ids"].to(device)
    attention_mask = tokenise["attention_mask"].to(device)
    
    with torch.no_grad(): #no gradients need to be tracked
        logits = model(input_ids, attention_mask)
        
        probabilities = torch.softmax(logits, dim=1) #converting outputs into probabilities
        
        probabilities = probabilities.squeeze().cpu().numpy() # removing unrequired dimensions moving to cpu and converting to numpy array
        
        sorted_indices = np.argsort(probabilities)
        sorted_indices = sorted_indices[::-1]   #sorting and reversing order for highest probability
        top_indicies = sorted_indices[:top_k]
        
        results = []
    
    for index in top_indicies:
        class_name = le.classes_[index]
        class_probability = float(probabilities[index])
        results.append((class_name, class_probability))
    
    return results

#Email function
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
load_dotenv()

sender_email = os.getenv("sender_email")
sender_password = os.getenv("sender_password")

def send_gp_email(patient_name, patient_email, gp_email, symptom_input, results, referral_msg):
    top_disease, top_conf = results[0]
    
    other_results = []
    for disease, confidence in results[1:]:
        format_result = f" - {disease}: {confidence * 100:.1f}%"
        other_results.append(format_result)
    
    others = "".join(other_results)
    
    body = f"""

SymptoMap - GP Referral Notifcation
------------------------------------

Patient Name: {patient_name}
Patient Email: {patient_email}

Described Symptoms:
{symptom_input}

Primary Prediction: {top_disease} ({top_conf*100:.1f}% confidence)
Referral Status: {referral_msg}

Other Possible Conditions:
{others}

------------------------------------
Please follow up with the patient directly at {patient_email}.

This is an automated referral from SymptoMap.
This tool is trained on synthetic data and is for research purposes only.
Clinical judgement should always take importance.
    """

    msg = MIMEMultipart()
    msg["Subject"] = f"SymptoMap Referral - {patient_name} - {top_disease}"
    msg["From"] = sender_email
    msg["To"] = gp_email
    msg.attach(MIMEText(body, "plain"))
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, gp_email, msg.as_string())


#User Interface
st.title("SymptoMap")
st.subheader("AI-Powered Symptom Triage & GP Referral Assistant")
st.caption("Research purposes only · 35 disease classes · Powered by BioClinicalBERT · Not a substitute for medical advice")
st.divider()

model, tokeniser, le, device = load_assets()

symptom_input = st.text_area(
    "Please describe your symptoms in as much detail as possible",
    placeholder="e.g. I've had a severe headache and the light bothers me...",
    height=140
)

predict_btn = st.button("Analyse Symptoms", type="primary", use_container_width=True)
    
confidence_threshold = 0.7

if predict_btn:
    st.session_state.results = None
    st.session_state.referral_msg = None
    st.session_state.colour = None
    st.session_state.symptom_saved = None
    
    if not symptom_input.strip():
        st.warning("Please enter your symptoms before submitting.")
    elif len(symptom_input.strip().split()) < 4:
        st.warning("Please describe your symptoms in more detail for a more accurate result.")
    else:
        with st.spinner("Analysing, please wait..."):
            results = predict(symptom_input, model, tokeniser, le, device)
        
        st.session_state.results = results
        st.session_state.symptom_input_saved = symptom_input
        st.session_state.referral_msg, st.session_state.color = get_referral(results[0][0])

if st.session_state.results:
    results = st.session_state.results
    referral_msg = st.session_state.referral_msg
    colour = st.session_state.colour
    symptom_input_saved = st.session_state.symptom_input_saved

    top_disease, top_conf = results[0]
    referral_msg, colour = get_referral(top_disease)
        
    st.divider()
        
    if top_conf < confidence_threshold:
        st.warning(f"Low confidence prediction ({top_conf*100:.1f}%). Your disease may not be in the classifier or you need to describe your symptoms in more detail.")
        st.markdown("### Possible Conditions")
        for disease, conf in results:
            st.markdown(f"**{disease}** - {conf*100:.1f}%")
            st.progress(conf)
    else:
        st.markdown("## Predicted Condition")
        st.markdown(f"## {top_disease}")
        st.progress(top_conf, text=f"Confidence: {top_conf*100:.1f}%")
        
        if colour == "red":
            st.error(referral_msg)
        elif colour == "orange":
            st.warning(referral_msg)
        else:
            st.success(referral_msg)
                
        st.divider()
        st.markdown("### Other Possible Conditions")
        for disease, conf in results[1:]:
            ref_msg, _ = get_referral(disease)
            st.markdown(f"**{disease}** - {conf*100:.1f}% | {ref_msg}")
            st.progress(conf)
            
    st.divider()
    st.caption("Results are generated by a model trained on synthetic data. Always make sure to consult a qualified healthcare professional.")
        
    #GP REFERRAL
    st.divider()
    st.markdown("### Send Referral to GP")
        
    with st.form("gp_form"):
        patient_name = st.text_input("Patient Name")
        patient_email = st.text_input("Patient Email")
        gp_email = st.text_input("GP Email Address")
        send_btn = st.form_submit_button("Send GP Referral", type="primary")

    if send_btn:
        if not patient_name or not gp_email or not patient_email:
            st.warning("Please fill in all fields before sending.")
        else:
            try:
                send_gp_email(
                    patient_name=patient_name,
                    patient_email=patient_email,
                    gp_email=gp_email,
                    symptom_input=symptom_input,
                    results=results,
                    referral_msg=referral_msg
                )
                st.success(f"Referral sent to {gp_email} on behalf of {patient_name}")
            except Exception as e:
                st.error(f"Failed to send email: {e}")
