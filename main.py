import pandas as pd 
import numpy as np
import joblib 
import streamlit as st

#title of the web app
st.title("Email spam classificatin Web App")
#load the model
model = joblib.load('model.pkl')
#load the vectorizer
Vectorizer = joblib.load('cv.pkl')
#take the input of email form the user
email = st.text_area("Enter the email text here:")

# make the prediction button
if st.button("predict"):
    if email.strip() == "":
        st.warning("please enter an email message")
    else:
        #convert the email text into vectors 
        email_vectors = vectorizer.transform([email])
        prediction = model.predict(email_vectors)
    # display the result 
    if prediction[0] == 'spam':
        st.error("🚨 This is a SPAM Email!")
    else:
        st.success("this is a ham email")