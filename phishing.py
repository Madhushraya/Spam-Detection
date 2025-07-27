import streamlit as st
import joblib
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🔍",
    layout="wide"
)

# Function to clean text (same as in your training code)
def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the saved model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load('phishing_predictor.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Main function
def main():
    st.title("📧 Phishing Email Detector")
    st.write("This app detects if an email is a phishing attempt or legitimate based on its content.")
    
    try:
        model, vectorizer = load_models()
        model_loaded = True
        st.success("Model and vectorizer loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Email Analysis", "Model Information", "Example Analysis"])
    
    with tab1:
        st.header("Analyze Email Content")
        
        # Text area for email input
        email_text = st.text_area("Enter email content to analyze:", 
                                  height=200,
                                  placeholder="Paste your email content here...")
        
        # Features visualization
        col1, col2, col3 = st.columns(3)
        
        if email_text:
            # Basic feature engineering
            text_length = len(email_text)
            num_special_chars = sum(not c.isalnum() and not c.isspace() for c in email_text)
            num_uppercase = sum(1 for c in email_text if c.isupper())
            
            # Display calculated features
            with col1:
                st.metric("Text Length", text_length)
            with col2:
                st.metric("Special Characters", num_special_chars)
            with col3:
                st.metric("Uppercase Letters", num_uppercase)
        
        # Analyze button
        if st.button("Analyze Email", type="primary") and email_text and model_loaded:
            st.subheader("Analysis Results:")
            
            # Clean the text
            cleaned_text = clean_text(email_text)
            
            # Transform text using the loaded vectorizer
            text_vectorized = vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = model.predict(text_vectorized)[0]
            probability = model.predict_proba(text_vectorized)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **PHISHING EMAIL DETECTED!**")
                    confidence = probability[1] * 100
                else:
                    st.success("✅ **LEGITIMATE EMAIL**")
                    confidence = probability[0] * 100
                
                st.write(f"Confidence: {confidence:.2f}%")
                
            with col2:
                # Create a gauge chart for visualization
                fig, ax = plt.subplots(figsize=(4, 3))
                
                # Plot probability
                colors = ['green', 'red']
                ax.bar(['Legitimate', 'Phishing'], probability, color=colors, alpha=0.7)
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title('Classification Probability')
                
                st.pyplot(fig)
    
    with tab2:
        st.header("Model Information")
        st.write("This application uses a pre-trained Random Forest model to classify emails as phishing or legitimate.")
        
        st.subheader("Model Details")
        st.write("""
        - **Model Type**: Random Forest Classifier
        - **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Features Used**: Email content (cleaned text)
        """)
        
        if model_loaded:
            # Display some model parameters
            st.subheader("Random Forest Parameters")
            st.code(f"""
            n_estimators: {model.n_estimators}
            max_depth: {model.max_depth if model.max_depth is not None else 'None'}
            min_samples_split: {model.min_samples_split}
            """)
            
            # Display vectorizer information
            st.subheader("TF-IDF Vectorizer")
            st.write(f"Vocabulary Size: {len(vectorizer.vocabulary_)}")
            
            # Display top features (if possible)
            st.subheader("Top 10 Important Words")
            try:
                feature_names = vectorizer.get_feature_names_out()
                importances = model.feature_importances_
                indices = importances.argsort()[-10:][::-1]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
                ax.set_title("Top 10 Important Words")
                ax.set_xlabel("Importance")
                st.pyplot(fig)
            except:
                st.info("Unable to display feature importances.")
    
    with tab3:
        st.header("Example Analysis")
        
        # Example phishing emails
        phishing_examples = [
            "Dear user, your account has been compromised. Click on this link to verify your information immediately: http://suspicious-link.com",
            "URGENT: Your payment has failed. Update your banking details within 24 hours to avoid account suspension. Click here: http://not-real-bank.com",
            "Congratulations! You've won a free iPhone. Claim your prize now by providing your personal information at our secure website."
        ]
        
        # Example legitimate emails
        legitimate_examples = [
            "Hello team, please find attached the quarterly report we discussed in yesterday's meeting. Let me know if you have any questions.",
            "Thank you for your order #12345. Your item has been shipped and should arrive within 3-5 business days.",
            "This is a reminder about our upcoming team building event on Friday at 3pm. Looking forward to seeing everyone there!"
        ]
        
        # Select examples
        example_type = st.radio("Select example type:", ["Phishing Email Examples", "Legitimate Email Examples"])
        
        if example_type == "Phishing Email Examples":
            example_list = phishing_examples
        else:
            example_list = legitimate_examples
        
        selected_example = st.selectbox("Choose an example:", range(len(example_list)), format_func=lambda i: example_list[i][:50] + "...")
        
        if st.button("Analyze Example") and model_loaded:
            example_text = example_list[selected_example]
            st.text_area("Example Text:", example_text, height=100)
            
            # Clean the text
            cleaned_text = clean_text(example_text)
            
            # Transform text using the loaded vectorizer
            text_vectorized = vectorizer.transform([cleaned_text])
            
            # Make prediction
            prediction = model.predict(text_vectorized)[0]
            probability = model.predict_proba(text_vectorized)[0]
            
            # Display result
            if prediction == 1:
                st.error("⚠️ **PHISHING EMAIL DETECTED!**")
                confidence = probability[1] * 100
            else:
                st.success("✅ **LEGITIMATE EMAIL**")
                confidence = probability[0] * 100
            
            st.write(f"Confidence: {confidence:.2f}%")
            
            # Display features
            text_length = len(example_text)
            num_special_chars = sum(not c.isalnum() and not c.isspace() for c in example_text)
            num_uppercase = sum(1 for c in example_text if c.isupper())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Text Length", text_length)
            with col2:
                st.metric("Special Characters", num_special_chars)
            with col3:
                st.metric("Uppercase Letters", num_uppercase)

# Run the app
if __name__ == "__main__":
    main()