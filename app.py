from flask import Flask, request, render_template, redirect, url_for, flash, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import google.generativeai as genai
from werkzeug.utils import secure_filename  # Import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the plant disease model
plant_disease_model = load_model('plant_disease_model.h5')

disease_classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
    "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Set up your Gemini API key
genai.configure(api_key=os.environ["API_KEY"])

# Initialize the Gemini model
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_disease_name(img_path):
    try:
        img_array = preprocess_image(img_path)
        predictions = plant_disease_model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        
        if predicted_class < len(disease_classes):
            disease_name = disease_classes[predicted_class]
        else:
            disease_name = "Unknown Disease"
        
        return disease_name
    except Exception as e:
        return f"Error: {str(e)}"

def get_disease_info(disease_name):
    prompt = f"Give me information and prevention tips about the plant disease called {disease_name}."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching data from Gemini API: {str(e)}"

def get_chat_response(user_question, disease_name):
    prompt = f"You are talking about {disease_name}. {user_question}"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching data from Gemini API: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)
            disease_name = predict_disease_name(filepath)
            os.remove(filepath)  # Clean up the uploaded file

            # Get additional info and prevention tips using Gemini API
            disease_info = get_disease_info(disease_name)
            # Save the disease name in session for use in the chat
            session['disease_name'] = disease_name
            session['chat_history'] = []  # Clear previous chat history
            return render_template('result.html', disease_name=disease_name, disease_info=disease_info)
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_question = request.form['question']
        disease_name = session.get('disease_name', '')
        
        # Get the chat history (if stored in the session)
        chat_history = session.get('chat_history', [])
        
        # Append the user's question to the chat history
        chat_history.append({'sender': 'user', 'text': user_question})
        
        # Generate response from Gemini API
        try:
            prompt = f"You are talking about {disease_name}. {user_question}"
            response = gemini_model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"Error fetching data from Gemini API: {str(e)}"
        
        # Append the system's response to the chat history
        chat_history.append({'sender': 'system', 'text': answer})
        
        # Store the updated chat history in the session
        session['chat_history'] = chat_history
        
        return render_template('chat.html', chat_history=chat_history, disease_name=disease_name)
    else:
        # For GET requests, just render the chat page with an empty history
        return render_template('chat.html', chat_history=[], disease_name=session.get('disease_name', ''))

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
