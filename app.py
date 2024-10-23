import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from bson import ObjectId
import bcrypt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from functools import wraps
from PIL import Image
import io

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["MONGO_URI"] = os.getenv("MONGO_URI")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "super-secret-key") 

client = MongoClient(app.config["MONGO_URI"])
db = client['garbageDB']  

jwt = JWTManager(app)

model = load_model('Garbage_classification.h5')
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

import jwt                  
import logging

from functools import wraps
from flask import request, jsonify
import jwt
from bson import ObjectId
import logging

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            user_id = data['sub']  
            current_user = db.users.find_one({'_id': ObjectId(user_id)})
            if current_user is None:
                return jsonify({'message': 'User not found!'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError as e:
            logging.error(f'Invalid Token: {e}')  
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(current_user, *args, **kwargs) 
    return decorated


def preprocess_image(image_content):
    image = Image.open(io.BytesIO(image_content))
    image = image.resize((224, 224)) 
    img_array = np.array(image) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array


# Signup route
@app.route('/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name, email, password = data['name'], data['email'], data['password']

    if db.users.find_one({"email": email}):
        return jsonify({'error': 'Email already in use'}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user = {'name': name, 'email': email, 'password': hashed_password}
    db.users.insert_one(user)

    return jsonify({'message': 'User created successfully'}), 201

# Login route
@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or 'email' not in data or 'password' not in data:
        return jsonify({'error': 'Missing email or password'}), 400

    email, password = data['email'], data['password']
    
    user = db.users.find_one({"email": email})

    if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({'error': 'Invalid credentials'}), 400

    access_token = create_access_token(identity=str(user['_id']), expires_delta=datetime.timedelta(days=1))
    
    return jsonify({'token': access_token, 'user': {'id': str(user['_id']), 'name': user['name'], 'email': user['email']}}), 200



# Logout route
@app.route('/auth/logout', methods=['POST'])
@token_required
def logout(current_user):
    
    return jsonify({'message': 'Logged out successfully'}), 200

# Get user info
@app.route('/auth/user', methods=['GET'])
@token_required
def get_user(current_user): 
    if not current_user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'id': str(current_user['_id']), 'name': current_user['name'], 'email': current_user['email']}), 200

import base64

@app.route('/predict', methods=['POST'])
@token_required
def predict(current_user):
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file:
        return jsonify({'error': 'File is empty'}), 400

    user_id = str(current_user['_id'])

    image_content = file.read()
    base64_image = base64.b64encode(image_content).decode('utf-8') 

    img = preprocess_image(image_content)  
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]

    result = {
        'userId': ObjectId(user_id),
        'prediction': predicted_class,
        'date': datetime.datetime.utcnow(),
        'image_base64': base64_image  
    }

    db.results.insert_one(result)

    return jsonify({'material': predicted_class, 'image_base64': base64_image}), 200


# Get user's classification history
@app.route('/history', methods=['GET'])
@token_required
def get_history(current_user):
    user_id = str(current_user['_id'])
    results = list(db.results.find({'userId': ObjectId(user_id)}).sort('date', -1).limit(20))

    history = [{
        'userId': str(r['userId']),  
        'prediction': r['prediction'],
        'date': r['date'],
        'image_base64': r['image_base64']
    } for r in results]

    return jsonify(history), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

