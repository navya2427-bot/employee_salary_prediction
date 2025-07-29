from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# ✅ Category mappings dictionary (flattened) - EXACTLY as in original
category_mappings = {
    # Education mappings - including ALL original options
    'preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'hs-grad': 9, 'some-college': 10, 'assoc-voc': 11, 'assoc-acdm': 12, 'bachelors': 13, 'masters': 14, 'prof-school': 15, 'doctorate': 16, 
    # Workclass mappings
    'federal-gov': 0, 'local-gov': 1, 'private': 2, 'self-emp-inc': 3, 'self-emp-not-inc': 4, 'state-gov': 5, 'without-pay': 6, 
    # Marital status mappings
    'divorced': 0, 'married-af-spouse': 1, 'married-civ-spouse': 2, 'married-spouse-absent': 3, 'never-married': 4, 'separated': 5, 'widowed': 6, 
    # Occupation mappings
    'adm-clerical': 0, 'armed-forces': 1, 'craft-repair': 2, 'exec-managerial': 3, 'farming-fishing': 4, 'handlers-cleaners': 5, 'machine-op-inspct': 6, 'other-service': 7, 'priv-house-serv': 8, 'prof-specialty': 9, 'protective-serv': 10, 'sales': 11, 'tech-support': 12, 'transport-moving': 13, 
    # Relationship mappings
    'husband': 0, 'not-in-family': 1, 'other-relative': 2, 'own-child': 3, 'unmarried': 4, 'wife': 5, 
    # Race mappings
    'amer-indian-eskimo': 0, 'asian-pac-islander': 1, 'black': 2, 'other': 3, 'white': 4, 
    # Gender mappings
    'female': 0, 'male': 1, 
    # Native country mappings - EXACTLY as in original
    'cambodia': 0, 'canada': 1, 'china': 2, 'columbia': 3, 'cuba': 4, 'dominican-republic': 5, 'ecuador': 6, 'el-salvador': 7, 'england': 8, 'france': 9, 'germany': 10, 'greece': 11, 'guatemala': 12, 'haiti': 13, 'holand-netherlands': 14, 'honduras': 15, 'hong': 16, 'hungary': 17, 'india': 18, 'iran': 19, 'ireland': 20, 'italy': 21, 'jamaica': 22, 'japan': 23, 'laos': 24, 'mexico': 25, 'nicaragua': 26, 'outlying-us(guam-usvi-etc)': 27, 'peru': 28, 'philippines': 29, 'poland': 30, 'portugal': 31, 'puerto-rico': 32, 'scotland': 33, 'south': 34, 'taiwan': 35, 'thailand': 36, 'trinadad&tobago': 37, 'united-states': 38, 'vietnam': 39, 'yugoslavia': 40
}

# ✅ Load the saved XGBoost model
try:
    with open("xgb_tuned_model.pkl", "rb") as f:
        loaded_xgb_model = pickle.load(f)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Error: xgb_tuned_model.pkl not found!")
    loaded_xgb_model = None
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    loaded_xgb_model = None

# ✅ Define prediction function
def predict_income(features):
    if loaded_xgb_model is None:
        raise ValueError("Model not loaded")
    
    input_array = np.array([features])
    prediction = loaded_xgb_model.predict(input_array)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_xgb_model is None:
        return jsonify({"error": "Model not available"}), 500
    
    try:
        form_data = request.json
        
        # Debug: Print received data
        print("Received form data:", form_data)
        
        # Convert all string inputs to lowercase - EXACTLY as in original
        form_data = {k: v.lower() if isinstance(v, str) else v for k, v in form_data.items()}
        
        # Create numerical data array in EXACT same order as original
        numerical_data = [
            int(form_data['age']),
            category_mappings[form_data['workclass']],
            category_mappings[form_data['education']],
            category_mappings[form_data['maritalStatus']],
            category_mappings[form_data['occupation']],
            category_mappings[form_data['relationship']],
            category_mappings[form_data['race']],
            category_mappings[form_data['gender']],
            int(form_data['hoursPerWeek']),
            category_mappings[form_data['nativeCountry']]
        ]
        
        # Debug: Print numerical data
        print("Numerical data for prediction:", numerical_data)
        
    except KeyError as e:
        print(f"KeyError: {e}")
        return jsonify({"error": f"Invalid input value: {e}"}), 400
    except ValueError as e:
        print(f"ValueError: {e}")
        return jsonify({"error": f"Invalid data type: {e}"}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Processing error: {e}"}), 400

    try:
        prediction = predict_income(numerical_data)
        print(f"Prediction result: {prediction}")
        
        result = {
            'prediction': int(prediction),
            'salary': '>50K' if prediction == 1 else '<=50K'
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# ✅ Health check endpoint
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": loaded_xgb_model is not None,
        "static_folder": app.static_folder,
        "template_folder": app.template_folder
    })

# ✅ Debug route to check mappings
@app.route('/debug/mappings')
def debug_mappings():
    return jsonify({
        "total_mappings": len(category_mappings),
        "education_options": [k for k in category_mappings.keys() if k in ['preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'hs-grad', 'some-college', 'assoc-voc', 'assoc-acdm', 'bachelors', 'masters', 'prof-school', 'doctorate']],
        "sample_mappings": dict(list(category_mappings.items())[:10])
    })

if __name__ == '__main__':
    # Check if static files exist
    static_files = ['confusion_matrix.png', 'feature.png', 'area.png', 'recall.png']
    for file in static_files:
        file_path = os.path.join('static', file)
        if os.path.exists(file_path):
            print(f"✅ {file} found")
        else:
            print(f"⚠️  {file} not found at {file_path}")
    
    print("🚀 Starting Flask server...")
    app.run()
