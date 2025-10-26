from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)
# Initialize Flask app

# --- Load Model and Encoder ---
# Construct paths relative to this script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'isolation_forest_model.joblib')
encoder_path = os.path.join(base_dir, '..', 'models', 'one_hot_encoder.joblib')

try:
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    print("✅ Model and encoder loaded successfully.")
    # Store the expected feature names after encoding (important!)
    # Load this from a file if saved during training, or get from encoder
    expected_columns = encoder.get_feature_names_out(["soil_type", "crop_type", "tillage_type", "season"])
    # Combine with numerical columns in the correct order the model expects
    numerical_cols = ["temp_7day_avg_f", "precip_7day_total_in"]
    # Adjust this list based on the actual order used during training!
    model_input_columns = numerical_cols + list(expected_columns)


except Exception as e:
    print(f"❌ Critical Error: Could not load model or encoder: {e}")
    model = None # Set to None to indicate failure
    encoder = None
    model_input_columns = []


# --- API Endpoint for Prediction ---
@app.route('/api/predict', methods=['POST'])
def handle_predict():
    if model is None or encoder is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        # Get data from the POST request body (expected as JSON)
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input to DataFrame (single row)
        input_df = pd.DataFrame([input_data])

        # Basic validation (check if required columns are present)
        required_input_cols = ["soil_type", "crop_type", "tillage_type", "season", "temp_7day_avg_f", "precip_7day_total_in"]
        if not all(col in input_df.columns for col in required_input_cols):
             return jsonify({"error": f"Missing required input fields. Need: {required_input_cols}"}), 400

        # --- Preprocessing ---
        categorical_cols = ["soil_type", "crop_type", "tillage_type", "season"]
        numerical_cols = ["temp_7day_avg_f", "precip_7day_total_in"]

        # Apply the loaded encoder
        encoded_cats = encoder.transform(input_df[categorical_cols])
        encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols), index=input_df.index)

        # Combine numerical and encoded categorical
        features_combined = pd.concat([input_df[numerical_cols], encoded_cats_df], axis=1)

        # Reindex to ensure columns match the model's training order exactly
        # This handles cases where input might lack a category seen during training
        features_final = features_combined.reindex(columns=model_input_columns, fill_value=0)


        # --- Prediction ---
        # decision_function: lower score = more anomalous (outlier)
        # score_samples: higher score = more anomalous (use if preferred)
        anomaly_score_raw = model.decision_function(features_final)[0] # Get score for the single row

        # --- Interpretation (Example) ---
        # Convert raw score to a 0-100 "risk" percentage
        # NOTE: This scaling is arbitrary and needs tuning based on score distribution!
        # Scores closer to -0.5 are outliers, closer to 0.5 are inliers for default contamination='auto'
        # Simple scaling: map roughly [-0.2 to 0.2] range (adjust based on observation)
        # to [100 down to 0] risk percentage.
        score_for_scaling = anomaly_score_raw
        min_score = -0.2 # Estimated threshold for definite anomaly
        max_score = 0.2  # Estimated threshold for definite normal point
        risk_percentage = 100 * (max_score - score_for_scaling) / (max_score - min_score)
        risk_percentage = max(0, min(100, risk_percentage)) # Clamp between 0 and 100


        # Return result
        return jsonify({
            "risk_percentage": round(risk_percentage),
            "raw_anomaly_score": float(anomaly_score_raw) # Include raw score for debugging
        })

    except Exception as e:
        print(f"❌ Error during prediction request: {e}")
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# Optional: Add a simple root route for testing if needed
@app.route('/', methods=['GET'])
def home():
    # Tell Flask to render the HTML file from the 'templates' folder
    try:
        return render_template('index.html')
    except Exception as e:
        # Basic error handling if the template isn't found
        print(f"Error rendering template: {e}")
        return "Error: Could not load the prediction form.", 500

# Vercel runs the file, finds the 'app' object, and serves it.
# No need for app.run() when deploying to Vercel.