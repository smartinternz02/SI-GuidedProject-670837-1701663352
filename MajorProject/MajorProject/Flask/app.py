from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__,template_folder="templates")
# Load the trained model and scaler
try:
    model=joblib.load('xgb_regressor_model.pkl')
    scaler=joblib.load('standard_scaler.pkl')
except Exception as e:
    print("Error loading model or scaler:", e)


@app.route('/')
def index():
    return render_template('input_form.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get Input data from the form
        feature_14_NH4 = float(request.form['14_NH4'])
        feature_14_NO2 = float(request.form['14_NO2'])
        feature_14_NO3 = float(request.form['14_NO3'])
        feature_15_NH4 = float(request.form['15_NH4'])
        feature_15_NO2 = float(request.form['15_NO2'])
        feature_15_NO3 = float(request.form['15_NO3'])
        feature_16_NH4 = float(request.form['16_NH4'])
        feature_16_NO2 = float(request.form['16_NO2'])

        # Create an Input data array for prediction
        input_data = np.array([
            [feature_14_NH4, feature_14_NO2, feature_14_NO3, feature_15_NH4,
             feature_15_NO2, feature_15_NO3, feature_16_NH4, feature_16_NO2]
        ])

        # Scale the input data using the loaded StandardScaler
        scaled_input_data = scaler.transform(input_data)

        # Perform predictions using your model
        predictions = model.predict(scaled_input_data)

        # Render results on a new page
        return render_template('result.html', predictions=predictions)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('error.html', error_message=error_message)


if __name__ == '__main__':
    app.run()
