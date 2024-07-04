from flask import Flask, request, render_template, redirect, url_for
from markupsafe import Markup
import joblib
import numpy as np

app = Flask(__name__)

# Explanation 
def explain_ghi(ghi_value):
    if ghi_value < 0:
        return "Invalid GHI value. Please enter a non-negative number."
    
    explanation = f"The Global Horizontal Irradiance (GHI) is {ghi_value} W/m². <br/>This indicates:<br/>"

    if ghi_value == 0:
        explanation += (
            "- There is no solar radiation reaching the horizontal surface, which is typical at night.<br/>"
        )
    elif ghi_value > 0 and ghi_value <= 100:
        explanation += (
            "- Very low solar radiation, which could occur during early morning or late evening hours.<br/>"
            "- It could also indicate heavily overcast or stormy weather conditions.<br/>"
        )
    elif ghi_value > 100 and ghi_value <= 400:
        explanation += (
            "- Moderate solar radiation. This level is typical during partially cloudy conditions or when the sun is lower in the sky (e.g., morning or late afternoon).<br/>"
            "- Adequate for some solar power generation but not optimal.<br/>"
        )
    elif ghi_value > 400 and ghi_value <= 700:
        explanation += (
            "- High solar radiation, indicating clear skies and optimal conditions for solar power generation.<br/>"
            "- This level is typically observed around midday when the sun is at a high angle in the sky.<br/>"
        )
    elif ghi_value > 700:
        explanation += (
            "- Very high solar radiation, which is ideal for solar energy generation.<br/>"
            "- Such high values are usually observed in regions with clear skies and strong sunlight, like desert areas or during peak solar hours.<br/>"
        )
    
    
    return explanation

# Load the trained model
model = joblib.load('ghi_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        features = [
            int(request.form['Year']),
            int(request.form['Month']),
            int(request.form['Day']),
            int(request.form['Hour']),
            int(request.form['Minute']),
            float(request.form['Temperature']),
            float(request.form['Dew Point']),
            float(request.form['DHI']),
            float(request.form['DNI']),
            float(request.form['Relative Humidity']),
            float(request.form['Solar Zenith Angle']),
            float(request.form['Surface Albedo']),
            float(request.form['Pressure']),
            float(request.form['Wind Speed']),
            float(request.form['Topocentric zenith angle']),
            float(request.form['Top. azimuth angle (eastward from N)']),
            float(request.form['Top. azimuth angle (westward from S)'])
        ]        

        # Convert features to a numpy array
        features_array = np.array(features)

        # Scaling the values
        array_scaled = scaler.transform([features_array])

        # Predict GHI
        prediction = model.predict(array_scaled)[0]
        print(prediction)
        
        # Redirect to the prediction page
        return redirect(url_for('results', prediction=prediction))
    
    return render_template('prediction-page.html')

@app.route('/results')
def results():
    # Get the prediction from the query parameters
    prediction = request.args.get('prediction')

    # Making sense of the predicted values
    explaination = explain_ghi(float(prediction))

    # Converting the explaination into a markup code
    explaination = Markup(explaination)

    return render_template('results.html', prediction=explaination)

if __name__ == '__main__':
    app.run(debug=True)
