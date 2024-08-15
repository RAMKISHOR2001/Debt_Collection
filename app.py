from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from feature_engineering import get_features, engineer_features
from data_preprocessing import preprocess_data
from optimization_strat import apply_optimization
from nlp_analysis import analyze_communications, generate_sample_communications

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('xgboost_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    trial = request.form
    trial = dict(trial)
    # print(trial)
    trial = list(trial.items())
    temp = trial[-2:]
    # print(temp)
    trial = trial[:len(trial)-2]
    trial = list(map(lambda x: (x[0],int(x[1])),trial))
    trial = dict(trial + temp)
    # print(trial)
    trial["last_contact_date"] += " 02:06:47.057132"
    if trial["communication_preference"] == "0":
        trial["communication_preference"] = "email"
    elif trial["communication_preference"] == "1":
        trial["communication_preference"] = "phone"
    else:
        trial["communication_preference"] = "mail"
    
    data = {"customer_id":1,"debt_amount":3807,"days_overdue":183,"credit_score":455,"age":74,"income":162408,"num_previous_loans":1,"num_previous_defaults":4,"communication_preference":"email","last_contact_date":"2024-07-13 02:06:47.057132"}
    df = pd.DataFrame([trial])
    # Preprocess and engineer features  
    df = preprocess_data(df)
    df = engineer_features(df)
    features = get_features()
    results = apply_optimization(df, model, scaler, features)
    results = dict(results)
    results = list(results.items())
    temp = []
    for i in results:
        temp.append([i[0],list(i[1])[0]])
    results = temp
    for i in range(len(results)):
        results[i] = list(results[i])
    for i in range(len(results)):
        temp = results[i][0]
        temp = temp.split("_")
        for j in range(len(temp)):
            temp[j] = temp[j].title()
            print(temp[j])
        temp = " ".join(temp)
        results[i][0] = temp
    
    results = dict(results)
    print(results)
    # return jsonify(results.to_dict(orient='records')[0])

    return render_template('test.html',data = results)    

@app.route('/batch_optimize', methods=['POST'])
def batch_optimize():
    file = request.files['file']
    df = pd.read_csv(file)
    
    # Preprocess and engineer features
    df = preprocess_data(df)
    df = engineer_features(df)
    
    features = get_features()
    results = apply_optimization(df, model, scaler, features)
    
    return results.to_json(orient='records')

@app.route('/analyze_communications', methods=['POST'])
def analyze_comms():
    communications = request.json.get('communications', [])
    if not communications:
        communications = generate_sample_communications(10)
    
    results, topics = analyze_communications(communications)
    
    return jsonify({
        'results': results.to_dict(orient='records'),
        'topics': topics
    })

if __name__ == '__main__':
    app.run(debug=True)