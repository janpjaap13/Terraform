from flask import Flask, jsonify, request
from service.ml_service import MLService
import os

app = Flask(__name__)

@app.route('/ml/predict', methods=['POST'])
def ml_predict():
    if not request.get_json():
        return jsonify(message='No data provided'), 400
    try:
        data = request.get_json()
        ml_service = MLService()
        output = ml_service.predict(data)
        return jsonify(output)
    except Exception as err:
        return jsonify(message=str(err)), 500
    
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)