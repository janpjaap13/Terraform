from flask import Flask, jsonify, request
from service.cv_service import CVService
from service.ml_service import MLService
import os
from flasgger import Swagger

app = Flask(__name__)

swagger = Swagger(app)

@app.route('/ml/predict', methods=['POST'])
def ml_predict_width():
    """
    Endpoint to predict the width of head thickness using machine learning.
    ---
    tags: 
        - Machine Learning
    parameters:
      - in: body
        name: body
        description: Input parameters for prediction
        required: true
        schema:
          type: array
          items:
            type: object
            properties:
              local time:
                type: string
                format: date-time
                example: "2024-06-06T08:00:00"
              irrigation_EC:
                type: number
                format: float
                example: 1.2
              energy_curtain:
                type: boolean
                example: true
              PAR_sum_plant:
                type: number
                format: float
                example: 350
              lower_circuit:
                type: number
                format: float
                example: 22.5
              outside_temperature:
                type: number
                format: float
                example: 18.3
              plant_PAR:
                type: number
                format: float
                example: 500
              humidity_deficit_above_curtain:
                type: number
                format: float
                example: 15
              vent_lee:
                type: boolean
                example: false
              difference_plant:
                type: number
                format: float
                example: 0.8
    responses:
      200:
        description: Successfully predicted width of head thickness
        schema:
          type: object
          properties:
            local_time:
              type: string
              format: date-time
              example: "2024-06-06T08:00:00"
            predicted_width:
              type: number
              format: float
              example: 10.014210213793165
    """
    
    if not request.get_json():
        return jsonify(message='No data provided'), 400
    
    try:
        data = request.get_json()
        ml_service = MLService()
        output = ml_service.predict_width(data)
        return jsonify(output)
    except Exception as err:
        return jsonify(message=str(err)), 500
    
    

if __name__ == '__main__':
    app.run(debug=True)
