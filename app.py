# import os
# import uvicorn  # This can be imported at the top
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi import Request
# from pydantic import BaseModel
# from Braintumor.utils.common import decodeImage
# from Braintumor.pipeline.predict import PredictionPipeline
# from fastapi.middleware.cors import CORSMiddleware


# # Setup environment
# os.putenv('LANG', 'en_US.UTF-8')
# os.putenv('LC_ALL', 'en_US.UTF-8')

# # Initialize FastAPI app
# app = FastAPI()

# # Allow CORS for all origins (can be restricted if needed)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

# class ClientApp:
#     def __init__(self):
#         self.filename = "inputImage.jpg"
#         self.classifier = PredictionPipeline(self.filename)

# # Pydantic model for input validation
# class ImageInput(BaseModel):
#     image: str

# # Initialize ClientApp instance
# clApp = ClientApp()

# @app.get("/")
# async def home():
#     return {"message": "Welcome to the Image Prediction API"}

# @app.get("/train")
# async def train_route():
#     os.system("python main.py")
#     return {"message": "Training done successfully!"}

# @app.post("/predict")
# async def predict_route(image_input: ImageInput):
#     image = image_input.image
#     # Decode the base64 image string
#     decodeImage(image, clApp.filename)
#     # Make prediction using the pipeline
#     result = clApp.classifier.predict()
#     # Return the result as a JSON response
#     return JSONResponse(content=result)


# # Run the app using uvicorn (this is the entry point)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from Braintumor.utils.common import decodeImage
from Braintumor.pipeline.predict import PredictionPipeline


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    # app.run(host='0.0.0.0', port=8080) #local host
    # app.run(host='0.0.0.0', port=8080) #for AWS
    app.run(host='0.0.0.0', port=80) #for AZURE