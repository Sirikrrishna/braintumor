# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import os


# class PredictionPipeline:
#     def __init__(self,filename):
#         self.filename =filename


    
#     def predict(self):
#         # load model
#         model = load_model(os.path.join("artifacts","training", "model.h5"))

#         imagename = self.filename
#         test_image = image.load_img(imagename, target_size = (224,224))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis = 0)
#         result = np.argmax(model.predict(test_image), axis=1)
#         print(result)

#         if result[0] == 1:
#             prediction = 'Yes'
#             return [{ "image" : prediction}]
#         else:
#             prediction = 'No'
#             return [{ "image" : prediction}]

# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import os

# class PredictionPipeline:
#     def __init__(self, filename):
#         self.filename = filename
    
#     def predict(self):
#         try:
#             model_path = os.path.join("artifacts", "training", "model.h5")
#             if not os.path.exists(model_path):
#                 raise FileNotFoundError(f"Model file not found at {model_path}")
            
#             # load model
#             model = load_model(model_path)
            
#             if not os.path.exists(self.filename):
#                 raise FileNotFoundError(f"Image file not found: {self.filename}")
            
#             test_image = image.load_img(self.filename, target_size=(224,224))
#             test_image = image.img_to_array(test_image)
#             test_image = np.expand_dims(test_image, axis=0)
#             result = np.argmax(model.predict(test_image), axis=1)
#             print(result)

#             prediction = 'Yes' if result[0] == 1 else 'No'
#             return [{"image": prediction}]
            
#         except Exception as e:
#             print(f"Error in prediction: {str(e)}")
#             return [{"error": str(e)}]


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model = self._load_model()

    def _load_model(self):
        """
        Load the pre-trained model
        """
        try:
            model_path = os.path.join("artifacts", "training", "model.h5")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            return load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self):
        """
        Predict brain tumor presence
        Returns a list of dictionaries with prediction details
        """
        try:
            # Load and preprocess the image
            test_image = image.load_img(self.filename, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)

            # Make prediction
            prediction_prob = self.model.predict(test_image)
            result = np.argmax(prediction_prob, axis=1)

            # Detailed analysis
            if result[0] == 1:
                tumor_presence = 'Tumor Detected'
                probability = round(prediction_prob[0][1] * 100, 2)
                details = 'The image shows characteristics consistent with a brain tumor. Medical consultation is recommended.'
            else:
                tumor_presence = 'No Tumor Detected'
                probability = round(prediction_prob[0][0] * 100, 2)
                details = 'The image appears to be free of tumor indicators.'

            # Construct result
            return [
                {
                    "tumor_presence": tumor_presence,
                    "probability": probability,
                    "details": details
                }
            ]

        except Exception as e:
            print(f"Prediction error: {e}")
            return [
                {
                    "tumor_presence": "Error",
                    "probability": 0,
                    "details": f"An error occurred during analysis: {str(e)}"
                }
            ]