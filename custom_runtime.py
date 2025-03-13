import base64
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.codecs import NumpyCodec

class CustomDiabeticRetinopathyModel(MLModel):
    async def load(self) -> bool:
        """Load the trained TensorFlow model."""
        model_path = "/app/model/diabetic_retinopathy_model.h5"
        self._model = tf.keras.models.load_model(model_path)
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """Process base64 encoded image, make predictions, and return results."""
        
        # Class names and descriptions for diabetic retinopathy stages
        class_names = {
            0: "No Diabetic Retinopathy (Normal)",
            1: "Mild Diabetic Retinopathy",
            2: "Moderate Diabetic Retinopathy",
            3: "Severe Diabetic Retinopathy",
            4: "Proliferative Diabetic Retinopathy"
        }

        class_descriptions = {
            0: "This class represents the absence of diabetic retinopathy. The retina appears normal, with no visible signs of damage.",
            1: "Mild diabetic retinopathy is characterized by small changes in the blood vessels of the retina, such as microaneurysms.",
            2: "Moderate diabetic retinopathy shows more extensive changes, including hemorrhages, and some retinal ischemia.",
            3: "Severe diabetic retinopathy involves large areas of ischemia, multiple hemorrhages, and cotton wool spots.",
            4: "Proliferative diabetic retinopathy involves abnormal new blood vessels, leading to bleeding and risk of retinal detachment."
        }

        try:
            # Decode base64 input
            encoded_image = payload.inputs[0].data[0]  # Extract first input
            image_bytes = base64.b64decode(encoded_image)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Preprocess image
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0  # Normalize pixel values
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Make prediction
            predictions = self._model.predict(image_array)
            predicted_label = int(np.argmax(predictions, axis=1)[0])
            confidence_scores = predictions.tolist()

            # Get the class name and description for the prediction
            class_name = class_names[predicted_label]
            class_description = class_descriptions[predicted_label]

            # Return response
            return InferenceResponse(
                model_name=self.name,
                outputs=[
                    ResponseOutput(
                        name="predicted_label",
                        shape=[1],
                        datatype="INT64",
                        data=[predicted_label]
                    ),
                    ResponseOutput(
                        name="confidence_scores",
                        shape=[1, len(confidence_scores[0])],
                        datatype="FP32",
                        data=confidence_scores[0]
                    ),
                    ResponseOutput(
                        name="class_name",
                        shape=[1],
                        datatype="BYTES",
                        data=[class_name]
                    ),
                    ResponseOutput(
                        name="class_description",
                        shape=[1],
                        datatype="BYTES",
                        data=[class_description]
                    )
                ]
            )
        except Exception as e:
            return InferenceResponse(
                model_name=self.name,
                outputs=[ResponseOutput(name="error", shape=[1], datatype="BYTES", data=[str(e)])]
            )
