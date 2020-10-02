from tensorflow.keras.models import model_from_json

import numpy as np

json_file = open("10_alpha_ASL_Model1.json", "r")
loaded_json_model = json_file.read()
json_file.close()

model = model_from_json(loaded_json_model)
model.load_weights("10_alpha_ASL_Model_weights1.h5")

labels = list("ABCDEFGHIJ")

def image_predict(image):
    return labels[np.argmax(model.predict(image))]