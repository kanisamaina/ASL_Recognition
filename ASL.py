from tensorflow.keras.models import model_from_json

import numpy as np

json_file = open("alphabet_ASL_Model.json", "r")
loaded_json_model = json_file.read()
json_file.close()

model = model_from_json(loaded_json_model)
model.load_weights("alphabet_ASL_Model_weights.h5")

labels = list("ABCDEFGHIJ")

def image_predict(image):
    return labels[np.argmax(model.predict(image))]
