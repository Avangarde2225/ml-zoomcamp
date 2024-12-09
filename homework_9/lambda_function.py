import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img):
    img = prepare_image(img)
    img_array = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(url):
    print("Downloading image...")
    img = download_image(url)
    
    print("Preprocessing image...")
    img_array = preprocess_image(img)
    
    print("Loading model and making prediction...")
    interpreter = tflite.Interpreter(model_path='model_2024_hairstyle_v2.tflite')
    interpreter.allocate_tensors()
    
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    
    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    
    preds = interpreter.get_tensor(output_index)
    result = float(preds[0][0])
    print(f"Raw prediction: {result:.3f}")
    return result


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    print(f"Model prediction score: {result:.3f}")  # Added print statement
    
    # After making prediction
    prediction = model.predict(X)
    print('Raw prediction scores:', prediction)  # This will show the raw scores
    
    # Return both the class and the confidence score
    return {
        'prediction': result,
        'probability': float(prediction.max())  # Convert to float for JSON serialization
    }