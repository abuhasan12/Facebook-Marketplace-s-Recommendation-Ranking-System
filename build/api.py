from fastapi import FastAPI
from fastapi import File
from fastapi import Form
import uvicorn
from fastapi import UploadFile
import torch
from PIL import Image
import pickle
from fastapi.responses import JSONResponse
from image_processor import ImageProcessor
from text_processor import TextProcessor
from image_classifier import ImageClassifier
from text_classifier import TextClassifier
from combined_classifier import ImageAndTextModel

image_processor = ImageProcessor()
# with open('model/image_decoder.pkl', 'rb') as f:
#     image_decoder = pickle.load(f)
with open('image_decoder.pkl', 'rb') as f:
    image_decoder = pickle.load(f)

n_classes = len(image_decoder)
img_classifier = ImageClassifier(num_classes=n_classes, decoder=image_decoder)
# img_classifier.load_state_dict(torch.load('model/state_dict_image_model.pt', map_location='cpu'), strict=False)
img_classifier.load_state_dict(torch.load('state_dict_image_model.pt', map_location='cpu'), strict=False)

text_processor = TextProcessor()
# with open('model/text_decoder.pkl', 'rb') as f:
#     text_decoder = pickle.load(f)
with open('text_decoder.pkl', 'rb') as f:
    text_decoder = pickle.load(f)

text_classifier = TextClassifier(num_classes=n_classes, decoder=text_decoder)
# text_classifier.load_state_dict(torch.load('model/state_dict_text_model.pt', map_location='cpu'), strict=False)
text_classifier.load_state_dict(torch.load('state_dict_text_model.pt', map_location='cpu'), strict=False)

cmbn_text_processor = TextProcessor(max_length=50)
with open('combined_decoder.pkl', 'rb') as f:
    combined_decoder = pickle.load(f)

cmbn_classifier = ImageAndTextModel(num_classes=n_classes, decoder=combined_decoder)
# cmbn_classifier.load_state_dict(torch.load('model/state_dict_combined_model.pt', map_location='cpu'), strict=False)
cmbn_classifier.load_state_dict(torch.load('state_dict_combined_model.pt', map_location='cpu'), strict=False)

app = FastAPI()
@app.post('/image')
def image_post(image: UploadFile = File(...)):
    img = Image.open(image.file)
    processed_img = image_processor(img)
    prediction = img_classifier.predict(processed_img)
    probs = img_classifier.predict_proba(processed_img)
    classes = img_classifier.predict_classes(processed_img)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(status_code=200, content={'prediction': prediction.tolist(), 'probs': probs.tolist(), 'classes': classes})

@app.post('/text')
def text_post(text: str = Form(...)):
    processed_text = text_processor(text)
    prediction = text_classifier.predict(processed_text)
    probs = text_classifier.predict_proba(processed_text)
    classes = text_classifier.predict_classes(processed_text)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(status_code=200, content={'prediction': prediction.tolist(), 'probs': probs.tolist(), 'classes': classes})

@app.post('/combined')
def text_post(image: UploadFile = File(...), text: str = Form(...)):
    img = Image.open(image.file)
    processed_img = image_processor(img)
    processed_text = cmbn_text_processor(text)
    prediction = cmbn_classifier.predict(processed_text, processed_img)
    probs = cmbn_classifier.predict_proba(processed_text, processed_img)
    classes = cmbn_classifier.predict_classes(processed_text, processed_img)
    print(prediction)
    print(probs)
    print(classes)
    return JSONResponse(status_code=200, content={'prediction': prediction.tolist(), 'probs': probs.tolist(), 'classes': classes})


if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8080)