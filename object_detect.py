# Object Detection
# pip install Pillow
# pip install transformers
# pip install timm
from PIL import Image
from transformers import pipeline
def Detect_Objects(img):
    detector = pipeline("object-detection")
    img = Image.open(img)
    results = detector(img)
    for result in results:
        print(result["label"], result["score"])
Detect_Objects("test.jpg")
