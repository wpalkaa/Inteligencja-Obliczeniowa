from ultralytics import YOLO
import cv2
import os
import json
from collections import defaultdict

# https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8 - klasy

def process_image(image_path, model):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Błąd wczytywania pliku {img}.")
        return

    results = model.predict(source=img, conf=0.003, save=False, verbose=False, classes=[4,14,33])
    
    birds_count = 0
    print(results[0].boxes)
    for result in results:
        birds_count += len(result.boxes)
    
    return birds_count
    
    
model = YOLO("yolov8n.pt")

img_source = "bird_miniatures"
files = os.listdir(img_source)

detections = defaultdict(int)

for f in files:
    img_path = os.path.join(img_source, f)
    
    birds = process_image(img_path, model)
    detections[f] = birds
    
print(json.dumps(detections, indent=4))