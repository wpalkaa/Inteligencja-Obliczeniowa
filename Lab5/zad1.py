from ultralytics import YOLO
import cv2
import os
import json
from collections import defaultdict

# cap = cv2.VideoCapture("street_yolo.mp4")

# while True:    
#     # Wczytywanie klatki, dzięki pętli robi się cały film
#     _, frame = cap.read()

#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(10)
    
#     # Zamknięcie filmu na ESC
#     if key == 27:
#         break
    
# cap.release()
# cv2.destroyAllWindows()

# Load a pretrained YOLO model (recommended for training)


# yolov8n - najnowsze yolo w wersji nano, wykrywa 80 klas, trenowany na COCO

output_dir = "results"

def process_image(image_path, model, thresholds):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Błąd wczytywania pliku {img}.")
        return
        
    for t in thresholds:
        results = model.predict(source=img, conf=t, save=False)
        
        detections = []
        img_drawn = img.copy()

        # Rysowanie obramowań
        for result in results:
            boxes = result.boxes # w boxes są wszystkie ramki - elementy wykryte
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist() # trzeba [0] bo yolo pakuje to w tensor
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "cordinatex": [x1, y1, x2, y2],
                })
                
                # Rysowanie
                cv2.rectangle(img_drawn, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name} - {confidence:.2f}"
                cv2.putText(img_drawn, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                
        # Zapis do JSON
        output_file_path = os.path.join(output_dir, f"image_detections_conf_{t}.json")
        with open(output_file_path, 'w') as f:
            json.dump(detections, f, indent=4)
            
        img_path = os.path.join(output_dir, f"image_annotated_conf_{t}.jpg")
        cv2.imwrite(img_path, img_drawn)
        print(f"Zapisano zdjęcie: {img_path} oraz plik {output_file_path}")
            

def process_video(video_path, model, thresholds):
    for t in thresholds:
        cap = cv2.VideoCapture(video_path) # zamiast wczytywać cały film, wczytuje go klatka po klatce
        
        if not cap.isOpened:
            print("Błąd wczytania filmu: ", video_path)
            return
        
        # Parametry dla outputu
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_path = os.path.join(output_dir, f"video_annotated_conf_{t}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        detections = []
        class_stats = defaultdict(int)
        frame_index = 0
        
        while cap.isOpened():
            ok, frame = cap.read()
            
            if not ok:
                break
            
            results = model.predict(source=frame, conf=t, save=False)
            frame_detections = []
            
            for result in results: # results ma zawsze jeden element bo tylko jedno zdjęcie wczytuje
                boxes = result.boxes # w boxes są wszystkie ramki - elementy wykryte
                
                for box in boxes:
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    frame_detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "cordinatex": [x1, y1, x2, y2],
                    })
                        
                    # statsytki
                    class_stats[class_name] += 1
                    
                    # Rysowanie
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{class_name} - {confidence:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)

            # Dodaje do spisu wykryć, wykrycia z danej klatki
            detections.append({
                "frame": frame_index,
                "detections": frame_detections
            })
            
            output.write(frame)
            frame_index += 1
            
        # Zamyka pliki, zapisuje je
        cap.release()
        output.release()
        
        # Zapisywanie
        json_path = os.path.join(output_dir, f"video_detections_conf_{t}.json")
        with open(json_path, 'w') as f:
            json.dump(detections, f, indent=4)
            
        stats_path = os.path.join(output_dir, f"video_stats_conf_{t}.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("Statystyki wykryć klas w całym filmie:\n")
            for cls, count in class_stats.items():
                line = f"- {cls}: {count}\n"
                f.write(line)
                
        print(f"Zapisano wideo: {output_path}\nZapisano JSON: {json_path}\nZapisano statystyki: {stats_path}\n")


os.makedirs(output_dir, exist_ok=True)

model = YOLO("yolov8n.pt")
thresholds = [0.1, 0.3, 0.5, 0.7]

process_image("office_yolo.png", model, thresholds)
process_video("street_yolo.mp4", model, thresholds)

print("Gotowe.")






"""
Najważniejsze atrybuty obiektu Results (results[0]):

.boxes: Obiekt zawierający informacje o wykrytych ramkach ograniczających (bounding boxes).

.masks: Obiekt z maskami segmentacyjnymi (jeśli używasz modelu do segmentacji, np. YOLOv8-seg).

.keypoints: Obiekt z punktami kluczowymi (jeśli używasz modelu do detekcji póz, np. YOLOv8-pose).

.probs: Obiekt z prawdopodobieństwami klasyfikacji (jeśli używasz modelu do klasyfikacji obrazu, np. YOLOv8-cls).

.names: Bardzo przydatny słownik mapujący ID klas na ich nazwy tekstowe (np. {0: 'person', 1: 'bicycle', ...}).

.orig_img: Oryginalny obraz (w postaci macierzy NumPy), który został przekazany do modelu.

.path: Ścieżka do oryginalnego pliku (jeśli źródłem była ścieżka na dysku).

Najważniejsze metody obiektu Results:

.plot(): Zwraca obraz (macierz NumPy) z narysowanymi wszystkimi detekcjami (ramkami, maskami, etykietami). To właśnie ta metoda jest wywoływana pod maską, gdy używasz save=True.

.show(): Otwiera okienko systemowe i natychmiast wyświetla obraz z nałożonymi wynikami.

.save(filename="..."): Zapisuje wygenerowany obraz z wynikami do pliku.

.cpu() / .numpy(): Przenosi wszystkie dane z karty graficznej (GPU) do pamięci RAM procesora (CPU) i konwertuje tensory na macierze NumPy.
"""
