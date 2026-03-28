import cv2
import os
from collections import defaultdict
import json

def count_birds(img_path):
    img = cv2.imread(img_path)
    
    if img is None:
        print("Nie można wczytać zdjęcia: ", img)
        return
    
    # Konwersja na skalę szarości
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold( # zmiana obrazu z szarości na czarno biały
        gray, 
        255, # maxValue - wartość, którą otrzymują pixele gdy spełniają warunek progowania
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # metoda adaptacyjna - GAUSSIAN_C próg dla danego piksela to suma ważona wartości sąsiednich pikseli
        # Wagi są określane przez okno (rozkład) Gaussa, co oznacza, 
        # że piksele znajdujące się bliżej środka rozpatrywanego obszaru mają większy wpływ na wynik niż te na jego brzegach.
        cv2.THRESH_BINARY_INV,  # Określa co się stanie po wyliczeniu progu
        # jeśli pixel ma wartość większą niż próg, jest czarny, jeśli nie to biały
        15, # Rozmiar bloku sąsiedztwa, analizuje kwadrat 15x15
        6   # Stała odejmowana od średniej - reguluje czułość na jasność
    ) # wylicza w kwadracie 15x15 lokalny próg, bada czy pixel < od progu i daje mu kolor biały lub czarny na tej podstawie

    base_name = os.path.basename(img_path)
    filename, ext = os.path.splitext(base_name)
    os.makedirs("thresh_miniatures", exist_ok=True)
    cv2.imwrite(f"thresh_miniatures/{filename}_thresh{ext}", thresh)

    # Zwraca listę konturów, znalezionych kształtów (listę punktów z których się składają)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    # RETR_EXTERNAL - tylko kontury obiektu, CHAIN - alg decydujący ile opisujących obiekt pkt ma być
    # print(contours)
    
    # Filtrowanie konturów: liczymy tylko te, które są większe od zera,
    # aby wyeliminować pojedyncze, przypadkowe piksele (szum).
    bird_count = 0
    for contour in contours:
        area = cv2.contourArea(contour) # liczy pole powierzchni obiektu
        if area > 0:
            bird_count += 1

    return bird_count
    
    
img_source = "bird_miniatures"
files = os.listdir(img_source)

detections = defaultdict(int)

for f in files:
    img_path = os.path.join(img_source, f)
    
    birds = count_birds(img_path)
    detections[f] = birds
    
print(json.dumps(detections, indent=4))