
import datetime
import math

def calculateBiometer(t, cycle):
    return math.sin( (2 * math.pi / cycle ) * t )

def analyseBiometer(value, nextDayValue, name): 
    if value > 0.5:
        print(f"Twój {name} biometr jest dziś bardzo wysoki! Brawo c:")
    else:
        print(f"Niestety twój {name} biometr jest dziś niski.", end=" ")
        if nextDayValue > 0.5:
            print(f"Nie martw się, jutro będzie lepiej, bo aż {round(nextDayValue, 2)}.")
        else:
            print(f"Niestety jutro też będzie słabo ({round(nextDayValue, 2)}).")
        

name = str(input("Podaj swoje imię: "))
y = int(input("Podaj rok urodzenia: "))
m = int(input("Podaj miesiąc urodzenia: "))
d = int(input("Podaj dzień urodzenia: "))

# dni życia
today = datetime.date.today()
dateOfBirth = datetime.date(y, m, d)

t = (today - dateOfBirth).days

print(f"Witaj {name}! Dzisiaj jest twój {t} dzień.")

# ===================== Biometr ============================

# a)
t_next = t + 1
# Fizyczna fala
yp = calculateBiometer(t, 23)
yp_next = calculateBiometer(t_next, 23)
# Emocjonalna fala
ye = calculateBiometer(t, 28)
ye_next = calculateBiometer(t_next, 28)
#Intelektualna fala
yi = calculateBiometer(t, 33)
yi_next = calculateBiometer(t_next, 33)

print(f"Dzisiejszy biometr:\n Fizyczna fala: {round(yp, 2)}\n Emocjonalna fala: {round(ye,2)}\n Intelektualna fala: {round(yi,2)}")

# b)
analyseBiometer(yp, yp_next, "fizyczny")
analyseBiometer(ye, ye_next, "emocjonalny")
analyseBiometer(yi, yi_next, "intelektyalny")

# c) 10 - 15 min

# d)
"""
import math
from datetime import date

def get_biorhythm(t, period):
    # Oblicza wartość funkcji sinus dla danego okresu (23, 28 lub 33)
    return math.sin((2 * math.pi / period) * t)

def analyze_score(score, next_score, name_pl):
    # Sprawdza i komentuje wynik zgodnie z wytycznymi
    if score > 0.5:
        print(f"Gratulacje! Twój cykl {name_pl} jest dzisiaj wysoki ({score:.2f}). Wykorzystaj ten czas!")
    elif score < -0.5:
        print(f"Twój cykl {name_pl} jest dzisiaj niski ({score:.2f}). Bądź dla siebie wyrozumiały/a.")
        # Jeśli wynik jest bardzo niski, sprawdzamy czy jutro będzie lepiej
        if next_score > score:
            print("Nie martw się. Jutro będzie lepiej!")

def main():
    # Pobieranie danych od użytkownika
    print("--- Kalkulator Biorytmów ---")
    name = input("Podaj swoje imię: ")
    
    try:
        year = int(input("Podaj rok urodzenia (np. 1990): "))
        month = int(input("Podaj miesiąc urodzenia (1-12): "))
        day = int(input("Podaj dzień urodzenia (1-31): "))
        
        birth_date = date(year, month, day)
    except ValueError:
        print("Błąd: Wprowadzono niepoprawną datę. Upewnij się, że wpisujesz liczby.")
        return

    today = date.today()
    
    # Obliczanie różnicy w dniach
    t = (today - birth_date).days
    
    if t < 0:
        print("Błąd: Data urodzenia nie może być w przyszłości!")
        return

    print(f"\nWitaj, {name}!")
    print(f"Dzisiaj jest Twój {t}. dzień życia.\n")

    # Obliczanie wyników na dziś
    phys_score = get_biorhythm(t, 23)
    emot_score = get_biorhythm(t, 28)
    intel_score = get_biorhythm(t, 33)

    # Obliczanie wyników na jutro (t + 1)
    t_next = t + 1
    phys_next = get_biorhythm(t_next, 23)
    emot_next = get_biorhythm(t_next, 28)
    intel_next = get_biorhythm(t_next, 33)

    # Wyświetlanie surowych wyników
    print("Twoje dzisiejsze wyniki biorytmów (od -1.0 do 1.0):")
    print(f"Fizyczny:      {phys_score:.2f}")
    print(f"Emocjonalny:   {emot_score:.2f}")
    print(f"Intelektualny: {intel_score:.2f}\n")

    # Analiza i komunikaty z pocieszeniem/gratulacjami
    analyze_score(phys_score, phys_next, "fizyczny")
    analyze_score(emot_score, emot_next, "emocjonalny")
    analyze_score(intel_score, intel_next, "intelektualny")

if __name__ == "__main__":
    main()
"""