import math

def sigmoid(x):
    return 1 / ( 1 + math.exp(-x))

# Wejście
x1 = 0.6
x2 = 0.1

# Wyjście
y_true = 0.8

# neuron h1
w1 = 0.2
w2 = -0.3
b1 = 0.4

#neuron h2
w3 = -0.5
w4 = 0.1
b2 = -0.2

# wyjście
w5 = 0.3
w6 = -0.4
b3 = 0.2

def forward_propagation(x1, x2):
    z1 = x1*w1 + x2*w2 + b1
    h1 = sigmoid(z1)
    
    z2 = x1*w3 + x2*w4 + b2
    h2 = sigmoid(z2)
    
    y = h1*w5 + h2*w6 + b3
    
    return h1, h2, y

def lost(y):
    return ( y - y_true)**2 / 2 
    

def back_propagation():
    global w1, w2, b1, w3, w4, b2, w5, w6, b3
    
    # Dla każdej wagi chcemy policzyć pochodną postaci - gradient:
    # ∂L/∂w
    # Taka pochodna mówi, jak bardzo zmieni się funkcja straty, gdy lekko zmienimy wagę w.
    # •Jeśli pochodna jest dodatnia, to zwiększanie wagi zwiększa stratę.
    # •Jeśli pochodna jest ujemna, to zwiększanie wagi zmniejsza stratę.
    
    # Dlatego w metodzie gradient descent aktualizujemy parametry według wzoru:
    # wnew = wold −η( ∂L /  ∂w )
    # gdzie η oznacza współczynnik uczenia.
    # eta * gradient wagi
    
    # Sygnał błędu mówi jak bardzo dany neuron przyczynił się do błędnego wyniku 
    h1, h2, y = forward_propagation(x1, x2)
    delta3 = y - y_true # sygnał błędu neuronu na wyjściu=> -0.5659
    
        
    # gradient wag
    dw5 = delta3 * h1
    dw6 = delta3 * h2
    db3 = delta3
    
    # błędy warstwy ukrytej = błąd następnego * waga po której szło * pochodna z sigmoid aktualnego neuronu
    delta1 = delta3 * w5 * ( h1 * (1 - h1) ) # -0.0400
    delta2 = delta3 * w6 * ( h2 * (1 - h2) ) # 0.0533
    
    # gradienty wag warstwy ukrytej
    dw1 = delta1 * x1 # -0.0240
    dw2 = delta1 * x2 # -0.0040
    db1 = delta1      # -0.0040
    
    dw3 = delta2 * x1 # 0.0320
    dw4 = delta2 * x2 # 0.0053
    db2 = delta2      # 0.0533
    
    # aktualizacja wag
    eta = 0.1 # współczynnik uczenia
    
    w1 = w1 - eta * dw1
    w2 = w2 - eta * dw2
    b1 = b1 - eta * db1
    
    w3 = w3 - eta * dw3
    w4 = w4 - eta * dw4
    b2 = b2 - eta * db2
    
    w5 = w5 - eta * dw5 # 0.3351
    w6 = w6 - eta * dw6 # -0.3785
    b3 = b3 - eta * db3

print(forward_propagation(x1, x2)) # zwraca h1, h2, y
back_propagation()
print(forward_propagation(x1, x2))
print(w1, w2, w3, w4, w5, w6)


    
    