from simpful import *

FS = FuzzySystem()

TLV = AutoTriangle(3, terms=['poor', 'average', 'good'], universe_of_discourse=[0,10])
FS.add_linguistic_variable("service", TLV)
FS.add_linguistic_variable("quality", TLV)
FS.add_linguistic_variable("food", TLV)

O1 = TriangleFuzzySet(0,0,10,   term="low")
O2 = TriangleFuzzySet(0,10,30,  term="medium") # to są granice zbiorów rozmytych, 0-13 niski, 13-25 sredni, 25-25 wysoki
O3 = TriangleFuzzySet(15,30,30, term="high")
FS.add_linguistic_variable("tip", LinguisticVariable([O1, O2, O3], universe_of_discourse=[0,25]))

FS.add_rules([
	"IF (quality IS poor) OR (service IS poor) OR (food IS poor) OR (food IS average) THEN (tip IS low)",
	"IF (service IS average) THEN (tip IS medium)",
	"IF (quality IS good) OR (service IS good) OR (food IS good) THEN (tip IS high)",
	])

FS.set_variable("quality", 6.5) 
FS.set_variable("service", 9.8) 
FS.set_variable("food", 2.0)

tip = FS.inference()
print(tip)
FS.plot_variable("quality")
FS.plot_variable("service") # stopień przynależności klasy napiwku na podstawie osi X