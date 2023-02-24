import matplotlib.pyplot as plt
import csv
from decimal import Decimal
import time
  
x = []
ref = []
pred = []


with open('outputs.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=';')
    for row in lines:
        x.append(round(Decimal(row[0]), 2))
        ref.append(float(row[1]))
        pred.append(float(row[2]))

plt.plot(x, ref, color = 'g', linestyle = 'dashed',label = "Reference")
plt.plot(x, pred, color = 'r', linestyle = 'dashed',label = "Predictions")

plt.xticks(rotation = 25)
plt.xlabel('X')
plt.ylabel('output')
plt.title('Network predictions', fontsize = 20)
plt.grid()
plt.legend()
plt.show()