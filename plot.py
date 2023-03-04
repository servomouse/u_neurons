import matplotlib.pyplot as plt
import csv
from decimal import Decimal
import time


def plot1():
    x = []
    ref = []
    pred = []
    effectiveness = 0
    ef_counter = 0
    result = 100
    treshold = 0.01
    commission = 0.006

    with open('outputs.csv','r') as csvfile:
        lines = csv.reader(csvfile, delimiter=';')
        for row in lines:
            x.append(round(Decimal(row[0]), 2))
            r = float(row[1])   # reference
            p = float(row[2])   # prediction
            ref.append(r)
            pred.append(p)
            ef_counter += 1
            if(r > 0 and p > 0) or (r < 0 and p < 0):
                effectiveness += 1
            if p > treshold:
                result *= 1 + ((r / 10) - commission)

    plt.plot(x, ref, color = 'g', linestyle = 'dashed',label = "Reference")
    plt.plot(x, pred, color = 'r', linestyle = 'dashed',label = "Predictions")

    plt.xticks()    # rotation = 25)
    plt.xlabel(f'Effectiveness {str(effectiveness)}/{str(ef_counter)}, result: {result}')
    plt.ylabel('output')
    plt.title('Network predictions', fontsize = 20)
    plt.grid()
    plt.legend()
    plt.show()


def plot2():
    x = []
    ref = []
    pred = []
    effectiveness = 0
    ef_counter = 0
    result = 100
    treshold = 0.01
    commission = 0.006

    with open('outputs_new.csv','r') as csvfile:
        lines = csv.reader(csvfile, delimiter=';')
        for row in lines:
            x.append(round(Decimal(row[0]), 2))
            r = float(row[1])   # reference
            p = float(row[2])   # prediction
            ref.append(r)
            pred.append(p)
            ef_counter += 1
            if(r > 0 and p > 0) or (r < 0 and p < 0):
                effectiveness += 1
            if p > treshold:
                result *= 1 + ((r / 10) - commission)

    plt.plot(x, ref, color = 'g', linestyle = 'dashed',label = "Reference")
    plt.plot(x, pred, color = 'r', linestyle = 'dashed',label = "Predictions")

    plt.xticks()    # rotation = 25)
    plt.xlabel(f'Effectiveness {str(effectiveness)}/{str(ef_counter)}, result: {result}')
    plt.ylabel('output')
    plt.title('Network predictions', fontsize = 20)
    plt.grid()
    plt.legend()
    plt.show()


plot1()
plot2()
