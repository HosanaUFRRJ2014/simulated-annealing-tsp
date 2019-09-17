from anneal import SimAnneal
import matplotlib.pyplot as plt
import random

coords = []
# TODO: Gerar vários arquivos grandes com coordenadas e colocar para ler em 
# paralelo. Pelo o que sabemos do GIL (Global Interpreter Locker), vai ficar
# mais rápido. (https://docs.python.org/3/glossary.html#term-global-interpreter-lock)
with open("coord.txt", "r") as f:
    for line in f.readlines():
        line = [float(x.replace("\n", "")) for x in line.split(" ")]
        coords.append(line)

if __name__ == "__main__":
    # coords = [[random.uniform(-1000, 1000), random.uniform(-1000, 1000)] for i in range(100)]
    sa = SimAnneal(coords, stopping_iter=5000)
    sa.anneal()
    sa.visualize_routes()
    sa.plot_learning()
