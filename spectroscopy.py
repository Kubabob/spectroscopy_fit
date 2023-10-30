import pygad as pg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy as sp

def derivative(f1,f2,diff=0.02): #liczenie pochodnej z definicji, mniejsza roznica > wieksza zgodnosc
    return (f1-f2)/diff

def second_der(E, m, A, E_ck, gamma, ph_angle): #wzor z JAP2017
    try:
        if m != 0:
            return (-m*(m-1)*A*np.exp(1j*ph_angle)*(E-E_ck+1j*gamma))**(m-2)
        else:
            return (A*np.exp(1j*ph_angle)*(E-E_ck+1j*gamma))**(-2)
    except:
        return 0


input_data = pd.read_csv('P3HT_e1_e2.csv',sep=';')

e1 = input_data['e1']
e2 = input_data['e2']
eV = input_data['eV']


def fitting_method(n,phase_angle=None):
    e1_first_der = np.array([derivative(e1[i], e1[i + 1]) for i in range(len(e1) - 1)])
    e1_second_der = np.array([derivative(e1_first_der[i], e1_first_der[i + 1]) for i in range(len(e1_first_der) - 1)])

    e2_first_der = np.array([derivative(e2[i], e2[i + 1]) for i in range(len(e2) - 1)])
    e2_second_der = np.array([derivative(e2_first_der[i], e2_first_der[i + 1]) for i in range(len(e2_first_der) - 1)])

    #kazda kolejna pochodna ma o 1 mniej punktow niz poprzednia

    def fitness_func(ga_instance, solution, solution_idx):
        #macierz oscylatorow

        if phase_angle:
            oscillators = np.array([[second_der(E, solution[0 * n + i], solution[1 * n + i], solution[2 * n + i],
                                                solution[3 * n + i], phase_angle) for E in eV] for i in range(n)])
        else:
            oscillators = np.array([[second_der(E, solution[0*n+i], solution[1*n+i], solution[2*n+i], solution[3*n+i],
                                                solution[-1]) for E in eV] for i in range(n)])
        #oscylator wyjsciowy
        oscillators_sum = oscillators.sum(axis=0)[:len(e1_second_der)]
        RMSE = np.sqrt(np.sum((np.real(oscillators_sum) - e1_second_der)**2 + (np.imag(oscillators_sum) - e2_second_der)**2))

        #oscylator SCPM bazuje na liczbach zespolonnych wiec liczymy dlugosc liczby zespolonej aby mierzyc odchylenie w kazdym kierunku
        return -np.sqrt(np.real(RMSE)**2 + np.imag(RMSE)**2)
        #nie wiem czy powyzszy pomysl ma sens wiec w razie czego jest tez mozliwosc mierzenia odchylenia tylko w plaszczyznie rzeczywistej
        #return -np.real(RMSE)

    gene_space = []
    gene_space.extend([[-0.5, 0, 0.5] for _ in range(n)]) # m
    gene_space.extend([np.arange(0.1, 20, 0.01) for _ in range(n)]) # A
    gene_space.extend([np.arange(0.2, 6, 0.01) for _ in range(n)]) # E_ck
    gene_space.extend([np.arange(0.1, 20, 0.01) for _ in range(n)]) # gamma
    #nie wiem ile wynosi phase angle wiec dodalem go jako kolejny parametr
    gene_space.append(np.arange(0.1,10,0.01)) # phase angle

    if phase_angle:
        ga_instance = pg.GA(
            num_generations=50,
            num_parents_mating=40*n,
            fitness_func=fitness_func,
            sol_per_pop=160*n,
            num_genes=4*n,
            gene_type=[float, 2],
            crossover_type='uniform',
            parent_selection_type='rank',
            gene_space=gene_space,
            keep_elitism=n,
            crossover_probability=0.8,
            mutation_probability=0.1,
            random_mutation_min_val=-0.5,
            random_mutation_max_val=0.5
        )
    else:
        ga_instance = pg.GA(
            num_generations=50,
            num_parents_mating=10*((4*n)+1),
            fitness_func=fitness_func,
            sol_per_pop=40*((4*n)+1),
            num_genes=(4*n)+1,
            gene_type=[float, 2],
            crossover_type='uniform',
            parent_selection_type='rank',
            gene_space=gene_space,
            keep_elitism=n,
            crossover_probability=0.8,
            mutation_probability=0.3,
            random_mutation_min_val=-0.5,
            random_mutation_max_val=0.5
        )
    st = time.time()
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f'solution: {solution}\nfitness: {solution_fitness}\ntime: {time.time()-st}')

    if phase_angle:
        oscillators_sum = np.array([[second_der(E, solution[0 * n + i], solution[1 * n + i], solution[2 * n + i],
                                                solution[3 * n + i], phase_angle) for E in eV] for i in range(n)]).sum(axis=0)[:len(e1_second_der)]
    else:
        oscillators_sum = np.array([[second_der(E, solution[0 * n + i], solution[1 * n + i], solution[2 * n + i],
                                        solution[3 * n + i], solution[-1]) for E in eV] for i in range(n)]).sum(axis=0)[:len(e1_second_der)]


    plt.plot(eV, e1, 'g')
    plt.plot(eV, e2, 'g--')

    plt.plot(eV[:-2], np.real(oscillators_sum), 'y')
    plt.plot(eV[:-2], e1_second_der, 'y--')

    plt.plot(eV[:-2], np.imag(oscillators_sum), 'r')
    plt.plot(eV[:-2], e2_second_der, 'r--')
    plt.grid()
    plt.show()


for i in range(5):
    fitting_method(5)
