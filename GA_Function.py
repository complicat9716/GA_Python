import numpy as np
from ToolClasses import ResultClass     # import tools

class EmptyIndividual():
    def __init__(self):
        self.position = None
        self.cost = None


def GA(problem, params):
    
    # Problem Definition
    CostFunction = problem.CostFunction     # Cost fucntion
    nVar = problem.nVar                     # Number of variables
    VarMin = problem.VarMin                 # Lower bound
    VarMax = problem.VarMax                 # Upper bound
    FindMin = problem.FindMin               # find minimum or maximum

    # GA Parameters
    MaxIt = params.MaxIt                    # Maximum number of iterations
    nPop = params.nPop                      # parent population
    beta = params.beta                      # Parent score coefficient for picking parents(<0 for find the minimum, >0 for find the maximum)
    c_percentage = params.c_percentage      # Percentage of offspring compare to Population size
    Mutation_rate = params.Mutation_rate    # Mutation rate of GA
    sigma = params.sigma                    # Standard deviation of the gaussian noise in the mutation
    ShowIterInfo = params.ShowIterInfo      # =true to show the iterations


    # Initial parent population with empty Individual Template
    parent_pop = []
    for i in range(nPop):
        empty_individual = EmptyIndividual()
        empty_individual.position = np.random.uniform(VarMin, VarMax, nVar)     # Generate random variables
        empty_individual.cost = CostFunction(empty_individual.position)         # get the score
        parent_pop.append(empty_individual)



    return parent_pop