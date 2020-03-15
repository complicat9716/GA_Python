import numpy as np
from ToolClasses import EmptyIndividual, ResultClass     # import tools


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

    # Best Individual (best parameters and best cost)
    BestIndividual = EmptyIndividual()

    # inf for find minimum, -inf for find the maximum
    if FindMin:
        BestIndividual.cost = np.inf
    else:
        BestIndividual.cost = -np.inf

    # flag for update the best
    UpdateFlag = None

    # Initial parent population with empty Individual Template
    parent_pop = []
    for i in range(nPop):
        empty_individual = EmptyIndividual()
        empty_individual.position = np.random.uniform(VarMin, VarMax, nVar)     # Generate random variables
        empty_individual.cost = CostFunction(empty_individual.position)         # Get the score
        parent_pop.append(empty_individual)                                     # Add individual into the population

        # find the minimum or maximum based on the user input
        if FindMin:
            UpdateFlag = empty_individual.cost < BestIndividual.cost
        else:
            UpdateFlag = empty_individual.cost > BestIndividual.cost

        # update the best individual
        if UpdateFlag:
            BestIndividual = empty_individual.deepcopy()


    # result
    result = ResultClass()
    result.last_iteration_pop = parent_pop

    return result