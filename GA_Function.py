import numpy as np
import copy
from ToolClasses import EmptyIndividual, ResultClass     # import tools

################################################################################################
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
    offspring_percentage = params.offspring_percentage      # Percentage of offspring compare to Population size
    offspring_PopSize = np.round(offspring_percentage*nPop/2)*2     # two parents have two offsprings
    Mutation_rate = params.Mutation_rate    # Mutation rate of GA
    ShowIterInfo = params.ShowIterInfo      # =true to show the iterations

    ################################################################################################
    # Best Individual (best parameters and best cost)
    BestIndividual = EmptyIndividual()

    # inf for find minimum, -inf for find the maximum
    if FindMin:
        BestIndividual.cost = np.inf
    else:
        BestIndividual.cost = -np.inf

    # flag for update the best
    UpdateFlag = None

    ################################################################################################
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

        # update the best individual (make a deep copy)
        if UpdateFlag:
            BestIndividual = copy.deepcopy(empty_individual)

    ################################################################################################
    # record the best cost and candidate along the iterations
    BestCosts_List = np.empty(MaxIt)
    BestCandidates_List = []

    ################################################################################################
    # Main loop
    for it in range(MaxIt):

        # offspring population
        offspring_pop = []

        for k in range(offspring_PopSize//2):
            
            # random permutation
            q = np.random.permutation(nPop)

            # pick two parents randomly
            parent1 = parent_pop[q[0]]
            parent2 = parent_pop[q[1]]

            ################################################################################################
            # Crossover
            offspring1, offspring2 = crossover(parent1, parent2)

            ################################################################################################
            # mutation


        

    ################################################################################################
    # result
    result = ResultClass()
    result.last_iteration_pop = parent_pop

    return result





# crossover function
def crossover(parent1, parent2):

    # make copies of parent 1 and 2
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)

    #extened range from [0, 1)
    gamma = 0.1

    # distinct call for random numbers
    alpha = np.random.uniform(-gamma, 1+gamma, *offspring1.position.shape)

    # crossover
    offspring1.position = alpha*parent1.position + (1-alpha)*parent2.position
    offspring2.position = (1-alpha)*parent1.position + alpha*parent2.position


    return offspring1, offspring2


# mutation function
def mutation(offspring, Mutation_rate):

    # make a copy of the offspring
    mutated_offspring = copy.deepcopy(offspring)

    # create a random list from 0 to 1
    flag = np.random.rand(*offspring.position.shape) <= Mutation_rate

    # find the index that is true
    indexes = np.argwhere(flag)

    # standard deviation of the gaussian noise
    sigma = 0.1

    # mutate the corresponding indexes
    mutated_offspring.position[indexes] = offspring.position[indexes] + sigma*np.random.randn(*indexes.shape)