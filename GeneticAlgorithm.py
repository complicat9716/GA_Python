import numpy as numpy
import matplotlib.pyplot as plt
from ToolClasses import ProblemClass, ParamsClass           # import tools
from CostFunction import *                                  # import cost function
from GA_Function import *                                   # import GA

################################################################################################
# Problem Definition
problem = ProblemClass()
problem.CostFunction = sphere
problem.nVar = 5
problem.VarMin = -10
problem.VarMax = 10
problem.FindMin = True

################################################################################################
# GA Parameters
params = ParamsClass()
params.MaxIt = 100
params.nPop = 20
params.beta = 1
params.c_percentage = 1
params.Mutation_rate = 0.001
params.sigma = 0.1
params.ShowIterInfo = True

################################################################################################
# Run GA
result = GA(problem, params)
print(result)


################################################################################################
# Results