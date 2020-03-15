# Problem class
class ProblemClass():
    def __init__(self):
        self.CostFunction = None
        self.nVar = None
        self.VarMin = None
        self.VarMax = None
        self.FindMin = None
        
# Parameters class
class ParamsClass():
    def __init__(self):
        self.MaxIt = None
        self.nPop = None
        self.beta = None
        self.c_percentage = None
        self.Mutation_rate = None
        self.sigma = None
        self.ShowIterInfo = None

# Result class
class ResultClass():
    def __init__(self):
        self.last_iteration_pop = None
        self.BestCosts_List = None
        self.BestCandidate_List = None
        self.GlobalBest = None