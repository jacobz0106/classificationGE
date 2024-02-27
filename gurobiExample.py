import pandas as pd
import numpy as np
print('import gurobi')
import gurobipy as gurobi

from gurobipy import Model, GRB

print('---')
# Create a new model
m = Model("example_model")

# Create variables
x = m.addVar(name="x", lb=0)  # x >= 0
y = m.addVar(name="y", lb=0)  # y >= 0

# Set objective
m.setObjective(2 * x + 3 * y, GRB.MAXIMIZE)

# Add constraint: x + y <= 4
m.addConstr(x + y <= 4, "c0")

# Add constraint: x - y >= 1
m.addConstr(x - y >= 1, "c1")

# Optimize model
m.optimize()