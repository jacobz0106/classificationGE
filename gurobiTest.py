from gurobipy import Model, GRB

# Create a new model
m = Model("my_model")

# Enable solver output (set OutputFlag to 1)
m.setParam('OutputFlag', 1)

# Define variables
x = m.addVar(name="x", vtype=GRB.CONTINUOUS)
y = m.addVar(name="y", vtype=GRB.CONTINUOUS)

# Set objective
m.setObjective(2 * x + y, GRB.MAXIMIZE)

# Add constraint: x + y <= 10
m.addConstr(x + y <= 10, "c0")

# Add constraint: x - y >= 3
m.addConstr(x - y >= 3, "c1")

# Optimize model
m.optimize()

# Print the solution
if m.status == GRB.OPTIMAL:
    print('Optimal objective: %g' % m.objVal)
    print('x:', x.X)
    print('y:', y.X)
else:
    print("Optimization was stopped with status", m.status)