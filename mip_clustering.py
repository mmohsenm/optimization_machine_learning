import pulp
import gurobipy as grb
import numpy as np
import time


def add_pulp_constraint(model, constraint):
    model.addConstraint(constraint)
    return constraint


class Instance(object):
    # Experiment instance class
    def __init__(self, clusters, number_of_data_points):
        self.clusters = clusters
        self.N = number_of_data_points
        self.two_dim_points = np.array([(np.random.normal(0, 1), np.random.normal(0, 1)) for _ in range(self.N)])
        self.distance = {(i, j): np.sqrt(sum((self.two_dim_points[i] - self.two_dim_points[j]) ** 2)) for i in
                         range(self.N) for j in range(self.N)}


instances = {(clusters, data_points): Instance(clusters, data_points) for clusters in [2, 4]
             for data_points in [10, 15, 20, 30]}

grb_runtimes = {}
pulp_runtimes = {}
for (clusters, data_points), instance in instances.items():
    C = range(instance.clusters)
    N = range(instance.N)
    # Setting up clustering MIP models using Gurobi
    grb_model = grb.Model()
    y = grb_model.addVars([(i, j, k) for i in N for j in N for k in C if j > i], ub=1, vtype=grb.GRB.BINARY, name="Y")
    z = grb_model.addVars([(i, k) for i in N for k in C],
                          ub=1,
                          vtype=grb.GRB.BINARY,
                          name="Z")
    inequality_constraints = grb_model.addConstrs(y[i, j, k] - z[i, k] - z[j, k] >= -1 for (i, j, k) in y)
    equality_constraints = grb_model.addConstrs(z.sum(i, "*") == 1 for i in N)
    obj = grb.quicksum(y[i, j, k] * instance.distance[i, j] for (i, j, k) in y)
    grb_model.setObjective(obj)
    grb_model.ModelSense = grb.GRB.MINIMIZE
    grb_model.update()
    grb_model.setParam(grb.GRB.Param.TimeLimit, 1500)
    # Solving Gurobi model
    start = time.time()
    grb_model.optimize()
    end = time.time()
    grb_runtimes[(clusters, data_points)] = end - start
    # Setting up clustering MIP models using Pulp-CBC
    pulp_model = pulp.LpProblem(name="clustering", sense=pulp.LpMinimize)
    y = {(i, j, k): pulp.LpVariable(name="Y[{}{}{}]".format(i, j, k),
                                    cat=pulp.LpContinuous,
                                    lowBound=0,
                                    upBound=1)
         for i in N
         for j in N
         for k in C if j > i}
    z = {(i, k): pulp.LpVariable(name="Z[{}{}]".format(i, k),
                                 cat=pulp.LpBinary,
                                 lowBound=0,
                                 upBound=1)
         for i in N
         for k in C}
    inequality_constraints_pulp = {(i, j, k): add_pulp_constraint(pulp_model, pulp.LpConstraint(
        e=y[i, j, k] - z[i, k] - z[j, k],
        sense=pulp.LpConstraintGE,
        name="inequality_constraint[{}{}{}]".format(i, j, k),
        rhs=-1)) for (i, j, k) in y}
    equality_constraints_pulp = {i: add_pulp_constraint(pulp_model, pulp.LpConstraint(
        e=pulp.lpSum(z[i, k] for k in C),
        sense=pulp.LpConstraintEQ,
        name="equality_constraint[{}]".format(i),
        rhs=1)) for i in N}
    obj = pulp.lpSum([y[i, j, k] * instance.distance[i, j] for (i, j, k) in y])
    pulp_model.setObjective(obj)
    # Solving Pulp-CBC model
    from pulp.solvers import pulp_cbc_path

    _solver = pulp.COIN_CMD(path=pulp_cbc_path, maxSeconds=1500)
    start = time.time()
    status = pulp_model.solve(solver=_solver)
    end = time.time()
    pulp_runtimes[(clusters, data_points)] = end - start
