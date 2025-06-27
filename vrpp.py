import gurobipy as gp
from gurobipy import GRB
import random
import time

# Problem Instance
random.seed(1234)
M = 1000  # Priority weighting for vehicle costs
C_L = 7   # Max locations per vehicle
locations = 10
task_count = 30

# Generate tasks with random demands and locations
tasks = [
    {'weight': random.randint(10, 50), 
     'volume': random.randint(20, 60),
     'loc': random.randint(0, locations-1)}
    for _ in range(task_count)
]

# Vehicle Fleet Parameters
num_large = 10
num_small = 10

# Vehicle specifications
large_vehicle_spec = {'type': 'large', 'weight_cap': 200, 'volume_cap': 300, 'cost': 350}
small_vehicle_spec = {'type': 'small', 'weight_cap': 100, 'volume_cap': 150, 'cost': 200}

# Generate vehicle fleet
vehicles = []
for i in range(num_large):
    vehicles.append(large_vehicle_spec.copy())
for i in range(num_small):
    vehicles.append(small_vehicle_spec.copy())

# ======================
# Gurobi Model Formulation (Silent Mode)
# ======================
def new_func(tasks, i, t, r, solution):
    if t[i, r].X > 0.5:
        solution[r]['tasks'].append(i)
        solution[r]['locations'].add(tasks[i]['loc'])

with gp.Env(empty=True) as env:
    env.setParam('OutputFlag', 0)      # Suppress all output
    env.setParam('LogToConsole', 0)    # Suppress console logging
    env.start()
    
    # Create Gurobi model
    model = gp.Model('VRP_Gurobi', env=env)
    
    # Decision Variables
    t = model.addVars(len(tasks), len(vehicles), vtype=GRB.BINARY, name="task_assign")
    o = model.addVars(len(vehicles), vtype=GRB.BINARY, name="vehicle_used")
    d = model.addVars(locations, len(vehicles), vtype=GRB.BINARY, name="location_visit")
    
    # Objective Function
    vehicle_cost = gp.quicksum(M * vehicles[r]['cost'] * o[r] for r in range(len(vehicles)))
    location_cost = gp.quicksum(d[l, r] for l in range(locations) for r in range(len(vehicles)))
    model.setObjective(vehicle_cost + location_cost, GRB.MINIMIZE)
    
    # Constraints
    for i in range(len(tasks)):
        model.addConstr(
            gp.quicksum(t[i, r] for r in range(len(vehicles))) == 1,
            name=f"task_assignment_{i}"
        )
    
    for r in range(len(vehicles)):
        model.addConstr(
            gp.quicksum(tasks[i]['weight'] * t[i, r] for i in range(len(tasks))) <= vehicles[r]['weight_cap'],
            name=f"weight_capacity_{r}"
        )
        model.addConstr(
            gp.quicksum(tasks[i]['volume'] * t[i, r] for i in range(len(tasks))) <= vehicles[r]['volume_cap'],
            name=f"volume_capacity_{r}"
        )
        model.addConstr(
            gp.quicksum(d[l, r] for l in range(locations)) <= C_L * o[r],
            name=f"location_limit_{r}"
        )
        model.addConstr(
            gp.quicksum(t[i, r] for i in range(len(tasks))) <= len(tasks) * o[r],
            name=f"vehicle_usage_{r}"
        )
    
    for l in range(locations):
        for r in range(len(vehicles)):
            model.addConstr(
                gp.quicksum(t[i, r] for i in range(len(tasks)) if tasks[i]['loc'] == l) <= len(tasks) * d[l, r],
                name=f"location_visit_{l}_{r}"
            )
    
    # Only MIPGap, no time limit
    model.Params.MIPGap = 0.01        # 1% optimality gap
    
    # Solve Model with timing
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    solve_time = end_time - start_time

    # ======================
    # Solution Extraction and Output
    # ======================
    if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
        print(f"\nSolve Time: {solve_time:.2f} seconds")
        print(f"Optimal Objective Value: {model.ObjVal}")
        used_vehicles = [r for r in range(len(vehicles)) if o[r].X > 0.5]
        print(f"Vehicles Used: {len(used_vehicles)}")
        solution = {r: {'tasks': [], 'locations': set()} for r in used_vehicles}
        for i in range(len(tasks)):
            for r in used_vehicles:
                new_func(tasks, i, t, r, solution)
        for r in used_vehicles:
            vehicle_type = "large" if r < num_large else "small"
            print(f"\nVehicle {r} ({vehicle_type}):")
            print(f"- Tasks: {sorted(solution[r]['tasks'])}")
            print(f"- Locations: {sorted(solution[r]['locations'])}")
            print(f"- Weight Used: {sum(tasks[i]['weight'] for i in solution[r]['tasks'])}/{vehicles[r]['weight_cap']}")
            print(f"- Volume Used: {sum(tasks[i]['volume'] for i in solution[r]['tasks'])}/{vehicles[r]['volume_cap']}")
        # Location constraint check
        print(f"\nLocation Visit Verification (Max allowed: {C_L}):")
        for r in used_vehicles:
            vehicle_type = "large" if r < num_large else "small"
            locations_visited = len(solution[r]['locations'])
            status = "" if locations_visited <= C_L else "VIOLATION"
            print(f"- Vehicle {r} ({vehicle_type}): {locations_visited} locations {status}")
    elif model.Status == GRB.INFEASIBLE:
        print("Model is infeasible!")
        model.computeIIS()
        model.write("infeasible_model.ilp")
    else:
        print(f"Optimization ended with status: {model.Status}")
