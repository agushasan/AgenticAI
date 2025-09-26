import numpy as np
import pandas as pd
import pulp as plp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

plt.rcParams.update({'font.size': 16})

# Simulate historical demand data for 14 days (each day has 24 hours)
np.random.seed(42)
days = 14
hours_per_day = 24
total_hours = days * hours_per_day

# Generate realistic daily patterns for each consumer type
time = np.arange(total_hours)

# Modify the demand for weekends (Saturday and Sunday)
def weekend_variation(day_of_week, demand):
    # Apply lower demand on weekends for C1 and C3 (household and business)
    # Apply higher demand on weekends for C2 (public facility) and C4 (industry)
    if day_of_week == 5 or day_of_week == 6:  # Saturday (5) and Sunday (6)
        demand_variation = demand * (1.1 if demand.name in ['C2', 'C4'] else 0.9)
    else:
        demand_variation = demand
    return demand_variation

# Base demand patterns for weekdays
base_demand = {
    "C1": 1000 + 3000 * np.exp(-0.5 * ((time % 24 - 13) / 4) ** 2) +
          100 * np.random.normal(0, 2, total_hours),
    "C2": 2000 + 1000 * np.sin((time % 24 - 7) * np.pi / 12) +
          1500 * np.sin((time % 24 - 18) * np.pi / 6) +
          100 * np.random.normal(0, 2, total_hours),
    "C3": 5000 + 1000 * np.sin((time % 24 - 6) * np.pi / 18) +
          2000 * np.sin((time % 24 - 12) * np.pi / 12) +
          100 * np.random.normal(0, 3, total_hours),
    "C4": 1500 + 500 * np.sin((time % 24 - 8) * np.pi / 10) +
          1000 * np.sin((time % 24 - 20) * np.pi / 8) +
          100 * np.random.normal(0, 2, total_hours),
}

# Ensure that each demand is positive
base_demand = {key: np.maximum(0, value) for key, value in base_demand.items()}

df = pd.DataFrame(base_demand)
df["Hour"] = time % 24
df["Day"] = time // 24

X = df[["Hour", "Day"]]
y = df[["C1", "C2", "C3", "C4"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=24, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

future_hours = np.arange(24)
future_X = pd.DataFrame({"Hour": future_hours, "Day": [7] * 24})
predicted_demand = model.predict(future_X)

# Apply minimum demand of 10 MW to each consumer's predicted demand
min_demand = 10
predicted_demand = np.maximum(predicted_demand, min_demand)

time_series_demand = {col: predicted_demand[:, i] for i, col in enumerate(y.columns)}
demand_df = pd.DataFrame(time_series_demand)

# Generator data
generators = {
    "G1": {"cost": 40, "min": 1000, "max": 4000}, # Wind
    "G2": {"cost": 42, "min": 2000, "max": 7000}, # Hydro
    "G3": {"cost": 47, "min": 1500, "max": 8000}, # Nuclear
    "G4": {"cost": 73, "min": 1000, "max": 10000}, # Geothermal
}

transmission = {
    ("G1", "C1"): 2000, ("G1", "C2"): 4000,
    ("G2", "C2"): 5000, ("G2", "C3"): 5000,
    ("G3", "C1"): 3000, ("G3", "C3"): 4000,
    ("G4", "C2"): 5000, ("G4", "C4"): 4000,
}

# Optimization
solutions = []
for hour in range(hours_per_day):
    prob = plp.LpProblem(f"LMP_Optimization_Hour_{hour}", plp.LpMinimize)
    power = plp.LpVariable.dicts("Power", transmission, lowBound=0, cat="Continuous")

    # Minimize the total cost of generation
    prob += plp.lpSum([power[(g, c)] * generators[g]["cost"] for (g, c) in transmission])

    # Generator limits
    for g in generators:
        total_generation = plp.lpSum([power[(g, c)] for c in demand_df.columns if (g, c) in transmission])
        prob += total_generation >= generators[g]["min"], f"MinGen_{g}_Hour_{hour}"
        prob += total_generation <= generators[g]["max"], f"MaxGen_{g}_Hour_{hour}"

    # Ensure demand equals supply at each hour
    total_demand = plp.lpSum([demand_df.at[hour, c] for c in demand_df.columns])
    total_supply = plp.lpSum([power[(g, c)] for (g, c) in transmission])
    prob += total_supply == total_demand, f"TotalSupplyDemand_Balance_Hour_{hour}"

    # Demand equality for each consumer
    for c in demand_df.columns:
        prob += (
            plp.lpSum([power[(g, c)] for g in generators if (g, c) in transmission]) == demand_df.at[hour, c],
            f"Demand_{c}_Hour_{hour}",
        )

    # Transmission capacity limits
    for (g, c) in transmission:
        prob += power[(g, c)] <= transmission[(g, c)], f"TransCap_{g}_{c}_Hour_{hour}"

    prob.solve()

    # Store the results
    hour_solution = {"Hour": hour}
    for (g, c) in power:
        hour_solution[f"Power_{g}_{c}"] = power[(g, c)].varValue
    solutions.append(hour_solution)

solutions_df = pd.DataFrame(solutions)

# Plotting

# Plotting training data used for prediction
plt.figure(figsize=(10, 5))
for consumer in y_train.columns:
    plt.plot(X_train.index, y_train[consumer], 'o-', label=f"Training {consumer}", linewidth=2)
plt.ylabel("Demand (MW)")
plt.xlabel("Hour of the Day")
plt.title("Training Data Used for Prediction")
plt.legend(('Training C1 (Household)','Training C2 (Business)','Training C3 (Industry)','Training C4 (Public Facility)'))
plt.show()

# Plot 1: Consumer Demand Over Time
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
for consumer in demand_df.columns:
    axs[0, 0].plot(demand_df.index, demand_df[consumer], 'o-', label=consumer, linewidth=3)
axs[0, 0].set_ylabel("Demand (MW)")
axs[0, 0].set_title("Consumer Demand Over Time")
axs[0, 0].legend(('C1 (Household)','C2 (Business)','C3 (Industry)','C4 (Public Facility)'))

# Plot 2: Power Flow from Generators to Consumers
for (g, c) in transmission:
    power_col = f"Power_{g}_{c}"
    axs[0, 1].plot(solutions_df["Hour"], solutions_df[power_col], 'o-', label=f"{g} to {c}", linewidth=3)
axs[0, 1].set_ylabel("Power Flow (MW)")
axs[0, 1].set_title("Power Flow from Generators to Consumers")
axs[0, 1].legend()

# Plot 3: Power Generated by Each Generator
for g in generators:
    gen_power = solutions_df[[col for col in solutions_df.columns if col.startswith(f"Power_{g}_")]].sum(axis=1)
    axs[1, 0].plot(solutions_df["Hour"], gen_power, 'o-', label=g, linewidth=3)
axs[1, 0].set_ylabel("Generated Power (MW)")
axs[1, 0].set_title("Power Generated by Each Generator")
axs[1, 0].legend(('G1 (Wind)','G2 (Hydro)','G3 (Nuclear)','C4 (Geothermal)'))

# Plot 4: Total Generation Cost Over Time
total_cost = solutions_df.apply(
    lambda row: sum(row[f"Power_{g}_{c}"] * generators[g]["cost"] for (g, c) in transmission), axis=1)
axs[1, 1].plot(solutions_df["Hour"], total_cost, 'o-', label="Total Cost", linewidth=3)
axs[1, 1].set_ylabel("Cost (EUR)")
axs[1, 1].set_title("Total Generation Cost Over Time")
axs[1, 1].legend()

plt.xlabel("Hour of the Day")
plt.tight_layout()
plt.show()