"""HySupply Inputs Optimisation

This script can be used to run optimisations on the HySupply Hydrogen Model class to determine the inputs that result
in the minimum LCH2 for a particular site.
"""

import pandas as pd
import numpy as np
import time
from HydrogenModel import HydrogenModel
from scipy import optimize
import itertools as it
import os


def optimise_lcoh():
    # Uses Scipy's brute force algorithm to run the Hydrogen Model for all inputs in an n-dimensional array.
    # A minimum value can be set for the electrolyser CF to find the minimum cost such that a set output is achieved.
    start_time = time.time()

    min_elec_cf = 0
    elec_type = "AE"
    inputs_dict = {'Elec_Capacity': slice(100, 1100, 100),
                   'Solar_Oversize': slice(0, 3, 0.5),
                   'Wind_Oversize': slice(0, 3, 0.5),
                   'Battery_Power': slice(0, 2.5, 0.5),
                   'Battery_Hours': slice(0, 4, 1)}

    if not os.path.exists('Results'):
        os.mkdir('Results')

    optimal_df = pd.DataFrame.from_dict(inputs_dict, orient='index', columns=['INPUTS'])
    optimal_df = optimal_df.append(pd.Series(name='LCH2', dtype=float))
    input_ranges = tuple(inputs_dict.values())
    #sites = pd.read_csv('Data/solar-traces.csv', nrows=0).columns[1:]
    sites = ['REZ-S1']

    for location in sites:
        best_inputs = optimize.brute(get_model_lcoh, input_ranges, (location, min_elec_cf, elec_type), Ns=3,
                                     finish=None, full_output=True, workers=-1)

        print(f"{location}: Lowest LCOH of {best_inputs[1]} found for inputs {best_inputs[0]}")

        optimal_df[location] = np.append(best_inputs[0], best_inputs[1])
    optimal_df.to_csv(f'Results/Optimisation_outputs_{start_time}.csv')
    df_cols = list(inputs_dict.keys()) + ['LCH2']
    get_full_outputs(best_inputs, df_cols, f'Results/Full_outputs_{start_time}_{sites[-1]}.csv')
    elapsed_time = (time.time() - start_time) / 60
    print(f"Process finished after {elapsed_time} minutes and outputs saved to the Results folder")


def get_model_lcoh(args_vector, *params):
    elec_capacity = args_vector[0]
    solar_capacity = args_vector[1] * elec_capacity
    wind_capacity = args_vector[2] * elec_capacity
    battery_hours = int(2 ** args_vector[4])
    battery_power = args_vector[3] * elec_capacity
    location = params[0]
    min_elec_cf = params[1]
    elec_type = params[2]
    if solar_capacity + wind_capacity == 0:
        return 500.0

    case = HydrogenModel(elec_capacity=elec_capacity, solar_capacity=solar_capacity, wind_capacity=wind_capacity,
                         battery_power=battery_power, battery_hours=battery_hours, location=location,
                         elec_type=elec_type)
    model_outputs = case.calculate_electrolyser_output()
    if model_outputs["Achieved Electrolyser Capacity Factor"] > min_elec_cf:
        return case.calculate_costs('fixed')
    else:
        # return an arbitrarily large LCH2 so that the case is rejected by the optimiser
        return 500.0


def get_full_outputs(optimisation_results, out_col_names, filename):
    opti_params = len(optimisation_results[0])
    grid = optimisation_results[2]
    jout = optimisation_results[3]

    cols = 'abcdefghijklmnopqrstuvwxyz'  # Not particularly robust...
    # columns is number of inputs + 2
    grid_cols = cols[:opti_params+2]

    df = pd.DataFrame(
        data=[(*axes, v.item()) for axes, v in zip(it.product(*[range(i) for i in grid.shape]), np.nditer(grid))],
        columns=tuple(grid_cols))
    df = df.pivot_table(columns=grid_cols[0], values=grid_cols[-1], index=list(grid_cols[1:-1]))

    df2 = pd.DataFrame(
        data=[(*axes, v.item()) for axes, v in zip(it.product(*[range(i) for i in jout.shape]), np.nditer(jout))],
        columns=tuple(grid_cols[1:]))
    df2 = df2.pivot_table(values=grid_cols[-1], index=list(grid_cols[1:-1]))
    df3 = df.merge(df2, left_index=True, right_index=True)
    df3.columns = out_col_names
    df3.to_csv(filename, index=False)


if __name__ == "__main__":
    optimise_lcoh()
