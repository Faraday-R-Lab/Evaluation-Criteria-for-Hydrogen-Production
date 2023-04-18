"""Example usage of the HySupply Tool

This script provides example usage of the HySupply Hydrogen Model class.
"""

from HydrogenModel import HydrogenModel


def example1():
    # Create an instance of the model for a 10 MW PEM electrolyser with 10 MW solar and 10 MW wind, with excess
    # generation sold to the grid for A$30/MWh
    model1 = HydrogenModel(elec_capacity=10, elec_type="PEM", solar_capacity=10, wind_capacity=10, spot_price=30)
    # Call "print" on the object to print a list of the user and default inputs
    print(model1)
    # Call calculate_electrolyser_output to receive a dictionary with a summary of the results of the model
    outputs = model1.calculate_electrolyser_output()
    print("=======================Outputs============================")
    for key in outputs:
        print(f"{key}: {round(outputs[key],2)}")
    # Call the method calculate_costs to get the LCH2
    print(f"LCH2 is {model1.calculate_costs('fixed')} A$/kg\n")
    # Call make_duration_curve with no parameters to display the annual duration curve for the generator
    model1.make_duration_curve()


def example2():
    # Create an instance of the model for a 10 MW AE electrolyser with 15 MW solar and a 10 MW 2 hour battery (20MWh)
    model2 = HydrogenModel(elec_capacity=10, elec_type="AE", solar_capacity=15, battery_power=10, battery_hours=2)
    # Calculate and print the LCH2
    print(f"LCH2 for model2 is {model2.calculate_costs('fixed')}\n")


def example3():
    # Create an instance of the model for a 10 MW grid-connected electrolyser with 15 MW wind farm bought via an
    # A$50/MWh PPA and print the results summary
    model3 = HydrogenModel(elec_capacity=10, solar_capacity=0, wind_capacity=100, ppa_price=50)
    outputs = model3.calculate_electrolyser_output()
    for key in outputs:
        print(f"{key}: {round(outputs[key],2)}")
    print(f"LCH2 for model3 is {model3.calculate_costs('fixed')}\n")


if __name__ == '__main__':
    example1()
    example2()
    example3()
