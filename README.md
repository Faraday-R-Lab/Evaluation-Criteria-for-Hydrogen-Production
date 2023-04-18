# HySupply Electrolyser Model

> Techno-economic model for green hydrogen electrolysis

The HySupply analysis tool has been developed to evaluate the cost of generating hydrogen through exclusively renewable powered electrolysis (using both Alkaline Electrolyte Electrolyser (AE) and Polymer Membrane Electrolyte Electrolyser (PEM)) in Australia through an in-depth techno-economic analysis.<br><br>
The tool can be used to create a grid-connected or standalone model based on a solar, wind or hybrid generator. There is also the option to add battery storage, and sell surplus generation to the grid. The key inputs are defined when creating the object though other variables such as the costs and electrolyser operating parameters can be changed in the config file.

## Installation

Download and unzip the folder. Running the model requires python 3. Other package requirements are listed in the requirements.txt.

## Example

```python
from HydrogenModel import HydrogenModel

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
print(f"LCH2 is {model1.calculate_costs('fixed')}")
# Call make_duration_curve with no parameters to display the annual duration curve for the generator
model1.make_duration_curve()
```

```
This model has inputs:
Location = REZ-N1
Electrolyser Capacity = 10
Solar Capacity = 10
Wind Capacity = 10
Battery Power = 0
Battery duration = 0
=======================Outputs============================
Generator Capacity Factor: 0.32
Time at rated Capacity: 0.19
Time operating: 0.87
Electrolyser Capacity Factor: 0.58
Energy Input [MWh/yr]: 50847.39
Surplus Energy [MWh/yr]: 4845.84
Hydrogen production (fixed) [t/yr]: 799.17
Hydrogen production (variable) [t/yr]: 862.94
LCH2 is 4.9 A$/kg
```
