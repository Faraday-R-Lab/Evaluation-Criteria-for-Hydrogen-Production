"""Hydrogen Model
Defines a class HydrogenModel which mimics the HySupply Cost Tool v1.3 Excel model.
"""

import os
import pandas as pd
import numpy as np
import yaml
import math
from matplotlib import pyplot as plt


class HydrogenModel:
    """
    A class that defines a techno-economic model for green hydrogen production in Australia.

    Attributes
    ----------
    solar_df : pandas dataframe
        a dataframe containing hourly solar traces for 1 year
    wind_df : pandas dataframe
        a dataframe containing hourly wind traces for 1 year
    location : str
        the ID for the location to be modelled. Needs to correspond to a column in solar_df and wind_df
        (default "REZ-N1")
    elecType : str
        the electrolyser type - either "AE" for Alkaline Electrolyte or "PEM" for Polymer Electrolyte Membrane
        (default "AE")
    elecCapacity : int
        the rated capacity of the electrolyser in MW (default 10)
    solarCapacity : float
        the rated capacity of the solar farm in MW (default 10.0)
    windCapacity : float
    the rated capacity of the wind farm in MW (default 0.0)
    batteryPower : float
        the rated power capacity of the battery. Set to 0 to remove battery from the model (default 0)
    batteryHours : int
        the time period that the battery can discharge at full power, must be 0, 1, 2, 4 or 8 (default 0)
    spotPrice : float
        Price that excess generation can be sold to the grid for, in A$/MWh (default 0.0)
    ppaPrice : float
        Price that electricity can be purchased for, in A$/MWh. Setting this value greater than zero results in all
        electricity being bought from the grid and hence CAPEX and OPEX for the generation are ignored (default 0.0)

    Methods
    -------
    calculate_electrolyser_output()
        returns a dictionary with summarised output values for the model including capacity factors, energy input and
        hydrogen production
    calculate_costs(specific_consumtion_type='fixed')
        returns the levelised cost of hydrogen for the model for either "fixed" or "variable" values for the
        specific energy consumption vs electrolyser load
    make_duration_curve(generator=True, electrolyser=False)
        creates annual duration curves for the generator and/or the electrolyser
    """

    def __init__(self, solardata=None, winddata=None, config='Config/config.yml',
                 location='REZ-N1', elec_type='AE', elec_capacity=10, solar_capacity=10.0, wind_capacity=0.0,
                 battery_power=0, battery_hours=0, spot_price=0.0, ppa_price=0.0):
        if solardata is not None:
            self.solar_df = solardata
        elif solar_capacity > 0:
            solarfile = 'Data/solar-traces.csv'
            if not os.path.exists(solarfile):
                raise FileNotFoundError
            self.solar_df = pd.read_csv(solarfile, header=[0], skiprows=[1], index_col=0)
        else:
            self.solar_df = pd.DataFrame()
        if winddata is not None:
            self.wind_df = winddata
        elif wind_capacity > 0:
            windfile = 'Data/wind-traces.csv'
            if not os.path.exists(windfile):
                raise FileNotFoundError
            self.wind_df = pd.read_csv(windfile, header=[0], skiprows=[1], index_col=0)
        else:
            self.wind_df = pd.DataFrame()
        with open(config, 'r') as config_file:
            config_dict = yaml.safe_load(config_file)
        if location in self.solar_df.columns or location in self.wind_df.columns:
            self.location = location
        else:
            raise KeyError("location not found in solar or wind file header.")
        if elec_type not in ['AE', 'PEM']:
            raise ValueError("elec_type must be 'AE' or 'PEM'")
        else:
            self.elecType = elec_type
        self.elecCapacity = elec_capacity
        self.solarCapacity = solar_capacity
        self.windCapacity = wind_capacity
        self.batteryPower = battery_power
        self.batteryHours = battery_hours
        self.spotPrice = spot_price
        self.ppaPrice = ppa_price

        # Fixed values
        try:
            # Electrolyser parameters
            self.elecMaxLoad = config_dict['elecMaxLoad'] / 100
            self.elecOverload = config_dict[self.elecType]['elecOverload'] / 100
            self.elecOverloadRecharge = config_dict[self.elecType]['elecOverloadRecharge']
            self.elecReferenceCap = config_dict['elecReferenceCapacity']
            self.elecCostReduction = config_dict['elecCostReduction']
            self.elecMinLoad = config_dict[self.elecType]['elecMinLoad'] / 100
            self.elecEff = config_dict['elecEff'] / 100
            self.specCons = config_dict[self.elecType]['specCons']  # kWh/Nm3
            self.H2VoltoMass = config_dict['H2VoltoMass']  # kg/m3
            self.MWtokW = 1000  # kW/MW
            self.hydOutput = self.H2VoltoMass * self.MWtokW * self.elecEff  # kg.kWh/m3.MWh
            self.hoursPerYear = 8760
            self.kgtoTonne = 1/1000
            self.stackLifetime = config_dict[self.elecType]['stackLifetime']  # hours before replacement
            self.waterNeeds = config_dict[self.elecType]['waterNeeds']  # kL/ton

            self.genCapacity = self.solarCapacity + self.windCapacity
            self.solarRatio = self.solarCapacity / self.genCapacity
            self.windRatio = self.windCapacity / self.genCapacity

            # Battery parameters
            self.batteryEnergy = self.batteryPower * self.batteryHours
            self.batteryEfficiency = config_dict['batteryEfficiency'] / 100
            self.battMin = config_dict['battMin'] / 100
            self.battLife = config_dict['battLifetime']

            # Costings inputs
            self.solarCapex = config_dict['solarCapex'] * self.MWtokW  # A$/MW
            self.solarCapex = self.__scale_capex(self.solarCapex, self.solarCapacity,
                                                 config_dict['powerplantReferenceCapacity'],
                                                 config_dict['powerplantCostReduction'])
            self.solarCapex = self.__get_capex(self.solarCapex, config_dict['powerplantEquip'],
                                               config_dict['powerplantInstall'], config_dict['powerplantLand'])
            self.solarOpex = config_dict['solarOpex']  # A$/MW
            self.windCapex = config_dict['windCapex'] * self.MWtokW  # A$/MW
            self.windCapex = self.__scale_capex(self.windCapex, self.windCapacity,
                                                config_dict['powerplantReferenceCapacity'],
                                                config_dict['powerplantCostReduction'])
            self.windCapex = self.__get_capex(self.windCapex, config_dict['powerplantEquip'],
                                              config_dict['powerplantInstall'], config_dict['powerplantLand'])
            self.windOpex = config_dict['windOpex']  # A$/MW
            self.batteryCapex = config_dict['batteryCapex']  # A$/kWh
            self.batteryCapex.update({n: self.batteryCapex[n] * self.MWtokW for n in self.batteryCapex.keys()})  # A$/MWh
            self.batteryOpex = config_dict['batteryOpex']  # A$/MW
            self.battReplacement = config_dict['batteryReplacement'] / 100 * self.batteryCapex[self.batteryHours] # A$/MWh
            electrolyserCapexUnscaled = config_dict[self.elecType]['electrolyserCapex'] * self.MWtokW  # A$/MW
            self.electrolyserCapex = self.__scale_capex(electrolyserCapexUnscaled, self.elecCapacity,
                                                        self.elecReferenceCap, self.elecCostReduction)
            self.electrolyserOandM = config_dict[self.elecType]['electrolyserOandM'] / 100 * self.electrolyserCapex  # A$/MW
            # get CAPEX including indirect costs
            self.electrolyserStackCost = config_dict['electrolyserStackCost'] / 100 * self.electrolyserCapex  # A$/MW
            self.electrolyserCapex = self.__get_capex(self.electrolyserCapex, config_dict['elecEquip'],
                                                      config_dict['elecInstall'], config_dict['elecLand'])
            self.waterCost = config_dict['waterCost']  # A$/kL
            self.discountRate = config_dict['discountRate'] / 100  # percentage as decimal
            self.projectLife = config_dict['projectLife']
        except KeyError as e:
            raise KeyError(f"Error: Entry {e} not found in config file.")

        # Empty variables
        self.operating_outputs = {}
        self.LCH2 = {}

    def __str__(self):
        return f"This model has inputs:\nLocation = {self.location}\nElectrolyser Capacity = {self.elecCapacity}\n" \
               f"Solar Capacity = {self.solarCapacity}\nWind Capacity = {self.windCapacity}\n" \
               f"Battery Power = {self.batteryPower}\nBattery duration = {self.batteryHours}"

    def calculate_electrolyser_output(self):
        """Calculates the hourly operation of the electrolyser and returns a dictionary with a summary of the results.

        Returns
        -------
        operating_outputs
            dictionary with keys 'Generator Capacity Factor', 'Time Electrolyser is at its Rated Capacity', 
            'Total Time Electrolyser is Operating', 'Achieved Electrolyser Capacity Factor',
            'Energy in to Electrolyser [MWh/yr]', 'Surplus Energy [MWh/yr]',
            'Hydrogen Output for Fixed Operation [t/yr]', 'Hydrogen Output for Variable Operation [t/yr]'
        """

        working_df = self.__calculate_hourly_operation()
        # Generate results table to mirror the one in the excel tool
        operating_outputs = self.__get_tabulated_outputs(working_df)
        return operating_outputs

    def calculate_costs(self, specific_consumption_type='fixed'):
        """Calculates the levelised cost of hydrogen production for the model

        Parameters
        ----------
        specific_consumption_type : str, optional
            the method by which the electrolyser specific consumption of energy is calculated. This must be either
            "fixed" for a constant value or "variable" for a value that depends on the operating load.

        Returns
        -------
        lcoh
            the LCOH in A$/kg rounded to two decimal places
        """

        if not self.operating_outputs:
            self.calculate_electrolyser_output()

        gen_capex = self.solarCapex * self.solarCapacity + self.windCapex * self.windCapacity
        gen_opex = self.solarOpex * self.solarCapacity + self.windOpex * self.windCapacity

        if specific_consumption_type == "variable":
            annual_hydrogen = self.operating_outputs["Hydrogen Output for Variable Operation [t/yr]"]
        elif specific_consumption_type == "fixed":
            annual_hydrogen = self.operating_outputs["Hydrogen Output for Fixed Operation [t/yr]"]
        else:
            raise ValueError("Specific consumption type not valid, please select either 'variable' or 'fixed'")

        # Calculate the annual cash flows as in the 'Costings' tab of the excel tool
        cash_flow_df = pd.DataFrame(index=range(self.projectLife + 1), columns=['Year', 'Gen_CAPEX', 'Elec_CAPEX',
                                                                                'Gen_OPEX', 'Elec_OandM', 'Power_cost',
                                                                                'Stack_replacement', 'Water_cost',
                                                                                'Battery_cost', 'Total'])
        cash_flow_df['Year'] = range(self.projectLife + 1)

        if self.ppaPrice > 0:
            cash_flow_df.loc[1:, 'Power_cost'] = self.operating_outputs["Energy in to Electrolyser [MWh/yr]"] * \
                                                 self.ppaPrice
        else:
            cash_flow_df.at[0, 'Gen_CAPEX'] = gen_capex
            cash_flow_df.loc[1:, 'Gen_OPEX'] = gen_opex
            cash_flow_df.loc[1:, 'Power_cost'] = -1 * self.operating_outputs[
                "Surplus Energy [MWh/yr]"] * self.spotPrice

        cash_flow_df.at[0, 'Elec_CAPEX'] = self.electrolyserCapex * self.elecCapacity
        cash_flow_df.loc[1:, 'Elec_OandM'] = self.electrolyserOandM * self.elecCapacity
        stack_years = self.__find_stack_replacement_years()
        cash_flow_df.loc[stack_years, 'Stack_replacement'] = self.electrolyserStackCost * self.elecCapacity
        cash_flow_df.loc[1:, 'Water_cost'] = annual_hydrogen * self.waterNeeds * self.waterCost
        cash_flow_df.at[0, 'Battery_cost'] = self.batteryCapex[self.batteryHours] * self.batteryEnergy
        cash_flow_df.loc[1:, 'Battery_cost'] = self.batteryOpex[self.batteryHours] * self.batteryPower
        cash_flow_df.at[10, 'Battery_cost'] += self.battReplacement * self.batteryEnergy
        cash_flow_df['Total'] = cash_flow_df.sum(axis=1)

        # Calculate the annual discounted cash flows for hydrogen and total costs
        discounted_flow = pd.DataFrame(index=range(self.projectLife + 1), columns=['Year', 'Hydrogen_kg',
                                                                                   'Hydrogen_kg_Discounted', 'Total'])
        discounted_flow['Year'] = range(self.projectLife + 1)
        discounted_flow.loc[1:, 'Hydrogen_kg'] = annual_hydrogen / self.kgtoTonne
        discounted_flow['Hydrogen_kg_Discounted'] = discounted_flow['Hydrogen_kg'] * \
            (1 / (1 + self.discountRate)) ** (discounted_flow['Year'])
        discounted_flow['Total'] = cash_flow_df['Total'] * (1 / (1 + self.discountRate)) ** discounted_flow['Year']

        # Calculate the LCH2 as the total discounted costs divided by the total discounted hydrogen produced over the
        # project lifetime
        lcoh = discounted_flow['Total'].sum() / discounted_flow['Hydrogen_kg_Discounted'].sum()
        self.LCH2 = round(lcoh, 2)
        return round(lcoh, 2)

    def __calculate_hourly_operation(self):
        """Private method- Creates a dataframe with a row for each hour of the year and columns Generator_CF,
        Electrolyser_CF, Hydrogen_prod_fixed and Hydrogen_prod_var
        """

        oversize = self.genCapacity / self.elecCapacity
        working_df = pd.DataFrame()
        if self.solarRatio == 1:
            working_df['Generator_CF'] = self.solar_df[self.location]
        elif self.windRatio == 1:
            working_df['Generator_CF'] = self.wind_df[self.location]
        else:
            working_df['Generator_CF'] = self.solar_df[self.location] * self.solarRatio + \
                                         self.wind_df[self.location] * self.windRatio

        has_excess_gen = working_df['Generator_CF'] * oversize > self.elecMaxLoad
        has_insufficient_gen = working_df['Generator_CF'] * oversize < self.elecMinLoad
        working_df['Electrolyser_CF'] = np.where(has_excess_gen, self.elecMaxLoad,
                                                 np.where(has_insufficient_gen, 0,
                                                          working_df['Generator_CF'] * oversize))
        if self.elecOverload > self.elecMaxLoad and self.elecOverloadRecharge > 0:
            working_df['Electrolyser_CF'] = self.__overloading_model(working_df, oversize)
        if self.batteryEnergy > 0:
            if self.batteryHours not in [1, 2, 4, 8]:
                raise ValueError("Battery storage length not valid. Please enter one of 1, 2, 4 or 8")

            working_df['Electrolyser_CF'] = self.__battery_model(oversize, working_df)

        working_df['Hydrogen_prod_fixed'] = working_df['Electrolyser_CF'] * self.hydOutput / self.specCons
        working_df['Hydrogen_prod_variable'] = working_df['Electrolyser_CF'].apply(
            lambda x: x * self.hydOutput / self.__electrolyser_output_polynomial(x))

        return working_df

    def __electrolyser_output_polynomial(self, x):
        """Private method - Calculates the specific energy consumption as a function of the electrolyser operating
        capacity factor
        """

        return 1.25 * x**2 - 0.4286 * x + self.specCons - 0.85

    def __find_stack_replacement_years(self):
        """Private method - Returns a list of the years in which the electrolyser stack will need replacing, defined as
        the total operating time surpassing a multiple of the stack lifetime.
        """

        if len(self.operating_outputs.keys()) == 0:
            self.calculate_electrolyser_output()
        op_hours_per_year = self.operating_outputs["Total Time Electrolyser is Operating"] * self.hoursPerYear
        stack_years = []
        for year in range(1, self.projectLife):
            if math.floor(op_hours_per_year * year / self.stackLifetime) - math.floor(op_hours_per_year * (year - 1) /
                                                                                      self.stackLifetime) == 1.0:
                stack_years.append(year)
        return stack_years

    def __battery_model(self, oversize, cf_profile_df):
        """Private method - Calculates and returns the hourly electrolyser operation profile when a battery is included
        in the model.
        """

        cf_profile_df = cf_profile_df.reset_index()
        index_name = cf_profile_df.columns[0]
        cf_profile_df['Excess_Generation'] = (cf_profile_df['Generator_CF'] * oversize -
                                              cf_profile_df['Electrolyser_CF']) * self.elecCapacity
        cf_profile_df['Battery_Net_Charge'] = 0.0
        cf_profile_df['Battery_SOC'] = 0.0
        cf_profile_df['Electrolyser_CF_batt'] = 0.0
        batt_losses = (1-(1-self.batteryEfficiency)/2)
        elec_min = self.elecMinLoad * self.elecCapacity
        elec_max = self.elecMaxLoad * self.elecCapacity

        cf_profile_df.at[0, 'Battery_Net_Charge'] = min(self.batteryPower,
                                                        cf_profile_df.at[0, 'Excess_Generation'] * batt_losses)
        cf_profile_df.at[0, 'Battery_SOC'] = cf_profile_df.at[0, 'Battery_Net_Charge'] / self.batteryEnergy

        for hour in range(1, len(cf_profile_df)):
            # Iterate over the hours of the year and determine which case it falls into and thus what the battery
            # charging behaviour should be.
            batt_soc = cf_profile_df.at[hour - 1, 'Battery_SOC']
            spill = cf_profile_df.at[hour, 'Excess_Generation']
            elec_cons = cf_profile_df.at[hour, 'Electrolyser_CF'] * self.elecCapacity
            batt_discharge_potential = min(self.batteryPower, (batt_soc - self.battMin) * self.batteryEnergy) * \
                batt_losses
            elec_just_operating = elec_cons > 0 or cf_profile_df.at[hour - 1, 'Battery_Net_Charge'] < 0 \
                or cf_profile_df.at[hour - 1, 'Electrolyser_CF'] > 0

            if elec_cons == 0 and spill + batt_discharge_potential > elec_min and elec_just_operating:
                # When the generation is insufficient alone but combined with battery power can power the electrolyser
                if spill + batt_discharge_potential > elec_max:
                    cf_profile_df.at[hour, 'Battery_Net_Charge'] = -1 * min(self.batteryPower,
                                                                            (elec_max-spill) * 1/batt_losses)
                else:
                    cf_profile_df.at[hour, 'Battery_Net_Charge'] = -1 * batt_discharge_potential * 1/batt_losses
            elif spill > 0 and batt_soc + spill/self.batteryEnergy * batt_losses > 1:
                # When spilled generation is enough to completely charge the battery
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = min(self.batteryPower,
                                                                   max(self.batteryEnergy * (1.0 - batt_soc), 0.0))
            elif spill > 0:
                # Any other cases when there is spilled generation
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = min(self.batteryPower, spill * batt_losses)
            elif elec_cons + batt_discharge_potential < elec_min or (spill == 0 and batt_soc <= self.battMin):
                # generation and battery together are insufficient to power the electrolyser or there is no
                # spilled generation and the battery is empty
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = 0
            elif spill == 0 and elec_max - elec_cons > (batt_soc - self.battMin) * batt_losses * self.batteryEnergy \
                    and elec_just_operating:
                # When the electrolyser is operating and the energy to get to max capacity is more than what is stored
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = -1 * batt_discharge_potential * 1/batt_losses
            elif spill == 0 and elec_just_operating:
                # When the stored power is enough to power the electrolyser at max capacity
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = -1 * min(self.batteryPower,
                                                                        (elec_max - elec_cons) * 1/batt_losses)
            elif spill == 0:
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = 0
            else:
                print("Error: battery configuration not accounted for")

            # Determine the battery state of charge based on the previous state of charge and the net change
            cf_profile_df.at[hour, 'Battery_SOC'] = cf_profile_df.at[hour - 1, 'Battery_SOC'] + \
                cf_profile_df.at[hour, 'Battery_Net_Charge'] / self.batteryEnergy
        cf_profile_df['Electrolyser_CF_batt'] = np.where(cf_profile_df['Battery_Net_Charge'] < 0,
                                                         cf_profile_df['Electrolyser_CF'] +
                                                         (-1*cf_profile_df['Battery_Net_Charge'] * batt_losses +
                                                         cf_profile_df['Excess_Generation']) / self.elecCapacity,
                                                         cf_profile_df['Electrolyser_CF'])
        cf_profile_df.set_index(index_name, inplace=True)
        return cf_profile_df['Electrolyser_CF_batt']

    def __overloading_model(self, cf_profile_df, oversize):
        """Private method - Calculates and returns the hourly electrolyser operation profile when overloading is
        included in the model
        """
        can_overload = cf_profile_df['Generator_CF'] * oversize > self.elecMaxLoad

        for hour in range(1, len(cf_profile_df)):
            for hour_i in range(1, min(hour, self.elecOverloadRecharge)+1):
                if can_overload[hour] and can_overload[hour-hour_i]:
                    can_overload[hour] = False
        cf_profile_df['Max_Overload'] = self.elecOverload
        cf_profile_df['Energy_generated'] = cf_profile_df['Generator_CF'] * oversize
        cf_profile_df['Energy_for_overloading'] = cf_profile_df[['Max_Overload', 'Energy_generated']].min(axis=1)
        cf_profile_df['Electrolyser_CF_overload'] = np.where(can_overload,
                                                             cf_profile_df['Energy_for_overloading'],
                                                             cf_profile_df['Electrolyser_CF'])

        return cf_profile_df['Electrolyser_CF_overload']

    def __scale_capex(self, unscaled_capex, capacity, reference_capacity, scale_factor):
        """Private method - Calculates the capital cost considering economies of scale
        """
        if capacity > 0:
            scaled_capex = unscaled_capex * reference_capacity * (capacity / reference_capacity) ** scale_factor / \
                           capacity
        else:
            scaled_capex = unscaled_capex
        return scaled_capex

    def __get_capex(self, equip_cost, equip_pc, install_pc, land_pc):
        """Private method - Calculates the capital cost for given indirect costs
        """
        capex = equip_cost * (1 + install_pc / equip_pc) * (1 + land_pc)
        return capex

    def __get_tabulated_outputs(self, working_df):
        """Private method- Generates results summary table as a dictionary
        """

        operating_outputs = dict()
        operating_outputs["Generator Capacity Factor"] = working_df['Generator_CF'].mean()
        operating_outputs["Time Electrolyser is at its Rated Capacity"] = \
            working_df.loc[working_df['Electrolyser_CF'] == self.elecMaxLoad,
                           'Electrolyser_CF'].count() / self.hoursPerYear
        operating_outputs["Total Time Electrolyser is Operating"] = working_df.loc[working_df['Electrolyser_CF'] > 0,
                                                             'Electrolyser_CF'].count() / self.hoursPerYear
        operating_outputs["Achieved Electrolyser Capacity Factor"] = working_df['Electrolyser_CF'].mean()
        operating_outputs["Energy in to Electrolyser [MWh/yr]"] = working_df['Electrolyser_CF'].sum() * \
                                                                  self.elecCapacity
        operating_outputs["Surplus Energy [MWh/yr]"] = working_df['Generator_CF'].sum() * self.genCapacity - \
            working_df['Electrolyser_CF'].sum() * self.elecCapacity
        operating_outputs["Hydrogen Output for Fixed Operation [t/yr]"] = working_df['Hydrogen_prod_fixed'].sum() * \
            self.elecCapacity * self.kgtoTonne
        operating_outputs["Hydrogen Output for Variable Operation [t/yr]"] = \
            working_df['Hydrogen_prod_variable'].sum() * self.elecCapacity * self.kgtoTonne
        self.operating_outputs = operating_outputs
        return operating_outputs

    def make_duration_curve(self, generator=True, electrolyser=False):
        """Opens a figure in a new window showing the annual duration curve(s) for the chosen configuration

        Parameters
        ----------
        generator : bool, optional
            a boolean that determines whether the generator duration curve should show
        electrolyser : bool, optional
            a boolean that determines whether the electrolyser duration curve should show
        """

        plots = []
        if generator:
            plots.append("Generator")
        if electrolyser:
            plots.append("Electrolyser")
        elif not generator:
            raise ValueError("generator or electrolyser must be True")

        if self.windCapacity == 0:
            tech = "solar"
        elif self.solarCapacity == 0:
            tech = "wind"
        else:
            tech = "hybrid"

        colours = {"solar": "goldenrod", "wind": "royalblue", "hybrid": "limegreen"}
        hourly_df = self.__calculate_hourly_operation()
        fig = plt.figure(1)
        for i in range(len(plots)):
            gen_elec = plots[i]
            generation = hourly_df[gen_elec + '_CF'].sort_values(ascending=False).reset_index(drop=True) * 100
            generation.index = generation.index / 8760 * 100
            ax = fig.add_subplot(1, len(plots), i+1)
            generation.plot(ax=ax, color=colours[tech])
            ax.set(title=f"{tech.capitalize()} {gen_elec} Capacity Factor - {self.location}",
                   xlabel="Proportion of year (%)", ylabel=f"{gen_elec} Capacity Factor (%)")
            ax.set_ylim(0, 100)
            ax.grid(axis='y', which='both')
        plt.show()
