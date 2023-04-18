import unittest
from HydrogenModel import HydrogenModel
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal


class TestModel(unittest.TestCase):
    def setUp(self):
        self.location = 'Test_City'
        self.dt_index = ['1/01/2019 0:30', '1/01/2019 1:30', '1/01/2019 2:30', '1/01/2019 3:30',
                         '1/01/2019 4:30', '1/01/2019 5:30', '1/01/2019 6:30', '1/01/2019 7:30',
                         '1/01/2019 8:30', '1/01/2019 9:30', '1/01/2019 10:30', '1/01/2019 11:30']
        self.solardata12 = pd.DataFrame(index=self.dt_index,
                                 data={'Test_City': [0, 0, 0.1, 0.12, 0.5, 0.6, 0.7, 0.6, 0.3, 0.2, 0.1, 0.05]})
        self.winddata12 = pd.DataFrame(index=self.dt_index,
                                data={'Test_City': [0, 0.1, 0.5, 0.9, 0.1, 0, 0, 0.7, 0.6, 0.1, 0.1, 0.05]})

    def test_1(self):
        """test battery calculation for 10MW solar, 10MW 1 hr battery and 10MW electrolyser plant"""
        model1 = HydrogenModel(location=self.location, solardata=self.solardata12, winddata=self.winddata12,
                               battery_power=10, battery_hours=1)
        model1.batteryEfficiency = 1.0
        in_df = pd.DataFrame(index=self.dt_index,
                             data={'Generator_CF': [0, 0, 0.1, 0.12, 0.5, 0.6, 0.7, 0.6, 0.3, 0.2, 0.1, 0.05],
                                   'Electrolyser_CF': [0, 0, 0, 0, 0.5, 0.6, 0.7, 0.6, 0.3, 0.2, 0, 0]})
        out = model1._HydrogenModel__battery_model(oversize=1.0, cf_profile_df=in_df)
        expected = pd.Series(name='Electrolyser_CF_batt',
                             index=self.dt_index,
                             data=[0.0, 0.0, 0.0, 0.0, 0.72, 0.6, 0.7, 0.6, 0.3, 0.2, 0.0, 0.0])
        expected = expected.rename_axis('index')
        assert_series_equal(out, expected)

    def test_2(self):
        """test battery calculation for 20MW solar, 10MW 1 hr battery and 10MW electrolyser plant"""
        model2 = HydrogenModel(location=self.location, solardata=self.solardata12, winddata=self.winddata12,
                               solar_capacity=20, battery_power=10, battery_hours=1)
        model2.batteryEfficiency = 1.0
        in_df = pd.DataFrame(index=self.dt_index,
                             data={'Generator_CF': [1, 0, 0.1, 0.6, 0.05, 0.6, 0.6, 0.6, 0.2, 0.2, 0.1, 0.05],
                                   'Electrolyser_CF': [1, 0, 0.2, 1, 0, 1.0, 1.0, 1.0, 0.4, 0.4, 0.2, 0]})
        out = model2._HydrogenModel__battery_model(oversize=2.0, cf_profile_df=in_df)
        expected = pd.Series(name='Electrolyser_CF_batt',
                             index=self.dt_index,
                             data=[1.0, 1.0, 0.2, 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 0.4, 0.2, 0.0])
        expected = expected.rename_axis('index')
        assert_series_equal(out, expected)

if __name__ == '__main__':
    unittest.main()
