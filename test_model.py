import unittest
from HydrogenModel import HydrogenModel
import pandas as pd
from pandas.util.testing import assert_frame_equal


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
        """test LCH2 calculation based on default inputs"""

        model1 = HydrogenModel(config='Config/config.yml')
        lch2 = model1.calculate_costs()
        expected_lch2 = 4.31
        self.assertEqual(lch2, expected_lch2)

    def test_2(self):
        """test hourly operation for 10MW solar 10MW electrolyser plant"""
        model2 = HydrogenModel(location=self.location, solardata=self.solardata12, winddata=self.winddata12,
                               config='Config/config.yml')
        out = model2._HydrogenModel__calculate_hourly_operation()
        out = out.iloc[:, :2]
        expected = pd.DataFrame(index=self.dt_index,
                                data={'Generator_CF': [0, 0, 0.1, 0.12, 0.5, 0.6, 0.7, 0.6, 0.3, 0.2, 0.1, 0.05],
                                      'Electrolyser_CF': [0, 0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.6, 0.3, 0.2, 0.0, 0]})
        assert_frame_equal(out, expected)

    def test_3(self):
        """test hourly operation for 5MW solar, 5MW wind 10MW electrolyser plant"""
        model3 = HydrogenModel(location=self.location, solardata=self.solardata12, winddata=self.winddata12,
                               solar_capacity=5.0, wind_capacity=5.0, config='Config/config.yml')
        out = model3._HydrogenModel__calculate_hourly_operation()
        out = out.iloc[:, :2]
        expected = pd.DataFrame(index=self.dt_index,
                                data={'Generator_CF': [0, 0.05, 0.3, 0.51, 0.3, 0.3, 0.35, 0.65, 0.45, 0.15, 0.1, 0.05],
                                      'Electrolyser_CF': [0, 0, 0.3, 0.51, 0.3, 0.3, 0.35, 0.65, 0.45, 0.0, 0.0, 0]})
        assert_frame_equal(out, expected)

    def test_4(self):
        """test hourly operation for 20MW wind 10MW electrolyser plant"""
        model4 = HydrogenModel(location=self.location, solardata=self.solardata12, winddata=self.winddata12,
                               solar_capacity=0, wind_capacity=20, config='Config/config.yml')
        out = model4._HydrogenModel__calculate_hourly_operation()
        out = out.iloc[:, :2]
        expected = pd.DataFrame(index=self.dt_index,
                                data={'Generator_CF': [0, 0.1, 0.5, 0.9, 0.1, 0, 0, 0.7, 0.6, 0.1, 0.1, 0.05],
                                      'Electrolyser_CF': [0, 0.2, 1, 1, 0.2, 0, 0, 1, 1, 0.2, 0.2, 0]})
        assert_frame_equal(out, expected)


if __name__ == '__main__':
    unittest.main()
