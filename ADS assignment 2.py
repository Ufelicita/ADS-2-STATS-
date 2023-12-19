# -*- coding: utf-8 -*-
"""
The code  analyzes World Bank data for selected  countries regarding
climate change using five indicators: GDP (current US$)
, Population density (people per sq. km of land area),GDP,
Arable land (hectares), Agricultural land (sq. km) ,CO2 emissions (kt) and 
Population total every 5 years for 25 years .

Created December 18, 2023

@author: Felicita Adeleke (Student Id : 22026653 )
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scipy
import os as os


def read(filename):
    """
    Reads the DataFrame in world bank format and returns two data frames with
    years as columns and countries as columns

    Args:
    ----------
    filename(str): The name of the file with the DataFrame

    Returns:
        years_df A :Transposed dataframe with years  as columns
        countries_df: Transposed dataframe with countries as columns
    """

    # Read csv file
    df_world_data = pd.read_csv(filename)

    # Set Series and Country name as index
    df_world_data.set_index(["Country Name", "Series Name"], inplace=True)

    # Transpose df so that countries become columns
    df_world_data_t1 = df_world_data.transpose()

    # Drop unnessary columns
    df_world_data_t1 = df_world_data_t1.drop(["Series Code", "Country Code"],
                                             axis=0)

    # check for NaN
    print(df_world_data_t1.isna().any())
    print(df_world_data_t1)

    # Check Data Type
    print(df_world_data_t1.dtypes)

    # change the data type
    df_world_data_t1 = df_world_data_t1.apply(pd.to_numeric, errors="coerce")
    print(df_world_data_t1.dtypes)
    return df_world_data,   df_world_data_t1

# Analyse the DataFrame
df_world_data, df_world_data_t1 = read("worldBDATA1.csv")


"""
Below is the summary statistics using the method descibe() for three indicators
in five selected countries over a period of 25 years every five years 
over a per 

"""
# Select the indicators and columns for use 
row_select =  df_world_data_t1.iloc[:, 0:5]
column_select1 =  df_world_data_t1.iloc[:, 20:25]
column_select2 =  df_world_data_t1.iloc[:, 50:55]

# Concat to get a single dataframe of the selection 
stats_df = pd.concat([row_select, column_select1, column_select2], axis=1)

stats_df = stats_df.rename(columns={"":"Year"})

# Set display options for legible format in the console
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

# Set display options for legible format in the console
#pd.set_option('display.float_format', lambda x: f'{x:.2f}')

df_summary_stats_describe = stats_df.describe()

df_summary_stats_skew = stats_df.skew()

df_summary_stats_kurtosis = stats_df.kurtosis()


# Display the concatenated DataFrame in the console
print("df_summary_stats_describe:")
print(df_summary_stats_describe)
print(df_summary_stats_skew)
print(df_summary_stats_kurtosis)

#Reset display options to default in order to save in excel
pd.reset_option('display.float_format')

# Save to Excel with a specified number format
excel_file_path = 'df_summary_stats_describe.xlsx'
df_summary_stats_describe.to_excel(excel_file_path, index=True, 
                               float_format='%.2f')
df_summary_stats_skew.to_excel(excel_file_path, index=True, 
                               float_format='%.2f')
df_summary_stats_kurtosis.to_excel(excel_file_path, index=True, 
                               float_format='%.2f')


print(f"\ndf_df_summary_stats_describe saved to {excel_file_path}")
print(f"\ndf_df_summary_stats_skew saved to {excel_file_path}")
print(f"\ndf_df_summary_stats_kurtosis saved to {excel_file_path}")
"""
# Calculate descriptive analysis for China
uk_descb = df_t1.xs("China", level=0, axis=1).agg("describe")
print(uk_descb)

# Calculate Skewness  for China
uk_skew = df_t1.xs("China", level=0, axis=1).agg("skew")
print(uk_skew)

# Calculate Kurtosis for China
uk_kurtosis = df_t1.xs("China", level=0, axis=1).agg("kurtosis")
print(uk_kurtosis)

# Calculate descriptive analysis for India
India_descb = df_t1.xs("India", level=0, axis=1).agg("describe")
print(India_descb)

# Calculate skewness for India
India_Skew = df_t1.xs("India", level=0, axis=1).agg("skew")
print(India_Skew)

# Calculate kurtosis for India
India_kurtosis = df_t1.xs("India", level=0, axis=1).agg("kurtosis")
print(India_kurtosis)

# Calculate descriptive analysis  for Japan
Japan_describe = df_t1.xs("Japan", level=0, axis=1).agg("describe")
print(Japan_describe)

# Calculate  Skew analysis for Japan
Japan_skew = df_t1.xs("Japan", level=0, axis=1).agg("skew")
print(Japan_skew)

# Calculate Kurtosis  analysis for Japan
Japan_kurtosis = df_t1.xs("Japan", level=0, axis=1).agg("kurtosis")
print(Japan_kurtosis)

"""
