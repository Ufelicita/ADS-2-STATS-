# -*- coding: utf-8 -*-
"""

The code  analyzes World Bank data for 10  countries regarding
climate change using five indicators: Forest Area, GDP, Population,
Total Greenhouse emmisions (kt of co2), Manufacturing Value Added.
 every 6 years for 24 years .

 Created December 1, 2023


@author: Felicita Adeleke (Student Id : 22026653 )

"""


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau


def read(filename, index_col):
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
    df = pd.read_csv(filename)

    # set index
    df.set_index(index_col, inplace=True)

    # Transpose df so that countries become columns
    df_t = df.transpose()

    return df, df_t


# Analyse the DataFrame
df, df_t = read("worldbankdata.csv", index_col=["Country Name", "Series Name"])

# Drop unnessary columns
df_t = df_t.drop(["Series Code", "Country Code"], axis=0)

# Check Data Type
print(df_t.dtypes)

# change the data type
df_t = df_t.apply(pd.to_numeric, errors="coerce")
print(df_t.dtypes)

# check for NaN
print(df_t.isna().sum())

# Change years format
df_t_yr_format = {"1995 [YR1995]": "1995", "2000 [YR2000]": "2000",
                  "2005 [YR2005]": "2005", "2010 [YR2010]": "2010",
                  "2010 [YR2010]": "2010"}

print(df_t.rename(df_t_yr_format,axis=0))

# Format DataFrame so that values show properly
df_t_formats = {"South Africa": "{:9.0f}".format, "India": "{:9.0f}".format,
                "korea Rep": "{:9.0f}".format, "Rwanda": "{:9.0f}".format,
                "United Kingdom": "{:9.0f}".format, "Guyana": "{:9.0f}".format,
                "Saudi Arabia": "{:9.0f}".format, "Germany": "{:9.0f}".format,
                "Netherlands": "{:9.0f}".format}

print(df_t.to_string(formatters=df_t_formats))


"""
Below is the summary statistics for three(3)countries :United Kingdom,
South Africa and India for all the indicators, Population total, 
GDP (current US$), Forest area (sq. km),Total greenhouse gas emissions
(kt of CO2 equivalent)

"""
   
       
# Calculate descriptive analysis for United Kindgom             
uk_descb = df_t.xs("United Kingdom", level=0, axis=1).agg("describe")
print(uk_descb)

# Calculate Skewness  for United Kingdom 
uk_skew = df_t.xs("United Kingdom", level=0, axis=1).agg("skew")
print(uk_skew)

# Calculate Kurtosis for United Kingdom 
uk_kurtosis = df_t.xs("United Kingdom", level=0, axis=1).agg("kurtosis")
print(uk_kurtosis)
      
# Calculate descriptive analysis for India 
India_descb = df_t.xs("India", level=0, axis=1).agg("describe")
print(India_descb)

# Calculate skewness for India 
India_Skew = df_t.xs("India", level=0, axis=1).agg("skew")
print(India_Skew)

# Calculate kurtosis for India 
India_kurtosis = df_t.xs("India", level=0, axis=1).agg("kurtosis") 
print(India_kurtosis)

# Calculate descriptive analysis  for South Africa
South_Africa_describe = df_t.xs("South Africa", level=0,
                                axis=1).agg("describe")  
print(South_Africa_describe)

# Calculate  Skew analysis for South Africa
South_Africa_skew = df_t.xs("South Africa", level=0,
                                axis=1).agg("skew")
print(South_Africa_skew)

# Calculate Kurtosis  analysis for South Africa
South_Africa_kurtosis = df_t.xs("South Africa", level=0,
                                 axis=1).agg("kurtosis")
print(South_Africa_kurtosis)







