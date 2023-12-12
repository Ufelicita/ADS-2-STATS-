# -*- coding: utf-8 -*-
"""

The code  analyzes World Bank data for 10 European countries
regarding climate change using four indicators: Nitrous Oxide  emissions,
agricultural land area, population, and life expectancy over 25 years.

 Created December 1, 2023


@author: Felicita Adeleke (Student Id : )
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau


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
    df = pd.read_csv(filename)

    # Drop columns not required
    df = df.drop(["Series Code", "Country Code"],axis=1)
    
    # Rename Columns names 
    df.columns = ["Indicator Name","Country Name","2003", "2004", "2005", 
                  "2006","2007","2008","2009", "2010", "2011", "2012", 
                  "2013", "2014", "2015", "2016", "2017", "2018","2019", 
                  "2020"]
    
   
    
    #set Coountry Name as index 
    df = df.set_index(["Country Name", "Indicator Name"])
  
   
    
    # Transpose df so that countries become columns 
    df_t = df.transpose()
    
    #check for missing data 
    print(df_t.isna().any())
    
    return df, df_t

# Analyse the DataFrame
df, df_t = read("worldbank.csv")

# Statis
df_t_decribe_stat = df_t.describe()
print(df_t_decribe_stat)
print(df_t.dtypes)

correlation = df_t.corr()

for series in df_t.columns.levels[1]:
    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 10))
    df_t.xs(series,level=1, axis=1).plot(ax=ax)
    ax.set_ylabel(series)
    plt.show()
    


