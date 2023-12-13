"""

The code  analyzes World Bank data for 10  countries regarding 
climate change using four indicators: Forest Area, GDP, Population,
Total Greenhouse emmisions (kt of co2), Manufacturing Value Added.
over 25 years.

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

    # Rename Columns names
    df.columns = ["Country Name", "Indicator Name", "Series Code",
                  "Country Code", "1991",  "1992", "1993", "1994", "1995",
                  "1996", "1997", "1998", "1999", "2000", "2001", "2002",
                  "2003", "2004", "2005", "2006", "2007", "2008", "2009",
                  "2010", "2011",  "2012", "2013", "2014", "2015"]

    # set Coountry Name as index
    #df = df.set_index(["Country Name", "Indicator Name"])

    # Transpose df so that countries become columns
    df_t = df.transpose()

    # Set Country names as header
    df_t.columns = df_t.iloc[0]

    # Drop unwanted rows
    df_t = df_t.drop(["Country Name", "Series Code", "Country Code"], axis=0)

    # check for missing values
    df_t.isna().any()

    # Check the class type of data
    print(df_t)
    print(type(df_t.index[1]))
    print(type(df_t.iloc[1, 1]))

    # Format DataFrame so that values show properly
    df_t_formats = {"South Africa": "{:9.0f}".format,
                    "India": "{:9.0f}".format,
                    "korea Rep": "{:9.0f}".format,
                    "United Kingdom": "{:9.0f}".format,
                    "Rwanda": "{:9.0f}".format,
                    "Saudi Arabia": "{:9.0f}".format,
                    "Netherlands": "{:9.0f}".format,
                    "Guyana": "{:9.0f}".format, "Chile": "{:9.0f}".format,
                    "Germany": "{:9.0f}".format}

    df_t.head().style.format(df_t_formats)
    print(df_t_formats)

    # Set Dataframe to standard float format
    textfile = open("country.txt", "w")
    df_t.to_string(textfile, formatters=df_t_formats)

    textfile.close()

    return df, df_t

# Check the class type of data
# print(df_t_sub)
# print(type(df_t_sub.index[1]))
#print(type(df_t_sub.iloc[1, 1]))
#


# Analyse the DataFrame
df, df_t = read("worldbankdata.csv")
