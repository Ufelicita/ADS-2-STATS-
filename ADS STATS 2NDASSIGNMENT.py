# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:20:15 2023

@author: User
"""

# -*- coding: utf-8 -*-
"""
The code  analyzes World Bank data for selected  countries regarding
climate change using five indicators: GDP (current US$),
Arable land (hectares) ,CO2 emissions (kt) , Population Density and
Population total every 5 years for 25 years .

Created December 20, 2023

@author: Felicita Adeleke (Student Id : 22026653 )
"""


# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scipy

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
Below is the summary statistics using the statiscal methods descibe(),
Kurtosis and  for three indicators in five selected countries over a period
of 25 years every five years
"""


# Select the indicators and countries for use
row_select = df_world_data_t1.iloc[:, 0:5]
column_select1 = df_world_data_t1.iloc[:, 20:25]
column_select2 = df_world_data_t1.iloc[:, 50:55]

# Concat to get a single dataframe of the selection
stats_df = pd.concat([row_select, column_select1, column_select2], axis=1)

# Set display options for legible format in the console
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

# Analyse decribe(), skew and kurtosis for data
df_summary_stats_describe = stats_df.describe()
df_summary_stats_skew = stats_df.skew()
df_summary_stats_kurtosis = stats_df.kurtosis()

# Display summary statistics  in the console
print(df_summary_stats_describe)
print(df_summary_stats_skew)
print(df_summary_stats_kurtosis)

# Reset display options to default in order to save in excel
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
using Kendall correlation and a heatmap to understand correlations (
or lack of) between indicators and trends change with time.
"""


df_worldbank = df_world_data.reset_index()
df_worldbank.dropna()

# Melt the DataFrame specifying multiple years in value_vars
df_worldbank_melt = pd.melt(df_worldbank, id_vars=["Country Name",
                                                   "Series Name"],
                            var_name="Year",
                            value_name="Value")

df_worldbank_melt = df_worldbank_melt.iloc[120:480, :]

df_worldbank_melt['Value'] = pd.to_numeric(df_worldbank_melt['Value'],
                                           errors='coerce')

# Pivot the melted DataFrame
pivot_df_worldbank = df_worldbank_melt.pivot_table(index=["Country Name"],
                                                   columns="Series Name",
                                                   values="Value",
                                                   aggfunc="mean")

pivot_df_worldbank = pivot_df_worldbank.reset_index()

print(pivot_df_worldbank)

# Extract numeric columns for correlation analysis
df_numeric_cols = pivot_df_worldbank.drop('Country Name', axis=1)

# Calculate Kendall correlation matrix
df_corr_matrix = df_numeric_cols.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)

plt.title("Selected Indicators Correlation Heatmap")
plt.show()


def plot_bar_plot(df, ylabel):
    """
    Plots a grouped plot to show the comparisons of Arable and Agricultural
    lands in 10 different countries in the world
    Args:
    - df(pd.DataFrame): The input DataFrame.

    Returns:
    None: Displays the Grouped Bar  plot.
    """

    plt.figure()

    # Plot graph and customer labels and title
    df.plot.bar()

    plt.title("Comparison between population Density & Arable lands")

    # Set legend and  Title
    plt.legend(bbox_to_anchor=(1.02, 1))

    # Save plot
    plt.savefig('bar_plot.png')

    # Display plot
    plt.show()
    plt.show()

    return


# Subset from orginal World Bank Data
df_bar = df_world_data_t1.iloc[:, 10:40]

df_bar_t = df_bar.transpose()


df_bar_t = df_bar_t.iloc[10:30, :]

# Reset Country Name and Series Name as Index
df_bar_t = df_bar_t.reset_index(["Country Name", "Series Name"])

# Melt the DataFrame to make it easier for use in plotting
df_bar_melt = pd.melt(df_bar_t, id_vars=["Country Name", "Series Name"],
                      var_name="Year", value_name="Value")

# Set Index
df_bar_melt = df_bar_melt.set_index("Country Name")
df_melt = df_bar_melt.pivot_table(index='Country Name',
                                        columns='Series Name',
                                        values='Value', aggfunc='mean')

# change the land units to per square meters
df_melt["Arable land(per sqm)"] = df_melt["Arable land (hectares)"]*0.0001

# Delete the Column with Arable land in hectares
df_melt.drop("Arable land (hectares)", axis=1, inplace=True)


# Plot group bar chart
df_melt.plot.bar(figsize=(10, 6))

plot_bar_plot(df_bar_melt, ylabel="square meteres ")


def plot_line_plot(df, title, ylabel):
    """
    Plots a line plot to show trend of a world bank indicator over some years 
    in some selected countries in the world 

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    None: Displays the line plot.
    """

    plt.figure(figsize=(10, 8))

    # Plot graph and customer labels and title
    df.plot(xlabel="Year", ylabel=ylabel, marker="o")
    plt.title("Title name")

    # Set legend , Grid and Title
    plt.legend(bbox_to_anchor=(1.02, 1), title="Select Countries")
    plt.grid(True)
    plt.title(title)

    # Set subplots to adjust boarders
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Save plot
    plt.savefig('line_plot.png')

    # Display plot
    plt.show()
    plt.show()

    return


# Subset a new DataFrame for preferred indicator from df_world_data
df_gdp = df_world_data.iloc[0:10]

# Reset Index and Transpose
df_gdp = df_gdp.reset_index()
df_gdp = df_gdp.transpose()

# Set Country name as headers, remove unneccesary rows and drop NANs
df_gdp.columns = df_gdp.iloc[0]
df_gdp = df_gdp.iloc[4:]
df_gdp = df_gdp.dropna()

# Set Year Index heaader and change to numeric data
df_gdp.index = pd.to_numeric(df_gdp.index)
df_gdp.index.names = ["Year"]

# call the line plot function with values or GDP(Current US$)
plot_line_plot(df_gdp,
               title="Trends in GDP in Selected  Countries in the World", ylabel="GDP (current US$")

# Subset a new DataFrame for preferred indicator from df_world_data
df_co2_emmisions = df_world_data.iloc[40:50]

# Reset Index and Transpose
df_co2_emmisions = df_co2_emmisions .reset_index()
df_co2_emmisions = df_co2_emmisions .transpose()

# Set Country name as headers, remove unneccesary rows and drop NANs
df_co2_emmisions.columns = df_co2_emmisions .iloc[0]
df_co2_emmisions = df_co2_emmisions .iloc[4:]
df_co2_emmisions = df_co2_emmisions .dropna()

# Set Year Index heaader and change to numric data
df_co2_emmisions .index = pd.to_numeric(df_gdp.index)
df_co2_emmisions.index.names = ["Year"]

# call the line plot function with values or GDP(Current US$)
plot_line_plot(df_co2_emmisions,
               title="Trends in C02 Emissions(Kt) in Selected Countries in the World", ylabel="CO2 emissions (kt)")
