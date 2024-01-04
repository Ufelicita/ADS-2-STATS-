
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


def plot_kendall_heatmap(df_world_data, country_name, title):
    """
    Generates a Kendall correlation heatmap for selected indicators.

    Args:
    - df_world_data (pd.DataFrame): Input DataFrame

    Returns:
    None: Displays the Kendall correlation heatmap.
    """

    df_worldbank = df_world_data.reset_index()
    df_worldbank.dropna()

    # Filter  DataFrame for  specific country
    df_country = df_worldbank[df_worldbank['Country Name'] == country_name]

    # Drop columns with missing values
    df_country = df_country.dropna(axis=1)

   # Melt the DataFrame specifying multiple years in value_vars
    df_country_melt = pd.melt(df_country, id_vars=['Country Name',
                                                   'Series Name',
                                                   'Series Code',
                                                   'Country Code'],
                              var_name='Year',
                              value_name='Value')
    df_pivoted = df_country_melt.pivot_table(index='Year',
                                             columns='Series Code',
                                             values='Value',
                                             aggfunc='mean')

    # Extract numeric columns for correlation analysis
    df_numeric_cols = df_pivoted.select_dtypes(include=['float64', 'int64'])

    # Calculate Kendall correlation matrix
    df_corr_matrix = df_numeric_cols.corr(method='kendall')

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    heat_map = sns.heatmap(df_corr_matrix, annot=True,
                           cmap="cool", linewidths=.5,
                           annot_kws={"weight": "bold", "size": 10,
                                      "color": 'black'})

   # Customize title
    plt.title(title, fontweight='bold')

    # Customize labels
    plt.xlabel("Indicators", fontweight='bold')
    plt.ylabel("Indicators", fontweight='bold')

    # Customize labels and ticks
    plt.xlabel("Indicators", fontsize=14,  fontweight='bold')
    plt.ylabel("Indicators", fontsize=14, fontweight='bold')
    heat_map.set_xticklabels(heat_map.get_xticklabels(),
                             fontsize=12, rotation=90)
    heat_map.set_yticklabels(heat_map.get_yticklabels(),
                             fontsize=12, rotation=0)

    # Create a legend mapping Series Code to Series Name
   # Add a legend mapping Series Code to Series Name
    legend_labels = df_country_melt[[
        'Series Code', 'Series Name']].drop_duplicates()
    legend_labels = dict(
        zip(legend_labels['Series Code'], legend_labels['Series Name']))

    # Add spaces after ":" in each label and adjust legend placement
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
                          # markersize=10,
                          label=f"${code}:$ {name}    ") for code,
               name in legend_labels.items()]

    # Adjust legend font size and style
    plt.legend(handles=handles, title='Legend',
               bbox_to_anchor=(1.05, -0.39),
               loc='lower left', fontsize=12, title_fontsize=14,
               prop={'weight': 'bold'})

    plt.show()

    return


plot_kendall_heatmap(df_world_data, country_name="China",
                     title=f"Kendall Correlation Heatmap for China")

plot_kendall_heatmap(df_world_data, country_name="United States",
                     title=f"Kendall Correlation Heatmap for United States")


def plot_bar_plot(df):
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
    df_melt.plot.bar(figsize=(10, 6), ylabel="Square Metres",
                     xlabel="Countries",)
    plt.title("Comparison between population Density & Arable lands")

    # Customize font size and weight for labels and title
    plt.ylabel("Square Metres", fontsize=14, fontweight='bold')
    plt.xlabel("Countries", fontsize=14, fontweight='bold')
    plt.title("Comparison between Population Density & Arable Lands",
              fontsize=16, fontweight='bold')

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

# plot Bar graph
plot_bar_plot(df_bar_melt)


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
               title="Trends in GDP in Selected  Countries ",
               ylabel="GDP (current US$")

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
               title="Trends in C02 Emissions(Kt) in Selected Countries",
               ylabel="CO2 emissions (kt)")
