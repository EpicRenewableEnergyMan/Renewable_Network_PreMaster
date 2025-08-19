import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.text import TextPath

from matplotlib.transforms import Affine2D
import matplotlib.colors as mcolors
from matplotlib.patches import PathPatch

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import geoplot as gplt
import os

import matplotlib.patches as patches
import geoplot.crs as gcrs
import textwrap

from scipy.stats import norm

import seaborn as sns


def Het_plot_Country(Het, Country, BusData,foldername,figname):
    # Get indexes of nodes in the specified country
    indexes_with_country = BusData[BusData['country'] == Country].index.tolist()

    # Extract gamma and alpha values for nodes in the specified country
    gamma = Het['gamma'][indexes_with_country]
    alpha = Het['alpha'][indexes_with_country]

    # Calculate wind and solar contributions
    wind_contributions = alpha * gamma
    solar_contributions = gamma * (1 - alpha)

    # Plotting
    fig, ax = plt.subplots()
    bar_width = 0.5
    index = np.arange(len(gamma))

    bars_wind = ax.bar(index, wind_contributions, bar_width, label='Wind', color='deepskyblue')
    bars_solar = ax.bar(index, solar_contributions, bar_width, bottom=wind_contributions, label='Solar', color='gold')

    ax.set_xlabel('Nodes')
    ax.set_ylabel('gamma_n')
    ax.set_title('Gamma Contributions by Source')
    ax.set_xticks(index)
    ax.set_xticklabels(indexes_with_country)  # Set tick labels as the indexes of nodes in the country
    ax.legend()

    plot_type = 'hetplot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }
    
    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Show plot
    plt.show()
    
    

def Het_plot_EU(alpha,gamma, countries,figname,ylabel,foldername):

    

    
    # Calculate contributions
    wind_contributions = alpha * gamma
    solar_contributions = gamma * (1 - alpha)
    
    # Prepare for plotting
    
    index = np.arange(len(countries))
    bar_width = 1
    
    #plt.style.use('seaborn-darkgrid')  # Use a seaborn style for a prettier plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size for readability
    
    # Plotting with updated colors and black outlines
    bars_wind = ax.bar(index, wind_contributions, bar_width, label='Wind', color='deepskyblue', edgecolor='black')
    bars_solar = ax.bar(index, solar_contributions, bar_width, bottom=wind_contributions, label='Solar', color='gold', edgecolor='black')
    
    # Enhance legibility
    ax.set_xlabel('Country', fontsize=23)
    ax.set_ylabel(ylabel, fontsize=32)
    #ax.set_title('Mean Gamma Contributions by Source for Each Country', fontsize=22)
    
    # Add y padding
    max_contribution = max(wind_contributions + solar_contributions)
    ax.set_ylim(0, max_contribution * 1.18)  # Add 10% padding above the highest bar
    
    ax.set_xlim(-1,len(gamma))
    
    
    ax.set_xticks(index)
    ax.set_xticklabels(countries, rotation=90, ha="center", fontsize=16)
    ax.legend(fontsize=18)
    
    ax.tick_params(axis='y', labelsize=16)  # Make y-axis tick labels larger
    ax.legend(fontsize=20)
    
    
    
    
    plt.tight_layout()
    plot_type = 'hetplot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }
    
    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Show plot
    plt.show()
    
    
  
    
    


def avgLCOEplot(LCOE, LoadData, figname, foldername, sort='load'):
    LoadDataMean = LoadData.sum(axis=0)

    # Extract LCOE data
    Backup = LCOE['LCOE_backup']
    Trans = LCOE['LCOE_trans']
    Wind = LCOE['LCOE_wind']
    Solar = LCOE['LCOE_solar']

    if sort == 'load':
        # Sort the indices of LoadDataMean in ascending order
        sorted_indices = LoadDataMean.argsort()
    elif sort == 'LCOE':
        # Sort the indices of LCOE in descending order
        sorted_indices = LCOE['LCOE'].argsort()[::-1]
    else:
        print('wrong sort type')
        return

    # Use the sorted indices to sort the LCOE arrays
    sorted_Backup = Backup[sorted_indices]
    sorted_Trans = Trans[sorted_indices]
    sorted_Wind = Wind[sorted_indices]
    sorted_Solar = Solar[sorted_indices]

    # Accumulate the LCOE data in the desired order
    cumulative_LCOE = np.cumsum([sorted_Trans, sorted_Backup, sorted_Solar, sorted_Wind], axis=0)

    # Create the area plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(range(len(sorted_Backup)), 0, cumulative_LCOE[0], label='Transmission', color='forestgreen', alpha=0.5)
    plt.fill_between(range(len(sorted_Backup)), cumulative_LCOE[0], cumulative_LCOE[1], label='Backup', color='rebeccapurple', alpha=0.5)
    plt.fill_between(range(len(sorted_Backup)), cumulative_LCOE[1], cumulative_LCOE[2], label='Solar', color='gold', alpha=0.5)
    plt.fill_between(range(len(sorted_Backup)), cumulative_LCOE[2], cumulative_LCOE[3], label='Wind', color='deepskyblue', alpha=0.5)

    if sort == 'load':
        plt.xlabel('Nodes (sorted by ascending mean load)', fontsize=16)
    elif sort == 'LCOE':
        plt.xlabel('Nodes (sorted by Total LCOE)', fontsize=16)

    plt.ylabel('Cumulative LCOE (€/MWh)', fontsize=16)
    plt.legend()
    plt.tight_layout()
    
    plot_type = 'LCOEplot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }
    
    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Show plot
    plt.show()
    
    
def avgLCOEplot2(LCOE, LoadData, Country, figname, foldername, ylim='none',ylabel='ylabel',xlabel='xlabel'):
    # Extract LCOE data
    Backup = LCOE['LCOE_backup']
    Trans = LCOE['LCOE_trans']
    Wind = LCOE['LCOE_wind']
    Solar = LCOE['LCOE_solar']

    # Identify unique countries in the dataset
    unique_countries = Country.unique()

    # Initialize arrays to store nodal sums with gaps
    node_Backup = []
    node_Trans = []
    node_Wind = []
    node_Solar = []
    updated_Country = []
    
    totalLCOE = Backup+Trans+Wind+Solar

    # Sort each category by LCOE for each country and prepare nodal sums with gaps
    for country in unique_countries:
        country_indices = np.where(Country == country)[0]
        sorted_indices = np.argsort(totalLCOE[country_indices])  # Sort by Backup LCOE (adjust as needed)
        sorted_country_indices = country_indices[sorted_indices]

        # Add 4 columns of zeros for gaps
        node_Trans.extend(Trans[sorted_country_indices])
        node_Backup.extend(Backup[sorted_country_indices])
        node_Solar.extend(Solar[sorted_country_indices])
        node_Wind.extend(Wind[sorted_country_indices])

        # Extend country labels
        updated_Country.extend([country] * len(country_indices))

        # Add gaps
        node_Trans.extend([0, 0, 0, 0])
        node_Backup.extend([0, 0, 0, 0])
        node_Solar.extend([0, 0, 0, 0])
        node_Wind.extend([0, 0, 0, 0])
        updated_Country.extend([''] * 4)

    # Convert to numpy arrays
    node_Trans = np.array(node_Trans)
    node_Backup = np.array(node_Backup)
    node_Solar = np.array(node_Solar)
    node_Wind = np.array(node_Wind)

    # Create the bar plot
    bar_width = 1  # Adjust bar width as needed
    indices = np.arange(len(node_Trans))

    plt.figure(figsize=(16, 3))

    # Plot each category with nodal accumulation and gaps
    plt.bar(indices, node_Trans, bar_width, color='forestgreen', alpha=1)
    plt.bar(indices, node_Backup, bar_width, bottom=node_Trans, color='rebeccapurple', alpha=1)
    plt.bar(indices, node_Solar, bar_width, bottom=node_Trans + node_Backup, color='gold', alpha=1)
    plt.bar(indices, node_Wind, bar_width, bottom=node_Trans + node_Backup + node_Solar, color='deepskyblue', alpha=1)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    # Set x-ticks and labels at the start of each country's data range
    ticks = []
    labels = []
    for i, country in enumerate(unique_countries):
        # Calculate mid position for each country
        country_indices = np.where(Country == country)[0]
        mid_pos = (np.mean(country_indices) + 4 * i)  # Adjust for gaps
        ticks.append(mid_pos)
        labels.append(country)

    plt.xticks(ticks=ticks, labels=labels, rotation=90, fontsize=10)
    plt.xlim(-1, len(indices) - 1)  # Adjust x-axis limit to ensure all bars are visible
    
    if ylim == 'none':
        plt.ylim(0,max(totalLCOE)*1.05)
    else:
        plt.ylim(0,ylim)
    plt.legend()
    plt.tight_layout()

    plot_type = 'LCOEplot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }

    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)

    # Show plot
    plt.show()


def avgGammaAlphaPlot2(kappa, LoadData, Country, figname, foldername, ylim='none'):
    # Extract gamma and alpha
    gamma = kappa['gamma_n']
    alpha = kappa['alpha_n']
    
    # Ensure gamma and alpha are numpy arrays of float64
    gamma = np.array(gamma, dtype=np.float64)
    alpha = np.array(alpha, dtype=np.float64)

    # Verify lengths match
    assert len(gamma) == len(alpha) == len(Country), "Lengths of gamma, alpha, and Country must match"

    # Calculate node values
    node_wind_temp = alpha * gamma
    node_solar_temp = (1 - alpha) * gamma

    # Check that the sum matches gamma
    total = node_wind_temp + node_solar_temp
    assert np.allclose(total, gamma), "Sum of wind and solar components does not equal gamma"

    # Initialize lists to store nodal sums with gaps for plotting
    node_wind_plot = []
    node_solar_plot = []
    updated_Country = []

    # Identify unique countries in the dataset
    unique_countries = np.unique(Country)

    # Sort each category by gamma for each country and prepare nodal sums with gaps
    for country in unique_countries:
        country_indices = np.where(Country == country)[0]
        sorted_indices = np.argsort(gamma[country_indices])  # Sort by gamma within the country
        sorted_country_indices = country_indices[sorted_indices]

        # Add sorted data for plotting
        node_wind_plot.extend(node_wind_temp[sorted_country_indices])
        node_solar_plot.extend(node_solar_temp[sorted_country_indices])
        updated_Country.extend([country] * len(sorted_country_indices))

        # Add gaps in plotting arrays
        node_wind_plot.extend([np.nan, np.nan, np.nan, np.nan])
        node_solar_plot.extend([np.nan, np.nan, np.nan, np.nan])
        updated_Country.extend([''] * 4)

    # Convert to numpy arrays for plotting
    node_wind_plot = np.array(node_wind_plot)
    node_solar_plot = np.array(node_solar_plot)

    # Create the bar plot
    bar_width = 1  # Adjust bar width as needed
    indices = np.arange(len(node_wind_plot))

    plt.figure(figsize=(16, 3))

    # Plot each category with nodal accumulation and gaps
    plt.bar(indices, node_wind_plot, bar_width, label=r'$\text{Wind}$', color='deepskyblue', alpha=1)
    plt.bar(indices, node_solar_plot, bar_width, bottom=node_wind_plot, label=r'$\text{Solar}$', color='gold', alpha=1)

    plt.xlabel(r'$\text{Nodes categorized by country and sorted by }\gamma_{n}$', fontsize=14)
    plt.ylabel(r'$\gamma_n$', fontsize=14)

    # Set x-ticks and labels at the start of each country's data range
    ticks = []
    labels = []
    offset = 0
    for country in unique_countries:
        country_indices = np.where(Country == country)[0]
        num_nodes = len(country_indices)
        mid_pos = offset + num_nodes / 2
        ticks.append(mid_pos)
        labels.append(country)
        offset += num_nodes + 4  # Account for the added gaps

    plt.xticks(ticks=ticks, labels=labels, rotation=90, fontsize=10)
    plt.xlim(-1, len(indices) - 1)  # Adjust x-axis limit to ensure all bars are visible
    
    if ylim == 'none':
        plt.ylim(0, max(gamma) * 1.05)
    else:
        plt.ylim(0, ylim)
    plt.legend()
    plt.tight_layout()

    plot_type = 'GammaAlphaPlot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }

    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)

    # Show plot
    plt.show()

def avgGammaAlphaPlot2(kappa, LoadData, Country, figname, foldername, ylim='none'):
    # Extract LCOE data
    gamma = kappa['gamma_n']
    alpha = kappa['alpha_n']
    
    # Ensure gamma and alpha are numpy arrays of float64
    gamma = np.array(gamma, dtype=np.float64)
    alpha = np.array(alpha, dtype=np.float64)

    # Verify lengths match
    assert len(gamma) == len(alpha) == len(Country), "Lengths of gamma, alpha, and Country must match"

    # Calculate node values
    node_wind_temp = alpha * gamma
    node_solar_temp = (1 - alpha) * gamma

    # Check that the sum matches gamma
    total = node_wind_temp + node_solar_temp
    assert np.allclose(total, gamma), "Sum of wind and solar components does not equal gamma"

    # Initialize lists to store nodal sums with gaps for plotting
    node_wind_plot = []
    node_solar_plot = []
    updated_Country = []

    # Identify unique countries in the dataset
    unique_countries = np.unique(Country)

    # Sort each category by LCOE for each country and prepare nodal sums with gaps
    for country in unique_countries:
        country_indices = np.where(Country == country)[0]
        sorted_indices = np.argsort(total[country_indices])  # Sort by LCOE
        sorted_country_indices = country_indices[sorted_indices]

        # Add sorted data with gaps for plotting
        node_wind_plot.extend(node_wind_temp[sorted_country_indices])
        node_solar_plot.extend(node_solar_temp[sorted_country_indices])
        updated_Country.extend([country] * len(country_indices))

        # Add gaps in plotting arrays
        node_wind_plot.extend([np.nan, np.nan, np.nan, np.nan])
        node_solar_plot.extend([np.nan, np.nan, np.nan, np.nan])
        updated_Country.extend([''] * 4)

    # Convert to numpy arrays for plotting
    node_wind_plot = np.array(node_wind_plot)
    node_solar_plot = np.array(node_solar_plot)

    # Create the bar plot
    bar_width = 1  # Adjust bar width as needed
    indices = np.arange(len(node_wind_plot))

    plt.figure(figsize=(16, 3))

    # Plot each category with nodal accumulation and gaps
    plt.bar(indices, node_wind_plot, bar_width, label=r'$\text{Wind}$', color='deepskyblue', alpha=1)
    plt.bar(indices, node_solar_plot, bar_width, label=r'$\text{Solar}$', color='gold', alpha=1)

    plt.xlabel(r'$\text{Nodes categorized by country and sorted by }\gamma_{n}$', fontsize=14)
    plt.ylabel(r'$\gamma_n$', fontsize=14)

    # Set x-ticks and labels at the start of each country's data range
    ticks = []
    labels = []
    offset = 0
    for country in unique_countries:
        country_indices = np.where(Country == country)[0]
        num_nodes = len(country_indices)
        mid_pos = offset + num_nodes / 2
        ticks.append(mid_pos)
        labels.append(country)
        offset += num_nodes + 4

    plt.xticks(ticks=ticks, labels=labels, rotation=90, fontsize=10)
    plt.xlim(-1, len(indices) - 1)  # Adjust x-axis limit to ensure all bars are visible
    
    if ylim == 'none':
        plt.ylim(0, max(total) * 1.05)
    else:
        plt.ylim(0, ylim)
    plt.legend()
    plt.tight_layout()

    plot_type = 'LCOEplot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }

    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)

    # Show plot
    plt.show()
















def kappaC_plot(data, figname, foldername):
    # Extract data
    countries = data['Bus']
    
    Wind = data['Wind']
    Solar = data['Solar']
    Backup = data['Backup']

    
    # Prepare for plotting
    index = np.arange(len(countries))
    bar_width = 1
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the bars with updated colors and black outlines
    bars_backup = ax.bar(index, Backup, bar_width, bottom=Wind + Solar, label='Backup', color='firebrick', edgecolor='black')
    bars_solar = ax.bar(index, Solar, bar_width, bottom=Wind, label='Solar', color='gold', edgecolor='black')
    bars_wind = ax.bar(index, Wind, bar_width, label='Wind', color='deepskyblue', edgecolor='black')
    
    
 
    # Enhance legibility
    ax.set_xlabel('Country', fontsize=22)
    ax.set_ylabel(r'$\kappa_{C}$', fontsize=22)
    
    # Add y padding
    max_contribution = max(Wind + Solar + Backup)
    ax.set_ylim(0, max_contribution * 1.05)  # Add 18% padding above the highest bar
    ax.set_xlim(-2, len(countries))
    
    # Set the x-axis ticks and labels
    ax.set_xticks(index)
    ax.set_xticklabels(countries, rotation=90, ha="center", fontsize=16)
    
    # Customize the legend
    ax.legend(fontsize=18)
    
    # Customize tick parameters
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Prepare directories for saving the plot
    plot_type = 'LCOEplot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }

    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)

    # Show plot
    plt.show()
    
    
    
    
    
def LCOE_C_plot(data, countries, figname, foldername,ylim=False,ylabel='ylabel'):
    # Extract data
    Wind = data['LCOE_wind']
    Solar = data['LCOE_solar']
    Backup = data['LCOE_backup']
    Trans = data['LCOE_trans']

    # Prepare for plotting
    index = np.arange(len(countries))
    bar_width = 1

    # Create the figure and axis for the main plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the bars with updated colors and black outlines
    bars_trans = ax.bar(index, Trans, bar_width, label='Transmission', color='forestgreen', edgecolor='black')
    bars_backup = ax.bar(index, Backup, bar_width, bottom=Trans, label='Backup', color='rebeccapurple', edgecolor='black')
    bars_solar = ax.bar(index, Solar, bar_width, bottom=Trans + Backup, label='Solar', color='gold', edgecolor='black')
    bars_wind = ax.bar(index, Wind, bar_width, bottom=Trans + Backup + Solar, label='Wind', color='deepskyblue', edgecolor='black')

    # Enhance legibility
    ax.set_xlabel('Country', fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)

    # Add y padding
    max_contribution = max(Wind + Solar + Backup + Trans)
    if ylim==False:
        ax.set_ylim(0, max_contribution * 1.18)  # Add 18% padding above the highest bar
    else:
        ax.set_ylim(0,ylim)
        
    ax.set_xlim(-1, len(countries))

    # Set the x-axis ticks and labels
    ax.set_xticks(index)
    ax.set_xticklabels(countries, rotation=90, ha="center", fontsize=16)

    # Customize tick parameters
    ax.tick_params(axis='y', labelsize=16)

    # Adjust layout
    plt.tight_layout()

    # Prepare directories for saving the plot
    plot_type = 'LCOEplot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }

    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # Save the main plot without the legend
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Create a separate figure for the legend
    fig_legend, ax_legend = plt.subplots(figsize=(12, 2))
    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc='center', fontsize=20, ncol=len(handles))
    ax_legend.axis('off')
    
    # Save the legend
    for savetype, folder in folders.items():
        legend_filepath = os.path.join(folder, figname + f'_legend.{savetype}')
        plt.savefig(legend_filepath, bbox_inches='tight', format=savetype, dpi=300)

    # Show the main plot and the legend plot
    plt.show()
    fig_legend.show()
    
    

def LCOE_EU_plot(data, figname, foldername, ylim=False, ylabel='ylabel'):
    # Extract data
    Wind = data['Wind']
    Solar = data['Solar']
    Backup = data['Backup']
    Trans = data['Trans']

    # Prepare for plotting
    index = 1
    bar_width = 0.5  # Reduced the bar width to make the plot narrower

    # Create the figure and axis for the main plot
    fig, ax = plt.subplots(figsize=(2, 8))  # Reduced width from 12 to 6 to make the plot narrower

    # Plot the bars with updated colors and black outlines
    bars_trans = ax.bar(index, Trans, bar_width, label='Transmission', color='forestgreen', edgecolor='black')
    bars_backup = ax.bar(index, Backup, bar_width, bottom=Trans, label='Backup', color='rebeccapurple', edgecolor='black')
    bars_solar = ax.bar(index, Solar, bar_width, bottom=Trans + Backup, label='Solar', color='gold', edgecolor='black')
    bars_wind = ax.bar(index, Wind, bar_width, bottom=Trans + Backup + Solar, label='Wind', color='deepskyblue', edgecolor='black')

    # Enhance legibility
    ax.set_xlabel('EU', fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)

    # Add y padding
    max_contribution = Wind + Solar + Backup + Trans
    if not ylim:
        ax.set_ylim(0, max_contribution * 1.18)  # Add 18% padding above the highest bar
    else:
        ax.set_ylim(0, ylim)

    # Customize tick parameters
    ax.tick_params(axis='y', labelsize=16)
    ax.axes.get_xaxis().set_ticks([])
      # Adjust layout
    plt.tight_layout()

    # Prepare directories for saving the plot
    plot_type = 'LCOEplot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }

    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    # Save the main plot without the legend
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)

    # Show the main plot and the legend plot
    plt.show()
















def CapacityBarPlot(results, LoadData,figname,foldername):
    fig, ax = plt.subplots()
    LoadDataMean = LoadData.mean(axis=0)

    # Normalize the energy generation data by the mean load for each node
    # Backup = abs(results['Avr_export_Backup_sch12'].sum(axis=0)) / LoadDataMean
    # Solar = abs(results['Avr_export_Solar_sch12'].sum(axis=0)) / LoadDataMean
    # Wind = abs(results['Avr_export_Wind_sch12'].sum(axis=0)) / LoadDataMean
    
    Backup = abs(results['Capacity_exp_Backup'].sum(axis=0)) / LoadDataMean
    Solar = abs(results['Capacity_exp_Solar'].sum(axis=0)) / LoadDataMean
    Wind = abs(results['Capacity_exp_Wind'].sum(axis=0)) / LoadDataMean
    

    # Sort the nodes by mean load in descending order
    sorted_indices = np.argsort(LoadDataMean)[::-1]
    
    # Sort the normalized energy data to match the sorted load
    BackupSorted = Backup[sorted_indices]
    SolarSorted = Solar[sorted_indices]
    WindSorted = Wind[sorted_indices]
    
    # Define the number of bins
    num_bins = 32
    bin_size = len(BackupSorted) // num_bins

    # Aggregate the sorted normalized data into the defined number of bins
    agg_Backup = np.array([BackupSorted[i:i + bin_size].mean() for i in range(0, len(BackupSorted), bin_size)])
    agg_Solar = np.array([SolarSorted[i:i + bin_size].mean() for i in range(0, len(SolarSorted), bin_size)])
    agg_Wind = np.array([WindSorted[i:i + bin_size].mean() for i in range(0, len(WindSorted), bin_size)])

    # Create the x-axis indices for the bars
    ind = np.arange(num_bins)

    # Plotting the stacked bar chart
    plt.figure(figsize=(14, 7))
    plt.bar(ind, agg_Wind, width=0.9, color='deepskyblue', label='Wind')
    plt.bar(ind, agg_Solar, width=0.9, color='gold', label='Solar', bottom=agg_Wind)
    plt.bar(ind, agg_Backup, width=0.9, color='rebeccapurple', label='Backup', bottom=(agg_Wind + agg_Solar))

    plt.xlabel('Node Groups Sorted by Load')
    plt.ylabel('Normalized Energy Generation (per node mean load)')
    plt.title('Normalized Energy Generation by Source per Group of Nodes')
    plt.xticks(ind, [f'Group {i+1}' for i in ind], rotation=45)
    plt.legend()
    plt.tight_layout()
 
    

    plot_type = 'CapacityBarPlot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }
    
    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Show plot
    plt.show()
    






class TwoSlopeNorm(mcolors.Normalize):
    def __init__(self, vcenter=None, vmin=None, vmax=None):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip=True)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))





def Compare(x_data, y_data, x_label, y_label, plot_title, figname, foldername,
            x_limits=None, y_limits=None, color='C0', point_size=10, alpha=0.9, grid=False, edgecolor='none',
            linewidth=0.5, tick_intervalx=100, tick_intervaly=100, color_data=None, zoom_window=None, point_scale=0, cmap='plasma',equal=True,zoom_place=[0.45, 0.62, 0.3, 0.3],Zoom_as=1):
    
    cmap_check = False
    
    if cmap == 'div' and color_data is not None:
        cnorm = TwoSlopeNorm(vcenter=color_data.mean(), vmin=min(color_data), vmax=max(color_data))
        cmap = 'RdBu_r'
        cmap_check = True
    
    # Normalize the color_data for size scaling
    if color_data is not None:
        sizes = (color_data - np.min(color_data)) / (np.max(color_data) - np.min(color_data)) * point_scale + point_size
    else:
        sizes = point_size

    # Sort data to plot higher color_data values on top
    if color_data is not None:
        sorted_indices = np.argsort(color_data)
        x_data = x_data[sorted_indices]
        y_data = y_data[sorted_indices]
        color_data = color_data[sorted_indices]
        sizes = sizes[sorted_indices]

    # Create a figure and axis with larger size
    fig, ax = plt.subplots(figsize=(8, 8))

    # Use color_data if provided
    if color_data is not None and cmap_check:
        scatter = ax.scatter(x_data, y_data, s=sizes, c=color_data, norm=cnorm, cmap=cmap, alpha=alpha, edgecolors=edgecolor, linewidth=linewidth, marker='o')
    elif color_data is not None:
        scatter = ax.scatter(x_data, y_data, s=sizes, c=color_data, alpha=alpha, edgecolors=edgecolor, linewidth=linewidth, cmap=cmap, marker='o')
    else:
        scatter = ax.scatter(x_data, y_data, s=sizes, c=color, alpha=alpha, edgecolors=edgecolor, linewidth=linewidth, marker='o')

    # Add color bar if color_data is provided
    if color_data is not None and cmap_check:
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',shrink=0.8, ticks=[cnorm.vmin, cnorm.vmax])
        cbar.set_label(r'$\langle L_n\rangle$',fontsize=16)

    # Add labels and title with updated font properties
    ax.set_xlabel(x_label, fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel(y_label, fontsize=18, fontweight='bold', labelpad=10)
    ax.set_title(plot_title, fontsize=20, fontweight='bold', pad=20)
    

    # Set axis limits if provided
    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)

    # Ensure the plot is square
    
    ax.set_box_aspect(Zoom_as)
    

        

    # Set ticks with updated interval and properties
    if x_limits is not None:
        ax.set_xticks(np.arange(x_limits[0], x_limits[1] + 1, tick_intervalx))
    if y_limits is not None:
        ax.set_yticks(np.arange(y_limits[0], y_limits[1] + 1, tick_intervaly))
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Add grid with improved style
    if grid:
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
        # Add minor ticks for better granularity if grid is enabled
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', alpha=0.5)

    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw a light grey box around the data area
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgrey')
        spine.set_linewidth(1.5)

    # Improved zoom-in plot in the upper right corner
    if zoom_window:
        zoom_ax = fig.add_axes(zoom_place)  # [left, bottom, width, height]

        x_min_zoom, x_max_zoom, y_min_zoom, y_max_zoom = zoom_window
        zoom_ax.scatter(x_data, y_data, s=sizes*0.8, c=color_data,norm=cnorm if color_data is not None else color, alpha=alpha, edgecolors=edgecolor, linewidth=linewidth, cmap=cmap if color_data is not None else None, marker='o')
        zoom_ax.set_xlim(x_min_zoom, x_max_zoom)
        zoom_ax.set_ylim(y_min_zoom, y_max_zoom)
        zoom_ax.set_xticks(np.linspace(x_min_zoom, x_max_zoom, 3))
        zoom_ax.set_yticks(np.linspace(y_min_zoom, y_max_zoom, 3))
        zoom_ax.tick_params(axis='both', which='major', labelsize=8)
        if equal==True:
            zoom_ax.set_aspect('equal')
        else:
            ax.set_box_aspect(equal)
            
        if grid:
            zoom_ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        zoom_ax.set_title('Zoomed In', fontsize=10)

        # Draw zoom box on the main plot
        zoom_box = patches.Rectangle((x_min_zoom, y_min_zoom), x_max_zoom - x_min_zoom, y_max_zoom - y_min_zoom,
                                     linewidth=1, edgecolor='k', alpha=0.5, facecolor='none', linestyle='--')
        ax.add_patch(zoom_box)

    plot_type = 'compare'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }
    
    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Show plot
    plt.show()







def histplot2(data, x_label, y_label, figname, foldername, x_limits=None, y_limits=None, num_bins=100, zoom_window=None, zoom_num_bins=50, zoom_place=[0.45, 0.62, 0.3, 0.3], zoom_ylim=None):
    plt.figure(figsize=(8, 8))
    
    if x_limits is not None:
        # Generate bin edges based on x_limits and num_bins
        bin_edges = np.linspace(x_limits[0], x_limits[1], num_bins + 1)
    else:
        # Use default number of bins
        bin_edges = num_bins

    # Plot the histogram with defined bin edges
    plt.hist(data, bins=bin_edges, edgecolor='black', alpha=0.75)
    
    # Set x-axis limits if provided
    if x_limits is not None:
        plt.xlim(x_limits)
    if y_limits is not None:
        plt.ylim([0, y_limits])
        
    # Customize the plot
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=13)
    plt.grid(False)
    
    ax = plt.gca()
    ax.set_box_aspect(0.25)
    
    # Draw a light grey box around the data area
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgrey')
        spine.set_linewidth(1.5)
    
    ax.set_xlabel(x_label, fontsize=17, fontweight='bold', labelpad=10)
    ax.set_ylabel(y_label, fontsize=17, fontweight='bold', labelpad=10)
    
    # Create directories if they do not exist
    plot_type = 'compare'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Add zoomed-in inset
    if zoom_window is not None:
        x_min_zoom, x_max_zoom, y_min_zoom, y_max_zoom = zoom_window

        # Create zoomed-in inset
        zoom_ax = plt.axes(zoom_place)  # [left, bottom, width, height]
        
        # Generate bin edges based on zoom_x_limits and zoom_num_bins
        zoom_bin_edges = np.linspace(x_min_zoom, x_max_zoom, zoom_num_bins + 1)

        # Plot the zoomed-in histogram with defined bin edges
        zoom_ax.hist(data, bins=zoom_bin_edges, edgecolor='black', alpha=0.75)
        
        # Set x-axis limits for the zoom window
        zoom_ax.set_xlim(x_min_zoom, x_max_zoom)
        if zoom_ylim is not None:
            zoom_ax.set_ylim(0, zoom_ylim)
        
        zoom_ax.set_xticks(np.linspace(x_min_zoom, x_max_zoom, 5))
        zoom_ax.set_yticks(np.linspace(y_min_zoom, y_max_zoom, 3))
        
        
        zoom_ax.tick_params(axis='both', which='major', labelsize=8)
        
        zoom_ax.set_xlabel('')
        zoom_ax.set_ylabel('')
        zoom_ax.grid(False)
        
        for spine in zoom_ax.spines.values():
            spine.set_edgecolor('lightgrey')
            spine.set_linewidth(1.5)
        
        # Draw vertical lines on the main plot to indicate zoom area
        ax.axvline(x_min_zoom, color='lightgrey', linestyle='--', linewidth=1.5)
        ax.axvline(x_max_zoom, color='lightgrey', linestyle='--', linewidth=1.5)

    # Save the figure with zoom inset in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'_zoom.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Show plot
    plt.show()
    
    
    
    
    
    
    

def Compare2(x_data, y_data, x_label, y_label, plot_title, figname, foldername,
             color='deepskyblue', point_size=50, alpha=0.7, grid=True, edgecolor='none',
             linewidth=0.5, tick_intervalx=100, tick_intervaly=100,
             min_aspect_ratio=1, color_data=None):
    
    # Create a DataFrame from the data
    df = pd.DataFrame({
        x_label: x_data,
        y_label: y_data})
    
    # Determine the limits for x and y axes
    max_lim = max(df[x_label].max(), df[y_label].max())
    min_lim = min(df[x_label].min(), df[y_label].min())
    

    
    # Create a joint plot with scatter plot and histograms
    #sns.set(style="whitegrid")
    plot = sns.jointplot(data=df, x=x_label, y=y_label, kind="scatter", color=color,
                         marginal_kws=dict(bins=100, fill=True), height=8, ratio=5, space=0,
                         marginal_ticks=False)
    
    # Access the Matplotlib axes object from the jointplot
    ax = plot.ax_joint
    #plot.ax_joint.set_aspect('equal')
    # Customize plot features
    plot.fig.suptitle(plot_title)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    
    # Remove grid if specified
    if not grid:
        ax.grid(False)
    
    folders = {'pdf': os.path.join(foldername, 'pdf'), 'png': os.path.join(foldername, 'png')}
    
    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
        
    plt.show()

def Compare3(x_data, y_data, x_label, y_label, plot_title, figname, foldername,
             backup_color='rebeccapurple', solar_color='gold', wind_color='deepskyblue',
             point_size=10, alpha=0.7, grid=True, edgecolor='none', linewidth=0.5,
             tick_intervalx=100, tick_intervaly=100, min_aspect_ratio=1):
    
    # Extract data for each category
    Backup_x = x_data['LCOE_backup']
    Solar_x = x_data['LCOE_solar']
    Wind_x = x_data['LCOE_wind']
    
    Backup_y = y_data['LCOE_backup']
    Solar_y = y_data['LCOE_solar']
    Wind_y = y_data['LCOE_wind']
    
    # Create DataFrames for each category
    df_backup = pd.DataFrame({
        x_label: Backup_x,
        y_label: Backup_y,
        'Category': 'Backup'
    })
    
    df_solar = pd.DataFrame({
        x_label: Solar_x,
        y_label: Solar_y,
        'Category': 'Solar'
    })
    
    df_wind = pd.DataFrame({
        x_label: Wind_x,
        y_label: Wind_y,
        'Category': 'Wind'
    })
    
    # Concatenate the DataFrames
    df = pd.concat([df_solar, df_wind, df_backup], ignore_index=True)  # Backup is now last in the concatenation
    
    # Create a figure and axis using seaborn
    plt.figure(figsize=(8, 8))
    
    # Plot using seaborn.scatterplot with hue
    sns.scatterplot(data=df, x=x_label, y=y_label, hue='Category',
                    palette={'Backup': backup_color, 'Solar': solar_color, 'Wind': wind_color},
                    s=point_size, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
    
    # Customize plot features
    plt.title(plot_title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    
    # Set axis limits to start from (0,0)
    min_lim = min(df[x_label].min(), df[y_label].min())
    max_lim = max(df[x_label].max(), df[y_label].max())
    plt.xlim(0, max_lim)
    plt.ylim(0, max_lim)
    
    # Set grid if specified
    if grid:
        plt.grid(True)
    
    # Add legend
    plt.legend()
    
    plot_type = 'compare'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }
    
    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Show plot
    plt.show()



def bellcurve(LCOE, WSD,figname,foldername, xlim=None, ylim=None):
    # Calculate means
    mean_total = LCOE['LCOE'].mean()
    mean_wind = LCOE['LCOE_wind'].mean()
    mean_solar = LCOE['LCOE_solar'].mean()
    mean_backup = LCOE['LCOE_backup'].mean()
    mean_trans = LCOE['LCOE_trans'].mean()
    
    # Calculate weighted standard deviations
    WSD_total = WSD['LCOE']
    WSD_wind = WSD['Wind']
    WSD_solar = WSD['Solar']
    WSD_backup = WSD['Backup']
    WSD_trans = WSD['Trans']
    
    # Generate data for the bell curves
    x = np.linspace(mean_total - 4*WSD_total, mean_total + 4*WSD_total, 1000)
    y_total = norm.pdf(x, mean_total, WSD_total)
    y_wind = norm.pdf(x, mean_wind, WSD_wind)
    y_solar = norm.pdf(x, mean_solar, WSD_solar)
    y_backup = norm.pdf(x, mean_backup, WSD_backup)
    y_trans = norm.pdf(x, mean_trans, WSD_trans)

    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Plot the bell curves
    ax.plot(x, y_total, label='Total LCOE', color='k',linewidth=1)
    ax.plot(x, y_wind,'--', label='Wind LCOE', color='deepskyblue',linewidth=0.5)
    ax.plot(x, y_solar,'--', label='Solar LCOE', color='gold',linewidth=0.5)
    ax.plot(x, y_backup,'--', label='Backup LCOE', color='rebeccapurple',linewidth=0.5)
    ax.plot(x, y_trans,'--', label='Transmission LCOE', color='forestgreen',linewidth=0.5)
    
    # Add labels and title
    ax.set_xlabel('LCOE')
    ax.set_ylabel('Probability Density')
    ax.set_title(figname)
    
    # Set x and y limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    # Add legend
    ax.legend()
    
    plot_type = 'bellcurve'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }
    
    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Show plot
    plt.show()
    





def Compare4(x, y, figname, foldername, x_label='CF wind', y_label='ARGH'):
    """
    Plots a scatter plot of two variables against each other.

    Parameters:
    x (array-like): Data for the x-axis.
    y (array-like): Data for the y-axis.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    figname (str): Name of the figure file.
    foldername (str): Path to the folder where figures will be saved.
    """
    plt.figure(figsize=(8, 8))  # Set the figure size to be square (6x6 inches)

    # Plot the data
    plt.scatter(x, y, alpha=0.7, edgecolors='none', s=15, c='b')
    
    # Set labels and grid
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.grid(False)
    
    # Adjust axis limits based on data
    plt.xlim(min(x)*1.01, max(x)*1.01)
    plt.ylim(0, max(y)*1.01)
    
    # Customize ticks and labels
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Save directories for different formats
    plot_type = 'LCOEplot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }
    
    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Show plot
    plt.tight_layout()  # Ensures everything fits within the figure area
    plt.show()




def LCOE_DIFF(diff, countries,figname,foldername):
    
    index = np.arange(len(countries))
    bar_width = 1
    
    #plt.style.use('seaborn-darkgrid')  # Use a seaborn style for a prettier plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size for readability
    
    # Plotting with updated colors and black outlines
    bars_diff = ax.bar(index, diff, bar_width, color='green', edgecolor='black')
    
    
    # Enhance legibility
    ax.set_xlabel('Country', fontsize=22)
    ax.set_ylabel(r'$\Delta\widetilde{LCOE}_c$', fontsize=22)
    #ax.set_title('Mean Gamma Contributions by Source for Each Country', fontsize=22)
    
    # Add y padding
    max_contribution = min(diff)
    ax.set_ylim(0, max_contribution * 1.02)  # Add 10% padding above the highest bar
    
    ax.set_xlim(-1,len(diff))
    
    
    ax.set_xticks(index)
    ax.set_xticklabels(countries, rotation=90, ha="center", fontsize=16)
    ax.legend(fontsize=18)
    
    ax.tick_params(axis='y', labelsize=16)  # Make y-axis tick labels larger
    ax.legend(fontsize=20)
    
    
    
    
    plt.tight_layout()
    plot_type = 'hetplot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }
    
    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Show plot
    plt.show()













def Worldmapplot(EVec, cmap, Title, figname, unit, Map, gdf, foldername, bounds_list='none', own_cap='none',Title_on=False,form='%.0f'):
    """
    Calls the Worldmapplot function multiple times with different bounds configurations,
    each time with a unique title and filename based on the bounds.

    Parameters:
    - EVec: The energy vector to plot.
    - cmap: Colormap for the plot.
    - Title: Base title of the plot. This will be modified for each bounds configuration.
    - figname: Base filename for saving plots. This will be modified for each bounds configuration.
    - unit: The unit of the values in EVec.
    - Map: The base map for plotting.
    - gdf: GeoDataFrame with the data to plot.
    - foldername: The folder where plots will be saved.
    - bounds_list: A list of bounds configurations. Each configuration is either a string or a dictionary specifying the mode and parameters.
    - own_cap: (Optional) Own capacity information for the title. Default is 'none'.
    """
    if not isinstance(bounds_list, list):
        bounds_list = [bounds_list]
        
    for idx, bounds in enumerate(bounds_list):
        # Modify figname for each configuration to ensure unique filenames
        unique_figname = f"{figname.rstrip('.png')}_bounds_{idx}"
        
        # Create a unique title for each plot based on bounds
        bounds_desc = " with bounds" if bounds == 'fixed' else ""
        unique_title = f"{Title}{bounds_desc}"
        
        # Call the original Worldmapplot function with the current bounds configuration and unique title
        WorldmapplotIndividual(EVec, cmap, unique_title, unique_figname, unit, Map, gdf, foldername, bounds=bounds, own_cap=own_cap,Title_on=Title_on,form=form)


#COLORMAPS USED (TODO put in seperate file)

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))






def WorldmapplotIndividual(EVec,cmap, Title,figname,unit, Map,  gdf,foldername, bounds='none',own_cap = 'none',Title_on=False,form='%.0f'): 
    
    cmap = plt.cm.get_cmap(cmap)
    # Directly modify the colormap to set NaN values to red
    
    
    
    
    # Longitude and lattitude of plot boundary
    MinLon, MaxLon = -10, 31.5 # -15, 35
    MinLat, MaxLat = 35.7, 68 # 35.7, 70
    extent=(MinLon, MinLat, MaxLon, MaxLat)
    
    
    vmin, vmax = min(EVec), max(EVec)
    vcenter=(vmin + vmax) / 2

    # Determine mode and parameters
    mode = bounds if isinstance(bounds, str) else bounds.get('mode', 'none')
    vmin = bounds.get('vmin', vmin) if isinstance(bounds, dict) else vmin
    vcenter = bounds.get('vcenter', vcenter) if isinstance(bounds, dict) else vcenter
    vmax = bounds.get('vmax', vmax) if isinstance(bounds, dict) else vmax
    max_cap = bounds.get('max_cap') if isinstance(bounds, dict) else False

    # Apply normalization based on mode
    if mode == 'div':
        cnorm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=vcenter)
    elif mode == 'div_new':
        cnorm = mcolors.TwoSlopeNorm(vmin=vmin,vmax=vmax,vcenter=vcenter)
    elif mode =='help':
        cnorm = mcolors.CenteredNorm(vcenter=vcenter)
    elif mode =='alpha':
        cnorm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=vcenter)
    elif mode == 'none':
        vcenter=(vmin + vmax) / 2
        cnorm = mcolors.TwoSlopeNorm(vmin=vmin,vcenter=vcenter,vmax=vmax)
    elif mode =='twoslope':
        cnorm = mcolors.TwoSlopeNorm(vmin=vmin,vcenter=vcenter,vmax=vmax)
    elif mode == 'fixed':
        cnorm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        vcenter=(vmin + vmax) / 2
    elif mode == 'log':
        cnorm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    elif mode == 'percentile':
        cnorm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    elif mode == 'asymmetric_div':
        vcenter = bounds.get('vcenter', (vmin + vmax) / 2) if isinstance(bounds, dict) else (vmin + vmax) / 2
        cnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        raise ValueError(f"Unsupported bounds option: {mode}")
    
    
    
    
        
    # Projection type
    #proj = gcrs.AlbersEqualArea()
    proj = gcrs.Miller(central_longitude=0)
    
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection':proj})
    
    EVec_copy = EVec.copy()
    
    
    
    if Title!='' and Title == True:
        ax.set_title(Title, fontsize=20)
    
    if own_cap != 'none':
        own_cap = int(own_cap)
        
        title = r"Own Capacity: {}\,\si{{MW}}".format(own_cap)
        wrapped_title = "\n".join(textwrap.wrap(title, 80))
        fig.suptitle(wrapped_title, fontsize=16,color='Black', y=0.8, x=0.3)
    
    if max_cap ==True:
        EVec_copy[EVec_copy > vmax] = np.nan
        EVec_copy[EVec_copy < vmin] = np.nan
    
    
    
    
    cmap.set_bad(color='Black')
    
    gplt.polyplot(Map, ax=ax, extent=extent, zorder=2, linewidth=0.4)
    plot = gplt.voronoi(gdf, clip=Map, hue=EVec_copy, projection=proj,
                  cmap=cmap, norm=cnorm, extent=extent, edgecolor='w', linewidth=0,
                  ax=ax)
    
    
    
    
    

    tick_place = [vmin, vcenter, vmax]
    
    
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap),
                    ax=ax, orientation='horizontal', pad=0.01, fraction=0.04,
                    ticks=tick_place, format=form,spacing='proportional')
    
    
    
    # Setting the tick labels with the specified format
    tick_labels = [form % x for x in tick_place]
    cbar.ax.set_xticklabels(tick_labels, fontsize=18)
    
    
        
    
    # Optionally set the colorbar label if `unit` is not empty
    if unit != '':
        cbar.set_label(unit,fontsize=28)
        
        
    
    
    plot_type = 'wordmap_plot'
    folders = {
        'pdf': os.path.join(foldername, 'pdf', plot_type),
        'png': os.path.join(foldername, 'png', plot_type)
    }
    
    # Create directories if they do not exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Save the figure in both formats
    for savetype, folder in folders.items():
        filepath = os.path.join(folder, figname + f'.{savetype}')
        plt.savefig(filepath, bbox_inches='tight', format=savetype, dpi=300)
    
    # Show plot
    plt.show()


































































