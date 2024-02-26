#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import json
import pingouin as pg
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


# In[2]:


def analyze_and_plot_tau_by_day(dataframe):
    sns.set_style("whitegrid")

    # Compute max_count and relative_counts for state
    max_count = dataframe['state'].value_counts().max()
    relative_counts = dataframe['state'].value_counts() / max_count

    # Get colormap
    cmap = plt.get_cmap('crest')

    # Create a color dictionary for states
    state_colors = {state: cmap(relative_counts[state]) for state in relative_counts.keys()}

    # Extract unique animals and sort days
    unique_animals = sorted(dataframe.index.unique())
    unique_days = sorted(dataframe['day'].unique())

    # Looping through each animal and each day in ascending order
    for animal in unique_animals:
        for day in unique_days:
            subset = dataframe.loc[animal]
            subset = subset[subset['day'] == day]

            # If both 'wake' and 'sleep' states are present for the day, then proceed
            if 'wake' in subset['state'].values and 'sleep' in subset['state'].values:
                # Plotting setup with GridSpec
                fig = plt.figure(figsize=(18, 6))
                gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
                ax3 = fig.add_subplot(gs[2])  # this will be used for the colorbar

                # Violin plot with relative colors and ensured order
                sns.violinplot(x='state', y='tau', data=subset, ax=ax1, inner="quartile",
                               palette=state_colors, order=["wake", "sleep"])
                ax1.set_ylim(bottom=0)  # Set the y-axis bottom limit to 0 for tau
                ax1.set_title(f'Animal: {animal} | Day: {day}', fontsize=16, fontweight='bold')
                ax1.set_xlabel('State', fontsize=14)
                ax1.set_ylabel('Tau', fontsize=14)

                # Box plot with relative colors and ensured order
                sns.boxplot(x='state', y='tau', data=subset, ax=ax2, width=0.4, 
                            palette=state_colors, order=["wake", "sleep"])
                ax2.set_ylim(bottom=0)  # Set the y-axis bottom limit to 0 for tau
                ax2.set_title(f'Animal: {animal} | Day: {day}', fontsize=16, fontweight='bold')
                ax2.set_xlabel('State', fontsize=14)
                ax2.set_ylabel('Tau', fontsize=14)

                # Adding colorbar for color intensity
                sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_count))
                sm.set_array([])
                cbar = plt.colorbar(sm, cax=ax3, orientation="vertical", label="Number of Data Points")

                # Adjust spacing between plots
                gs.tight_layout(fig, rect=[0, 0, 0.95, 1])

                plt.show()

                # Statistical Analysis with Pingouin pairwise_tests
                pairwise_results = pg.pairwise_tests(data=subset, dv='tau', between='state', subject=None, correction='auto', parametric='auto')

                # Display pairwise test results for the plot
                display(pairwise_results)

def analyze_and_plot_average_epochstate(dataframe, metric):
    sns.set_style("whitegrid")

    # Compute max_count and relative_counts for state
    max_count = dataframe['state'].value_counts().max()
    relative_counts = dataframe['state'].value_counts() / max_count

    # Get colormap
    cmap = plt.get_cmap('crest')

    # Create a color dictionary for states
    state_colors = {state: cmap(relative_counts[state]) for state in relative_counts.keys()}

    # Plotting setup with GridSpec
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])  # this will be used for the colorbar

    # Violin plot with relative colors
    sns.violinplot(x='state', y=metric, data=dataframe, ax=ax1, inner="quartile",
                   palette=state_colors, order=["wake", "sleep"])
    ax1.set_title(f'{metric.capitalize()} Distribution by State', fontsize=16, fontweight='bold')
    ax1.set_xlabel('State', fontsize=14)
    ax1.set_ylabel(metric.capitalize(), fontsize=14)

    # Bar plot for average values with relative colors
    avg_data = dataframe.groupby('state')[metric].mean().reset_index()
    sns.barplot(x='state', y=metric, data=avg_data, 
                ax=ax2, palette=state_colors, order=["wake", "sleep"])
    ax2.set_title(f'Average {metric.capitalize()} by State', fontsize=16, fontweight='bold')
    ax2.set_xlabel('State', fontsize=14)
    ax2.set_ylabel(f'Average {metric.capitalize()}', fontsize=14)

    # Adjust y-axis based on metric
    if metric == 'tau':
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
    elif metric == 'branching_factor':
        ax1.set_ylim(bottom=0.9, top=1)
        ax2.set_ylim(bottom=0.9, top=1)

    # Aesthetic improvements
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=12)

    # Adding colorbar for color intensity
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_count))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax3, orientation="vertical", label="Number of Data Points")

    # Adjust spacing between plots
    gs.tight_layout(fig, rect=[0, 0, 0.95, 1])

    plt.show()

    # Statistical Analysis with Pingouin
    # Add a dummy "group" column to use with pairwise_tests (all values set to the same group)
    dataframe['group'] = 'comparison'
    pairwise_results = pg.pairwise_tests(data=dataframe, dv=metric, between='state', subject=None, correction='auto', parametric='auto')

    # Display pairwise test results for the plot
    display(pairwise_results)
    
def analyze_and_plot_average_epochstate_by_area(dataframe, metric):
    sns.set_style("whitegrid")

    # Compute counts for state
    state_counts = dataframe['state'].value_counts()
    max_count = state_counts.max()

    # Get colormap
    cmap = plt.get_cmap('crest')

    # Create a color dictionary for states with relative coloring
    state_colors = {state: cmap(count / max_count) for state, count in state_counts.items()}

    # Extract unique areas
    unique_areas = dataframe['area'].unique()

    # Looping through each area
    for area in unique_areas:
        subset = dataframe[dataframe['area'] == area]

        # Group by state and compute mean for the specified metric
        avg_data = subset.groupby('state')[metric].mean().reset_index()

        # Plotting setup with GridSpec
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])  # this will be used for the colorbar

        # Violin plot with relative colors and ensured order
        sns.violinplot(x='state', y=metric, data=subset, ax=ax1, inner="quartile",
                       palette=state_colors, order=["wake", "sleep"])
        
        ax1.set_title(f'{area} - {metric.capitalize()} Distribution', fontsize=16, fontweight='bold')
        ax1.set_xlabel('State', fontsize=14)
        ax1.set_ylabel(metric.capitalize(), fontsize=14)

        # Set y-axis limits based on metric
        if metric == 'tau':
            ax1.set_ylim(bottom=0)
        elif metric == 'branching_factor':
            ax1.set_ylim(bottom=0.9, top=1.0)

        # Bar plot for average values
        sns.barplot(x='state', y=metric, data=avg_data, ax=ax2, palette=state_colors, order=["wake", "sleep"])
        ax2.set_title(f'{area} - Average {metric.capitalize()}', fontsize=16, fontweight='bold')
        ax2.set_xlabel('State', fontsize=14)
        ax2.set_ylabel(f'Average {metric.capitalize()}', fontsize=14)

        # Set y-axis limits based on metric
        if metric == 'tau':
            ax2.set_ylim(bottom=0)
        elif metric == 'branching_factor':
            ax1.set_ylim(bottom=0.9, top=1.0)
            ax2.set_ylim(bottom=0.9, top=1.0)

        # Aesthetic improvements
        for ax in [ax1, ax2]:
            ax.tick_params(axis='both', which='major', labelsize=12)

        # Adding colorbar for color intensity with absolute data point numbers
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_count))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax3, orientation="vertical", label="Number of Data Points")
        cbar.set_ticks(np.linspace(0, max_count, 5))
        cbar.set_ticklabels([f"{int(count)}" for count in np.linspace(0, max_count, 5)])

        # Adjust spacing between plots
        gs.tight_layout(fig, rect=[0, 0, 0.95, 1])

        plt.show()

        # Statistical Analysis with Pingouin
        results = pg.pairwise_tests(data=subset, dv=metric, between='state', subject=None, correction='auto', parametric='auto')
        display(results)
        
        
def analyze_and_plot_bf_by_day(dataframe):
    sns.set_style("whitegrid")

    # Compute max_count and relative_counts for state
    max_count = dataframe['state'].value_counts().max()
    relative_counts = dataframe['state'].value_counts() / max_count

    # Get colormap
    cmap = plt.get_cmap('crest')

    # Create a color dictionary for states
    state_colors = {state: cmap(relative_counts[state]) for state in relative_counts.keys()}

    # Extract unique animals and sort days
    unique_animals = sorted(dataframe.index.unique())
    unique_days = sorted(dataframe['day'].unique())

    # Looping through each animal and each day in ascending order
    for animal in unique_animals:
        for day in unique_days:
            subset = dataframe.loc[animal]
            subset = subset[subset['day'] == day]

            # If both 'wake' and 'sleep' states are present for the day, then proceed
            if 'wake' in subset['state'].values and 'sleep' in subset['state'].values:
                # Plotting setup with GridSpec
                fig = plt.figure(figsize=(18, 6))
                gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
                ax3 = fig.add_subplot(gs[2])  # this will be used for the colorbar

                # Violin plot with relative colors and ensured order
                sns.violinplot(x='state', y='branching_factor', data=subset, ax=ax1, inner="quartile",
                               palette=state_colors, order=["wake", "sleep"])

                # Set y-axis limits for 'branching_factor'
                ax1.set_ylim(bottom=0.9, top=1)
                ax1.set_title(f'Animal: {animal} | Day: {day}', fontsize=16, fontweight='bold')
                ax1.set_xlabel('State', fontsize=14)
                ax1.set_ylabel('Branching Factor', fontsize=14)

                # Box plot with relative colors and ensured order
                sns.boxplot(x='state', y='branching_factor', data=subset, ax=ax2, width=0.4, 
                            palette=state_colors, order=["wake", "sleep"])
                
                ax2.set_ylim(bottom=0.9, top=1)
                ax2.set_title(f'Animal: {animal} | Day: {day}', fontsize=16, fontweight='bold')
                ax2.set_xlabel('State', fontsize=14)
                ax2.set_ylabel('Branching Factor', fontsize=14)

                # Adding colorbar for color intensity
                sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_count))
                sm.set_array([])
                cbar = plt.colorbar(sm, cax=ax3, orientation="vertical", label="Number of Data Points")

                # Adjust spacing between plots
                gs.tight_layout(fig, rect=[0, 0, 0.95, 1])

                plt.show()

                # Statistical Analysis with Pingouin pairwise_tests
                pairwise_results = pg.pairwise_tests(data=subset, dv='branching_factor', between='state', subject=None, correction='auto', parametric='auto')

                # Display pairwise test results for the plot
                display(pairwise_results)
                

def analyze_and_plot_average_epochstate_by_animal(dataframe, metric='tau'):
    sns.set_style("whitegrid")

    # Compute max_count and relative_counts for state
    max_count = dataframe['state'].value_counts().max()
    relative_counts = dataframe['state'].value_counts() / max_count

    # Get colormap
    cmap = plt.get_cmap('crest')

    # Create a color dictionary for states
    state_colors = {state: cmap(relative_counts[state]) for state in relative_counts.keys()}

    # Extract unique animals and sort them
    unique_animals = sorted(dataframe.index.unique())

    # Looping through each animal in ascending order
    for animal in unique_animals:
        subset = dataframe.loc[animal].copy() # Create a copy of the slice
        subset['group'] = 'comparison'
        
        # Plotting setup with GridSpec
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])  # this will be used for the colorbar

        # Violin plot with relative colors and ensured order
        sns.violinplot(x='state', y=metric, data=subset, ax=ax1, inner="quartile",
                       palette=state_colors, order=["wake", "sleep"])
        
        ax1.set_title(f'Animal: {animal} | {metric.capitalize()} Distribution', fontsize=16, fontweight='bold')
        ax1.set_xlabel('State', fontsize=14)
        ax1.set_ylabel(metric.capitalize(), fontsize=14)

        # Box plot with relative colors and ensured order
        sns.boxplot(x='state', y=metric, data=subset, ax=ax2, width=0.4,
                    palette=state_colors, order=["wake", "sleep"])
        ax2.set_title(f'Animal: {animal} | Average {metric.capitalize()}', fontsize=16, fontweight='bold')
        ax2.set_xlabel('State', fontsize=14)
        ax2.set_ylabel(metric.capitalize(), fontsize=14)
        
        if metric == 'tau':
            ax1.set_ylim(bottom=0)
            ax2.set_ylim(bottom=0)
        elif metric == 'branching_factor':
            ax1.set_ylim(bottom=0.9, top=1)
            ax2.set_ylim(bottom=0.9, top=1)
                         
        # Aesthetic improvements
        for ax in [ax1, ax2]:
            ax.tick_params(axis='both', which='major', labelsize=12)

        # Adding colorbar for color intensity
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_count))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax3, orientation="vertical", label="Number of Data Points")

        # Adjust spacing between plots
        gs.tight_layout(fig, rect=[0, 0, 0.95, 1])

        plt.show()

        # Statistical Analysis with Pingouin pairwise_tests
        pairwise_results = pg.pairwise_tests(data=subset, dv=metric, between='state', subject=None, correction='auto', parametric='auto')

        # Display pairwise test results for the plot
        display(pairwise_results)


# In[ ]:




