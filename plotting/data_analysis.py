import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_stance_analysis(df, labels_columns):
    df_tmp = df[labels_columns].loc[df['Stance'] == 'against'] 
    df_tmp = df_tmp.drop('Argument ID', axis=1)
    sums_favor = df_tmp.sum().astype(float)

    df_tmp = df[labels_columns].loc[df['Stance'] != 'against'] 
    df_tmp = df_tmp.drop('Argument ID', axis=1)
    sums_against = df_tmp.sum().astype(float)

    fig, ax = plt.subplots()

    bar_width = 0.35
    bar_positions1 = np.arange(len(sums_favor))
    bar_positions2 = [p + bar_width for p in bar_positions1]
    
    labels = ['in favor of', 'against']

    # plot the bar chart for df1
    barlist_favor = ax.bar(bar_positions1, sums_favor.values, bar_width, label=labels[0])

    # plot the bar chart for df2
    barlist_against = ax.bar(bar_positions2, sums_against.values, bar_width, label=labels[1])

    for i, bar in enumerate(zip(barlist_favor, barlist_against)):
        if barlist_favor[i].get_height() > barlist_against[i].get_height():
            barlist_against[i].set(linestyle='--', color='grey')
        elif barlist_favor[i].get_height() < barlist_against[i].get_height():
            barlist_favor[i].set(linestyle='--', color='grey')

    # set the x-axis tick labels
    ax.set_xticks([p + bar_width / 2 for p in bar_positions1])
    ax.set_xticklabels(sums_favor.index, rotation=90)

    # set the chart title and axis labels
    ax.set_title('Sum of values in each column')
    ax.set_xlabel('Human Value')
    ax.set_ylabel('Count of labels')

    # add a legend
    ax.legend()

    plt.show()