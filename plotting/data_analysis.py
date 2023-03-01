import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_stance_analysis(df, labels_columns):
    df_tmp = df[labels_columns].loc[df['Stance'] == 'against'] 
    sums_favor = df_tmp.sum().astype(float)

    df_tmp = df[labels_columns].loc[df['Stance'] != 'against'] 
    sums_against = df_tmp.sum().astype(float)

    fig, ax = plt.subplots()

    bar_width = 0.35
    bar_positions1 = np.arange(len(sums_favor))
    bar_positions2 = [p + bar_width for p in bar_positions1]

    labels = ['in favor of', 'against']

    # plot the bar chart for df1
    barlist_favor = ax.bar(bar_positions1, sums_favor.values, bar_width, label=labels[0], hatch='')

    # plot the bar chart for df2
    barlist_against = ax.bar(bar_positions2, sums_against.values, bar_width, label=labels[1], hatch='')

    # highlight the smaller bar with a different hatch pattern
    for i, bar in enumerate(zip(barlist_favor, barlist_against)):
        if barlist_favor[i].get_height() > barlist_against[i].get_height():
            barlist_against[i].set(hatch='//')
        elif barlist_favor[i].get_height() < barlist_against[i].get_height():
            barlist_favor[i].set(hatch='//')

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

def plot_label_distribution(labels):
    label_counts = labels.iloc[:,1:].astype(int).sum().sort_values(ascending=False)

    # Plot the label distribution as a bar chart
    plt.figure(figsize=(10,6))
    plt.bar(label_counts.index, label_counts.values)
    plt.xticks(rotation=90)
    plt.xlabel('Label Category')
    plt.ylabel('Number of Arguments')
    plt.title('Label Distribution in Training Set')
    plt.show()

    # Print the label counts as a table
    print('Label Counts:\n', label_counts)

def plot_argument_lengths(arguments):
    # Create a new column for argument length
    arguments['arg_length'] = arguments['Conclusion'].str.split().apply(len) + \
                                    arguments['Stance'].str.split().apply(len) + \
                                    arguments['Premise'].str.split().apply(len)

    # Plot the histogram of argument lengths
    plt.hist(arguments['arg_length'], bins=50)
    plt.title('Distribution of Argument Lengths')
    plt.xlabel('Argument Length (in words)')
    plt.ylabel('Frequency')
    plt.show()

def plot_labels_correlation(labels):
    # compute the correlation matrix
    corr_matrix = labels.corr()

    plt.figure(figsize=(10, 8))

    # set font size
    sns.set(font_scale=1)

    # plot heatmap
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='')

    # set axis labels and title
    plt.xlabel('Label Categories')
    plt.ylabel('Label Categories')
    plt.title('Correlation between Label Categories')

    # rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # show plot
    plt.show()