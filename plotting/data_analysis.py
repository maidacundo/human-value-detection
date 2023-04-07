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


def plot_classification_reports(reports, names, title):
    # Get the list of labels from one of the reports
    labels = list(reports[0].keys())[:20]

    # Create an array to hold the f1-scores for each report and each label
    f1_scores = np.zeros((len(reports), len(labels)))

    # Calculate the support for each label
    support = np.array([reports[0][label]['support'] for label in labels])
    support_scaled = support / support.max()  # Scale the support values

    # Populate the f1_scores array with the f1-scores for each report and each label
    for i, report in enumerate(reports):
        for j, label in enumerate(labels):
            f1_scores[i][j] = report[label]['f1-score']

    # Set up the figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Plot the f1-scores for each label as lines on the top y-axis
    for i in range(len(reports)):
        ax1.plot(f1_scores[i], label=names[i])
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, rotation=90, ha='center')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('F1-score')
    ax1.legend()

    # Plot the histogram of label support on the bottom y-axis
    ax2.bar(np.arange(len(support)), support_scaled, alpha=0.15, color='purple', width=0.5)
    ax2.yaxis.set_tick_params(length=0)
    ax2.set_yticklabels([])

    # Add a secondary y-axis with the real support values
    ax3 = ax2.secondary_yaxis('right', functions=(lambda x: x * support.max(), lambda x: x / support.max()), color='purple')
    ax3.set_ylabel('Support')
    ax3.tick_params(axis='y', color='purple')

    # Adjust the layout to make room for the x-axis labels
    fig.subplots_adjust(bottom=0.4)

    fig.suptitle(title)
    # Show the plot
    plt.show()

def print_MultilabelConfusionMatrix(matrices, labels, title):
    rows = 5
    cols = 4
    f, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.ravel()
    for i in range(len(matrices)):
        disp = ConfusionMatrixDisplay(matrices[i])
        disp.plot(ax=axes[i])
        disp.ax_.set_title(labels[i])
        if i<len(matrices)-cols:
            disp.ax_.set_xlabel('')
        if i%cols!=0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.05, hspace=0.5)
    f.colorbar(disp.im_, ax=axes)
    f.suptitle(title)
    plt.show()

def print_global_report(reports, names):
    dfs = []
    for i, report in enumerate(reports):
        if i == len(reports)-1:
            df = pd.DataFrame.from_dict(report, orient='index', columns=['precision', 'recall', 'f1-score', 'support'])
            dfs.append(df)
        else:
            df = pd.DataFrame.from_dict(report, orient='index', columns=['precision', 'recall', 'f1-score'])
            dfs.append(df)

    pd.options.display.float_format = '{:,.2f}'.format
    # Concatenate the dataframes into a single table
    df_combined = pd.concat(dfs, axis=1, keys=names)
    # Print the combined table
    print(df_combined.to_string())