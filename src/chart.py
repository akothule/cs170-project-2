import pandas as pd
import matplotlib.pyplot as plt


def create_chart(accuracy_percentages, feature_set_labels, title, xlabel, ylabel, filename, max_labels=20):
    # create a bar chart with improved styling
    plt.figure(figsize=(12, 6), facecolor='white')

    # use a more professional style
    plt.style.use('seaborn-v0_8')

    # create bars with better colors and edge
    bars = plt.bar(range(len(accuracy_percentages)), accuracy_percentages,
                   color='#606060', edgecolor='#404040', linewidth=1)

    # add a thin horizontal line at y=0
    plt.axhline(y=0, color='#888888', linestyle='-', linewidth=0.5)

    # improve axis labels and title with better fonts
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)

    # handle x-tick labels for large datasets
    num_items = len(accuracy_percentages)

    # store which indices will get x-tick labels
    x_tick_indices = []

    if num_items <= max_labels:
        # for small datasets, show all labels
        plt.xticks(range(num_items), feature_set_labels, rotation=45, ha='center', fontsize=10)
        x_tick_indices = list(range(num_items))
    else:
        # for large datasets, show a subset of labels
        # calculate how many labels to show from beginning and end
        half_count = max_labels // 2

        # add first half_count indices
        for i in range(min(half_count, num_items)):
            x_tick_indices.append(i)

        # add last half_count indices, but don't duplicate if the dataset isn't large enough
        for i in range(max(half_count, num_items - half_count), num_items):
            if i not in x_tick_indices:  # Avoid duplicates
                x_tick_indices.append(i)

        # sort indices to maintain proper order
        x_tick_indices = sorted(x_tick_indices)

        # create sparse labels (empty string for non-selected indices)
        sparse_labels = ['' for _ in range(num_items)]
        for idx in x_tick_indices:
            sparse_labels[idx] = feature_set_labels[idx]

        plt.xticks(range(num_items), sparse_labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=4)
        # add more bottom margin to accommodate vertical labels
        plt.subplots_adjust(bottom=0.3)

    # set y-axis limits with a bit of padding
    plt.ylim(0, max(accuracy_percentages) * 1.1)

    # add a subtle grid
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # always use x_tick_indices to determine which bars get data labels
    # this ensures any bar with an x-axis label also gets a data value label
    if num_items > max_labels:
        # add the labels to the same bars that have x-tick labels
        for i in x_tick_indices:
            bar = bars[i]
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=5)
    else:
        # for smaller datasets, label all bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # add a subtle box around the plot
    plt.box(True)

    # ensure everything fits
    plt.tight_layout()

    # higher resolution and quality
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    # print(f"Chart saved to {filename}")



def test():
    # read the CSV file, skipping comment lines if present
    csv_file = 'data/forward_selection_results_small.csv'
    output_file = csv_file.replace('.csv', '.png')
    df = pd.read_csv(csv_file, comment='#')
    print(f"Successfully loaded {len(df)} rows from {csv_file}")

    # extract the data
    feature_set_labels = df['Feature Set'].tolist()
    accuracy_values = df['Accuracy (%)'].tolist()

    title = 'Accuracy of increasingly small subsets of features discovered by backward elimination'
    xlabel = 'Current Feature Set: Backward Selection'
    ylabel = 'Accuracy (%)'

    # create the chart
    print(f"Creating chart: {title}")
    create_chart(accuracy_values, feature_set_labels, title, xlabel, ylabel, output_file)

    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # main()
    test()
