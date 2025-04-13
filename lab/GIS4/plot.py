import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_columns_of_interest(df: pd.DataFrame) -> list:
    return [col for col in df.columns if col.startswith('THAW') or col.startswith('SNOW')]


def boxplot_snow_melt(data):
    columns_of_interest = get_columns_of_interest(data)

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        data[columns_of_interest], 
        tick_labels=columns_of_interest, 
        meanline=True, 
        meanprops={'color': 'blue'}, 
        showmeans=True,
    )
    plt.title('Boxplot of Snow and Thaw Data')
    plt.xlabel('Variables')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_snow_melt_statistics(data: pd.DataFrame):
    columns_of_interest = get_columns_of_interest(data)

    stats = data[columns_of_interest].describe().T
    stats['variance'] = data[columns_of_interest].var()
    stats['range'] = stats['max'] - stats['min']

    thaw_stats = stats[stats.index.str.startswith('THAW')]


    thaw_stats[['mean', 'variance', 'range']].plot(
        kind='line',
        marker='o', 
        figsize=(12, 6), 
        color=['skyblue', 'orange', 'green'], 
        title='Mean, Variance, and Range for Thaw Each Month'
    )

    plt.ylabel('Value')
    plt.xlabel('Month')
    plt.legend(title='Statistics')
    plt.tight_layout()
    plt.show()


def plot_snow_melt_regression_lines(data: pd.DataFrame):
    columns_of_interest = get_columns_of_interest(data)

    plt.figure(figsize=(12, 8))

    regression_data = []

    # Define a colormap
    colors = plt.cm.tab10_r(np.linspace(0, 1, len([col for col in columns_of_interest if col.startswith('THAW')])))

    fig, (ax_scatter, ax_regression) = plt.subplots(2, 1, figsize=(16, 12), sharey=True)

    snow200705 = data['SNOW200705']

    for i, thaw_column in enumerate([col for col in columns_of_interest if col.startswith('THAW')]):
        # Fit 1st, 2nd, and 3rd order polynomials
        poly1 = np.polyfit(snow200705, data[thaw_column], 1)
        poly2 = np.polyfit(snow200705, data[thaw_column], 2)
        poly3 = np.polyfit(snow200705, data[thaw_column], 3)

        regression_data.append((thaw_column, poly1, poly2, poly3))
        
        # Scatter plot
        ax_scatter.scatter(snow200705, data[thaw_column], label=f'{thaw_column}', color=colors[i])
        
        # Plot regression lines
        x = np.linspace(snow200705.min(), snow200705.max(), 100)
        ax_regression.plot(x, np.polyval(poly1, x), label=f'...{thaw_column[-2:]} (1st order)', linestyle='-', color=colors[i], linewidth=2)
        ax_regression.plot(x, np.polyval(poly2, x), label=f'...{thaw_column[-2:]} (2nd order)', linestyle='--', color=colors[i], linewidth=2)
        # ax_regression.plot(x, np.polyval(poly3, x), label=f'...{thaw_column[-2:]} (3rd order)', linestyle=':', color=colors[i], linewidth=2)

    # Customize scatter plot axis
    ax_scatter.set_title('Scatter Plot of SNOW200705 vs THAW* Columns', fontsize='xx-large')
    ax_scatter.set_xlabel('SNOW200705')
    ax_scatter.set_ylabel('THAW Values')
    ax_scatter.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-large')
    ax_scatter.grid(alpha=0.5)

    # Customize regression plot axis
    ax_regression.set_title('Regression Lines for SNOW200705 vs THAW* Columns', fontsize='xx-large')
    ax_regression.set_xlabel('SNOW200705')
    ax_regression.set_ylabel('THAW Values')
    ax_regression.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-large')
    ax_regression.grid(alpha=0.5)

    plt.tight_layout()
    plt.show()

    return regression_data