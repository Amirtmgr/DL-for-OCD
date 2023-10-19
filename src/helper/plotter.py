import pandas as pd
import os
import numpy as np
import itertools
from src.helper.logger import Logger
import plotly.express as px
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.markers as markers
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from src.utils.config_loader import config_loader as cl
import datetime as dt
import plotly.graph_objects as go
import plotly.io as pio
import uuid

def plot_3d(df, axes=['acc x','acc y', 'acc z'], opacity=0.75):
    # 3D scatter plot using Plotly
    fig = px.scatter_3d(df, x=axes[0], y=axes[1], z=axes[2], color='relabeled', opacity=0.75)
    fig.update_layout(title='Sensor data 3D-Scatter Plot')
    fig.show()


def bar_unique(df, title="Distribution Plot", size=(8,6)):
    value_counts = df.value_counts()
    plt.figure(figsize=size)
    plt.bar(value_counts.index, value_counts.values)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()
    
def pdf(df,cols,title="PDF"):
    # plotting the scatterplot of before and after Min Max Scaling
    plt.figure(figsize=(7,5))
    plt.title("PDF Before Min Max Scaling", fontsize=18)
    for col in cols:
        sns.kdeplot(data = df[col], label=col)
    plt.legend()
    plt.grid(True)
    plt.show()


def arrays(data_lists, title="", x_label="", y_label="", legend_labels=None, save_fig=True, grid=True):
    """
    Plot lists of lists of numbers using Matplotlib.

    Args:
        data_lists (list of lists): The data to plot, where each list represents a data series.
        title (str): The title of the plot (optional).
        x_label (str): The label for the x-axis (optional).
        y_label (str): The label for the y-axis (optional).
        legend_labels (list): List of legend labels for each data series (optional).
        save_path (bool): Save option (default is True).
        grid (bool): Toggle grid on or off (default is True).

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

    for i, data in enumerate(data_lists):
        label = legend_labels[i] if legend_labels and i < len(legend_labels) else f"List {i + 1}"
        plt.plot(data, label=label)

    # Set plot title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Add legend if legend_labels are provided
    if legend_labels:
        plt.legend()

    # Toggle grid on or off
    plt.grid(grid)
    plt.tight_layout()
    
    # Save the plot as a file
    if save_fig:
        save_plot(plt,title)
        plt.close()
    else:
        # Show plot
        plt.show()
    

def plot_cm(cm, classes, save_fig=True, title='Confusion Matrix', cmap=plt.cm.Blues):
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels= classes)
    
    disp.plot(cmap=cmap)

    plt.title(title)
   

   # Save the plot as a file
    if save_fig:
        save_plot(plt,title)
        plt.close()
    else:
        plt.show()


def confusion_matrix(cm, classes, title = "Confusion Matrix", save_fig=True):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_fig:
        save_plot(plt,title)
    else:
        plt.show()


def plot_multiple_pies(dataframes, names, num_cols=1):
    """
    Plots multiple histograms in rows within one figure.

    Args:
        dataframes (list): List of DataFrames, each containing label data in a single column.
        names (list): List of names corresponding to each DataFrame for labeling the subplots.
    """
    num_dataframes = len(dataframes)
    
    labels = ['others', 'rHW', 'cHW']
    
    # Determine the number of rows and columns for subplots
    num_rows = num_dataframes//num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.subplots_adjust(hspace=0.5)
    
    # Colors for the pie chart
    colors = ["#47B39C", "#FFC154", "#EC6B56"]

    for i, df in enumerate(dataframes):
        other_totals = len(df[df["relabeled"]==0])
        rHW_totals = len(df[df["relabeled"]==1])
        cHW_totals = len(df[df["relabeled"]==2])
        category_counts = [other_totals, rHW_totals, cHW_totals]
        overall_percentages = [(count / len(df)) * 100 for count in category_counts]

        row = i//num_cols
        col = i%num_cols
        if num_rows==1 or num_cols ==1:
            ax = axes[i]
        else:
            ax = axes[row,col]
        
        #ax.pie(category_counts, labels=categories, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'lightpink'])
        #ax.axis('equal')
        wedges, _ ,_ = ax.pie(overall_percentages, colors=colors, startangle=90, autopct="",wedgeprops={'edgecolor': 'black'})

        # Create legend with values
        legend_labels = [f"{label}: {percentage:.1f}%" for label, percentage in zip(labels, overall_percentages)]
        ax.legend(wedges, legend_labels, loc="center", bbox_to_anchor=(0.5, -0.1))

        # Remove text from pie chart
        #ax.setp(wedges, width=0.4) 

        #ax.hist(category_counts, bins=categories, rwidth=0.8, alpha=0.7, color='skyblue')
        #ax.set_xlabel('Labels')
        #ax.set_ylabel('Frequency')
        ax.set_title(names[i])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        
    # Remove empty subplots if any
    for i in range(num_dataframes, num_rows * num_cols):
        fig.delaxes(axes[i // num_cols, i % num_cols])
    
    # Show the legend
    plt.axis('off')
    plt.show()


def plot_multiple_histograms(dataframes, names, num_cols=1):
    """
    Plots multiple histograms in rows within one figure.

    Args:
        dataframes (list): List of DataFrames, each containing label data in a single column.
        names (list): List of names corresponding to each DataFrame for labeling the subplots.
    """
    num_dataframes = len(dataframes)
    
    # Determine the number of rows and columns for subplots
    num_rows = num_dataframes//num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.subplots_adjust(hspace=0.5)

    for i, df in enumerate(dataframes):
        row = i//num_cols
        col = i%num_cols
        if col>2:
            ax = axes[row,col]
        else:
            ax = axes[row]
        ax.hist(df.iloc[:, -2], bins=len(df[df.columns[-2]].unique()), rwidth=0.8, alpha=0.7, color='skyblue')
        ax.set_xlabel('Labels')
        ax.set_ylabel('Frequency')
        ax.set_title(names[i])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    # Remove empty subplots if any
    for i in range(num_dataframes, num_rows * num_cols):
        fig.delaxes(axes[i // num_cols, i % num_cols])

    plt.show()


def save_plot(plt, title):
    """
    Saves a plot to a file.

    Args:
        plt (matplotlib.pyplot): The plot to save.
        file_name (str): The name of the file to save the plot to.
    """

    path = cl.config.charts_path
    file_name = title.replace(" ", "-").lower() + "_" + dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + str(uuid.uuid4().hex) + ".png"

    if not os.path.exists(path):
        os.makedirs(path)

    try:
        plt.savefig(path + "/" + file_name)
        plt.close()
        Logger.info(f"Plot saved as {file_name} inside path {path}")
    except Exception as e:
        Logger.error(f"{e} while saving plot as {file_name}")

     

def plot_sensor_data(input_data, ground_truth, predictions, sampling_rate=50, save = False, title=None):
    """
    Plot sensor data with expanded predictions and ground truth using Plotly.

    Parameters:
        input_data (numpy.ndarray): Input data of shape (10, 250, 6).
        predictions (numpy.ndarray): Predicted values for each window (shape: (10,)).
        ground_truth (numpy.ndarray): Ground truth values for each window (shape: (10,)).
        sampling_rate (int): Sampling rate in Hz (default is 50).
        window_size (int): Size of each window (default is 250).
        save_path (str): File path for saving the figure (e.g., 'plot.png').

    Returns:
        None
    """
    num_windows = input_data.shape[0]
    window_size = input_data.shape[1]
    num_channels = input_data.shape[2]

    # Calculate the time axis based on the sampling rate and data length
    time_axis = np.arange(1, (window_size*num_windows)+1) / sampling_rate
    
    # Expand predictions and ground truth to match the window size
    expanded_predictions = np.repeat(predictions, window_size)
    expanded_ground_truth = np.repeat(ground_truth, window_size)

    # Flatten all channels from the input data
    flattened_data = input_data.reshape(-1, num_channels)

    print(f"Time axis shape: {time_axis.shape}")
    print(f"Expanded predictions shape: {expanded_predictions.shape}")
    print(f"Expanded ground truth shape: {expanded_ground_truth.shape}")
    print(f"Flattened data shape: {flattened_data.shape}")

    # Create a DataFrame to make it easier to work with the data
    df = pd.DataFrame({'Time': time_axis, 'Predictions': expanded_predictions, 'Ground Truth': expanded_ground_truth})

    # Create a figure using Plotly
    fig = go.Figure()

    lighter_colors = [
    '#D3D3D3',  # Light Gray
    '#87CEFA',  # Light Sky Blue
    '#7FFFD4',  # Aquamarine
    '#DDA0DD',  # Plum
    '#ADFF2F',  # Green Yellow
    '#F08080',  # Light Coral
    '#FFA07A',  # Light Salmon
    '#CD853F',  # Peru
    '#B0E0E6',  # Powder Blue
    '#E6E6FA',  # Lavender
    ]

    light_colors = ['#E6E6E6', '#FFD700', '#98FB98', '#ADD8E6' , '#F5DEB3', '#FFA07A', '#F0E68C', '#FFC0CB', '#D3D3D3', '#00CED1']
    lighter_colors_2 = [
    '#EDEDED',  # Light Gray
    '#B0E2FF',  # Light Sky Blue
    '#AFEEEE',  # Pale Turquoise
    '#E6C9E6',  # Pale Plum
    '#DFFFBF',  # Light Green Yellow
    '#FFCCCC',  # Light Light Coral
    '#FFDAB9',  # Light Light Salmon
    '#DAA520',  # Light Goldenrod
    '#C0E0F2',  # Light Powder Blue
    '#F0F0FF',  # Alice Blue
    ]

    light_shades = [
    '#F0E8E8',  # Light Grayish Pink
    '#B4B4B4',  # Light Gray
    '#A0A0A0',  # Light Silver
    '#E6E0D8',  # Light Beige
    '#C4B0A8',  # Light Mauve
    '#F2EFEF',  # Light Platinum
    '#D8D0C0',  # Light Tan
    '#D8D8E6',  # Light Lavender Gray
    '#D2C9C9',  # Light Pinkish Gray
    '#E0E0D8',  # Light Grayish Blue
    ]

    channel_colors = light_colors
    channels = ['acc x', 'acc y', 'acc z', 'gyro x', 'gyro y', 'gyro z']
    pallete = ["ffa69e","faf3dd","b8f2e6","aed9e0","5e6472"]
    
    # Add traces for data points, predictions, and ground truth for each channel
    for i in range(num_channels):
        channel_name = channels[i]
        fig.add_trace(go.Scatter(x=df['Time'], y=flattened_data[:, i], mode='lines', name=channel_name,
                                 line=dict(color=channel_colors[i], width=1), showlegend=True))

    fig.add_trace(go.Scatter(x=df['Time'], y=df['Predictions'], mode='lines', name='Predictions',
                             line=dict(color='red', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Ground Truth'], mode='lines', name='Ground Truth',
                             line=dict(color='green', width=2, dash='dash')))

    # Adjust the y-axis labels to be 0 or 1
    fig.update_yaxes(tickvals=[0, 1])

    # Set the layout with better font style
    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Value',
        title='Sensor Data, Predictions, and Ground Truth',
        font=dict(family='Arial, sans-serif', size=14),
    )

    # Customize the grid settings
    fig.update_layout(
        xaxis=dict(
            showgrid=True,  # Show the x-axis grid lines
            gridwidth=1,  # Width of major grid lines
            gridcolor='white',  # Color of major grid lines
            dtick=1,  # Spacing of grid lines based on x-axis values
        ),
        yaxis=dict(
            showgrid=True,  # Show the y-axis grid lines
            gridwidth=1,  # Width of major grid lines
            gridcolor='white',  # Color of major grid lines
            dtick=1,  # Spacing of grid lines based on y-axis values
        )
    )

    # Show the interactive plot or save it as an image
    if save:
        title = "Personalization Ground Truth vs Predictions" if title is None else title
        save_path = cl.config.charts_path + "/" + title.split()[0] + "_personalization_" + dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + str(uuid.uuid4().hex) + ".png"  
        pio.write_image(fig, save_path, format='png', width=6400, height=2000)
        print(f"Figure saved as {save_path}")
    else:
        fig.show()
