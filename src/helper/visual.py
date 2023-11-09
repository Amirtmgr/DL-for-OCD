import plotly.graph_objects as go; import numpy as np
#from plotly_resampler import FigureResampler, FigureWidgetResampler
from src.helper import df_manager as dfm
from src.helper import data_preprocessing as dp
from src.helper import plotter as pl
import torch
from torch.utils.data import ConcatDataset, TensorDataset
from src.helper.logger import Logger
from src.utils.config_loader import config_loader as cl
from src.helper import data_preprocessing as dp
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def show():
    shelf_name = cl.config.dataset.name
    # Load python dataset
    X_dict, y_dict = dp.load_shelves(shelf_name)

    # All subjects
    subjects = list(X_dict.keys())
    X = np.concatenate([X_dict[sub_id] for sub_id in subjects])

    # Normalize    
    # scaler = StandardScaler()
    # X_norm = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    X_norm = X
    
    #scaler = MinMaxScaler()
    #X_norm = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    prev_idx = 0

    for sub_id in subjects:
        X_total = X_dict[sub_id].shape[0]
        end = prev_idx + X_total
        X = X_norm[prev_idx:end]
        prev_idx = end
        y = y_dict[sub_id]
        if sub_id != '15':
            continue
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float())
        train_loader = dp.load_dataloader(dataset)
        
        print("===="*20)
        print(f"Subject {sub_id} - Total samples: {len(dataset)}")
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Subject {sub_id} - Batch {batch_idx}")
            title = f"sub_{sub_id}_b_{batch_idx}"
            counts = np.bincount(target)
            majority = np.argmax(counts)
            
            if majority == 0:
                label = "Null" if cl.config.dataset.task_type > 2 else "rHW"
            elif majority == 1:
                label = "rHW" if cl.config.dataset.task_type > 2 else "cHW"
            elif majority == 2:
                label = "cHW"

            pl.plot_sensor_data(data.numpy(), target.numpy(), predictions=None, save=True, title=f"sub_{sub_id}_batch_{batch_idx}_{label}", sensor='acc', batch_idx=batch_idx)
            pl.plot_sensor_data(data.numpy(), target.numpy(), predictions=None, save=True, title=f"sub_{sub_id}_batch_{batch_idx}_{label}", sensor='gyro',batch_idx=batch_idx)
        print(f"Subject {sub_id} - Total batches: {batch_idx + 1}")
        

def view(df, save = False, title=None, sensor='both'):
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

    # OPTION 1 - FigureWidgetResampler: dynamic aggregation via `FigureWidget.layout.on_change`
    fig = FigureWidgetResampler(go.Figure())
    # fig.add_trace(go.Scattergl(name='noisy sine', showlegend=True), hf_x=x, hf_y=noisy_sin)

    # fig
    
    channels = ['acc x', 'acc y', 'acc z', 'gyro x', 'gyro y', 'gyro z']
   

    # Add traces for data points, predictions, and ground truth for each channel
    for i in channel_ids:
        channel_name = channels[i]
        fig.add_trace(go.Scatter(x=df['datetime'], y=df[channel_name], mode='lines', name=channel_name,
                                 line=dict( width=1), showlegend=True))

    fig.add_trace(go.Scatter(x=df['Time'], y=df['relabeled'], mode='lines', name='Predictions',
                             line=dict(color='red', width=2.5, dash='dot')))
    # fig.add_trace(go.Scatter(x=df['Time'], y=df['Ground Truth'], mode='lines', name='Ground Truth',
    #                          line=dict(color='blue',width=2.5, dash='dash')))

    # Adjust the y-axis labels to be 0 or 1
    #fig.update_yaxes(tickvals=[0, 1])

    # Set the layout with better font style
    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Value',
        title='Sensor Data and Ground Truth',
        font=dict(family='Arial, sans-serif', size=24, color='black'),
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    # Customize the grid settings
    fig.update_layout(
        xaxis=dict(
            showgrid=True,  # Show the x-axis grid lines
            gridwidth=1,  # Width of major grid lines
            gridcolor='white',  # Color of major grid lines
            dtick=5,  # Spacing of grid lines based on x-axis values
        ),
        yaxis=dict(
            showgrid=True,  # Show the y-axis grid lines
            gridwidth=1,  # Width of major grid lines
            gridcolor='white',  # Color of major grid lines
            dtick=1,  # Spacing of grid lines based on y-axis values
        )
    )
    
    # Black ticks
    #fig.update_xaxes(gridcolor='black', griddash='dash', minor_griddash="dot")

    # Show the interactive plot or save it as an image
    if save:
        title = "Personalization Ground Truth vs Predictions" if title is None else title
        save_path = cl.config.charts_path + "/" + sensor + "_" + title.split()[0] + "_personalization_" + dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + str(uuid.uuid4().hex) + ".png"  
        pio.write_image(fig, save_path, format='png', width=1200, height=500)
        print(f"Figure saved as {save_path}")
    else:
        fig.show()
