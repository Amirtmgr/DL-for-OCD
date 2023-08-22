import pandas as pd
import os
from logger import logger
import plotly.express as px
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as markers
import seaborn as sns

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
    
