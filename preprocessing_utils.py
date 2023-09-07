import pandas as pd
import plotly.graph_objs as go

def MVAnalyzer(df, target, height=1000, margin=dict(l=20, r=20, t=50, b=20)):
    """
    Generate a heatmap to visualize missing values in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    target (str): The name of the target column.
    height (int): The height of the plot in pixels. Default is 1000.
    margin (dict): The margin of the plot. Default is {'l': 20, 'r': 20, 't': 50, 'b': 20}.

    Returns:
    None
    """
    # Generate a matrix to represent missing values
    missing_matrix = df.isnull().astype(int)
    
    # Calculate total rows
    total_rows = len(df)
    
    # Calculate total missing values for each feature
    total_missing_per_feature = df.isnull().sum()

    # Create a DataFrame to hold the counts of missing values per class, per feature
    classes = df[target].dropna().unique()
    missing_counts_per_class = {cls: (df[df[target] == cls].isnull().sum()) for cls in classes}

    # Create the hover text
    hover_text = []
    for i in range(missing_matrix.shape[0]):
        hover_text.append([])
        for j in range(missing_matrix.shape[1]):
            feature = missing_matrix.columns[j]
            total_missing = total_missing_per_feature[j]
            
            # Calculate the percentage of total missing values for the feature
            total_missing_percent = (total_missing / total_rows) * 100
            
            text = f"Row: {i+1}, Feature: {feature}<br>Total Missing: {total_missing} ({total_missing_percent:.2f}%)<br>Missing counts per {target}:<br>"
            for cls in classes:
                missing_for_class = missing_counts_per_class[cls][j]
                if total_missing != 0:
                    percent_missing = (missing_for_class / total_missing) * 100
                else:
                    percent_missing = 0
                text += f"{target} {cls}: {missing_for_class} ({percent_missing:.2f}%)<br>"
            hover_text[-1].append(text)

    # Create the Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Heatmap(z=missing_matrix.values,
                             x=missing_matrix.columns,
                             y=missing_matrix.index,
                             hoverinfo="text",
                             hovertext=hover_text,
                             colorscale="Viridis",
                             showscale=False))

    # Add title and labels
    fig.update_layout(title="Missing Value Heatmap",
                      xaxis_title="Columns",
                      yaxis_title="Rows",
                      height=height,
                      margin=margin)

    # Show the figure
    fig.show()

"""" Adding test"""
