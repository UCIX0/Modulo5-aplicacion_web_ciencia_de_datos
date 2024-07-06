import streamlit as st
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

### FUNCTION DEFINITIONS ###

@st.cache_data(ttl=3600)
def load_employees(nrows=500):
    """
    Load employee data from a CSV file.
    This function loads employee data from a specified CSV file and caches the
    result to improve performance. The caching time-to-live (TTL) is set to 3600 seconds (1 hour).
    Parameters:
    -----------
    nrows : int, optional
        The number of rows of the CSV file to read. The default is 500.
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the employee data.
    """
    return pd.read_csv("https://drive.usercontent.google.com/download?id=1WvRu_ZpxBARd0T9k8W9NnUgWHYCAyMzF&export=download", nrows=nrows)


def animate_plot(df, x_col, y_col, bar_color):
    """
    Animate a bar chart in Streamlit.
    This function animates the growth of bars in a bar chart using Streamlit.
    It gradually increases the values in the specified y-axis column and updates
    the chart in real-time.

    Parameters:
    df : pandas.DataFrame
        The DataFrame containing the data to be plotted and animated.
    x_col : str
        The column name in df to be used for the x-axis data.
    y_col : str
        The column name in df to be used for the y-axis data.
    bar_color : list or str
        The color(s) to be used for the bars in the chart. This can be a list of colors or a single color.

    Example:
    --------
    >>> df = pd.DataFrame({
            'Ages': ['18-19', '20-29', '30-39', '40-49', '50-59', '60-69'],
            'Counts': [5, 10, 15, 20, 25, 30]
        })
    >>> animate_plot(df, 'Ages', 'Counts', ["#54BAB9"])
    """
    container = st.empty()
    df_temp = df.copy()
    df_temp[y_col] = 0
    if df[y_col].max() > 8:
        step = 1
        step_time = 0.3
    else:
        step = 0.1
        step_time = 0.1
    for nrow in range(df.shape[0]):
        for current_value in np.arange(0, df[y_col].iloc[nrow], step):
            df_temp.at[nrow, y_col] = current_value + step
            container.bar_chart(data=df_temp, x=x_col, y=y_col, color=bar_color)
            time.sleep(step_time/df[y_col].max())


def ages_hist(df, animetcheck):
    """
    Prepare and plot a histogram of ages from the given DataFrame.
    This function prepares a histogram of ages by binning the ages into specified intervals,
    and then either animates the histogram or displays it statically based on the `animate` flag.
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to be plotted.
    animetcheck : bool
        If True, the histogram is animated. If False, a static histogram is displayed.
    """
    df_cleaned = df.dropna(subset=['Age'])
    bin_edges = [18, 20, 30, 40, 50, 60, 70]
    hist_values, _ = np.histogram(df_cleaned['Age'], bins=bin_edges)
    age_ranges = ["18-19", "20-29", "30-39", "40-49", "50-59", "60-69"]
    hist_df = pd.DataFrame({
        'Ages': age_ranges,
        'Counts': hist_values
    })
    if animetcheck:
        animate_plot(hist_df, 'Ages', 'Counts', ["#957DAD"])
    else:
        st.bar_chart(data=hist_df, x='Ages', y='Counts', color=["#957DAD"])


def unit_bar(df, animetcheck):
    """
    Prepare and plot a bar chart of unit frequencies from the given DataFrame.
    This function prepares a bar chart of unit frequencies by counting the occurrences
    of each unit, and then either animates the chart or displays it statically based on the `animate` flag.
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to be plotted.
    animate : bool
        If True, the bar chart is animated. If False, a static bar chart is displayed.
    """
    unit_counts = df['Unit'].value_counts().reset_index()
    unit_counts.columns = ['Unit', 'Counts']
    unit_counts = unit_counts.sort_values('Unit').reset_index(drop=True)
    if animetcheck:
        animate_plot(unit_counts, 'Unit', 'Counts', ["#54BAB9"])
    else:
        st.bar_chart(data=unit_counts, x='Unit', y='Counts', color=["#54BAB9"])


@st.cache_data(ttl=3600)
def mean_line_scatter(df, x_column, _custom_cmap):
    """
    Create a scatter plot with a mean line of attrition rate, colored by count.
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted.
    x_column : str
        Column name to be used for the x-axis.
    _custom_cmap : LinearSegmentedColormap
        Custom colormap for the scatter plot.
    Returns:
    --------
    None
        The function displays the plot using Streamlit.
    """
    # Group the data by x_column and calculate mean attrition rate and count
    grouped_df = df.groupby(x_column).agg(
        Mean_attrition_rate=('Attrition_rate', 'mean'),
        Count=(x_column, 'count')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the line
    sns.lineplot(x=grouped_df[x_column], y=grouped_df['Mean_attrition_rate'], color="#D79BE5", ax=ax)
    # Plot the scatter points
    sns.scatterplot(
        x=grouped_df[x_column],
        y=grouped_df['Mean_attrition_rate'],
        size=grouped_df['Count'],
        sizes=(10, 200),
        hue=grouped_df['Count'],
        palette=_custom_cmap,
        legend=True,
        alpha=1.0,
        ax=ax
    )
    # Ensure scatter points are on top of the line
    plt.setp(ax.collections, zorder=10)
    # Set the x-ticks
    min_x = int(grouped_df[x_column].min())
    max_x = int(grouped_df[x_column].max())
    ax.set_xticks(np.arange(min_x, max_x + 1, 2))

    st.pyplot(fig)

@st.cache_data(ttl=3600)
def simple_lineplot(df):
    """
    Create a simple line plot of mean attrition rate by hometown.
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted with columns 'Hometown' and 'Attrition_rate'.
    Returns:
    --------
    None
        The function displays the plot using Streamlit.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    df_hometown = df.groupby('Hometown').mean().reset_index()
    sns.lineplot(data= df_hometown, x='Hometown', y='Attrition_rate', marker='X', markersize=15, color="#7469b6")
    st.pyplot(fig)



### MAIN CODE ##


# Set up the logo
st.logo("./data/logo.jpeg", icon_image="./data/logo.jpeg")
# Set the page title and icon
st.set_page_config(page_title="Employee analysis", page_icon="📈")
# Set the main title of the app
st.title("Employee analysis")
# Set the header for the data frame section
st.header("Data Frame")


# Load the employee data
df_employees = load_employees()
# Create a copy of the DataFrame for filtering
df_employees_filtred = df_employees


# Sidebar text input to filter by Employee ID
text_input_id = st.sidebar.text_input("Employee ID")
if text_input_id:
	df_employees_filtred = df_employees_filtred[df_employees_filtred['Employee_ID'].str.contains(text_input_id, case=False)]


# Sidebar selectbox to filter by Hometown
selectbox_hometown = st.sidebar.selectbox("Hometown", ["All"] + sorted(df_employees["Hometown"].unique()))
if selectbox_hometown != "All":
	df_employees_filtred = df_employees_filtred[df_employees_filtred['Hometown'] == selectbox_hometown]


# Sidebar selectbox to filter by Unit
selectbox_unit = st.sidebar.selectbox("Unit", ["All"] + sorted(df_employees["Unit"].unique()))
if selectbox_unit != "All":
	df_employees_filtred = df_employees_filtred[df_employees_filtred['Unit'] == selectbox_unit]


# Display the filtered DataFrame
st.dataframe(df_employees_filtred, hide_index=True)


# Sidebar checkbox to display raw data
raw_chekbox = st.sidebar.checkbox("Raw Data")
raw_df = load_employees(None)
if raw_chekbox:
    st.header("Raw Data Frame")
    st.dataframe(raw_df, hide_index=True)

#animete plot button
animetbutton = st.sidebar.button("Animate plots", type="primary")


# Header and call to histstrim to display histogram by age
st.header("Histogram by age")
ages_hist(df_employees_filtred, animetbutton)


# Header and call to unitbar to display unit frequency chart
st.header("Unit Frequency Chart")
unit_bar(df_employees_filtred, animetbutton)


# Set the style and parameters for the seaborn
sns.set_style("darkgrid")
plt.rcParams.update({
    'axes.facecolor': '#ffe6e6',  # Background color of the plot
    'figure.facecolor': '#ffe6e6',  # Background color of the figure
    'text.color': '#393356',  # Color of the text
    'axes.labelcolor': '#393356',  # Color of the axis labels
    'xtick.color': '#393356',  # Color of the x-axis ticks
    'ytick.color': '#393356',  # Color of the y-axis ticks
    'font.family': 'monospace',  # Monospace font
    'grid.color': '#e1afd1',  # Color of the grid lines
    'axes.edgecolor': '#393356'  # Color of the axis edges
})

#lineplot Attrition_rate vs Hometown
st.header("City Attrition Rate")
simple_lineplot(raw_df[["Hometown", "Attrition_rate"]])

#Create custom_cmap
colors_list = ["#D72CFF","#C658E0", "#4F2F95","#000000"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors_list)

#Create scatter plot Attrition_rate vs Time_of_service
st.header("Attrition rate vs Time service")
mean_line_scatter(raw_df, 'Time_of_service', custom_cmap)

#Create scatter plot Age vs Time_of_service
st.header("Age vs Time service")
mean_line_scatter(raw_df, 'Age', custom_cmap)


# Easteregg
balloonsbutton= st.button(":balloon:")
if balloonsbutton:
    st.balloons()