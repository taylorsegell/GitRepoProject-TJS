# Collection of functions I have created to for ease and also many contributions from other data dorks such as myself.
# Credits can be found at the end of each function


# import libraries general libraries
from IPython.display import Markdown, display
from tabnanny import verbose
import warnings
import pandas as pd
import numpy as np

# Modules for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot, \
    plot  # iplot() = plots the figure(fig) that is created by data and layout
import plotly.express as px

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

# plt.rcParams['figure.figsize'] = [6, 6]

# ignore DeprecationWarning Error Messages

warnings.filterwarnings('ignore')

'''fontdict = dict(family='Courier New Bold', color='lightblue', size=18)
fontdict2 = dict(family='Courier New Bold', color='darkblue', size=15)

titledict = dict(family='Courier New Bold', color='lightblue', size=30)
titledict2 = dict(family='Courier New Bold', color='darkblue', size=25)
titledict3 = dict(family='Courier New Bold', color='darkblue', size=35)'''

# style: https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.background_gradient.html
# color: https://matplotlib.org/stable/tutorials/colors/colormaps.html
cell_hover = {  # for row hover use <tr> instead of <td>
    'selector': 'td:hover',
    'props': [('background-color', '#7d8a35')]
}
index_names = {
    'selector': '.index_name',
    'props': 'font-style: italic; color: lightgrey; font-weight:normal;'
}
headers = {
    'selector': 'th:not(.index_name)',
    'props': 'background-color: #7d8a35; color: white; font-size: 1.05em;'
}

stylez = [cell_hover, index_names, headers]


def tbl(df):
    return df.style.set_table_styles(stylez)


def clean_header(df):
    """
    This functions removes weird characters and spaces from column names, while keeping everything lower case
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(
        ' ', '_').str.replace('(', '').str.replace(')', '')


def infoOut(data, details=False):
    dfInfo = data.columns.to_frame(name='Column')
    dfInfo['Non-Null Count'] = data.notna().sum()
    dfInfo['Dtype'] = data.dtypes
    dfInfo.reset_index(drop=True, inplace=True)
    if details:
        rangeIndex = (dfInfo['Non-Null Count'].min(),
                      dfInfo['Non-Null Count'].min())
        totalColumns = dfInfo['Column'].count()
        dtypesCount = dfInfo['Dtype'].value_counts()
        totalMemory = dfInfo.memory_usage().sum()
        return dfInfo, rangeIndex, totalColumns, dtypesCount, totalMemory
    else:
        return dfInfo


def table(df):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='#7d8a35',
                    align='left'),
        cells=dict(values=df.transpose().values.tolist(),
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout()
    fig.show()


def printmd(string, color=None, size='100%', under=None):
    colorstr = "<span style='text-decoration:{};text-align:center;font-size:{};color:{}'>{}</span>".format(
        under, size, color, string)
    display(Markdown(colorstr))


def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'


def data_research(data, data_name='data', un=False):

    printmd(("**{}**".format(data_name)), '#42358a', '28px', 'underline')
    printmd("______________________________________________________",
            '#7d8a35', '20px')
    printmd("**{}**".format('Basic Info'), '#42358a', '22px')
    printmd(("**Shape** = {}".format(data.shape)), '#000000', '14px')

    printmd('**Columns** = {}'.format(str(list(data.columns)
                                          ).replace("'", '').replace('[', '').replace(']', '')), '#000000', '14px')

    x = data.memory_usage().sum()
    b, l = format_bytes(x)
    printmd('**Memory Usage** = {}, {}'.format(b.round(2), l), '#000000', '14px')

    display(infoOut(data).style.set_table_styles(stylez))

    printmd("______________________________________________________",
            '#7d8a35', '20px')
    printmd('**Head**', '#42358a', '22px')

    display(data.head().style.set_table_styles(stylez))
    printmd("______________________________________________________",
            '#7d8a35', '20px')
    print()
    printmd('**Describe**', '#42358a', '22px')
    display(data.describe().style.set_table_styles(stylez))
    # display(data.columns)
    printmd("______________________________________________________",
            '#7d8a35', '20px')
    print()
    printmd('**Duplicates**', '#42358a', '22px')
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        printmd('{}'.format('There are no duplicated entries.'), "#000000", '14px')
    else:
        printmd('**There are {} duplicates.**'.format(duplicates),
                "#000000", '14px')

    # missing
    printmd("______________________________________________________",
            '#7d8a35', '20px')
    printmd('**Missing**', '#42358a', '22px')
    null_data = pd.DataFrame({'Percent_Missing': (
        ((data.isnull().sum().sort_values(ascending=False))/len(data))*100).round(2)})
    display(null_data[null_data['Percent_Missing'] != 0]
            ['Percent_Missing'].apply(lambda x: str(x) + '%').to_frame().style.set_table_styles(stylez))
    printmd("______________________________________________________",
            '#7d8a35', '20px')
    printmd('**Unique**', '#42358a', '22px')
    # Checking number of unique rows in each feature

    features = data.columns
    nu = data[features].nunique().sort_values()
    nf = []
    cf = []
    nnf = 0
    ncf = 0  # numerical & categorical features

    for i in range(data[features].shape[1]):
        if nu.values[i] <= 7:
            cf.append(nu.index[i])
        else:
            nf.append(nu.index[i])

    printmd('**Observation:** The Datset has {} numerical & {} categorical features.'.format(
        len(nf), len(cf)), "#000000", '14px')
    printmd("______________________________________________________",
            '#7d8a35', '20px')
    printmd('**Quantiles**', '#42358a', '22px')

    display(data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]
                          ).T.style.set_table_styles(stylez))
    printmd("______________________________________________________",
            '#7d8a35', '20px')
    print()


def get_redundant_pairs(data):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = data.columns
    for i in range(0, data.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(data, n):
    au_corr = data.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(data)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(
        ascending=False)
    return au_corr[0:n]


def heatmaps_tri(df, temp):
    corr = round(df.corr(), 3)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    df_mask = corr.mask(mask)

    fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(),
                                      x=df_mask.columns.tolist(),
                                      y=df_mask.columns.tolist(),
                                      colorscale=px.colors.diverging.Portland,
                                      showscale=True, ygap=1, xgap=1
                                      )

    fig.update_xaxes(side="bottom")

    fig.update_layout(
        title_text='Correlations',
        title_x=0.5,
        width=1000,
        height=1000,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed',
        template=temp
    )

    # NaN values are not handled automatically and are displayed in the figure
    # So we need to get rid of the text manually
    for i in range(len(fig.layout.annotations)):
        if fig.layout.annotations[i].text == 'nan':
            fig.layout.annotations[i].text = ""

    fig.show()


def style(table):
    """
    quick styling
    """
    view = table.style.background_gradient(cmap='Pastel1')
    return view


def percentage(s):
    """
    Converts a series to round off - percentage string format.
    """
    x = s.apply(lambda x: round(x / s[:].sum() * 100, 2))
    x = x.apply(lambda x: str(x) + '%')
    return x


def query_this(df, col, look):
    """
    Easy == Query
    """
    query_to_return = df.query('{} == "{}"'.format(col, look))
    return query_to_return


def missing_bar(df) -> go.Figure:
    """Plots Missing Data for Whole Dataset."""
    title = 'Stack Overflow Developer Survey Results 2021 <b>Missing Data by Features</b>'

    # counts missing data
    missing_data = df.isna().sum()
    missing_data = missing_data.to_frame().reset_index().rename(
        columns={'index': 'data_cols', 0: 'counts'})
    missing_data = missing_data.sort_values(by='counts', ascending=False)
    missing_perc = np.round(
        (df.isna().sum().sum() / df.size) * 100, 2)

    # figure colors
    colors = ['#f2f0eb'] * len(missing_data)
    colors[:10] = ['blue']

    # create figure
    fig = go.Figure()
    for labels, values \
            in zip(missing_data.data_cols.to_list(), (missing_data.counts/len(df)*100)):
        fig.add_trace(go.Bar(
            y=[labels],
            x=[values],
            name=labels,
            orientation='h'))

    # tweak layout
    fig.update_traces(marker_colorscale=px.colors.diverging.Portland)
    fig.update_xaxes(title='Missing Amount (Percentage)')
    fig.update_yaxes(title='Features', tickmode='linear')
    fig.update_layout(font=fontdict,
                      title_font=titledict,
                      title_x=0.5,
                      height=800,
                      width=1000,
                      showlegend=False)
    fig.add_annotation(xref='paper', yref='paper',
                       x=0.5, y=1.10, text='Total Data Missing: '+str(missing_perc)+'%',
                       font=titledict3,
                       bordercolor="#c7c7c7",
                       borderwidth=2,
                       borderpad=4,
                       bgcolor="white",
                       opacity=0.9,
                       showarrow=False)

    # add_bubble(fig)

    fig.show()
    # return paste_px_format(
    #    fig, title=title, height=1000, showlegend=False)


def missing_percentage(df):
    for col in df.columns:
        missing_percentage = df[col].isnull().mean()
        print(f'{col} - {missing_percentage :.1%}')


# Describing data
def group_median_aggregation(df, group_var, agg_var):
    # Grouping the data and taking median
    grouped_df = df.groupby([group_var])[
        agg_var].median().sort_values(ascending=False)
    return grouped_df


# Create the scatter plot
def scatter_plot(df):
    sns.lmplot(x="YearsCode", y="CompTotal", data=df)
    # Remove excess chart lines and ticks for a nicer looking plot
    sns.despine()


def split_multicolumn(col_series):
    result_df = col_series.to_frame()
    options = []
    # Iterate over the column
    for idx, value in col_series[col_series.notnull()].iteritems():
        # Break each value into list of options
        for option in value.split(';'):
            # Add the option as a column to result
            if not option in result_df.columns:
                options.append(option)
                result_df[option] = False
            # Mark the value in the option column as True
            result_df.at[idx, option] = True
    return result_df[options]


# function that imputes median
def impute_median(series):
    return series.fillna(series.median())


def calculate_min_max_whisker(df):
    """
    Calculates the values of the 25th and 75th percentiles
    It takes the difference between the two to get the interquartile range (IQR).
    Get the length of the whiskers by multiplying the IQR by 1.5
    Calculate the min and max whisker value by subtracting
    Add the whisker length from the 25th and 75th percentile values.
    """
    q_df = df.quantile([.25, .75])
    q_df.loc['iqr'] = q_df.loc[0.75] - q_df.loc[0.25]
    q_df.loc['whisker_length'] = 1.5 * q_df.loc['iqr']
    q_df.loc['max_whisker'] = q_df.loc['whisker_length'] + q_df.loc[0.75]
    q_df.loc['min_whisker'] = q_df.loc[0.25] - q_df.loc['whisker_length']
    return q_df


# credit: https://www.kaggle.com/jaepin
# Visualization to check % of missing values in each column
def paste_px_format(figure, **kwargs):
    """Updates Layout of the Figure with custom setting"""
    return figure.update_layout(**kwargs,
                                font={'color': 'Gray', 'size': 10},
                                width=780, margin={'pad': 10})


def add_bubble(fig, **kwargs):
    """Creates shape ontop of the figure"""
    return fig.add_shape(
        type="circle",
        line_color="white",
        fillcolor="lightblue",
        opacity=0.6,
        xref='paper', yref='paper',
        x0=0.5, y0=0.5)


def whitespace_remover(df):
    """
    The function will remove extra leading and trailing whitespace from the data.
    Takes the data frame as a parameter and checks the data type of each column.
    If the column's datatype is 'Object.', apply strip function; else, it does nothing.
    Use the whitespace_remover() process on the data frame, which successfully removes the extra whitespace from the columns.
    https://www.geeksforgeeks.org/pandas-strip-whitespace-from-entire-dataframe/
    """
    # iterating over the columns
    for i in df.columns:

        # checking datatype of each columns
        if df[i].dtype == 'str':

            # applying strip function on column
            df[i] = df[i].map(str.strip)
        else:
            # if condition is False then it will do nothing.
            pass


def confusion_matrix(data, actual_values, model):
    """
        Confusion matrix
        Parameters
        ----------
        data: data frame or array
        data is a data frame formatted in the same way as your input data (without the actual values)
        e.g. const, var1, var2, etc. Order is essential!
        actual_values: data frame or array
        These are the actual values from the test_data
        In the case of logistic regression, it should be a single column with 0s and 1s
        model: a LogitResults object
        this is the variable where you have the fitted model
        , e.g., results_log in this course
        ----------
        Predict the values using the Logit model
    """
    pred_values = model.predict(data)
    # Specify the bins
    bins = np.array([0, 0.5, 1])
    # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
    # if they are between 0.5 and 1, they will be considered 1
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    # Calculate the accuracy
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    # Return the confusion matrix and the accuracy
    return cm, accuracy
