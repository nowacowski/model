import snowflake.connector as sf
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")


########################################################################
######################### functions ####################################

# import data from SF
@st.cache_data
def import_data(schema, project_name1, project_name2, user, password, account):

    conn = sf.connect(
        user = user,
        password = password,
        account = account,
        role = st.secrets["role"],
        warehouse = st.secrets["warehouse"],
    )

    query = st.secrets["query"] % (schema, project_name1, schema, project_name2)

    df0 = pd.read_sql(query, conn)

    conn.close()

    return df0

# get last 'full' month
@st.cache_data
def last_full_month():
    current_date = pd.to_datetime('today').date()
    last_full_month = current_date.replace(day=1) - pd.DateOffset(months=1)
    while (current_date - pd.to_datetime(last_full_month + pd.offsets.MonthEnd()).date()).days < 30:
        last_full_month -= pd.DateOffset(months=1)
    return last_full_month

# DF transforms
@st.cache_data
def df_category(df, plat, us_ww, traffic=None):
    if traffic is not None:
        df_cat = df[(df['PLATFORM']==plat) & (df['US_WW']==us_ww) & (df['TRAFFIC']==traffic)].groupby(['MONTH_DATE','SPEND','ACCOUNTS_CREATED','DAY_OF_GAME_UTC'], as_index=False).sum()
    else:
        df_cat = df[(df['PLATFORM']==plat) & (df['US_WW']==us_ww)].groupby(['MONTH_DATE','SPEND','ACCOUNTS_CREATED','DAY_OF_GAME_UTC'], as_index=False).sum()
    return df_cat

@st.cache_data
def df_category_pivot(df_cat):
    df_cat_pivot = df_cat.pivot(index=['MONTH_DATE','SPEND','ACCOUNTS_CREATED'],columns='DAY_OF_GAME_UTC',values='REVENUE_NET').sort_index(ascending=False)
    return df_cat_pivot

# DoG to MoG
@st.cache_data
def dog_to_mog(group):
    chunks = []
    dogs = group['DAY_OF_GAME_UTC'].values
    first_chunk_sum = group.loc[dogs <= 30, 'REVENUE_NET'].sum()
    chunks.append(first_chunk_sum)
    for i in range(30, dogs.max(), 30):
        chunk_sum = group.loc[(dogs > i) & (dogs <= i+30), 'REVENUE_NET'].sum()
        chunks.append(chunk_sum)
    return pd.Series(chunks, index=[i for i in range(len(chunks))])

@st.cache_data
def df_mog_0(df_d_0):
    df_mog_0 = df_d_0.groupby(['MONTH_DATE', 'SPEND','ACCOUNTS_CREATED']).apply(dog_to_mog)
    df_mog_0 = df_mog_0.reset_index()

    df_mog_0['MONTH_DATE'] = pd.to_datetime(df_mog_0['MONTH_DATE'])
    # remove not comleted MoG
    current_date = pd.to_datetime('today')
    last_day_of_month = df_mog_0['MONTH_DATE'] + pd.offsets.MonthEnd(1)
    date_condition = last_day_of_month + pd.to_timedelta((df_mog_0['level_3'] + 1) * 30, unit='D')
    filtered_df = df_mog_0[date_condition < current_date]
    return filtered_df

@st.cache_data
def df_mog_pivot(df_mog_0):
    df_mog = df_mog_0.pivot(index=['MONTH_DATE','SPEND','ACCOUNTS_CREATED'],columns='level_3',values=0).sort_index(ascending=False)
    df_mog = df_mog.reset_index()
    return df_mog


# % from M0
@st.cache_data
def ratio_with_0(df0):
    df = df0.copy()
    columns = list(df.columns)[3:]
    for col in columns:
        df[col] = df0[col] / df0[0]
    return df


# fitting
# exp function
def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

# fit
def make_df_months_no(df, months_no):
    df_months_no = df[['MONTH_DATE']].sort_values(by='MONTH_DATE', ascending=False)
    df_months_no['INCLUDE'] = False
    df_months_no.iloc[1:months_no, 1] = True
    return df_months_no

def fitting2(df, df_months_no):
    merged_df = pd.merge(df_months_no, df, on='MONTH_DATE')
    merged_df_2 = merged_df[merged_df['INCLUDE']==True]
    df_avg = merged_df_2.iloc[:, 5:].mean(axis=0).dropna()
    params, covariance = curve_fit(exponential_decay, df_avg.index, df_avg)
    a_fit, b_fit = params
    return a_fit, b_fit, df_avg

# fit plot
def fit_plot(df_avg, a_fit, b_fit):
    fig, ax = plt.subplots()
    ax.scatter(df_avg.index, df_avg, label='Data')
    ax.plot(df_avg.index, exponential_decay(df_avg.index, a_fit, b_fit), 'r-', label='Fitted curve')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('Exponential Decay Fit')
    # plt.show()
    st.pyplot(fig)

# # fit plot
# def fit_plot(df_avg, a_fit, b_fit):
#     fig = px.scatter(df_avg)
#     line_trace = px.line(
#             x=df_avg.index, 
#             y=exponential_decay(df_avg.index, a_fit, b_fit)
#             ).data[0]
#     line_trace.update(line=dict(color='red'))
#     fig.update_traces(legendgroup='Avg')
#     line_trace.update(legendgroup='Fit')
#     fig.add_trace(line_trace)
#     fig.update_layout(
#         title = 'Exponential Decay Fit',
#         yaxis_title = '% Rev M0',
#         xaxis_title = 'MoG',
#         height = 800
#     )

#     st.plotly_chart(fig, use_container_width=True)


# new % rev avg+fit
def new_avg_fit(a_fit, b_fit, df_avg):
    x_fit = np.arange(df_avg.index.max()+1,37)
    fitted = exponential_decay(x_fit, a_fit, b_fit)
    fitted_series = pd.Series(data = fitted, index = x_fit)
    new_avg_fit_df = pd.concat([df_avg, fitted_series])
    return new_avg_fit_df


def series_to_df_transp(series, col_name, val_name):
    series_df = series.reset_index()
    series_df.columns = [col_name, val_name]
    series_df_transp = series_df.T
    series_df_transp.columns = series_df_transp.iloc[0]
    series_df_transp = series_df_transp[1:]
    return series_df_transp

# adjust param array
def adjust_params():
    column_names = [float(i) for i in range(1, 37)]
    data = {col: 1.0 for col in column_names}
    data['index'] = ['% of M0']
    df = pd.DataFrame(data)
    df.set_index('index', inplace=True)
    df.index.name = None
    return df


# plotting fitted and historical data
def plot_data_fitted(df_mog_pct,new_mog_pct_avg_fit):
    fig, ax = plt.subplots(figsize=(20, 12))
    for index, row in df_mog_pct.iterrows():
        name = row['MONTH_DATE']  # Extract name for legend
        x_values = df_mog_pct.columns[3:]  # Use column names from 2nd column onwards as x-values
        y_values = row[3:]  # Use values from 2nd column onwards as y-values
        plt.plot(x_values, y_values, label=name)
    ax.plot(new_mog_pct_avg_fit, linewidth=4, label='Avg+fit')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)




# final results
# multi 3y
def multi_3y(new_mog_pct_avg_fit):
    multiplier_y3 = new_mog_pct_avg_fit.sum()+1
    return multiplier_y3

# multiplier
def multiplier(df_d0,months_no, df_mog, multiplier_y3):
    zz = df_d0.copy()
    zz['MONTH_DATE'] = pd.to_datetime(zz['MONTH_DATE'])

    max_date = zz['MONTH_DATE'].max()
    min_date = max_date - pd.DateOffset(months=months_no)

    zz = zz[(zz['MONTH_DATE']>=min_date) & (zz['MONTH_DATE']<=max_date) & (zz['DAY_OF_GAME_UTC'] <= 7)]
    zz_d7 = zz.groupby(by=['MONTH_DATE'], as_index=False).sum('REVENUE_NET')[['MONTH_DATE','REVENUE_NET']]

    zz_m = df_mog.copy()
    zz_m['MONTH_DATE'] = pd.to_datetime(zz_m['MONTH_DATE'])
    zz_m = zz_m[(zz_m['MONTH_DATE']>=min_date) & (zz_m['MONTH_DATE']<=max_date)]
    zz_m0 = zz_m[['MONTH_DATE',0]]

    merged_df = pd.merge(zz_d7, zz_m0, on='MONTH_DATE')
    merged_df['Result'] = merged_df['REVENUE_NET'] / merged_df[0]
    multiplier_d7 = merged_df['Result'].mean()

    multiplier = multiplier_y3 / multiplier_d7

    return multiplier_d7, multiplier

# D(n) to D7 growth
def d7_growth(df_d, months_no):
    d7_growth_0 = df_d.reset_index()
    d7_growth_0['MONTH_DATE'] = pd.to_datetime(d7_growth_0['MONTH_DATE'])

    max_date = d7_growth_0['MONTH_DATE'].max()
    min_date = max_date - pd.DateOffset(months=months_no)
    d7_growth_0 = d7_growth_0[(d7_growth_0['MONTH_DATE']>=min_date) & (d7_growth_0['MONTH_DATE']<max_date)]
    cols = ['MONTH_DATE'] + list(d7_growth_0.columns)[3:11]
    d7_growth_0 = d7_growth_0[cols]
    
    d7_growth__cumm = d7_growth_0.copy()
    d7_growth__cumm.iloc[:, 1:] = d7_growth__cumm.iloc[:, 1:].cumsum(axis=1)
    
    d7_growth = d7_growth__cumm.copy()
    for i in range(0, len(d7_growth.columns)-1):
        d7_growth[i] = d7_growth[7] / d7_growth[i]
    
    d7_growth_mean = d7_growth.iloc[:,1:].mean()

    return d7_growth_mean

# ROAS goal and SKAD ROAS goal net
def roas_goal(ROAS_goal_Y3, multiplier, d7_growth_mean):
    ROAS_goal = ROAS_goal_Y3/(multiplier*d7_growth_mean)
    SKAD_ROAS_goal_net = ROAS_goal[1:3].mean()
    return ROAS_goal, SKAD_ROAS_goal_net




########################################################################
######################### app stuff ####################################


########################################################################
# sidebar
with st.sidebar:

    # user, password, account
    user = st.text_input(label='User')
    password = st.text_input(label='Password', type='password')
    account = st.text_input(label='Account')

    # project select box
    project = st.selectbox(
        'Project',
        ('TG','SC')
    )

########################################################################
# parameters for query
if project == 'SC':
    project_name1 = "where project_name = 'soccerclash-api'"
    project_name2 = "and project_name = 'soccerclash-api'"
    schema = 'NETTO'
else:
    project_name1 = ''
    project_name2 = ''
    schema = project

df0 = import_data(schema, project_name1, project_name2, user, password, account)
df = df0.copy()
df['MONTH_DATE'] = pd.to_datetime(df['MONTH_DATE'])
last_full_month = last_full_month()
df = df[df['MONTH_DATE']<=last_full_month]

########################################################################
########################################################################

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["iOS WW", "iOS US", "Android WW Organic", "Android WW Paid", "Android US Organic", "Android US Paid"])

with tab1:
    plat_1 = 'ios'
    us_ww_1 = 'ww'

    df_d_0_1 = df_category(df, plat_1, us_ww_1)
    df_d_pivot_1 = df_category_pivot(df_d_0_1)

    df_mog_0_1 = df_mog_0(df_d_0_1)
    df_mog_1 = df_mog_pivot(df_mog_0_1)

    df_mog_pct_1 = ratio_with_0(df_mog_1)

    with st.expander('Net Revenue per DoG'):
        st.dataframe(df_d_pivot_1)

    with st.expander('Net Revenue per MoG'):
        st.dataframe(df_mog_1)

    with st.expander('% of M0 Net Revenue per MoG'):
        st.dataframe(df_mog_pct_1)


    with st.expander('Fitting'):
        col1, col2 = st.columns(2)

        with col1:
            months_no_1 = st.number_input('No of recent months for fitting', value=9, key='months_no_1')
        
            st.write('Adjust months to average')
            df_months_no_0_1 = make_df_months_no(df_mog_pct_1, months_no_1)

            df_months_no_1 = st.data_editor(
                df_months_no_0_1,
                disabled=["MONTH_DATE"],
                key='df_months_no_1'
                )

            a_fit_1, b_fit_1, df_avg_1 = fitting2(df_mog_pct_1, df_months_no_1)

            st.metric('Amplitude', a_fit_1)
            st.metric('Decay', b_fit_1)
        with col2:
            fit_plot(df_avg_1, a_fit_1, b_fit_1)
            
        with st.container():
            # adjust_parameters_df_0_1 = adjust_params()
            # adjust_parameters_df_1 = st.data_editor(adjust_parameters_df_0_1, key='adjust_parameters_df_1')

            new_mog_pct_avg_fit_1 = new_avg_fit(a_fit_1, b_fit_1, df_avg_1)
            df_new_avg_fit_0_1 = series_to_df_transp(new_mog_pct_avg_fit_1, 'MoG', '% of M0')
            df_new_avg_fit_1 = df_new_avg_fit_0_1
            st.write('Avg + Fitted % of Revenue')
            st.dataframe(df_new_avg_fit_1)

            plot_data_fitted(df_mog_pct_1, new_mog_pct_avg_fit_1)


    with st.expander('Mulipliers / Goals'):

        

        col1b, col2b = st.columns(2)

        with col1b:
            ROAS_goal_Y3_1 = st.number_input('ROAS Goal Y3', value=1.3, key='roas goal y3 1')
        
        with col2b:
            multiplier_y3_1 = multi_3y(new_mog_pct_avg_fit_1)
            multiplier_d7_1, multiplier_1 = multiplier(df_d_0_1, months_no_1, df_mog_1, multiplier_y3_1)

            st.metric('Multiplier Y3', multiplier_y3_1)
            st.metric('Multiplier D7', multiplier_d7_1)
            st.metric('Multiplier D7', multiplier_1)
        
        d7_growth_mean_1 = d7_growth(df_d_pivot_1, months_no_1)
        d7_growth_mean_df_1 = series_to_df_transp(d7_growth_mean_1, 'MoG', 'D(n) to D7 Growth')
        st.write('D(n) to D7 Growth')
        st.dataframe(d7_growth_mean_df_1)

        ROAS_goal_1, SKAD_ROAS_goal_net_1 = roas_goal(ROAS_goal_Y3_1, multiplier_1, d7_growth_mean_1)
        
        ROAS_goal_df_1 = series_to_df_transp(ROAS_goal_1, 'MoG', 'ROAS Goal')
        st.write('ROAS Goal')
        st.dataframe(ROAS_goal_df_1)

        st.metric('SKAD ROAS Goal Net', SKAD_ROAS_goal_net_1)


with tab2:
    plat_2 = 'ios'
    us_ww_2 = 'us'

    df_d_0_2 = df_category(df, plat_2, us_ww_2)
    df_d_pivot_2 = df_category_pivot(df_d_0_2)

    df_mog_0_2 = df_mog_0(df_d_0_2)
    df_mog_2 = df_mog_pivot(df_mog_0_2)

    df_mog_pct_2 = ratio_with_0(df_mog_2)

    with st.expander('Net Revenue per DoG'):
        st.dataframe(df_d_pivot_2)

    with st.expander('Net Revenue per MoG'):
        st.dataframe(df_mog_2)

    with st.expander('% of M0 Net Revenue per MoG'):
        st.dataframe(df_mog_pct_2)


    with st.expander('Fitting'):
        col1, col2 = st.columns(2)

        with col1:
            months_no_2 = st.number_input('No of recent months for fitting', value=9, key='months_no_2')

            st.write('Adjust months to average')
            df_months_no_0_2 = make_df_months_no(df_mog_pct_2, months_no_2)

            df_months_no_2 = st.data_editor(
                df_months_no_0_2,
                disabled=["MONTH_DATE"],
                key='df_months_no_2'
                )

            a_fit_2, b_fit_2, df_avg_2 = fitting2(df_mog_pct_2, df_months_no_2)

            st.metric('Amplitude', a_fit_2)
            st.metric('Decay', b_fit_2)
        with col2:
            fit_plot(df_avg_2, a_fit_2, b_fit_2)
            
        with st.container():
            new_mog_pct_avg_fit_2 = new_avg_fit(a_fit_2, b_fit_2, df_avg_2)
            df_new_avg_fit_2 = series_to_df_transp(new_mog_pct_avg_fit_2, 'MoG', '% of M0')
            st.write('Avg + Fitted % of Revenue')
            st.dataframe(df_new_avg_fit_2)

            plot_data_fitted(df_mog_pct_2, new_mog_pct_avg_fit_2)


    with st.expander('Mulipliers / Goals'):

        

        col1b, col2b = st.columns(2)

        with col1b:
            ROAS_goal_Y3_2 = st.number_input('ROAS Goal Y3', value=1.3, key='roas goal y3 2')
        
        with col2b:
            multiplier_y3_2 = multi_3y(new_mog_pct_avg_fit_2)
            multiplier_d7_2, multiplier_2 = multiplier(df_d_0_2, months_no_2, df_mog_2, multiplier_y3_2)

            st.metric('Multiplier Y3', multiplier_y3_2)
            st.metric('Multiplier D7', multiplier_d7_2)
            st.metric('Multiplier D7', multiplier_2)
        
        d7_growth_mean_2 = d7_growth(df_d_pivot_2, months_no_2)
        d7_growth_mean_df_2 = series_to_df_transp(d7_growth_mean_2, 'MoG', 'D(n) to D7 Growth')
        st.write('D(n) to D7 Growth')
        st.dataframe(d7_growth_mean_df_2)

        ROAS_goal_2, SKAD_ROAS_goal_net_2 = roas_goal(ROAS_goal_Y3_2, multiplier_2, d7_growth_mean_2)
        
        ROAS_goal_df_2 = series_to_df_transp(ROAS_goal_2, 'MoG', 'ROAS Goal')
        st.write('ROAS Goal')
        st.dataframe(ROAS_goal_df_2)

        st.metric('SKAD ROAS Goal Net', SKAD_ROAS_goal_net_2)

    
with tab3:
    plat_3 = 'android'
    us_ww_3 = 'ww'
    traffic_3 = 'Organic'

    df_d_0_3 = df_category(df, plat_3, us_ww_3, traffic_3)
    df_d_pivot_3 = df_category_pivot(df_d_0_3)

    df_mog_0_3 = df_mog_0(df_d_0_3)
    df_mog_3 = df_mog_pivot(df_mog_0_3)

    df_mog_pct_3 = ratio_with_0(df_mog_3)

    with st.expander('Net Revenue per DoG'):
        st.dataframe(df_d_pivot_3)

    with st.expander('Net Revenue per MoG'):
        st.dataframe(df_mog_3)

    with st.expander('% of M0 Net Revenue per MoG'):
        st.dataframe(df_mog_pct_3)


    with st.expander('Fitting'):
        col1, col2 = st.columns(2)

        with col1:
            months_no_3 = st.number_input('No of recent months for fitting', value=9, key='months_no_3')

            st.write('Adjust months to average')
            df_months_no_0_3 = make_df_months_no(df_mog_pct_3, months_no_3)

            df_months_no_3 = st.data_editor(
                df_months_no_0_3,
                disabled=["MONTH_DATE"],
                key='df_months_no_3'
                )

            a_fit_3, b_fit_3, df_avg_3 = fitting2(df_mog_pct_3, df_months_no_3)

            st.metric('Amplitude', a_fit_3)
            st.metric('Decay', b_fit_3)
        with col2:
            fit_plot(df_avg_3, a_fit_3, b_fit_3)
            
        with st.container():
            new_mog_pct_avg_fit_3 = new_avg_fit(a_fit_3, b_fit_3, df_avg_3)
            df_new_avg_fit_3 = series_to_df_transp(new_mog_pct_avg_fit_3, 'MoG', '% of M0')
            st.write('Avg + Fitted % of Revenue')
            st.dataframe(df_new_avg_fit_3)

            plot_data_fitted(df_mog_pct_3, new_mog_pct_avg_fit_3)


    with st.expander('Mulipliers / Goals'):

        

        col1b, col2b = st.columns(2)

        with col1b:
            ROAS_goal_Y3_3 = st.number_input('ROAS Goal Y3', value=1.3, key='roas goal y3 3')
        
        with col2b:
            multiplier_y3_3 = multi_3y(new_mog_pct_avg_fit_3)
            multiplier_d7_3, multiplier_3 = multiplier(df_d_0_3, months_no_3, df_mog_3, multiplier_y3_3)

            st.metric('Multiplier Y3', multiplier_y3_3)
            st.metric('Multiplier D7', multiplier_d7_3)
            st.metric('Multiplier D7', multiplier_3)
        
        d7_growth_mean_3 = d7_growth(df_d_pivot_3, months_no_3)
        d7_growth_mean_df_3 = series_to_df_transp(d7_growth_mean_3, 'MoG', 'D(n) to D7 Growth')
        st.write('D(n) to D7 Growth')
        st.dataframe(d7_growth_mean_df_3)

        ROAS_goal_3, SKAD_ROAS_goal_net_3 = roas_goal(ROAS_goal_Y3_3, multiplier_3, d7_growth_mean_3)
        
        ROAS_goal_df_3 = series_to_df_transp(ROAS_goal_3, 'MoG', 'ROAS Goal')
        st.write('ROAS Goal')
        st.dataframe(ROAS_goal_df_3)

        st.metric('SKAD ROAS Goal Net', SKAD_ROAS_goal_net_3)


with tab4:
    plat_4 = 'android'
    us_ww_4 = 'ww'
    traffic_4 = 'Paid'

    df_d_0_4 = df_category(df, plat_4, us_ww_4, traffic_4)
    df_d_pivot_4 = df_category_pivot(df_d_0_4)

    df_mog_0_4 = df_mog_0(df_d_0_4)
    df_mog_4 = df_mog_pivot(df_mog_0_4)

    df_mog_pct_4 = ratio_with_0(df_mog_4)

    with st.expander('Net Revenue per DoG'):
        st.dataframe(df_d_pivot_4)

    with st.expander('Net Revenue per MoG'):
        st.dataframe(df_mog_4)

    with st.expander('% of M0 Net Revenue per MoG'):
        st.dataframe(df_mog_pct_4)


    with st.expander('Fitting'):
        col1, col2 = st.columns(2)

        with col1:
            months_no_4 = st.number_input('No of recent months for fitting', value=9, key='months_no_4')

            st.write('Adjust months to average')
            df_months_no_0_4 = make_df_months_no(df_mog_pct_4, months_no_4)

            df_months_no_4 = st.data_editor(
                df_months_no_0_4,
                disabled=["MONTH_DATE"],
                key='df_months_no_4'
                )

            a_fit_4, b_fit_4, df_avg_4 = fitting2(df_mog_pct_4, df_months_no_4)

            st.metric('Amplitude', a_fit_4)
            st.metric('Decay', b_fit_4)
        with col2:
            fit_plot(df_avg_4, a_fit_4, b_fit_4)
            
        with st.container():
            new_mog_pct_avg_fit_4 = new_avg_fit(a_fit_4, b_fit_4, df_avg_4)
            df_new_avg_fit_4 = series_to_df_transp(new_mog_pct_avg_fit_4, 'MoG', '% of M0')
            st.write('Avg + Fitted % of Revenue')
            st.dataframe(df_new_avg_fit_4)

            plot_data_fitted(df_mog_pct_4, new_mog_pct_avg_fit_4)


    with st.expander('Mulipliers / Goals'):

        

        col1b, col2b = st.columns(2)

        with col1b:
            ROAS_goal_Y3_4 = st.number_input('ROAS Goal Y3', value=1.3, key='roas goal y3 4')
        
        with col2b:
            multiplier_y3_4 = multi_3y(new_mog_pct_avg_fit_4)
            multiplier_d7_4, multiplier_4 = multiplier(df_d_0_4, months_no_4, df_mog_4, multiplier_y3_4)

            st.metric('Multiplier Y3', multiplier_y3_4)
            st.metric('Multiplier D7', multiplier_d7_4)
            st.metric('Multiplier D7', multiplier_4)
        
        d7_growth_mean_4 = d7_growth(df_d_pivot_4, months_no_4)
        d7_growth_mean_df_4 = series_to_df_transp(d7_growth_mean_4, 'MoG', 'D(n) to D7 Growth')
        st.write('D(n) to D7 Growth')
        st.dataframe(d7_growth_mean_df_4)

        ROAS_goal_4, SKAD_ROAS_goal_net_4 = roas_goal(ROAS_goal_Y3_4, multiplier_4, d7_growth_mean_4)
        
        ROAS_goal_df_4 = series_to_df_transp(ROAS_goal_4, 'MoG', 'ROAS Goal')
        st.write('ROAS Goal')
        st.dataframe(ROAS_goal_df_4)

        st.metric('SKAD ROAS Goal Net', SKAD_ROAS_goal_net_4)


with tab5:
    plat_5 = 'android'
    us_ww_5 = 'us'
    traffic_5 = 'Organic'

    df_d_0_5 = df_category(df, plat_5, us_ww_5, traffic_5)
    df_d_pivot_5 = df_category_pivot(df_d_0_5)

    df_mog_0_5 = df_mog_0(df_d_0_5)
    df_mog_5 = df_mog_pivot(df_mog_0_5)

    df_mog_pct_5 = ratio_with_0(df_mog_5)

    with st.expander('Net Revenue per DoG'):
        st.dataframe(df_d_pivot_5)

    with st.expander('Net Revenue per MoG'):
        st.dataframe(df_mog_5)

    with st.expander('% of M0 Net Revenue per MoG'):
        st.dataframe(df_mog_pct_5)


    with st.expander('Fitting'):
        col1, col2 = st.columns(2)

        with col1:
            months_no_5 = st.number_input('No of recent months for fitting', value=9, key='months_no_5')

            st.write('Adjust months to average')
            df_months_no_0_5 = make_df_months_no(df_mog_pct_5, months_no_5)

            df_months_no_5 = st.data_editor(
                df_months_no_0_5,
                disabled=["MONTH_DATE"],
                key='df_months_no_5'
                )

            a_fit_5, b_fit_5, df_avg_5 = fitting2(df_mog_pct_5, df_months_no_5)

            st.metric('Amplitude', a_fit_5)
            st.metric('Decay', b_fit_5)
        with col2:
            fit_plot(df_avg_5, a_fit_5, b_fit_5)
            
        with st.container():
            new_mog_pct_avg_fit_5 = new_avg_fit(a_fit_5, b_fit_5, df_avg_5)
            df_new_avg_fit_5 = series_to_df_transp(new_mog_pct_avg_fit_5, 'MoG', '% of M0')
            st.write('Avg + Fitted % of Revenue')
            st.dataframe(df_new_avg_fit_5)

            plot_data_fitted(df_mog_pct_5, new_mog_pct_avg_fit_5)


    with st.expander('Mulipliers / Goals'):

        

        col1b, col2b = st.columns(2)

        with col1b:
            ROAS_goal_Y3_5 = st.number_input('ROAS Goal Y3', value=1.3, key='roas goal y3 5')
        
        with col2b:
            multiplier_y3_5 = multi_3y(new_mog_pct_avg_fit_5)
            multiplier_d7_5, multiplier_5 = multiplier(df_d_0_5, months_no_5, df_mog_5, multiplier_y3_5)

            st.metric('Multiplier Y3', multiplier_y3_5)
            st.metric('Multiplier D7', multiplier_d7_5)
            st.metric('Multiplier D7', multiplier_5)
        
        d7_growth_mean_5 = d7_growth(df_d_pivot_5, months_no_5)
        d7_growth_mean_df_5 = series_to_df_transp(d7_growth_mean_5, 'MoG', 'D(n) to D7 Growth')
        st.write('D(n) to D7 Growth')
        st.dataframe(d7_growth_mean_df_5)

        ROAS_goal_5, SKAD_ROAS_goal_net_5 = roas_goal(ROAS_goal_Y3_5, multiplier_5, d7_growth_mean_5)
        
        ROAS_goal_df_5 = series_to_df_transp(ROAS_goal_5, 'MoG', 'ROAS Goal')
        st.write('ROAS Goal')
        st.dataframe(ROAS_goal_df_5)

        st.metric('SKAD ROAS Goal Net', SKAD_ROAS_goal_net_5)


with tab6:
    plat_6 = 'android'
    us_ww_6 = 'us'
    traffic_6 = 'Paid'

    df_d_0_6 = df_category(df, plat_6, us_ww_6, traffic_6)
    df_d_pivot_6 = df_category_pivot(df_d_0_6)

    df_mog_0_6 = df_mog_0(df_d_0_6)
    df_mog_6 = df_mog_pivot(df_mog_0_6)

    df_mog_pct_6 = ratio_with_0(df_mog_6)

    with st.expander('Net Revenue per DoG'):
        st.dataframe(df_d_pivot_6)

    with st.expander('Net Revenue per MoG'):
        st.dataframe(df_mog_6)

    with st.expander('% of M0 Net Revenue per MoG'):
        st.dataframe(df_mog_pct_6)


    with st.expander('Fitting'):
        col1, col2 = st.columns(2)

        with col1:
            months_no_6 = st.number_input('No of recent months for fitting', value=9, key='months_no_6')

            st.write('Adjust months to average')
            df_months_no_0_6 = make_df_months_no(df_mog_pct_6, months_no_6)

            df_months_no_6 = st.data_editor(
                df_months_no_0_6,
                disabled=["MONTH_DATE"],
                key='df_months_no_6'
                )

            a_fit_6, b_fit_6, df_avg_6 = fitting2(df_mog_pct_6, df_months_no_6)

            st.metric('Amplitude', a_fit_6)
            st.metric('Decay', b_fit_6)
        with col2:
            fit_plot(df_avg_6, a_fit_6, b_fit_6)
            
        with st.container():
            new_mog_pct_avg_fit_6 = new_avg_fit(a_fit_6, b_fit_6, df_avg_6)
            df_new_avg_fit_6 = series_to_df_transp(new_mog_pct_avg_fit_6, 'MoG', '% of M0')
            st.write('Avg + Fitted % of Revenue')
            st.dataframe(df_new_avg_fit_6)

            plot_data_fitted(df_mog_pct_6, new_mog_pct_avg_fit_6)


    with st.expander('Mulipliers / Goals'):

        

        col1b, col2b = st.columns(2)

        with col1b:
            ROAS_goal_Y3_6 = st.number_input('ROAS Goal Y3', value=1.3, key='roas goal y3 6')
        
        with col2b:
            multiplier_y3_6 = multi_3y(new_mog_pct_avg_fit_6)
            multiplier_d7_6, multiplier_6 = multiplier(df_d_0_6, months_no_6, df_mog_6, multiplier_y3_6)

            st.metric('Multiplier Y3', multiplier_y3_6)
            st.metric('Multiplier D7', multiplier_d7_6)
            st.metric('Multiplier D7', multiplier_6)
        
        d7_growth_mean_6 = d7_growth(df_d_pivot_6, months_no_6)
        d7_growth_mean_df_6 = series_to_df_transp(d7_growth_mean_6, 'MoG', 'D(n) to D7 Growth')
        st.write('D(n) to D7 Growth')
        st.dataframe(d7_growth_mean_df_6)

        ROAS_goal_6, SKAD_ROAS_goal_net_6 = roas_goal(ROAS_goal_Y3_6, multiplier_6, d7_growth_mean_6)
        
        ROAS_goal_df_6 = series_to_df_transp(ROAS_goal_6, 'MoG', 'ROAS Goal')
        st.write('ROAS Goal')
        st.dataframe(ROAS_goal_df_6)

        st.metric('SKAD ROAS Goal Net', SKAD_ROAS_goal_net_6)