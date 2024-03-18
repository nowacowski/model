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
def df_mog_0_func(df_d_0):
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
    for index, row in df.iterrows():
        if row.loc[0] == 0:
            # non_null_indices = row[3:].notnull()
            # df.iloc[index, 3:][non_null_indices] = 0
            df.loc[index, columns] = np.nan
        else:
            df.loc[index, columns] = row[3:] / row[0]
    return df


########################################################################

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
    try:
        merged_df = pd.merge(df_months_no, df, on='MONTH_DATE')
        merged_df_2 = merged_df[merged_df['INCLUDE']==True]
        df_avg = merged_df_2.iloc[:, 5:].mean(axis=0).dropna()

        if not df_avg.empty:
            params, covariance = curve_fit(exponential_decay, df_avg.index.astype(float), df_avg)
            a_fit, b_fit = params
        else:
             a_fit, b_fit = 0.0, 0.0
    except TypeError:
        a_fit, b_fit = 0.0, 0.0
    return a_fit, b_fit, df_avg

# fit plot
def fit_plot(df_avg, a_fit, b_fit):
    fig, ax = plt.subplots()
    ax.scatter(df_avg.index, df_avg, label='Data')
    ax.plot(df_avg.index, exponential_decay(df_avg.index.astype(float), a_fit, b_fit), 'r-', label='Fitted curve')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title('Exponential Decay Fit')
    # plt.show()
    st.pyplot(fig)



# new % rev avg+fit

def new_avg_fit(a_fit, b_fit, df_avg):
    x_fit = np.arange(1,37)
    fitted = exponential_decay(x_fit, a_fit, b_fit)
    fitted_series = pd.Series(data = fitted, index = x_fit)
    new_avg_fit_df = fitted_series.copy()
    new_avg_fit_df.loc[new_avg_fit_df.index.isin(df_avg.index)] = df_avg

    return new_avg_fit_df


def series_to_df_transp(series, val_name):
    series_df = series.rename(val_name)
    # series_df = series.reset_index()
    series_df_transp = series_df.to_frame().T
    return series_df_transp

# adjust param array
def adjust_params(value):
    data = {'% of M0': [value]*37}
    df = pd.DataFrame(data).iloc[1:].T
    return df

def multiply_adjusted(df1, df2):
    df1.columns = df1.columns.astype(str)
    df2.columns = df2.columns.astype(str)
    df_multi = df1.mul(df2)
    return df_multi



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



########################################################################

# final results
# multi 3y
def multi_3y(new_mog_pct_avg_fit):
    multiplier_y3 = new_mog_pct_avg_fit.sum()+1
    return multiplier_y3

# multiplier
def multiplier_func(df_d0, df_months_no, df_mog, multiplier_y3):
    zz0 = df_d0.copy()
    zz01 = pd.merge(zz0, df_months_no, on='MONTH_DATE')
    zz = zz01[zz01['INCLUDE']]
    zz = zz.drop(columns='INCLUDE')

    zz = zz[zz['DAY_OF_GAME_UTC'] <= 7]
    zz_d7 = zz.groupby(by=['MONTH_DATE'], as_index=False).sum('REVENUE_NET')[['MONTH_DATE','REVENUE_NET']]

    zz_m00 = df_mog.copy()
    zz_m01 = pd.merge(zz_m00, df_months_no, on='MONTH_DATE')
    zz_m = zz_m01[zz_m01['INCLUDE']]
    zz_m = zz_m.drop(columns='INCLUDE')
    zz_m0 = zz_m[['MONTH_DATE',0]]

    merged_df = pd.merge(zz_d7, zz_m0, on='MONTH_DATE')
    merged_df['Result'] = merged_df['REVENUE_NET'] / merged_df[0]

    if merged_df.empty or merged_df['Result'].isnull().all():
        multiplier_d7 = 0.0  # Assign a default value if merged_df is empty
    else:
        multiplier_d7 = merged_df['Result'].mean().item()

    multiplier = (multiplier_y3 / multiplier_d7).item()

    return multiplier_d7, multiplier


# D(n) to D7 growth
def d7_growth(df_d, df_months_no):
    d7_growth_00 = df_d.reset_index()

    d7_growth_0 = pd.merge(d7_growth_00, df_months_no, on='MONTH_DATE')
    d7_growth_0 = d7_growth_0[d7_growth_0['INCLUDE']]
    d7_growth_0 = d7_growth_0.drop(columns='INCLUDE')

    # cols = ['MONTH_DATE'] + list(d7_growth_0.columns)[3:11]
    cols = [col for col in d7_growth_0.iloc[:,3:].columns if 0 <= int(col) <= 7]
    d7_growth_0 = d7_growth_0[cols]
    
    d7_growth__cumm = d7_growth_0.copy()
    # d7_growth__cumm.iloc[:, 1:] = d7_growth__cumm.iloc[:, 1:].cumsum(axis=1)
    d7_growth__cumm = d7_growth__cumm.cumsum(axis=1)
    
    d7_growth = d7_growth__cumm.copy()
    for index, row in d7_growth.iterrows():
        for i, col_name in enumerate(d7_growth.columns):
            if row[7] == 0:
                d7_growth.at[index, col_name] = np.nan
            else:
                if row[col_name] == 0:
                    d7_growth.at[index, col_name] = np.nan
                else:
                    d7_growth.at[index, col_name] = row[7] / row[col_name]
        # d7_growth[i] = d7_growth[7] / d7_growth[i]
    
    # d7_growth_mean = d7_growth.iloc[:,1:].mean()
    d7_growth_mean = d7_growth.mean()

    return d7_growth_mean

# ROAS goal and SKAD ROAS goal net
def roas_goal(ROAS_goal_Y3, multiplier, d7_growth_mean):
    ROAS_goal = ROAS_goal_Y3/(multiplier*d7_growth_mean)
    SKAD_ROAS_goal_net = ROAS_goal[1:3].mean()
    return ROAS_goal, SKAD_ROAS_goal_net

########################################################################
########################################################################
# Tabs display
def tab_func(df, plat, us_ww, tab_no, traffic):

    df_d_0 = df_category(df, plat, us_ww, traffic)
    df_d_pivot = df_category_pivot(df_d_0)

    df_mog_0 = df_mog_0_func(df_d_0)
    df_mog = df_mog_pivot(df_mog_0)

    df_mog_pct = ratio_with_0(df_mog)

    with st.expander('Net Revenue per DoG'):
        st.dataframe(df_d_pivot)

    with st.expander('Net Revenue per MoG'):
        st.dataframe(df_mog)

    with st.expander('% of M0 Net Revenue per MoG'):
        st.dataframe(df_mog_pct)


    with st.expander('Fitting'):
        col1, col2 = st.columns(2)

        with col1:
            months_no = st.number_input('No of recent months for fitting', value=9, key='months_no_'+tab_no)
        
            st.write('Adjust months to average')
            df_months_no_0 = make_df_months_no(df_mog_pct, months_no)

            df_months_no = st.data_editor(
                df_months_no_0,
                disabled=["MONTH_DATE"],
                key='df_months_no_'+tab_no
                )

            a_fit, b_fit, df_avg = fitting2(df_mog_pct, df_months_no)

            st.metric('Amplitude', a_fit)
            st.metric('Decay', b_fit)
        with col2:
            fit_plot(df_avg, a_fit, b_fit)
            
        with st.container():

            new_mog_pct_avg_fit = new_avg_fit(a_fit, b_fit, df_avg)
            df_new_avg_fit_0 = series_to_df_transp(new_mog_pct_avg_fit, '% of M0')
            

            st.write('Avg + Fitted % of Revenue')
            st.dataframe(df_new_avg_fit_0)

            value = st.number_input('Adjust parameter intitial value:', min_value=0.0, value=1.0, key='value_'+tab_no)
            adjust_parameters_df_0 = adjust_params(value)
            adjust_parameters_df = st.data_editor(adjust_parameters_df_0, key='adjust_parameters_df_'+tab_no)

            df_new_avg_fit = multiply_adjusted(df_new_avg_fit_0, adjust_parameters_df)
            df_new_avg_fit_b = df_new_avg_fit.copy()
            df_new_avg_fit_b.columns = df_new_avg_fit.columns.astype(int)
            df_new_avg_fit_b = df_new_avg_fit_b.T

            st.write('Adjusted Avg + Fitted % of Revenue')
            st.dataframe(df_new_avg_fit)

            plot_data_fitted(df_mog_pct, df_new_avg_fit_b)


    with st.expander('Mulipliers / Goals'):

        col1b, col2b = st.columns(2)

        with col1b:
            ROAS_goal_Y3 = st.number_input('ROAS Goal Y3', value=1.3, key='roas goal y3 '+tab_no)
        
        with col2b:
            multiplier_y3 = multi_3y(df_new_avg_fit_b)
            # multiplier_d7, multiplier = multiplier_func(df_d_0, df_months_no, df_mog, multiplier_y3)
            multiplier_d7, multiplier = multiplier_func(df_d_0, df_months_no, df_mog, multiplier_y3)

            st.metric('Multiplier Y3', multiplier_y3)
            st.metric('Multiplier D7', multiplier_d7)
            st.metric('Multiplier', multiplier)
        
        d7_growth_mean = d7_growth(df_d_pivot, df_months_no)
        d7_growth_mean_df = series_to_df_transp(d7_growth_mean, 'D(n) to D7 Growth')
        st.write('D(n) to D7 Growth')
        st.dataframe(d7_growth_mean_df)

        ROAS_goal, SKAD_ROAS_goal_net = roas_goal(ROAS_goal_Y3, multiplier, d7_growth_mean)
        
        ROAS_goal_df = series_to_df_transp(ROAS_goal, 'ROAS Goal')
        st.write('ROAS Goal')
        st.dataframe(ROAS_goal_df)

        st.metric('SKAD ROAS Goal Net', SKAD_ROAS_goal_net)


########################################################################
######################### app stuff ####################################


########################################################################

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
# sidebar
with st.sidebar:
    
    with st.form(key='form'):
        # user, password, account
        user = st.text_input(label='User')
        password = st.text_input(label='Password', type='password')
        account = st.text_input(label='Account')

        # project select box
        project = st.selectbox(
            'Project',
            ('TG','SC')
        )

        submit_button = st.form_submit_button(label='Submit')

########################################################################
        
        if submit_button:
            st.session_state.form_submitted = True

if st.session_state.form_submitted:

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
        tab_no_1 = '1'
        traffic_1=None

        tab_func(df, plat_1, us_ww_1, tab_no_1, traffic_1)

    with tab2:
        plat_2 = 'ios'
        us_ww_2 = 'us'
        tab_no_2 = '2'
        traffic_2=None

        tab_func(df, plat_2, us_ww_2, tab_no_2, traffic_2)

    with tab3:
        plat_3 = 'android'
        us_ww_3 = 'ww'
        traffic_3 = 'Organic'
        tab_no_3 = '3'

        tab_func(df, plat_3, us_ww_3, tab_no_3, traffic_3)

    with tab4:
        plat_4 = 'android'
        us_ww_4 = 'ww'
        tab_no_4 = '4'
        traffic_4 = 'Paid'

        tab_func(df, plat_4, us_ww_4, tab_no_4, traffic_4)

    with tab5:
        plat_5 = 'android'
        us_ww_5 = 'us'
        traffic_5 = 'Organic'
        tab_no_5 = '5'

        tab_func(df, plat_5, us_ww_5, tab_no_5, traffic_5)

    with tab6:
        plat_6 = 'android'
        us_ww_6 = 'us'
        tab_no_6 = '6'
        traffic_6 = 'Paid'

        tab_func(df, plat_6, us_ww_6, tab_no_6, traffic_6)
    
