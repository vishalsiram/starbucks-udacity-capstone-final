import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def clean_portfolio(portfolio):
    """ Clean the portfolio dataset.
    - It makes columns for the channels
    - Changes the name of the id column to offer_id
    Input:
    - portfolio: original dataset
    Returns:
    - portfolio_clean
    """
    portfolio_clean = portfolio.copy()
    # Create dummy columns for the channels column
    d_chann = pd.get_dummies(portfolio_clean.channels.apply(pd.Series).stack(),
                             prefix="channel").sum(level=0)
    portfolio_clean = pd.concat([portfolio_clean, d_chann], axis=1, sort=False)
    portfolio_clean.drop(columns='channels', inplace=True)
    # Change column name
    portfolio_clean.rename(columns={'id':'offer_id'}, inplace=True)

    return portfolio_clean


def clean_profile(profile):
    """ Clean the profile dataset.
    - Fix the date format
    - Change the column name id to customer_id
    - Create column to identify customers with demographic data
    - Add dummy columns for gender
    Input:
    - profile: original dataset
    Returns:
    - profile_clean
    """
    profile_clean = profile.copy()
    # Transform date from int to datetime
    date = lambda x: pd.to_datetime(str(x), format='%Y%m%d')
    profile_clean.became_member_on = profile_clean.became_member_on.apply(date)
    # Create column that separates customers with valida data
    profile_clean['valid'] = (profile_clean.age != 118).astype(int)
    # Change the name of id column to customer_id
    profile_clean.rename(columns={'id':'customer_id'}, inplace=True)
    # Create dummy columns for the gender column
    dummy_gender = pd.get_dummies(profile_clean.gender, prefix="gender")
    profile_clean = pd.concat([profile_clean, dummy_gender], axis=1, sort=False)
    return profile_clean


def clean_transcript(transcript):
    """ Clean the transcript dataset.
    - Split value in several columns for offers and transactions
    - Split event column into sevelar columns
    - Change column name person to customer_id
    Input:
    - transcript: original dataset
    Returns:
    - transcript_clean
    """
    transcript_clean = transcript.copy()
    # Split event into several dummy columns
    transcript_clean.event = transcript_clean.event.str.replace(' ', '_')
    dummy_event = pd.get_dummies(transcript_clean.event, prefix="event")
    transcript_clean = pd.concat([transcript_clean, dummy_event], axis=1,
                                 sort=False)
    transcript_clean.drop(columns='event', inplace=True)
    # Get the offer_id data from the value column
    transcript_clean['offer_id'] = [[*v.values()][0]
                                    if [*v.keys()][0] in ['offer id',
                                                          'offer_id'] else None
                                    for v in transcript_clean.value]
    # Get the transaction amount data from the value column
    transcript_clean['amount'] = [np.round([*v.values()][0], decimals=2)
                                  if [*v.keys()][0] == 'amount' else None
                                  for v in transcript_clean.value]
    transcript_clean.drop(columns='value', inplace=True)
    # Change the name of person column to customer_id
    transcript_clean.rename(columns={'person':'customer_id'}, inplace=True)
    return transcript_clean

def merge_datasets(portfolio_clean, profile_clean, transcript_clean):
    """ Merge the three data sets into one
    Input:
    - portfolio_clean
    - profile_clean
    - transcript_clean
    Output:
    - df: merged dataframe
    """
    trans_prof = pd.merge(transcript_clean, profile_clean, on='customer_id',
                          how="left")
    df = pd.merge(trans_prof, portfolio_clean, on='offer_id', how='left')
    # Change the offer ids to a simplied form
    offer_id = {'ae264e3637204a6fb9bb56bc8210ddfd': 'B1',
                '4d5c57ea9a6940dd891ad53e9dbe8da0': 'B2',
                '9b98b8c7a33c4b65b9aebfe6a799e6d9': 'B3',
                'f19421c1d4aa40978ebb69ca19b0e20d': 'B4',
                '0b1e1539f2cc45b7b9fa7c272da2e1d7': 'D1',
                '2298d6c36e964ae4a3e7e9706d1fb8c2': 'D2',
                'fafdcd668e3743c1bb461111dcafc2a4': 'D3',
                '2906b810c7d4411798c6938adc9daaa5': 'D4',
                '3f207df678b143eea3cee63160fa8bed': 'I1',
                '5a8bc65990b245e5a138643cd4eb9837': 'I2'}
    df.offer_id = df.offer_id.apply(lambda x: offer_id[x] if x else None)

    return df


def get_offer_cust(df, offer_type=None):
    """
    Get offer data (received, viewed and completed) per customer and
    offer type
    Inputs:
    - df: dataframe of merged transactions, portfolio and profile data
    - offer_type: informational, bogo or discount
    Output:
    - aggregated data per customer and offer type
    """
    data = dict()
    for e in ['received', 'viewed', 'completed']:
        # Informational offers don't have completed data
        if offer_type == 'informational' and e == 'completed':
            continue
        flag = (df['event_offer_{}'.format(e)] == 1)
        key = e
        if offer_type:
            flag = flag & (df.offer_type == offer_type)
            key = '{}_'.format(offer_type) + key
        data[key] = df[flag].groupby('customer_id').offer_id.count()
    # Informational offers don't have reward data
    flag = (df.event_offer_completed == 1)
    if offer_type != 'informational':
        key = 'reward'
        if offer_type:
            flag = flag & (df.offer_type == offer_type)
            key = '{}_'.format(offer_type) + key
        data[key] = df[flag].groupby('customer_id').reward.sum()

    return data


def get_offer_id_cust(df, offer_id):
    """
    Get offer data (received, viewed and completed) per customer
    and offer id
    Inputs:
    - df: dataframe of merged transactions, portfolio and profile data
    - offer_id: B1, B2, ...
    Output:
    - aggregated data per customer and offer id
    """
    data = dict()

    for e in ['received', 'viewed', 'completed']:
        # Informational offers don't have completed data
        if offer_id in ['I1', 'I2'] and e == 'completed':
            continue
        event = 'event_offer_{}'.format(e)
        flag = (df[event] == 1) & (df.offer_id == offer_id)
        key = '{}_{}'.format(offer_id, e)
        data[key] = df[flag].groupby('customer_id').offer_id.count()

    # Informational offers don't have reward data
    flag = (df.event_offer_completed == 1) & (df.offer_id == offer_id)
    if offer_id not in ['I1', 'I2']:
        key = '{}_reward'.format(offer_id)
        data[key] = df[flag].groupby('customer_id').reward.sum()

    return data


def round_age(x):
    """
    Round age to the 5th of each 10th (15, 25,..., 105)
    Input:
    - x: age
    Output:
    - rounded age. Returns 0 if the value is less than 15 or more than 105
    """
    for y in range(15, 106, 10):
        if x >= y and x < y+10:
            return y
    return 0


def round_income(x):
    """
    Round income to the lower 10000th
    Intput:
    - income
    Output:
    - lower 10000th of the income. Return 0 if the income
    is less than 30,000 or more than 120,000
    """
    for y in range(30, 130, 10):
        if x >= y*1000 and x < (y+10)*1000:
            return y*1000
    return 0


def per_customer_data(df, profile):
    """ Build a dataframe with aggregated purchase and offer data and demographics
    Input:
    - df: merged dataframe with transactions, customer and offer data
    Output:
    - customer: dataframe with aggregated data
    """
    cust_dict = dict()
    # Get total transaction data
    transactions = df[df.event_transaction == 1].groupby('customer_id')
    cust_dict['total_expense'] = transactions.amount.sum()
    cust_dict['total_transactions'] = transactions.amount.count()
    # Get  aggr offer data
    cust_dict.update(get_offer_cust(df))
    # Get offer type data
    for ot in ['bogo', 'discount', 'informational']:
        cust_dict.update(get_offer_cust(df, ot))
    # Get offer id data
    for oi in ['B1', 'B2', 'B3', 'B4', 'D1', 'D2', 'D3', 'D4', 'I1', 'I2']:
        cust_dict.update(get_offer_id_cust(df, oi))

    customers = pd.concat(cust_dict.values(), axis=1, sort=False);
    customers.columns = cust_dict.keys()
    customers.fillna(0, inplace=True)

    # Add demographic data
    customers = pd.merge(customers, profile.set_index('customer_id'),
                         left_index=True, right_index=True)
    customers['age_group'] = customers.age.apply(round_age)
    customers['income_group'] = customers.income.apply(round_income)
    customers['net_expense'] = customers['total_expense'] - customers['reward']

    return customers


def get_offer_stat(customers, stat, offer):
    """ Get any column for customers that received but not viewed an offer,
    viewed but not completed the offer, and those that viewed and completed
    the offer
    Input:
    - customers: dataframe with aggregated data of the offers
    - stat: column of interest
    - offer: offer of interest
    Output:
    - (received, viewed, completed): tuple with the corresponding column
    """
    valid = (customers.valid == 1)
    rcv_col = '{}_received'.format(offer)
    vwd_col = '{}_viewed'.format(offer)
    received = valid & (customers[rcv_col] > 0) & (customers[vwd_col] == 0)
    cpd = None
    if offer not in ['informational', 'I1', 'I2']:
        cpd_col = '{}_completed'.format(offer)
        viewed = valid & (customers[vwd_col] > 0) & (customers[cpd_col] == 0)
        completed = valid & (customers[vwd_col] > 0) & (customers[cpd_col] > 0)
        cpd = customers[completed][stat]
    else:
        viewed = valid & (customers[vwd_col] > 0)

    return customers[received][stat], customers[viewed][stat], cpd


def get_average_expense(customers, offer):
    """ Get the average expense for customers that received but not
    viewed an offer, viewed but not completed the offer, and those
    that viewed and completed the offer
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    Output:
    - (received, viewed, completed): tuple with the average expense
    """
    rcv_total, vwd_total, cpd_total = get_offer_stat(customers,
                                                     'total_expense', offer)
    rcv_trans, vwd_trans, cpd_trans = get_offer_stat(customers,
                                                     'total_transactions',
                                                     offer)

    rcv_avg = rcv_total / rcv_trans
    rcv_avg.fillna(0, inplace=True)
    vwd_avg = vwd_total / vwd_trans
    vwd_avg.fillna(0, inplace=True)

    cpd_avg = None
    if offer not in ['informational', 'I1', 'I2']:
        cpd_avg = cpd_total / cpd_trans

    return rcv_avg, vwd_avg, cpd_avg


def get_average_reward(customers, offer):
    """ Get the average reward received by customers that completed the offer
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    Output:
    - reward: average reward
    """
    cpd_col = '{}_completed'.format(offer)
    rwd_col = '{}_reward'.format(offer)
    completed = customers[(customers.valid == 1) & (customers[cpd_col] > 0)]

    return completed[rwd_col] / completed[cpd_col]


def get_offer_stat_by(customers, stat, offer, by_col, aggr='sum'):
    """ Get any column for customers that received but not viewed an offer,
    viewed but not completed the offer, and those that viewed and completed
    the offer, grouped by a column
    Input:
    - customers: dataframe with aggregated data of the offers
    - stat: column of interest
    - offer: offer of interest
    - by_col: column used to group the data
    - aggr: aggregation method sum or mean
    Output:
    - (received, viewed, completed): tuple with sum aggregation
    """
    valid = (customers.valid == 1)
    rcv_col = '{}_received'.format(offer)
    vwd_col = '{}_viewed'.format(offer)
    received = valid & (customers[rcv_col] > 0) & (customers[vwd_col] == 0)
    cpd = None
    if offer not in ['informational', 'I1', 'I2']:
        cpd_col = '{}_completed'.format(offer)
        viewed = valid & (customers[vwd_col] > 0) & (customers[cpd_col] == 0)
        completed = valid & (customers[cpd_col] > 0)
        if aggr == 'sum':
            cpd = customers[completed].groupby(by_col)[stat].sum()
        elif aggr == 'mean':
            cpd = customers[completed].groupby(by_col)[stat].mean()
    else:
        viewed = valid & (customers[vwd_col] > 0)
    if aggr == 'sum':
        rcv = customers[received].groupby(by_col)[stat].sum()
        vwd = customers[viewed].groupby(by_col)[stat].sum()
    elif aggr == 'mean':
        rcv = customers[received].groupby(by_col)[stat].mean()
        vwd = customers[viewed].groupby(by_col)[stat].mean()

    return rcv, vwd, cpd


def get_average_expense_by(customers, offer, by_col):
    """ Get the average expense for customers that received but not
    viewed an offer, viewed but not completed the offer, and those
    that viewed and completed the offer, group by a column
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    - by_col: column used to group the data
    Output:
    - (received, viewed, completed): tuple with the average expense
    """
    rcv_total, vwd_total, cpd_total = get_offer_stat_by(customers,
                                                        'total_expense',
                                                        offer, by_col)
    rcv_trans, vwd_trans, cpd_trans = get_offer_stat_by(customers,
                                                        'total_transactions',
                                                        offer, by_col)

    rcv_avg = rcv_total / rcv_trans
    rcv_avg.fillna(0, inplace=True)
    vwd_avg = vwd_total / vwd_trans
    vwd_avg.fillna(0, inplace=True)

    cpd_avg = None
    if offer not in ['informational', 'I1', 'I2']:
        cpd_avg = cpd_total / cpd_trans

    return rcv_avg, vwd_avg, cpd_avg


def get_average_reward_by(customers, offer, by_col):
    """ Get the average reward received by customers that completed
    the offer, grouped by a column
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    - by_col: column used to group the data
    Output:
    - reward: average reward
    """
    cpd_col = '{}_completed'.format(offer)
    rwd_col = '{}_reward'.format(offer)
    completed = customers[(customers.valid == 1) &
                          (customers[cpd_col] > 0)].groupby(by_col)

    return completed[rwd_col].sum() / completed[cpd_col].count()



def plot_offer_expense(customers, offer):
    """ Plot the histograms of the total expense and the average
    expense per transaction incurred by customers that have received,
    viewed and completed an offer.
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    """
    rcv, vwd, cpd = get_offer_stat(customers, 'total_expense', offer)
    rcv_avg, vwd_avg, cpd_avg = get_average_expense(customers, offer)

    plt.figure(figsize=(16, 5))
    bins = 100

    plt.subplot(121)
    plt.hist(rcv, bins, alpha=0.5, label='{}-received'.format(offer))
    plt.hist(vwd, bins, alpha=0.5, label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.hist(cpd, bins, alpha=0.5, label='{}-completed'.format(offer))
    plt.legend(loc='best')
    ax = plt.gca();
    ax.set_xlim(0, 600);
    plt.title('Total Transaction ($)')
    plt.grid();

    plt.subplot(122)
    plt.hist(rcv_avg, bins, alpha=0.5, label='{}-received'.format(offer))
    plt.hist(vwd_avg, bins, alpha=0.5, label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.hist(cpd_avg, bins, alpha=0.5, label='{}-completed'.format(offer))
    plt.legend(loc='best')
    ax = plt.gca();
    ax.set_xlim(0, 50);
    plt.title('Average Transaction ($)')
    plt.grid();


def plot_offer_reward(customers, offer):
    """ Plot the histograms of the total reward and the average
    reward received by customers that completed an offer.
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    """
    plt.figure(figsize=(16, 5))
    bins = 10

    key = '{}_completed'.format(offer)
    key_avg = '{}_reward'.format(offer)
    rwd = customers[(customers.valid == 1) & (customers[key] > 0)][key_avg]
    rwd_avg = get_average_reward(customers, offer)

    plt.subplot(121)
    plt.hist(rwd, bins, alpha=0.5, label=offer)
    plt.title('Total Reward ($)');
    plt.legend(loc='best');
    plt.grid();

    plt.subplot(122)
    plt.hist(rwd_avg, bins, alpha=0.5, label=offer)
    plt.title('Average Reward ($)');
    plt.legend(loc='best');
    plt.grid();


def plot_offer_expense_by(customers, offer):
    """ Plot the total expense and the average expense per transaction
    incurred by customers that have received, viewed and completed an offer.
    The plots are separated by age, income and gender.
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    """
    rcv_by = dict()
    vwd_by = dict()
    cpd_by = dict()
    rcv_avg_by = dict()
    vwd_avg_by = dict()
    cpd_avg_by = dict()

    for key in ['age_group', 'income_group', 'gender']:
        rcv_by[key], vwd_by[key], cpd_by[key] = get_offer_stat_by(customers,
                                                                  'net_expense',
                                                                  offer, key,
                                                                  aggr='mean')
        by_data = get_average_expense_by(customers, offer, key)
        rcv_avg_by[key], vwd_avg_by[key], cpd_avg_by[key] = by_data

    plt.figure(figsize=(16, 10))

    plt.subplot(231)
    plt.plot(rcv_by['age_group'], label='{}-received'.format(offer))
    plt.plot(vwd_by['age_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_by['age_group'], label='{}-completed'.format(offer))
    plt.legend(loc='best')
    plt.title('Net Expense');
    plt.grid();

    plt.subplot(232)
    plt.plot(rcv_by['income_group'], label='{}-received'.format(offer))
    plt.plot(vwd_by['income_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_by['income_group'], label='{}-completed'.format(offer))
    plt.legend(loc='best')
    plt.title('Net Expense');
    plt.grid();

    index = np.array([0, 1, 2])
    bar_width = 0.3
    plt.subplot(233)
    plt.bar(index, rcv_by['gender'].reindex(['M', 'F', 'O']),
            bar_width, label='{}-received'.format(offer))
    plt.bar(index + bar_width, vwd_by['gender'].reindex(['M', 'F', 'O']),
            bar_width, label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.bar(index + 2*bar_width, cpd_by['gender'].reindex(['M', 'F', 'O']),
                bar_width, label='{}-completed'.format(offer))
    plt.grid();
    plt.legend(loc='best');
    plt.title('Net Expense');
    plt.xticks(index + bar_width, ('M', 'F', 'O'));

    plt.subplot(234)
    plt.plot(rcv_avg_by['age_group'], label='{}-received'.format(offer))
    plt.plot(vwd_avg_by['age_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_avg_by['age_group'], label='{}-completed'.format(offer))
    plt.legend(loc='best')
    plt.title('Average Transaction Value');
    plt.grid();

    plt.subplot(235)
    plt.plot(rcv_avg_by['income_group'], label='{}-received'.format(offer))
    plt.plot(vwd_avg_by['income_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_avg_by['income_group'], label='{}-completed'.format(offer))
    plt.legend(loc='best')
    plt.title('Average Transaction Value');
    plt.grid();

    plt.subplot(236)
    plt.bar(index, rcv_avg_by['gender'].reindex(['M', 'F', 'O']), bar_width,
            label='{}-received'.format(offer))
    plt.bar(index + bar_width, vwd_avg_by['gender'].reindex(['M', 'F', 'O']),
            bar_width, label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.bar(index+2*bar_width, cpd_avg_by['gender'].reindex(['M', 'F', 'O']),
                bar_width, label='{}-completed'.format(offer))
    plt.grid();
    plt.legend(loc='best');
    plt.title('Average Transaction Value');
    plt.xticks(index + bar_width, ('M', 'F', 'O'));


def plot_offer_reward_by(customers, offer):
    """ Plot the total and average reward received by all customers that
    completed an offer in a specific group. The plots are separated by age,
    income and gender.
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    """
    rwd_by = dict()
    rwd_avg_by = dict()

    for key in ['age_group', 'income_group', 'gender']:
        key_cpd = '{}_completed'.format(offer)
        key_rwd = '{}_reward'.format(offer)
        offer_cpd = customers[(customers.valid == 1) &
                              (customers[key_cpd] > 0)].groupby(key)
        rwd_by[key] = offer_cpd[key_rwd].mean()
        rwd_avg_by[key] = get_average_reward_by(customers, offer, key)

    plt.figure(figsize=(16, 10))

    plt.subplot(231)
    plt.plot(rwd_by['age_group'], label=offer)
    plt.title('Total Reward ($)');
    plt.legend(loc='best');
    plt.grid();

    plt.subplot(232)
    plt.plot(rwd_by['income_group'], label=offer)
    plt.title('Total Reward ($)');
    plt.legend(loc='best');
    plt.grid();

    index = np.array([0, 1, 2])
    bar_width = 0.3
    plt.subplot(233)
    plt.bar(index, rwd_by['gender'].reindex(['M', 'F', 'O']), bar_width,
            label=offer)
    plt.grid();
    plt.title('Total Reward ($)');
    plt.legend(loc='best');
    plt.xticks(index, ('M', 'F', 'O'));

    plt.subplot(234)
    plt.plot(rwd_avg_by['age_group'], label=offer)
    plt.title('Average Reward');
    plt.legend(loc='best');
    plt.grid();
    ax = plt.gca();
    ymax = rwd_avg_by['age_group'].max() + 1
    ax.set_ylim(0, ymax);

    plt.subplot(235)
    plt.plot(rwd_avg_by['income_group'], label=offer)
    plt.title('Average Reward');
    plt.legend(loc='best');
    plt.grid();
    ax = plt.gca();
    ymax = rwd_avg_by['income_group'].max() + 1
    ax.set_ylim(0, ymax);

    plt.subplot(236)
    plt.bar(index, rwd_avg_by['gender'].reindex(['M', 'F', 'O']), bar_width,
            label=offer)
    plt.grid();
    plt.title('Average Reward');
    plt.legend(loc='best');
    plt.xticks(index, ('M', 'F', 'O'));


def get_net_expense(customers, offer, q=0.5):
    """ Get the net_expense for customers that viewed and completed and offer
    Input:
    - offer: offer of interest
    - q: quantile to be used
    Returns:
    - net_expense median
    """
    flag = (customers['{}_viewed'.format(offer)] > 0)
    flag = flag & (customers.net_expense > 0)
    flag = flag & (customers.total_transactions >= 5)
    if offer not in ['I1', 'I2']:
        flag = flag & (customers['{}_completed'.format(offer)] > 0)
    return customers[flag].net_expense.quantile(q)


def get_most_popular_offers(customers, n_top=2, q=0.5, offers=None):
    """ Sort offers based on the ones that result in the highest net_expense
    Input:
    - customers: dataframe with aggregated data of the offers
    - n_top: number of offers to be returned (default: 2)
    - q: quantile used for sorting
    - offers: list of offers to be sorted
    Returns:
    - sorted list of offers, in descending order according to the median net_expense
    """
    if not offers:
        offers = ['I1', 'I2', 'B1', 'B2', 'B3',
                  'B4', 'D1', 'D2', 'D3', 'D4']
    offers.sort(key=lambda x: get_net_expense(customers, x, q), reverse=True)
    offers_dict = {o: get_net_expense(customers, o, q) for o in offers}
    return offers[:n_top], offers_dict


def get_most_popular_offers_filtered(customers, n_top=2, q=0.5, income=None,
                                     age=None, gender=None):
    """ Sort offers based on the ones that result in the highest net_expense
    Input:
    - customers: dataframe with aggregated data of the offers
    - n_top: number of offers to be returned (default: 2)
    - income_range: tuple with min and max income
    - age_range: tuple with min and max age
    - gender:  'M', 'F', or 'O'
    Returns:
    - sorted list of offers, in descending order according to the
    median net_expense
    """
    flag = (customers.valid == 1)
    if income:
        income_gr = round_income(income)
        if income_gr > 0:
            flag = flag & (customers.income_group == income_gr)
    if age:
        age_gr = round_age(age)
        if age_gr > 0:
            flag = flag & (customers.age_group == age_gr)
    if gender:
        flag = flag & (customers.gender == gender)
    return get_most_popular_offers(customers[flag], n_top, q)