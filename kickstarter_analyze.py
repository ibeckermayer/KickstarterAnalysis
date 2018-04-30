import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def fun1():
    fig1, ax1 = plt.subplots()
    ax1.scatter(df['usd_goal_real'], df['state_binary'])
    ax1.set_xlabel('Goal (USD)')
    ax1.set_ylabel('State (1=success, 0 = failure/cancelled)')
    ax1.set_title('State vs Goal')
    plt.show(block=False)


# given our graph, this is somewhat surprising. It's obvious that almost all the successes are
# clustered in a certain goal range. Let's choose some bin ranges along the goal axis and plot
# a bar graph of the percentage of campaigns that were successful in each bin in order to see if anything
# stands out

# bin ranges: 0-10, 11-100, 101-1000, 1001-10000, 10001-100000, 100001-1000000

bin_ranges = ((0, 10), (11, 100), (101, 1000), (1001, 10000), (10001, 100000), (100001, 1000000))
def get_att_in_range(df, lower, upper):
    return df[df['usd_goal_real'].between(lower, upper, inclusive=True)].shape[0]

def get_succ_in_range(df, lower, upper):
    return sum(df[df['usd_goal_real'].between(lower, upper, inclusive=True)]['state_binary'])

def get_perc_suc_in_range(df, lower, upper):
    return get_succ_in_range(df, lower, upper)/get_att_in_range(df, lower, upper) * 100
def fun2():
    fig2, ax2 = plt.subplots()
    percent_succ_in_range = [get_perc_suc_in_range(df, br[0], br[1]) for br in bin_ranges]
    b1, b2, b3, b4, b5, b6 = ax2.bar(np.arange(1,7), percent_succ_in_range, color='r', alpha=0.5)
    x_labs = ["\$"+ str(br[0]) + " to \$" +str(br[1]) for br in bin_ranges]
    ax2.set_xticks(np.arange(1,7))
    ax2.set_xticklabels(x_labs, rotation=30, ha='right')
    ax2.set_xlabel('Fundraising Goal')
    ax2.set_ylabel('Percent Successful')
    ax2.set_title('Percentage Successful vs Fundraising Goal')
    plt.tight_layout()
    plt.show(block=False)

# This is somewhat interesting, but may be misleading because it's not obvious
# how many kickstarter attempts were made in each range

# If we instead plot successful over total attempts in each range we get a
# better view of the data

def label_rects(ax, rects, labs):
    i = 0
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                labs[i], ha='center', va='bottom')
        i+=1
def fun3():
    fig3, ax3 = plt.subplots()
    att_in_range = np.array([get_att_in_range(df, br[0], br[1]) for br in bin_ranges])
    succ_in_range = np.array([get_succ_in_range(df, br[0], br[1]) for br in bin_ranges])
    ax3.bar(np.arange(1,7), att_in_range, color='b', alpha=0.5)
    rects = ax3.bar(np.arange(1,7), succ_in_range, color='r', alpha=0.5)
    x_labs = ["\$"+ str(br[0]) + " to \$" +str(br[1]) for br in bin_ranges]
    bar_labs = ["{:.2f}".format(perc) + "%" for perc in percent_succ_in_range]
    ax3.set_xticks(np.arange(1,7))
    ax3.set_xticklabels(x_labs, rotation=30, ha='right')
    ax3.set_xlabel('Fundraising Goal')
    ax3.set_ylabel('Total Attempts and Successful Attempts')
    ax3.set_title('Total Attempts and Successful Attempts vs Fundraising Goal')
    label_rects(ax3, rects, bar_labs)
    plt.show(block=False)

# From that we can see that most of the interesting data lies in kickstarter campaigns
# between $100 and $100,000

# Let's create a histogram to get a higher resolution image of what's going on in that range
def get_df_att_in_range(df, lower, upper):
    return df[df['usd_goal_real'].between(lower, upper, inclusive=True)]

def get_df_succ_in_range(df, lower, upper):
    temp = df[df['usd_goal_real'].between(lower, upper, inclusive=True)]
    return temp.loc[temp['state_binary'] == True]

def label_bins(ax, n_perc, bins, n_succ):
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for perc, x, n in zip(n_perc, bin_centers, n_succ):
        ax.annotate("{:.0f}".format(perc)+"%", xy=(x, n), xycoords='data',
            xytext=(0, 10), textcoords='offset points', va='top', ha='center')

def fun4():
    df_succ = get_df_succ_in_range(df, 100, 100000)
    df_att = get_df_att_in_range(df, 100, 100000)
    fig4, ax4 = plt.subplots()
    hist_bins = [x for x in range(1, 100001, 5000)]
    n_att, bins2, p2 = ax4.hist(df_att['usd_goal_real'], bins=hist_bins, color='b', alpha=0.5)
    n_succ, bins, p = ax4.hist(df_succ['usd_goal_real'], bins=hist_bins, color='r', alpha=0.5)
    n_perc = n_succ/n_att * 100
    ax4.set_xlabel('Fundraising Goal (USD)')
    ax4.set_ylabel('Total Attempts and Successful Attempts')
    ax4.set_title('Total Attempts and Successful Attempts vs Fundraising Goal')
    label_bins(ax4, n_perc, bins, n_succ)

# Data looks as expected, the less you ask for the more likely you are to meet your goal
# There could be more there but we're tight on time so it makes sense to explore
# other realms

def get_att_by_month(df, month):
    return df.loc[df['month_launched'] == month].shape[0]


def get_succ_by_month(df, month):
    temp = df.loc[df['month_launched'] == month]
    return temp.loc[temp['state_binary'] == True].shape[0]

def get_perc_suc_by_month(df, month):
    return get_succ_by_month(df, month)/get_att_by_month(df, month) * 100

def fun5():
    # let's look at success vs month launched for example
    import calendar
    df['month_launched'] = pd.to_datetime(df['launched']).dt.month
    fig5, ax5 = plt.subplots()
    mon_succ = [get_succ_by_month(df, i) for i in range(1, 13)]
    mon_att = [get_att_by_month(df, i) for i in range(1, 13)]
    bar_labs = ["{:.0f}".format(get_perc_suc_by_month(df, i)) + "%" for i in range(1, 13)]
    ax5.bar(np.arange(1,13), mon_att, color='b', alpha=0.5)
    rects = ax5.bar(np.arange(1,13), mon_succ, color='r', alpha=0.5)
    x_labs = [calendar.month_name[i] for i in range(1, 13)]
    ax5.set_xticks(np.arange(1,13))
    ax5.set_xticklabels(x_labs, rotation=30, ha='right')
    ax5.set_xlabel('Month Launched')
    ax5.set_ylabel('Total Attempts and Successful Attempts')
    ax5.set_title('Total Attempts and Successful Attempts vs Month Launched')
    label_rects(ax5, rects, bar_labs)
    plt.show(block=False)

# Cool, so we learned that you don't want to start your campaigns in December
# How about category by category?
def get_att_by_cat(df, cat):
    return df.loc[df['main_category'] == cat].shape[0]


def get_succ_by_cat(df, cat):
    temp = df.loc[df['main_category'] == cat]
    return temp.loc[temp['state_binary'] == True].shape[0]

def get_perc_suc_by_cat(df, cat):
    return get_succ_by_cat(df, cat)/get_att_by_cat(df, cat) * 100

def fun6():
    fig6, ax6 = plt.subplots()
    unique_categories = df['main_category'].unique()
    cat_succ = [get_succ_by_cat(df, i) for i in unique_categories]
    cat_att = [get_att_by_cat(df, i) for i in unique_categories]
    bar_labs = ["{:.0f}".format(get_perc_suc_by_cat(df, i)) + "%" for i in unique_categories]
    ax6.bar(unique_categories, cat_att, color='b', alpha=0.5)
    rects = ax6.bar(np.arange(0, df['main_category'].nunique()), cat_succ, color='r', alpha=0.5)
    x_labs = [i for i in unique_categories]
    ax6.set_xticks(np.arange(0, df['main_category'].nunique()))
    ax6.set_xticklabels(x_labs, rotation=30, ha='right')
    ax6.set_xlabel('Main Category')
    ax6.set_ylabel('Total Attempts and Successful Attempts')
    ax6.set_title('Total Attempts and Successful Attempts vs Main Category')
    label_rects(ax6, rects, bar_labs)
    plt.show(block=False)

# average goal by category
def get_avg_goal_by_cat(df, cat):
    return df[df['main_category'] == cat]['usd_goal_real'].mean()

def fun7():
    fig7, ax7 = plt.subplots()
    unique_categories = df['main_category'].unique()
    avg_goals = [get_avg_goal_by_cat(df, cat) for cat in unique_categories]
    avg_goals_succ = [get_avg_goal_by_cat(df_succ, cat) for cat in unique_categories]
    avg_goals_fail = [get_avg_goal_by_cat(df_fail, cat) for cat in unique_categories]
    x_tix = np.array([x for x in range(0, len(unique_categories))])
    thick = 0.25
    ax7.bar(x_tix-thick, avg_goals_succ, width=thick, color='g', alpha=0.5, label='successful')
    ax7.bar(x_tix, avg_goals_fail, width=thick, color='r', alpha=0.5, label='failed/cancelled')
    ax7.bar(x_tix+thick, avg_goals, width=thick, color='b', alpha=0.5, label='together')
    ax7.set_xlabel('Main Category')
    ax7.set_ylabel('Goal (USD)')
    ax7.set_title('Average Goal of Successful vs. Failed/Cancelled by Category')
    ax7.set_xticks(np.arange(0, df['main_category'].nunique()))
    ax7.legend()
    ax7.set_xticklabels(unique_categories, rotation=30, ha='right')
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

if __name__ == "__main__":
    df = pd.read_csv("Data/ks-projects-201801.csv")
    # Choose a topic that may be interesting to study:
    # Let's get a visual on successes and failures based on goal size
    df['state_binary'] = df['state'].apply(lambda s: True if s == "successful" else False)
    df_succ = df[df['state_binary'] == True]
    df_fail = df[df['state_binary'] == False]
    fun7()
    plt.show()

