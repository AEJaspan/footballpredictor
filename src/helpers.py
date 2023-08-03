import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  
from scipy.stats import poisson,skellam
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
import sklearn.metrics as metrics
teams = pd.read_csv('../data/teams.csv')

def permutaiton_importance(model, X, y, title, model_name):
  result = permutation_importance(
      model, X, y, n_repeats=10, random_state=42, n_jobs=2
  )
  sorted_idx = result.importances_mean.argsort()

  fig, ax = plt.subplots()
  ax.boxplot(
      result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx]
  )
  ax.set_title(f"Permutation Importances ({title} set)")
  fig.tight_layout()
  plt.savefig(f"../plots/{model_name}_permutation_importance_{title}_set.png")


def plot_roc_curve(clf, X_test, y_test, model_name):
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.savefig(f"../plots/{model_name}_roc_curve.png")

def plot_cm(y, pred, model_name):
  cm=confusion_matrix(y, pred)
  ax= plt.subplot()
  sns.heatmap(cm, annot=True, fmt='g', ax=ax);
  ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
  ax.set_title('Confusion Matrix'); 
  ax.xaxis.set_ticklabels(['negative', 'positive']); ax.yaxis.set_ticklabels(['negative', 'positive']);
  plt.savefig(f"../plots/{model_name}_confusion_matrix.png")
  return cm

def get_total_goals_per_match(df, home_goals_column='HomeScore', away_goals_column = 'AwayScore', season=1):
    df = df[df.SeasonID == season]
    lst_teams = list(set(list(df.HomeTeamID.unique()) + list(df.AwayTeamID.unique())))
    for team in lst_teams:
        df[team] = 0
    for idx, r in df.iterrows():
        df.loc[[idx], [r.HomeTeamID]] += r[home_goals_column]
        df.loc[[idx], [r.AwayTeamID]] += r[away_goals_column]
    return pd.concat([df['MatchID'], df.iloc[:,-28:]], axis=1)


def get_goal_difference_table(df, home_gd_column='NetHomeGoals', season=1):
    df = df[df.SeasonID == season]
    lst_teams = list(set(list(df.HomeTeamID.unique()) + list(df.AwayTeamID.unique())))
    for team in lst_teams:
        df[team] = 0
    for idx, r in df.iterrows():
        df.loc[[idx], [r.HomeTeamID]] += r[home_gd_column]
        df.loc[[idx], [r.AwayTeamID]] += -1*(r[home_gd_column])
    return pd.concat([df['MatchID'], df.iloc[:,-28:]], axis=1).groupby('MatchID').agg('min').cumsum()


def get_rankings_table(df, result_column='result', season=1):
    df = df[df.SeasonID == season]
    lst_teams = list(set(list(df.HomeTeamID.unique()) + list(df.AwayTeamID.unique())))
    for team in lst_teams:
        df[team] = 0
    for idx, r in df.iterrows():
        if r[result_column] == 1:
            df.loc[[idx], [r.HomeTeamID]] = 3
        if r[result_column] == 0:
            df.loc[[idx], [r.AwayTeamID]] = 3
        if r[result_column] == 0:
            df.loc[[idx], [r.HomeTeamID]] = 1
            df.loc[[idx], [r.AwayTeamID]] = 1
    return pd.concat([df['MatchID'], df.iloc[:,-28:]], axis=1).groupby('MatchID').agg('min').cumsum()
    
def get_max_probas_table(df, season=1, odds_columns=['PoissonHomeOdds', 'PoissonAwayOdds', 'PoissonDrawOdds']):
    home, away, draw = odds_columns
    df = df[df.SeasonID == season]
    lst_teams = list(set(list(df.HomeTeamID.unique()) + list(df.AwayTeamID.unique())))
    for team in lst_teams:
        df[team] = 0
    for idx, r in df.iterrows():
        df.loc[[idx], [r.HomeTeamID]] = max(r[home], r[away],  r[draw])
    return df.iloc[:,-28:]


def get_confs_table(max_probas):
    lower_bound = max_probas.mean() - 1.96*max_probas.std()
    upper_bound = max_probas.mean() + 1.96*max_probas.std()
    return pd.DataFrame(lower_bound), pd.DataFrame(upper_bound)


def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_pred = foot_model.get_prediction(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,
                                                            'home':1},
                                                            index=[1]))
    away_goals_pred = foot_model.get_prediction(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,
                                                            'home':0},
                                                            index=[1]))
    home_goals_frame = home_goals_pred.summary_frame(alpha=0.05)
    away_goals_frame = away_goals_pred.summary_frame(alpha=0.05)
    home_goals_avg = home_goals_frame['mean'].values[0]
    away_goals_avg = away_goals_frame['mean'].values[0]
    net_home_goals = -1*(home_goals_avg - away_goals_avg)
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    res = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    home = np.sum(np.tril(res, -1))
    draw = np.sum(np.diag(res))
    away = np.sum(np.triu(res, 1))
    return 1/home, 1/draw, 1/away, net_home_goals # decimal odds

def get_beach_week(rankings):
    i=0
    for idx, r in rankings.iloc[::-1].iterrows():
        winners_points = r.iloc[0]
        i+=1
        r = r.iloc[1:]
        if (r + (i*3) > winners_points).any():
            return (r.head(1).index.item(), idx)


def add_overall_rank(df, rankings_place):
    df = df.merge(rankings_place, left_on=['HomeTeamID', 'SeasonID'], right_on=['TeamID', 'SeasonID'], how='left')
    df = df.rename(columns={'OverallRank':'HomeOverallRank'})
    df = df.merge(rankings_place, left_on=['AwayTeamID', 'SeasonID'], right_on=['TeamID', 'SeasonID'], how='left')
    df = df.rename(columns={'OverallRank':'AwayOverallRank'})
    df = df[['SeasonID', 'MatchID', 'HomeTeamID', 'AwayTeamID', 'HomeScore',
        'HomeShots', 'AwayScore', 'AwayShots', 'HomeYield', 'AwayYield',
        'NetHomeGoals', 'AbsGd', 'Sin_GW', 'Cos_GW', 'result',
        'HomeOverallRank', 'AwayOverallRank']]
    return df

