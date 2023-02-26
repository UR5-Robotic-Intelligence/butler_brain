import pandas as pd
import pickle
import glob
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np


experiment_names = ['from_description_and_request_use_experience_use_type',
                    'from_description_and_request_use_experience',
                    'from_description_and_request_use_type',
                    'from_description_and_request',
                    'from_description_use_experience_use_type',
                    'from_description_use_experience',
                    'from_description_use_type',
                    'from_description',
                    'from_request_use_experience_use_type',
                    'from_request_use_experience',
                    'from_request_use_type',
                    'from_request']
figure_titles = ['Predict From Description&Req (experience and type)',
                 'Predict From Description&Req (experience)',
                 'Predict From Description&Req (type)',
                 'Predict From Description&Req',
                 'Predict From Description (experience and type)',
                 'Predict From Description (experience)',
                 'Predict From Description (type)',
                 'Predict From Description',
                 'Predict From Request (experience and type)',
                 'Predict From Request (experience)',
                 'Predict From Request (type)',
                 'Predict From Request']
experiment_mistakes = []

for i, experiment_name in enumerate(experiment_names):
    experiment_file_names_list = glob.glob('/home/bass/experiments_data_{}_at_*.pkl'.format(experiment_name))
    if len(experiment_file_names_list) == 0:
        continue
    all_exp_dt_str = [file_name.split('_at_')[1].split('.')[0] for file_name in experiment_file_names_list]
    all_exp_dt_str = list(filter(lambda x: dt.strptime(x, '%Y%m%d-%H%M%S'), all_exp_dt_str))
    exp_file_name = '/home/bass/experiments_data_{}_at_{}.pkl'.format(experiment_name, max(all_exp_dt_str))
    with open(exp_file_name, 'rb') as f:
        experiment = pickle.load(f)
    for k, v in experiment.items():
        if 'n_mistakes' not in v:
            v['n_mistakes'] = 0
    print(experiment_name)
    print(sum([v['n_mistakes'] for v in experiment.values()]))
    # experiment['experiment_name'] = experiment_name
    # experiment['file_name'] = exp_file_name
    # number_of_steps = sum([len(v) for act in experiment.values() for k, v in act.items() if k == 'steps'])
    mistakes = [v['n_mistakes'] for v in experiment.values()]
    cs_bert_score = [v['cs_bert_score'] for v in experiment.values() if 'cs_bert_score' in v]
    if len(cs_bert_score) > 0:
      print("avg_cs_bert_score = ", sum(cs_bert_score)/len(cs_bert_score))
    a_bert_score = [v['bert_score'] for v in experiment.values() if 'bert_score' in v]
    if len(a_bert_score) > 0:
      a_bert_score  = a_bert_score[0]
      print("avg_bert_score = ", sum(a_bert_score)/len(a_bert_score))
    # fuzzy_scores = [v['fuzzy_score'] for v in experiment.values()]
    # print("avg_fuzzy_scire = ", sum(fuzzy_scores)/len(fuzzy_scores))
    activities = [k for k, v in experiment.items()]
    for k, v in experiment.items():
        for k1, v1 in v.items():
          if type(v1) == list:
            if type(v1[0]) in [list, tuple]:
              experiment[k][k1] = [str(list(vv)) for vv in v1]
              experiment[k][k1] = "\n".join(experiment[k][k1])
            else:
              experiment[k][k1] = str(v1)
    dict_of_dataframes = [v for k, v in experiment.items()]
    df = pd.DataFrame(dict_of_dataframes, index=experiment.keys())
    # print(df)
    X_axis = np.arange(len(activities))

    # plt.bar(X_axis - 0.6 + 0.4*i, np.array(mistakes), 0.4, label = figure_titles[i])
    # df.to_excel('experiment_{}.xlsx'.format(experiment_name), index=True)
    # if i == 3:
    #   break
    
    if i >= 8:
      plt.bar(X_axis - 0.6 + 0.4*(i-8), np.array(mistakes), 0.4, label = figure_titles[i])
      df.to_excel('experiment_{}.xlsx'.format(experiment_name), index=True)

plt.xticks(rotation=30)
plt.xticks(X_axis, activities)
plt.title("Mistakes/Activity")
plt.xlabel('Activity Name')
plt.ylabel('Number of Mistakes')
plt.legend()
plt.show()