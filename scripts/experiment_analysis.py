import pandas as pd
import pickle
import glob
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from copy import deepcopy


experiment_names = ['from_description_and_request_use_experience',
                    'from_description_use_experience',
                    'from_request_use_experience',
                    'from_description_and_request_use_experience_use_type',
                    'from_description_use_experience_use_type',
                    'from_request_use_experience_use_type',
                    'from_description_and_request',
                    'from_description',            
                    'from_request',
                    'from_description_and_request_use_type',
                    'from_description_use_type',
                    'from_request_use_type']
# experiment_names = ['from_description_and_request_use_experience',
#                     'from_description_use_experience',
#                     'from_request_use_experience',
#                     'from_description_and_request_use_experience_use_type',
#                     'from_description_use_experience_use_type',
#                     'from_request_use_experience_use_type',
#                     'from_description_and_request',
#                     'from_description',            
#                     'from_request',
#                     'from_description_and_request_use_type',
#                     'from_description_use_type',
#                     'from_request_use_type']
figure_titles = ['(experience and type)',
                 '(experience)',
                 '(type)',
                 'No Additional Information',
                 '(experience and type)',
                 '(experience)',
                 '(type)',
                 'No Additional Information',
                 '(experience and type)',
                 '(experience)',
                 '(type)',
                 'No Additional Information']

def get_experiment_data(data_dir="/home/bass/experiments/with both/2", sort_keys=False, sort_by='fuzzy_score',last_n_act=None, test=False):
  experiments_data = {}
  avg_scores = {}
  for i, core_experiment_name in enumerate(experiment_names):
      for j, cond in enumerate(['_use_reasoning', '']):
        experiment_name = core_experiment_name + cond
        
        experiment_file_names_list = glob.glob(os.path.join(data_dir,f'experiments_data_{experiment_name}_at_*.pkl'))
        if len(experiment_file_names_list) == 0:
            continue
        all_exp_dt_str = [file_name.split('_at_')[1].split('.')[0] for file_name in experiment_file_names_list]
        all_exp_dt_str = list(filter(lambda x: dt.strptime(x, '%Y%m%d-%H%M%S'), all_exp_dt_str))
        exp_file_name = os.path.join(data_dir,f'experiments_data_{experiment_name}_at_{max(all_exp_dt_str)}.pkl')
        with open(exp_file_name, 'rb') as f:
            experiment = pickle.load(f)
        n = 0
        if test:
          new_experiment_name = experiment_name + '_0'
        else:
          new_experiment_name = experiment_name
        avg_scores[new_experiment_name] = {'fuzzy_score':0,'n_mistakes':0,'bert_score':0, 'prompt_length':0}
        exp_cp = deepcopy(experiment)
        test_idx = 0
        prev_test_idx = 0
        new_exp = {}
        for e_i, (k, v) in enumerate(exp_cp.items()):
            if last_n_act is not None:
              if e_i < len(exp_cp) - last_n_act:
                del experiment[k]
                continue
            if test:
              if 'test' not in k:
                del experiment[k]
                continue
              new_exp[k] = v
              test_idx = int(k.split('_')[-1])
              if test_idx != prev_test_idx:
                experiments_data[new_experiment_name] = new_exp
                new_exp = {}
                new_experiment_name = experiment_name + f'_{test_idx}'
                prev_test_idx = test_idx
                avg_scores[new_experiment_name] = {'fuzzy_score':0,'n_mistakes':0,'bert_score':0, 'prompt_length':0}
                n = 0
            if 'n_mistakes' not in v:
                v['n_mistakes'] = 0
            v['n_mistakes'] /= len(v['rob_commands'])
            avg_scores[new_experiment_name]['fuzzy_score'] += v['fuzzy_score']
            avg_scores[new_experiment_name]['n_mistakes'] += v['n_mistakes']
            avg_scores[new_experiment_name]['bert_score'] += v['bert_score'][-1]
            avg_scores[new_experiment_name]['prompt_length'] += sum([len(v1) for k1, v1 in v.items() if k1.endswith('prompt')]) / len(v['rob_commands'])
            if 'from_request' in new_experiment_name:
              for k1, v1 in v.items():
                if k1.endswith('prompt'):
                  print(f'{new_experiment_name}, {k}: {k1}: {v1}')
                if k1.endswith('res'):
                  print(f'{new_experiment_name}, {k}: {k1}: {v1}')
              print(v['reasoning_correction'])
              print(v['steps'])
              print(v['n_mistakes'])
              print(v['fuzzy_score'])
              print('------------------')
            n += 1
        for e_i, e_name in enumerate(avg_scores.keys()):
          for k, v in avg_scores[e_name].items():
              avg_scores[e_name][k] /= n
        if test:
          experiments_data[new_experiment_name] = new_exp
        else:
          experiments_data[new_experiment_name] = experiment
  if sort_keys:
    keys = list(experiments_data.keys())
    reverse = True if sort_by == 'n_mistakes' else False
    experiments_data = {k: experiments_data[k] for k in sorted(keys,key=lambda x: avg_scores[x][sort_by],reverse=reverse)}
  return experiments_data

def filter_experiments(experiments_data, all_of=[''], any_of=[''], none_of=[]):
  experiment_cond = lambda x: all([a in x for a in all_of]) and all([a not in x for a in none_of]) and any([a in x for a in any_of])
  filtered_experiments_data = {}
  for experiment_name, experiment_data in experiments_data.items():
    if experiment_cond(experiment_name):
      filtered_experiments_data[experiment_name] = experiment_data
  return filtered_experiments_data


def dict_to_excel(experiment, experiment_name):
  for k, v in experiment.items():
      for k1, v1 in v.items():
        if type(v1) == list:
          if len(v1) > 0:
            if type(v1[0]) in [list, tuple]:
              experiment[k][k1] = [str(list(vv)) for vv in v1]
              experiment[k][k1] = "\n".join(experiment[k][k1])
          else:
            experiment[k][k1] = str(v1)
  dict_of_dataframes = [v for k, v in experiment.items()]
  df = pd.DataFrame(dict_of_dataframes, index=experiment.keys())
  # print(df)
  df.to_excel('experiment_{}.xlsx'.format(experiment_name), index=True)

def get_fig_title_from_exp_name(exp_name, not_added):
  if 'from_description_and_request' in exp_name:
    label = 'Description & Request'
  elif 'from_request' in exp_name:
    label = 'Request'
  else:
    label = 'Description'
  if 'experience' in exp_name and 'experience' not in not_added:
    label += ' & Experience'
  if 'reasoning' in exp_name and 'reasoning' not in not_added:
    label += ' & Reasoning'
  if 'type' in exp_name and 'type' not in not_added:
    label += ' (Type Filtered)'
  return label

def analyze_experiments(bar_plot=True, data_dir=['/home/bass/experiments/with both/new_born_agent/2'], data_names=['new_born_agent'],
                        must_have=[''], none_of=[], input_type='', use_activities=False, last_n_act=None,
                        score_type='n_mistakes', compare_by='reasoning',
                        plot=True, sort_keys=False, sort_by=None, test=False, use_regions=False):
  
  if input_type == 'description':
    none_of += ['request']
  elif input_type == 'request':
    none_of += ['description']
  else:
    none_of += []
  
  all_exps = []
  all_exps_names = []
  if type(data_dir) == str:
    data_dir = [data_dir]
  if type(data_names) == str:
    data_names = [data_names]
  for name, ddir in zip(data_names, data_dir):
    experiments_data = get_experiment_data(sort_by=sort_by, sort_keys=sort_keys, data_dir=ddir, last_n_act=last_n_act, test=test)
    experiments = filter_experiments(experiments_data, all_of=[f'from_{input_type}', compare_by]+must_have, none_of=none_of)
    all_exps.append(experiments)
    all_exps_names.append(name)
    experiments = filter_experiments(experiments_data, all_of=[f'from_{input_type}']+must_have, none_of=none_of+[compare_by])
    all_exps.append(experiments)
    all_exps_names.append(name)
  
  # Restructure data
  if use_activities:
    all_exps_cp = deepcopy(all_exps)
    for i, exps in enumerate(all_exps_cp):
      new_experiment_dict = {}
      for exp_name, exp_data in exps.items():
        for act_i, (act_name, act_data) in enumerate(exp_data.items()):
          if act_name not in new_experiment_dict:
            new_experiment_dict[act_name] = {}
          new_experiment_dict[act_name].update({exp_name:act_data})
      all_exps[i] = new_experiment_dict
  
  
  # experiment_1 vs experiment_2 in terms of score
  experiment_scores_dict = {}
  experiment_names = []
  if type(score_type) not in [list, tuple]:
    score_type = [score_type]
  n = 1
  for i, exps in enumerate(all_exps):
    experiment_names.append([])
    prev_scores = []
    for experiment_name, experiment in exps.items():
      if '0' in experiment_name:
        prev_scores = []
      # scores_dict = {score_type_:v[score_type_] for v in experiment.values() for score_type_ in score_type}
      for score_type_ in score_type:
        scores = [v[score_type_] for v in experiment.values() if score_type_ in v]
        if len(scores) > 0:
          if score_type_ == 'n_mistakes':
            print(f"sum_{score_type_} = ", sum(scores))
          else:
            print(f"mean_{score_type_} = ", np.mean(scores))
        # scores = sum(scores) if score_type_ == 'n_mistakes' else np.mean(scores)
        new_scores = np.mean(scores)
        scores = new_scores
        if len(prev_scores) >= n-1:
          all_scores = prev_scores+[scores]
          scores = np.mean(all_scores[-n:])
        prev_scores.append(new_scores)
        if score_type_ not in experiment_scores_dict:
          experiment_scores_dict[score_type_] = {j:[] for j in range(len(all_exps))}
        experiment_scores_dict[score_type_][i].append(scores)
      experiment_names[i].append(experiment_name)
  
  if not plot:
    return experiment_names, experiment_scores_dict

  fs = 20
  tfs = 15
  # rotation = 0 if input_type != '' else 90
  rotation = 90
  comp_by_title = ['']
  if compare_by != '':
    if compare_by == 'experience':
      comp_by_title = ['Augmented Prompt', 'Fixed Prompt']
    else:
      c = compare_by.capitalize()
      comp_by_title = [f'With {c}', f'No {c}']
  for e_i in range(len(all_exps_names)):
      all_exps_names[e_i] += ' ' + comp_by_title[e_i%2]

  fig_titles = all_exps_names
  new_fig_titles = []
  if len(score_type) > 1:
    for score_type_ in score_type:
      for i,ft in enumerate(fig_titles):
        new_fig_titles.append(f'{ft} ({score_type_})')
  fig_titles.extend(new_fig_titles)
  
  color = ['blue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
  if compare_by == 'experience':
    title = 'Pompt Augmentation Vs Fixed Prompt'
    if 'reasoning' in must_have:
      title += '\n(Both have Fact Correction & Knowledge Gain)'
  else:
    c = compare_by.capitalize()
    title = f'{c} vs No {c}' #if len(all_exps) == 2 else 'Effect of Additional Information (All have reasoning)'
  if test:
    x_label = 'Test Index (A test after each new activity)'
  elif not use_activities:
    x_label = f'Additional Information (All have activity {input_type})' if input_type != '' else 'Additional Information'
  else:
    x_label = f'Activity (All have activity {input_type})' if input_type != '' else 'Activity'
  x_label += f' (sorted by {sort_by})' if sort_keys else ''
  x_ticks = []
  if input_type != '':
    x_ticks = [name.replace(f'from_{input_type}_use_','').replace('use','and').replace(f'from_{input_type}','nothing') for name in all_exps[1].keys()]
  elif use_activities:
    for name in all_exps[0].keys():
      name = name.replace('Making-', '')
      x_ticks.append(name)
  elif use_regions:
    if test:
      for name in all_exps[0].keys():
        x_ticks.append(name.split('_')[-1])
  else:
    p4 = 'from_description'
    p6 = 'from_request'
    p2 = 'from_description_and_request'
    p1, p3, p5 = p2 + '_use', p4 + '_use', p6 + '_use'
    for name in all_exps[0].keys():
      name = re.sub(f'({p1})|({p2})','d&r',name)
      name = re.sub(f'({p3})|({p4})','d',name)
      name = re.sub(f'({p5})|({p6})','r',name)
      name = name.replace(f'_use_{compare_by}','').replace('_use_','_').replace(f'_{compare_by}','')
      for nn in none_of:
        name = name.replace(f'_{nn}','')
      for mh in must_have:
        if mh != '':
          name = name.replace(f'_{mh}','')
      name = name.replace('experience','exp')
      x_ticks.append(name)
  name = f'{title}_using_{score_type}'
  name += f'_and_{input_type}' if input_type != '' else ''
  # name += f'_comparing_by_{compare_by}' if compare_by != '' else ''
  name += f'_activities' if use_activities else ''
  name += f'_sorted_by_{sort_by}' if sort_keys else ''
  name += f'_have_{must_have}' if must_have != [''] else ''
  name += f'_without_{none_of}' if none_of != [] else ''
  name += '.png'
  ax = plt.gca()
  ax.set_xlabel(f'{x_label}', fontsize=fs, fontweight='bold')
  if test:
    n_test = max(list(map(lambda x: int(x.split('_')[-1]), experiment_names[0]))) + 1
    exp_prefix = list(dict.fromkeys(map(lambda x: '_'.join(x.split('_')[:-1]), experiment_names[0])))
  if use_regions:
    # regions = 'additional_info_types'
    if test:
      regions = 'test_input_types'
    else:
      regions = 'input_types'
    alpha = 0.3
    if regions == 'test_input_types':
      for exp_i, exp_name in enumerate(exp_prefix):
        label = get_fig_title_from_exp_name(exp_name,[compare_by]+must_have+none_of)
        x1 = exp_i*n_test
        x2 = x1 + n_test - 1
        ax.axvspan(x1, x2, color=color[exp_i], alpha=alpha, label=label) 
    elif regions == 'input_types':
      ax.axvspan(0, 4, color='red', alpha=alpha, label='Description & Request')
      ax.axvspan(4, 8, color='blue', alpha=alpha, label='Description')
      ax.axvspan(8, 12, color='green', alpha=alpha, label='Request')
    elif regions == 'additional_info_types':
      ax.axvspan(-0.5, 2.5, color='red', alpha=alpha, label='Experience')
      ax.axvspan(2.5, 5.5, color='blue', alpha=alpha, label='Experience & Type')
      ax.axvspan(5.5, 8.5, color='green', alpha=alpha, label='No Additional Info')
      ax.axvspan(8.5, 11.5, color='orange', alpha=alpha, label='Type')
  l = 0
  for sc_i, (score_type_, experiment_scores) in enumerate(experiment_scores_dict.items()):
    plt.yticks(fontsize=tfs, fontweight='bold')
    x_axis = []
    h = []
    reverse = False if score_type_ == 'fuzzy_score' else True
    if sc_i == 1:
      ax = ax.twinx()
    y_label = 'Number of Mistakes' if score_type_ == 'n_mistakes' else 'Fuzzy Score'
    if bar_plot:
      bars_per_tick = len(all_exps) # reasoning vs no reasoning
      w = 0.5
      shift = int(w*bars_per_tick + w*2)
      z = (bars_per_tick)/2.0 - w
      if bar_plot:
        for i in range(len(all_exps)):
          x_axis.append(np.arange(len(experiment_scores[i])*shift, step=shift))
          h.append(np.array(experiment_scores[i]))
      if score_type == 'fuzzy_score':
        bottom = np.mean(np.concatenate(h))
        bottom = np.round(bottom, 1)
        ax.axhline(bottom, linestyle='--', color='k') # horizontal lines
        yt = ax.get_yticks()
        yt=np.append(yt,bottom)
        ax.set_yticks(yt)
        ax.set_yticklabels(yt)
      
      for i, (xi, hi) in enumerate(zip(x_axis, h)):
        plt.bar(xi - z*w + i*w, hi-bottom, w, bottom=bottom, label=fig_titles[i], color=color[i+sc_i])
        
      x_axis = x_axis[0]
      ax.set_ylabel(f'{y_label}', fontsize=fs, fontweight='bold')
      if sc_i == 0:
        ax.set_xticks(x_axis, x_ticks, rotation=rotation, fontsize=tfs, fontweight='bold')
    else:
      ax.set_ylabel(f"{y_label}",
            color=color[sc_i] if len(score_type) > 1 else 'black',
            fontsize=fs, fontweight='bold')

      for i in range(len(all_exps)):
        x_order = np.arange(len(experiment_names[i]))
        y = np.array(experiment_scores[i])
        marker = 'o' if sc_i == 0 else 'x'
        linestyle = '-' if i % 2 == 0 else '--'
        if sc_i == 0 and i == 0:
          ax.set_xticks(x_order, x_ticks, rotation=rotation, fontsize=tfs, fontweight='bold')
        if test:
          label = fig_titles[l]
          for exp_i, exp_name in enumerate(exp_prefix):
            x1 = exp_i*n_test
            x2 = x1 + n_test - 1
            ax.plot(x_order[x1:x2+1], y[x1:x2+1], label=label, color=color[l], marker=marker, markersize=5, linewidth=2, linestyle=linestyle)
            label = None
        else:
          ax.plot(x_order, y, label=fig_titles[l], color=color[l], marker=marker, markersize=5, linewidth=2, linestyle=linestyle)
        l+=1
      if len(score_type) > 1:
        loc = (-0.03, 1.02) if sc_i == 0 else (0.81, 1.02)
      else:
        loc = (-0.03, 1.09)
      ax.legend(fontsize=tfs, loc=loc, ncol=2)
  plt.title(f"{title}", fontsize=fs, fontweight='bold')
  # plt.legend(fontsize=tfs)
  plt.grid()
  figure = plt.gcf() # get current figure
  figure.set_size_inches(15, 10)
  if len(data_dir) > 1:
    new_data_dir = ""
    all_len = [len(d) for d in data_dir]
    min_len = min(all_len)
    for i, c in enumerate(data_dir[0]):
      if i == min_len:
        break
      same_char = c == data_dir[1][i]
      for ddir in data_dir[2:]:
        same_char = same_char and c == ddir[i]
      if same_char:
        new_data_dir += c
      else:
        break
  else:
    new_data_dir = data_dir[0]
  plt.savefig(os.path.join(new_data_dir,name),dpi=300,bbox_inches='tight')
  plt.show()

if __name__ == '__main__':
  data_dir = []
  data_names = []
  # data_dir.append('/home/bass/experiments/on_test_set/experienced_agent/1')
  # data_names.append('Experienced')
  # data_dir.append('/home/bass/experiments/on_test_set/new_born_agent/2')
  # data_names.append('New Born')
  # data_dir.append('/home/bass/experiments/with both/new_born_agent/4')
  # data_names.append('New Born')
  # data_dir.append('/home/bass/experiments/on_test_set/new_born_agent/2')
  # data_names.append('New Born')
  # data_dir.append('/home/bass/experiments/on_test_set/experienced_agent/2')
  # data_names.append('Experienced')
  data_dir.append('/home/bass/experiments/continuous_testing/new_born_agent/2')
  data_names.append('')
  input_type = ''
  # input_type = 'request'
  # input_type = 'description'
  # input_type = 'description_and_request'
  # score_type = ['n_mistakes', 'fuzzy_score']
  score_type = 'fuzzy_score'
  # score_type = 'n_mistakes'
  compare_by = ''
  compare_by = 'experience'
  # compare_by = 'reasoning'
  must_have = ['']
  must_have.append('reasoning')
  # must_have.append('experience')
  none_of = []
  # none_of.append('experience')
  # none_of.append('reasoning')
  none_of.append('from_request')
  analyze_experiments(data_dir=data_dir, data_names=data_names, input_type=input_type, score_type=score_type,
                      use_activities=False, compare_by=compare_by, must_have=must_have, none_of=none_of,
                      plot=True, bar_plot=False, sort_keys=False, sort_by='prompt_length', last_n_act=None,
                      test=True, use_regions=True)
