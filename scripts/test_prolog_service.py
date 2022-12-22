#!/usr/bin/env python

import rospy
from rosprolog_client import Prolog
from owl_test.robot_activities import prepareADrink, prepareAMeal, bringObject
from owl_test.utils import text_to_speech, text_to_keywords, speach_to_text, get_top_matching_candidate, cos_sim, write_embeddings
from owl_test.ontology_utils import OntologyUtils
import fuzzywuzzy.fuzz as fuzz
import argparse
import os
from sentence_transformers import SentenceTransformer
import torchtext.vocab as vocab
import torch
import time
import pickle

if __name__ == "__main__":
  rospy.init_node('test_rosprolog')
  ou = OntologyUtils()
  prolog = ou.prolog
  ns = ou.ns
  args = argparse.ArgumentParser(description='Test the rosprolog service')
  args.add_argument('-v', '--verbose', action='store_true', help='Print the explanations and intermediate results')
  args.add_argument('-s', '--save_embeddings', action='store_true', help='Save the query embeddings to a file')
  args.add_argument('-l', '--load_embeddings', action='store_true', help='Load the query embeddings from a file', default=True)
  args.add_argument('-lq', '--load_query_results', action='store_true', help='Load the query results from a file',default=True)
  args.add_argument('-sq', '--save_query_results', action='store_true', help='Save the query results to a file')
  verbose = args.parse_args().verbose
  load_embeddings = args.parse_args().load_embeddings
  save_embeddings = args.parse_args().save_embeddings
  load_query_results = args.parse_args().load_query_results
  save_query_results = args.parse_args().save_query_results
  model = SentenceTransformer('bert-base-nli-mean-tokens')
  data_path = os.path.join(os.getcwd(), 'uji_butler_wokring_memory.txt')
  query_results_path = os.path.join(os.getcwd(), 'query_results.pkl')

  # parse the output of GPT-3
  # output_of_gpt3 = "1.cup\n2.coffee"
  # output_of_gpt3 = "1.cup\n2.tea"
  # output_of_gpt3 = input("Enter the output of GPT-3: ")
  # user_request = input("Enter your request: ")
  
  text_to_speech("Press enter to start the test", verbose=verbose)
  input()
  text_to_speech("Please say your request:", verbose=verbose)
  user_request = speach_to_text(verbose=verbose)
  output_of_gpt3 = text_to_keywords(user_request.strip(), verbose=verbose)
  if verbose:
    print(output_of_gpt3)
  output_components = output_of_gpt3.strip().split("\n")
  if verbose:
    print(output_components)
  output_components = [x.split(".")[-1].strip() for x in output_components]
  if verbose:
    print(output_components)
  
  # output_components = ['chocolate', 'milk']
  # output_components = ['tea', 'beverage']
  # output_components = ['drinking']
  
  comp_enc = model.encode([component.lower() for component in output_components], device='cuda')
  
  ##########################################################################################################################################
  # This is like the working memory of the robot.                                                                                          #
  # A more realistic implementation would be to detect the objects in the environment and update the working memory.                       #
  # according to the objects that are detected, and the environment, the robot will generate a different query.                            #
  # For example, if the robot is in the kitchen, it will generate a query that includes the activities that are performed in the kitchen.  #
  ##########################################################################################################################################
  
  if load_query_results:
    with open(query_results_path, 'rb') as f:
      query_results = pickle.load(f)
  
  else:
    # Find the activity that outputs the components, and the name of the components in the ontology.
    # The output of GPT-3 is not necessarily the same as the name of the components in the ontology.
    # For example, GPT-3 may output "coffee", but the name of the coffee in the ontology is "Coffee-Beverage".
    # so we need to find the activity that outputs an object the has the word "coffee" in its name.
    # we first find all the activities that create a final product.
    # then we search for the activity that outputs an object that has the word "coffee" in its name.
    event_that_has_outputs_created = "is_restriction(A, some(" + ns + "outputsCreated\", C)), subclass_of(B, A), subclass_of(C, Sc)"
    that_are_subclass_of_food_or_drink = "subclass_of(Sc, " + ns + "FoodOrDrink\"), subclass_of(Other, Sc), \\+is_restriction(_, some(" + ns + "outputsCreated\", Other))"
    has_objects_acted_on = "is_restriction(D, some(" + ns + "objectActedOn\", E)), subclass_of(B, D), subclass_of(B, Sb), \\+subclass_of(Sb, D)"
    is_subclass_of_preparing_food_or_drink = "subclass_of(Sb,  " + ns + "PreparingFoodOrDrink\")"
    # non_activity_objects = "(subclass_of(NAO, " + ns + "FoodOrDrinkOrIngredient\"))" #, \\+(subclass_of(NAO, B); subclass_of(NAO, Sb)))"
    
    query_string = "(" + event_that_has_outputs_created + ", " + that_are_subclass_of_food_or_drink + \
        ", " + has_objects_acted_on + ", " + is_subclass_of_preparing_food_or_drink + ")." #; " + non_activity_objects
    # query = prolog.query(query_string)
    query_results = prolog.all_solutions(query_string)
    if save_query_results:
      with open(query_results_path, 'wb') as f:
        pickle.dump(query_results, f)
  # votes represent the number of components from the output of GPT-3 that appear in the name of the objects or activities in the ontology.
  # the activity with the highest number of votes is the activity that we are looking for.
  activities = {}
  super_activities = {}
  super_objects = {}
  other_objects = {}
  is_component_used = {component:0 for component in output_components}
  sim_thresh = 0.66
  data = {}
  encoded_before = {}
  if load_embeddings:
    trained_embeddings = vocab.Vectors(name = data_path,
                                   cache = 'custom_embeddings',
                                   unk_init = torch.Tensor.normal_)
  
  for solution in query_results:
    encodings = []
    # print(solution)
    # remove the namespace from the name of the activity
    B, C, E = solution['B'].split('#')[-1], solution['C'].split('#')[-1],  solution['E'].split('#')[-1]
    print("B: {}, C: {}, E: {}".format(B, C, E))
    Sc, Sb = solution['Sc'].split('#')[-1], solution['Sb'].split('#')[-1]
    Other = solution['Other'].split('#')[-1]
    if not load_embeddings:
      # encode the activity and the objects
      to_encode = [B.lower(), C.lower(), E.lower(), Sc.lower(), Sb.lower(), Other.lower()]
      encodings = model.encode(to_encode, device='cuda:0')
    else:
      tokens = [B, C, E, Sc, Sb, Other]
      for token in tokens:
        if token in encoded_before:
          encodings.append(encoded_before[token])
        else:
          token_idx = trained_embeddings.stoi[token]
          encodings.append(trained_embeddings.vectors[token_idx].numpy())
      # encodings = trained_embeddings.get_vecs_by_tokens(tokens=tokens,lower_case_backup=True).numpy()

    for token, enc in zip(tokens, encodings):
      if token not in encoded_before:
        encoded_before[token] = enc

    B_enc, C_enc, E_enc = encodings[0], encodings[1], encodings[2]
    Sc_enc, Sb_enc = encodings[3], encodings[4]
    Other_enc = encodings[5]
    
    if save_embeddings:
      data[B], data[C], data[E] = B_enc, C_enc, E_enc
      data[Sc], data[Sb] = Sc_enc, Sb_enc
      data[Other] = Other_enc
    
    for component, enc in zip(output_components, comp_enc):
      act_sim = cos_sim(enc, B_enc)
      output_sim = cos_sim(enc, C_enc)
      acted_on_sim = cos_sim(enc, E_enc)
      sup_act_sim = cos_sim(enc, Sb_enc)
      sup_obj_sim = cos_sim(enc, Sc_enc)
      other_obj_sim = cos_sim(enc, Other_enc)
      if (act_sim >= sim_thresh) or (output_sim >= sim_thresh) or (acted_on_sim >= sim_thresh):
        is_component_used[component] = 1
        if B not in activities.keys():
          activities[B] = {'output':C,\
                           'sim':[max(act_sim, output_sim, acted_on_sim)],\
                           'objectActedOn':[E],\
                           'level':'activity',\
                           'components':[component],\
                           'votes':1,\
                           'super_activities':[Sb],\
                           'super_objects':[Sc]}
        else:
          if E not in activities[B]['objectActedOn']:
            activities[B]['objectActedOn'].append(E)
          if component not in activities[B]['components']:
            activities[B]['components'].append(component)
            activities[B]['votes'] += 1
            activities[B]['sim'].append(max(act_sim, output_sim, acted_on_sim))
          if Sb not in activities[B]['super_activities']:
            activities[B]['super_activities'].append(Sb)
          if Sc not in activities[B]['super_objects']:
            activities[B]['super_objects'].append(Sc)
      if sup_act_sim >= sim_thresh:
        is_component_used[component] = 1
        if Sb not in super_activities.keys():
          super_activities[Sb] = {
              'level': 'superActivity', 'components': [component], 'votes': 1, 'sim': [sup_act_sim]}
        elif component not in super_activities[Sb]['components']:
          super_activities[Sb]['components'].append(component)
          super_activities[Sb]['votes'] += 1
          super_activities[Sb]['sim'].append(sup_act_sim)
      if sup_obj_sim >= sim_thresh:
        is_component_used[component] = 1
        if Sc not in super_objects.keys():
          super_objects[Sc] = {'level': 'superObject',
                                'components': [component], 'votes': 1, 'sim': [sup_obj_sim]}
        elif component not in super_objects[Sc]['components']:
          super_objects[Sc]['components'].append(component)
          super_objects[Sc]['votes'] += 1
          super_objects[Sc]['sim'].append(sup_obj_sim)
        # print("Found activity {} that outputs {} and acts on {}".format(B, C, E))
      if other_obj_sim >= sim_thresh:
        is_component_used[component] = 1
        if Other not in other_objects.keys():
          other_objects[Other] = {'level': 'other',
                                'components': [component], 'votes': 1, 'sim': [other_obj_sim], 'super_object':Sc}
        elif component not in other_objects[Other]['components']:
          other_objects[Other]['components'].append(component)
          other_objects[Other]['votes'] += 1
          other_objects[Other]['sim'].append(other_obj_sim)

  if save_embeddings:
    write_embeddings(data_path, list(data.values()), list(data.keys()))
  
  # TODO: handle the case where the output of GPT-3 does not match any of the components in the ontology.
  # for component, used in is_component_used.items():
  #   if used == 0:
  #     text_to_speech("I could not find a Food or Drink that matches your request for {}".format(component),verbose=verbose)
  #     exit()
  
  activities_list = sorted([(key, val) for key, val in activities.items()], key=lambda x: x[1]['votes'], reverse=True)
  super_activities_list = sorted([(key, val) for key, val in super_activities.items()], key=lambda x: x[1]['votes'], reverse=True)
  super_objects_list = sorted([(key, val) for key, val in super_objects.items()], key=lambda x: x[1]['votes'], reverse=True)
  other_objects_list = sorted([(key, val) for key, val in other_objects.items()], key=lambda x: x[1]['votes'], reverse=True)
  sorted_candidates = activities_list + super_activities_list + super_objects_list + other_objects_list
  if len(sorted_candidates) < 1:
    text_to_speech("No activities or Food or Drink found to match your request",verbose=verbose)
    exit()
  
  if verbose:
    print("Cadidates are: ", [(name, v['level']) for name, v in sorted_candidates])
  
  # Keep only the activities that have the highest number of votes.
  # highest_vote = -1
  # for key, val in sorted_candidates:
  #   if val['votes'] > highest_vote:
  #     highest_vote = val['votes']
  #     best_candidate = key
  # filtered_sorted_candidates = list(filter(lambda x: x[1]['votes'] == highest_vote, sorted_candidates))
  filtered_sorted_candidates = sorted_candidates
  
  #########################################################################################################
  # Remove all lower level candidates                                                                     #
  # since they have the same number of votes as a higher level class, but they are more specific.         #
  # and a higher level class is safer to use. such that we don't end up with a class that is too specific.#
  # and the user can always ask for a more specific class if needed.                                      #
  # TODO:: but take care, if the user asks for a specific class, and we don't have it, we will not be able#
  #        to find a class that matches the request.                                                      #
  #        so we need to handle this case, by asking the user to define the new unknown class.            #
  #        For example for "make a cold drink", since we don't have a class for "cold drink", we will end #
  #        up with "make a drink" which is not what the user wants.                                       #
  #        so we need to ask the user to define the new class "cold drink" and then we can use it.        # 
  #########################################################################################################
  
  # remove all lower level candidates
  # if len(filtered_sorted_candidates) > 1:
  #   if verbose:
  #     print("More than one candidate with the same number of votes, removing lower level candidates")
  #   super_class_exists = False
  #   for key2, val2 in filtered_sorted_candidates:
  #       if val2['level'] in ['superActivity', 'superObject']:
  #         super_class_exists = True
  #         break
  #   if super_class_exists:
  #     filtered_sorted_candidates = list(filter(lambda x: x[1]['level'] in ['superActivity', 'superObject'], filtered_sorted_candidates))
  
  # score the candidates based on the similarity between the components of the candidate and the candidate name.
  if len(filtered_sorted_candidates) > 1:
    if verbose:
      print("More than one candidate with the same number of votes, scoring candidates based on the similarity between the components of the candidate and the candidate name")
    for key, val in filtered_sorted_candidates:
      val['score'] = sum(val['sim'])
    filtered_sorted_candidates = sorted(filtered_sorted_candidates, key=lambda x: x[1]['score'], reverse=True)

    if verbose:
      print("Scored cadidates are: ")
      print([(candidate_name, v['score']) for candidate_name, v in filtered_sorted_candidates])

    highest_score = -1
    for key, val in filtered_sorted_candidates:
      if val['score'] > highest_score:
        highest_score = val['score']
        best_candidate = key
    filtered_sorted_candidates = list(
        filter(lambda x: x[1]['score'] == highest_score, filtered_sorted_candidates))
  
  if verbose:
    print("After filtering, cadidates are: ", [(name, v['level']) for name, v in filtered_sorted_candidates])

  if len(filtered_sorted_candidates) > 1:
    print("I am not sure which activity you want me to perform. Please choose one of the following activities:")
    for i, candidate in enumerate(filtered_sorted_candidates):
      print("{}. {}".format(i+1, candidate[0]))
    choice = int(input("Enter your choice: "))
    if choice == 1:
      filtered_sorted_candidates = filtered_sorted_candidates[:1]
    elif choice == 2:
      filtered_sorted_candidates = filtered_sorted_candidates[1:]
    else:
      print("Invalid choice. Please try again.")
      exit()
  
  if verbose:
    print("Final candidates are: ", filtered_sorted_candidates)
  
  chosen_activity = filtered_sorted_candidates[0][1]
  chosen_activity["name"] = filtered_sorted_candidates[0][0]
  chosen_activity_name = chosen_activity["name"]
  sorted_candidates_dict = dict(sorted_candidates)
  
  if chosen_activity['level'] in ['superActivity', 'superObject']:
    if chosen_activity['level'] == 'superObject':
      query_string = "is_restriction(A, some(" + ns + "outputsCreated\", " + ns + chosen_activity_name + "\")), subclass_of(B, A), \\+((subclass_of(Sb, A), subclass_of(B, Sb)))"
      query = prolog.query(query_string)
      for solution in query.solutions():
        chosen_activity_name = solution["B"].split("#")[1]
    activity_name, output_name = ou.handle_super_activity(chosen_activity_name, verbose=verbose)
    if activity_name is None:
      exit()
    if activity_name in sorted_candidates_dict.keys():
      chosen_activity = sorted_candidates_dict[activity_name]
      chosen_activity['name'] = activity_name
    else:
      text_to_speech("Will make you {}".format(output_name), verbose=verbose)
      exit()
    
  if chosen_activity['level'] == 'other':
    if chosen_activity_name in sorted_candidates_dict.keys():
      chosen_activity = sorted_candidates_dict[chosen_activity_name]
      text_to_speech("I know that {} is a {}, but I don't know the steps for preparing it, could you please tell me the steps?".format(chosen_activity_name, chosen_activity['super_object']), verbose=verbose)
      exit()
  # Find if it is a Drink or a Food
  found = False
  for val in ['Drink', 'Food']:
    objects = list(chosen_activity['objectActedOn'])
    if 'output' in chosen_activity.keys():
      objects.append(chosen_activity['output'])
    for obj in objects:
      query = prolog.query("subclass_of(" + ns + obj + "\", " + ns + val + "\").")
      for solution in query.solutions():
        chosen_activity['type'] = val
        found = True
        break
      query.finish()
      if found:
        break
    if found:
      break
  
  # Perform the activities
  filtered_objects = []
  for obj in chosen_activity['objectActedOn']:
    t = "Drinking" if chosen_activity['type'] == "Drink" else chosen_activity['type']
    query_string = "subclass_of(" + ns + obj + "\", " + ns + t + "Ingredient" + "\")"
    query_string += "; subclass_of(" + ns + obj + "\", " + ns + t + "Vessel" + "\")."
    query = prolog.query(query_string)
    for solution in query.solutions():
      filtered_objects.append(obj)
    query.finish()
  chosen_activity['objectActedOn'] = filtered_objects
  if chosen_activity['type'] == 'Drink':
    prepareADrink(chosen_activity)
  elif chosen_activity['type'] == 'Food':
    prepareAMeal(chosen_activity)
  else:
    bringObject(chosen_activity)