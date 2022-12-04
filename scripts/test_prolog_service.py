#!/usr/bin/env python

import rospy
from rosprolog_client import Prolog
from owl_test.robot_activities import prepareADrink, prepareAMeal, bringObject
from owl_test.utils import text_to_speech
import fuzzywuzzy.fuzz as fuzz
import argparse
import openai
import os



if __name__ == "__main__":
  rospy.init_node('test_rosprolog')
  prolog = Prolog()
  ns = "\"http://ias.cs.tum.edu/kb/knowrob.owl#"
  args = argparse.ArgumentParser(description='Test the rosprolog service')
  args.add_argument('-v', '--verbose', action='store_true', help='Print the explanations and intermediate results')
  verbose = args.parse_args().verbose
  openai.api_key = os.getenv("OPENAI_API_KEY")
  
  # parse the output of GPT-3
  # output_of_gpt3 = "1.cup\n2.coffee"
  # output_of_gpt3 = "1.cup\n2.tea"
  output_of_gpt3 = input("Enter the output of GPT-3: ")
  output_components = output_of_gpt3.split("\\n")
  output_components = [x.split(".")[-1] for x in output_components]
  
  # Find the activity that outputs the components, and the name of the components in the ontology.
  # The output of GPT-3 is not necessarily the same as the name of the components in the ontology.
  # For example, GPT-3 may output "coffee", but the name of the coffee in the ontology is "Coffee-Beverage".
  # so we need to find the activity that outputs an object the has the word "coffee" in its name.
  # we first find all the activities that create a final product.
  # then we search for the activity that outputs an object that has the word "coffee" in its name.
  event_that_has_outputs_created = "is_restriction(A, some(" + ns + "outputsCreated\", C)), subclass_of(B, A), subclass_of(C, Sc)"
  that_are_subclass_of_food_or_drink = "subclass_of(Sc, " + ns + "FoodOrDrink\")" 
  has_objects_acted_on = "is_restriction(D, some(" + ns + "objectActedOn\", E)), subclass_of(B, D), subclass_of(B, Sb)" # \\+((subclass_of(Sb, D), subclass_of(B, Sb)))"
  is_subclass_of_preparing_food_or_drink = "subclass_of(Sb,  " + ns + "PreparingFoodOrDrink\")"
  
  query_string = event_that_has_outputs_created + ", " + that_are_subclass_of_food_or_drink + \
      ", " + has_objects_acted_on + ", " + is_subclass_of_preparing_food_or_drink
  query = prolog.query(query_string)
  
  # votes represent the number of components from the output of GPT-3 that appear in the name of the objects or activities in the ontology.
  # the activity with the highest number of votes is the activity that we are looking for.
  activities = {}
  super_activities = {}
  super_objects = {}
  for solution in query.solutions():
    # remove the namespace from the name of the activity
    B, C, E = solution['B'].split('#')[-1], solution['C'].split('#')[-1],  solution['E'].split('#')[-1]
    # print("B: {}, C: {}, E: {}".format(B, C, E))
    Sc, Sb = solution['Sc'].split('#')[-1], solution['Sb'].split('#')[-1]
    for component in output_components:
      if (component.lower() in B.lower()) or (component.lower() in C.lower()):
        if B not in activities.keys():
          activities[B] = {'output':C, 'objectActedOn':[E], 'level':'activity', 'components':[component], 'votes':1, 'super_activities':[Sb], 'super_objects':[Sc]}
        else:
          if E not in activities[B]['objectActedOn']:
            activities[B]['objectActedOn'].append(E)
          if component not in activities[B]['components']:
            activities[B]['components'].append(component)
            activities[B]['votes'] += 1
          if Sb not in activities[B]['super_activities']:
            activities[B]['super_activities'].append(Sb)
          if Sc not in activities[B]['super_objects']:
            activities[B]['super_objects'].append(Sc)
      if component.lower() in Sb.lower():
        if Sb not in super_activities.keys():
          super_activities[Sb] = {
              'level': 'superActivity', 'components': [component], 'votes': 1}
        elif component not in super_activities[Sb]['components']:
          super_activities[Sb]['components'].append(component)
          super_activities[Sb]['votes'] += 1
      if component.lower() in Sc.lower():
        if Sc not in super_objects.keys():
          super_objects[Sc] = {'level': 'superObject',
                                'components': [component], 'votes': 1}
        elif component not in super_objects[Sc]['components']:
          super_objects[Sc]['components'].append(component)
          super_objects[Sc]['votes'] += 1
        # print("Found activity {} that outputs {} and acts on {}".format(B, C, E))

  query.finish()

  activities_list = sorted([(key, val) for key, val in activities.items()], key=lambda x: x[1]['votes'], reverse=True)
  super_activities_list = sorted([(key, val) for key, val in super_activities.items()], key=lambda x: x[1]['votes'], reverse=True)
  super_objects_list = sorted([(key, val) for key, val in super_objects.items()], key=lambda x: x[1]['votes'], reverse=True)
  sorted_candidates = activities_list + super_activities_list + super_objects_list
  if len(sorted_candidates) < 1:
    text_to_speech("No activities or Food or Drink found to match your request",verbose=verbose)
    exit()
  
  if verbose:
    print("Cadidates are: ", [(name, v['level']) for name, v in sorted_candidates])
  
  # Keep only the activities that have the highest number of votes.
  highest_vote = -1
  for key, val in sorted_candidates:
    if val['votes'] > highest_vote:
      highest_vote = val['votes']
      best_candidate = key
  filtered_sorted_candidates = list(filter(lambda x: x[1]['votes'] == highest_vote, sorted_candidates))
  
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
  if len(filtered_sorted_candidates) > 1:
    if verbose:
      print("More than one candidate with the same number of votes, removing lower level candidates")
    super_class_exists = False
    for key2, val2 in filtered_sorted_candidates:
        if val2['level'] in ['superActivity', 'superObject']:
          super_class_exists = True
          break
    if super_class_exists:
      filtered_sorted_candidates = list(filter(lambda x: x[1]['level'] in ['superActivity', 'superObject'], filtered_sorted_candidates))
  
  # score the candidates based on the similarity between the components of the candidate and the candidate name.
  if len(filtered_sorted_candidates) > 1:
    if verbose:
      print("More than one candidate with the same number of votes, scoring candidates based on the similarity between the components of the candidate and the candidate name")
    for key, val in filtered_sorted_candidates:
      for comp in val['components']:
        if 'score' not in val.keys():
          val['score'] = 0
        val['score'] += fuzz.ratio(comp, key)
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
  
  if chosen_activity['level'] == 'superObject':
    text_to_speech("Please specify what type of {} you want me to prepare".format(chosen_activity['name']), verbose=verbose)
    exit()
  
  if chosen_activity['level'] == 'superActivity':
    text_to_speech("Please specify what type of {} you want me to prepare".format(chosen_activity['components'][0]), verbose=verbose)
    text_to_speech("I can prepare the following types of {}:".format(
        chosen_activity['components'][0]))
    query_string = "subclass_of(A, " + ns + chosen_activity['name'] + "\")."
    query = prolog.query(query_string)
    possible_activities = []
    for solution in query.solutions():
      possible_activities.append(solution['A'].split('#')[-1])
    for i, candidate in enumerate(possible_activities):
      text_to_speech("{}. {}".format(i+1, candidate), verbose=True)
    choice = int(input(text_to_speech("Enter your choice: ", verbose=True)))
    chosen_activity['name'] = possible_activities[choice-1]
    sorted_candidates_dict = dict(sorted_candidates)
    if chosen_activity['name'] in sorted_candidates_dict.keys():
      chosen_activity = sorted_candidates_dict[chosen_activity['name']]
    else:
      text_to_speech("Will search for the activity in the ontology", verbose=verbose)
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
    prepareADrink(chosen_activity['output'])
  elif chosen_activity['type'] == 'Food':
    prepareAMeal(chosen_activity['output'])
  else:
    bringObject(chosen_activity['output'])