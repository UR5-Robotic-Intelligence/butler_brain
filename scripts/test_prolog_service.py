#!/usr/bin/env python

import rospy
from rosprolog_client import prolog
from owl_test.robot_activities import RobotActivities
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
from copy import deepcopy


class ButlerBrain():
  def __init__(self) -> None:
    rospy.init_node('test_rosprolog')
    ou = OntologyUtils()
    # self.ra = RobotActivities()
    self.prolog = ou.prolog
    self.ns = ou.ns
    args = argparse.ArgumentParser(description='Test the rosprolog service')
    args.add_argument('-v', '--verbose', action='store_true', help='Print the explanations and intermediate results')
    args.add_argument('-s', '--save_embeddings', action='store_true', help='Save the query embeddings to a file')
    args.add_argument('-l', '--load_embeddings', action='store_true', help='Load the query embeddings from a file')
    args.add_argument('-lq', '--load_query_results', action='store_true', help='Load the query results from a file')
    args.add_argument('-sq', '--save_query_results', action='store_true', help='Save the query results to a file')
    self.verbose = args.parse_args().verbose
    self.load_embeddings = args.parse_args().load_embeddings
    self.save_embeddings = args.parse_args().save_embeddings
    self.load_query_results = args.parse_args().load_query_results
    self.save_query_results = args.parse_args().save_query_results
    self.model = SentenceTransformer('bert-base-nli-mean-tokens')
    self.data_path = os.path.join(os.getcwd(), 'uji_butler_wokring_memory.txt')
    self.query_results_path = os.path.join(os.getcwd(), 'query_results.pkl')
    self.new_activities_path = os.path.join(os.getcwd(), 'new_activities.pkl')
    self.new_activities = {}
    if self.load_query_results:
        with open(self.query_results_path, 'rb') as f:
          self.query_results = pickle.load(f)
        with open(self.new_activities_path, 'rb') as f:
          self.new_activities = pickle.load(f)
  
  def add_new_activity(self):
    text_to_speech("Please tell me the output created from the new activity, to add it to my knowledge",verbose=self.verbose)
    output_name = speach_to_text(verbose=self.verbose).capitalize().split()
    output_name = "".join(output_name)
    act_name = "Making-"+output_name
    print(act_name)
    text_to_speech("please tell me the type of the new activity",verbose=self.verbose)
    type_txt = speach_to_text(verbose=self.verbose)
    text_to_speech("Please describe how is the activity performed",verbose=self.verbose)
    act_description = speach_to_text(verbose=self.verbose)
    output_of_gpt3 = text_to_keywords(act_description.strip(), verbose=self.verbose)
    if self.verbose:
      print(output_of_gpt3)
    output_components = output_of_gpt3.strip().split("\n")
    if self.verbose:
      print(output_components)
    output_components = [x.split(".")[-1].strip() for x in output_components]
    print(output_components)
    # ra.gptActivity(chosen_activity)
    self.new_activities[act_name] = {'output':output_name,\
                              'objectActedOn':output_components,\
                              'level':'activity',
                              'type':type_txt}
  
  def main(self):    
    while not rospy.is_shutdown():
    
      text_to_speech("Press enter to start the test", verbose=self.verbose)
      input()
      text_to_speech("Please say your request:", verbose=self.verbose)
      user_request = speach_to_text(verbose=self.verbose)
      output_of_gpt3 = text_to_keywords(user_request.strip(), verbose=self.verbose)
      if self.verbose:
        print(output_of_gpt3)
      output_components = output_of_gpt3.strip().split("\n")
      if self.verbose:
        print(output_components)
      output_components = [x.split(".")[-1].strip() for x in output_components]
      if self.verbose:
        print(output_components)
      
      # output_components = ['chocolate', 'milk']
      # output_components = ['tea', 'beverage']
      # output_components = ['drinking']
      # output_components = ['coffee', 'shop']
      # output_components = ['juice']
      
      comp_enc = self.model.encode([component.lower() for component in output_components], device='cuda')
      
      ##########################################################################################################################################
      # This is like the working memory of the robot.                                                                                          #
      # A more realistic implementation would be to detect the objects in the environment and update the working memory.                       #
      # according to the objects that are detected, and the environment, the robot will generate a different query.                            #
      # For example, if the robot is in the kitchen, it will generate a query that includes the activities that are performed in the kitchen.  #
      ##########################################################################################################################################
      
      if self.load_query_results:
        query_results = deepcopy(self.query_results)
      
      else:
        # Find the activity that outputs the components, and the name of the components in the ontology.
        # The output of GPT-3 is not necessarily the same as the name of the components in the ontology.
        # For example, GPT-3 may output "coffee", but the name of the coffee in the ontology is "Coffee-Beverage".
        # so we need to find the activity that outputs an object the has the word "coffee" in its name.
        # we first find all the activities that create a final product.
        # then we search for the activity that outputs an object that has the word "coffee" in its name.
        event_that_has_outputs_created = "is_restriction(A, some(" + self.ns + "outputsCreated\",C)),subclass_of(B,A),subclass_of(C,Sc)"
        that_are_subclass_of_food_or_drink = "subclass_of(Sc," + self.ns + "FoodOrDrink\"),subclass_of(Other,Sc),\\+is_restriction(_,some(" + self.ns + "outputsCreated\", Other))"
        has_objects_acted_on = "is_restriction(D,some(" + self.ns + "objectActedOn\",E)),subclass_of(B,D),subclass_of(B,Sb),\\+subclass_of(Sb,D)"
        must_be_subclass_of_ingredints_or_vessel = "(subclass_of(E," + self.ns + "DrinkingIngredient" + "\");subclass_of(E," + self.ns + "DrinkingVessel" + "\");"
        must_be_subclass_of_ingredints_or_vessel += "subclass_of(E," + self.ns + "FoodIngredient" + "\");subclass_of(E," + self.ns + "FoodVessel" + "\"))"
        is_subclass_of_preparing_food_or_drink = "subclass_of(Sb," + self.ns + "PreparingFoodOrDrink\")"
        # non_activity_objects = "(subclass_of(NAO, " + self.ns + "FoodOrDrinkOrIngredient\"))" #, \\+(subclass_of(NAO, B); subclass_of(NAO, Sb)))"
        
        query_string = "(" + event_that_has_outputs_created + "," + that_are_subclass_of_food_or_drink + \
            "," + has_objects_acted_on + "," + must_be_subclass_of_ingredints_or_vessel + "," + is_subclass_of_preparing_food_or_drink + ")." #; " + non_activity_objects
        # query = self.prolog.query(query_string)
        query_results = self.prolog.all_solutions(query_string)
        if self.save_query_results:
          with open(self.query_results_path, 'wb') as f:
            pickle.dump(query_results, f)
          with open(self.new_activities_path, 'wb') as f:
            pickle.dump(self.new_activities, f)
      # votes represent the number of components from the output of GPT-3 that appear in the name of the objects or activities in the ontology.
      # the activity with the highest number of votes is the activity that we are looking for.
      activities = deepcopy(self.new_activities)
      super_activities = {}
      super_objects = {}
      other_objects = {}
      is_component_used = {component:0 for component in output_components}
      sim_thresh = 0.66
      data = {}
      encoded_before = {}
      if self.load_embeddings:
        trained_embeddings = vocab.Vectors(name = self.data_path,
                                      cache = 'custom_embeddings',
                                      unk_init = torch.Tensor.normal_)
      
      for solution in query_results:
        encodings = []
        # print(solution)
        # remove the namespace from the name of the activity
        B, C, E = solution['B'].split('#')[-1], solution['C'].split('#')[-1],  solution['E'].split('#')[-1]
        # print("B: {}, C: {}, E: {}".format(B, C, E))
        Sc, Sb = solution['Sc'].split('#')[-1], solution['Sb'].split('#')[-1]
        Other = solution['Other'].split('#')[-1]
        tokens = [B, C, E, Sc, Sb, Other]
        if not self.load_embeddings:
          # encode the activity and the objects
          to_encode = [B.lower(), C.lower(), E.lower(), Sc.lower(), Sb.lower(), Other.lower()]
          encodings = self.model.encode(to_encode, device='cuda:0')
        else:
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
        
        if self.save_embeddings:
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
          if (component in B.lower()) or (component in C.lower()) or (component in E.lower())\
          or (B.lower() in component) or (C.lower() in component) or (E.lower() in component):
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
          if (component in Sb.lower()) or (Sb.lower() in component):
            is_component_used[component] = 1
            if Sb not in super_activities.keys():
              super_activities[Sb] = {
                  'level': 'superActivity', 'components': [component], 'votes': 1, 'sim': [sup_act_sim]}
            elif component not in super_activities[Sb]['components']:
              super_activities[Sb]['components'].append(component)
              super_activities[Sb]['votes'] += 1
              super_activities[Sb]['sim'].append(sup_act_sim)
          if (component in Sc.lower()) or (Sc.lower() in component):
            is_component_used[component] = 1
            if Sc not in super_objects.keys():
              super_objects[Sc] = {'level': 'superObject',
                                    'components': [component], 'votes': 1, 'sim': [sup_obj_sim]}
            elif component not in super_objects[Sc]['components']:
              super_objects[Sc]['components'].append(component)
              super_objects[Sc]['votes'] += 1
              super_objects[Sc]['sim'].append(sup_obj_sim)
            # print("Found activity {} that outputs {} and acts on {}".format(B, C, E))
          if (component in Other.lower()) or (Other.lower() in component):
            is_component_used[component] = 1
            if Other not in other_objects.keys():
              other_objects[Other] = {'level': 'other',
                                    'components': [component], 'votes': 1, 'sim': [other_obj_sim], 'super_object':Sc}
            elif component not in other_objects[Other]['components']:
              other_objects[Other]['components'].append(component)
              other_objects[Other]['votes'] += 1
              other_objects[Other]['sim'].append(other_obj_sim)

      if self.save_embeddings:
        write_embeddings(self.data_path, list(data.values()), list(data.keys()))
      
      for component, used in is_component_used.items():
        if used == 0:
          text_to_speech("I could not find a Food or Drink that matches your request for {}".format(component),verbose=self.verbose)
          text_to_speech(f"Which means {component} is not in my database",verbose=self.verbose)
          self.add_new_activity()
      
      activities_list = sorted([(key, val) for key, val in activities.items()], key=lambda x: x[1]['votes'], reverse=True)
      super_activities_list = sorted([(key, val) for key, val in super_activities.items()], key=lambda x: x[1]['votes'], reverse=True)
      super_objects_list = sorted([(key, val) for key, val in super_objects.items()], key=lambda x: x[1]['votes'], reverse=True)
      other_objects_list = sorted([(key, val) for key, val in other_objects.items()], key=lambda x: x[1]['votes'], reverse=True)
      sorted_candidates = activities_list + super_activities_list + super_objects_list + other_objects_list
      if len(sorted_candidates) < 1:
        text_to_speech("No activities or Food or Drink found to match your request",verbose=self.verbose)
        exit()
      
      if self.verbose:
        print("Cadidates are: ", [(name, v['level']) for name, v in sorted_candidates])
      
      # Keep only the activities that have the highest number of votes.
      highest_vote = -1
      for key, val in sorted_candidates:
        if val['votes'] > highest_vote:
          highest_vote = val['votes']
          best_candidate = key
      filtered_sorted_candidates = list(filter(lambda x: x[1]['votes'] == highest_vote, sorted_candidates))
      print("Vote Filtered Cadidates are: ", [(name, v['level'], v['votes']) for name, v in filtered_sorted_candidates])
      
      num_of_activities = 0
      for key, val in filtered_sorted_candidates:
        if val['level'] in ['activity']:
          num_of_activities += 1
      
      if num_of_activities > 0:
        filtered_sorted_candidates = list(filter(lambda x: x[1]['level'] in ['activity', 'superActivity'], filtered_sorted_candidates))

      choice = None
      if (len(filtered_sorted_candidates) > 1):
        text_to_speech("It looks like I don't have a unique match for your request",verbose=self.verbose)
        speech = "Please choose from:"
        for key, val in filtered_sorted_candidates:
          speech += " " + key + ","
        speech += " or tell me to add a new activity"
        text_to_speech(speech,verbose=self.verbose)
        choose_from = [key for key, val in filtered_sorted_candidates]
        choose_from.append("add a new activity")
        while not rospy.is_shutdown():
          txt = speach_to_text(verbose=self.verbose)
          _, top_match_idx, top_ratio = get_top_matching_candidate(txt, choose_from, bert=True, bert_model=self.model, verbose=self.verbose)
          if top_ratio > 0.9:
            choice = choose_from[top_match_idx]
            if choice != "add a new activity":
              filtered_sorted_candidates = [filtered_sorted_candidates[top_match_idx]]
            break
          else:
            text_to_speech("I did not understand your request, please try again",verbose=self.verbose)
        
      if (num_of_activities == 0) or choice == "add a new activity":
        if num_of_activities == 0:
          text_to_speech("I looks like I do not have an activity to match your request",verbose=self.verbose)
        self.add_new_activity()
      
      if self.verbose:
        print("Final candidates are: ", filtered_sorted_candidates)
      
      chosen_activity = filtered_sorted_candidates[0][1]
      chosen_activity["name"] = filtered_sorted_candidates[0][0]

      # Find if it is a Drink or a Food
      found = False
      for val in ['Drink', 'Food']:
        objects = list(chosen_activity['objectActedOn'])
        if 'output' in chosen_activity.keys():
          objects.append(chosen_activity['output'])
        for obj in objects:
          query = self.prolog.query("subclass_of(" + self.ns + obj + "\", " + self.ns + val + "\").")
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
      chosen_activity['objects_details'] = {}
      for obj in chosen_activity['objectActedOn']:
        chosen_activity['objects_details'][obj] = []
        query_string = "holds(" + self.ns + obj + "\", A, B)."
        query = self.prolog.query(query_string)
        for solution in query.solutions():
            if "Description" not in solution["B"]:
              chosen_activity['objects_details'][obj].append((solution["A"].split('#')[-1], solution["B"].split('#')[-1]))
      print("chosen activity is: ", chosen_activity)
      # if chosen_activity['type'] == 'Drink':
      #   ra.prepareADrink(chosen_activity)
      # elif chosen_activity['type'] == 'Food':
      #   ra.prepareAMeal(chosen_activity)
      # else:
      #   ra.bringObject(chosen_activity)

if __name__ == "__main__":
  butler_brain = ButlerBrain()
  butler_brain.main()