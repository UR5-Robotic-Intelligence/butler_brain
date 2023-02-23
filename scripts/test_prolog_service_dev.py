#!/usr/bin/env python

import rospy
from owl_test.robot_activities import RobotActivities
from owl_test.utils import text_to_speech, gpt, speach_to_text, get_top_matching_candidate, cos_sim, write_embeddings, tell_me_one_of
from owl_test.ontology_utils import OntologyUtils
import argparse
import os
from sentence_transformers import SentenceTransformer
import torchtext.vocab as vocab
import torch
import pickle
from copy import deepcopy
from std_msgs.msg import String


class ButlerBrain():
  def __init__(self) -> None:
    rospy.init_node('test_rosprolog')
    args = argparse.ArgumentParser(description='Test the rosprolog service')
    args.add_argument('-v', '--verbose', action='store_true', help='Print the explanations and intermediate results')
    args.add_argument('-s', '--save_embeddings', action='store_true', help='Save the query embeddings to a file')
    args.add_argument('-l', '--load_embeddings', action='store_true', help='Load the query embeddings from a file')
    args.add_argument('-lq', '--load_query_results', action='store_true', help='Load the query results from a file')
    args.add_argument('-sq', '--save_query_results', action='store_true', help='Save the query results to a file')
    args.add_argument('-la', '--load_activities', action='store_true', help='load the new activities from a file')
    args.add_argument('-sa', '--save_activites', action='store_true', help='Save the new_activities to a file')
    args.add_argument('-ld', '--load_manip_data', action='store_true', help='load the manip data from a file')
    args.add_argument('-sd', '--save_manip_data', action='store_true', help='Save the manip data to a file')
    self.verbose = args.parse_args().verbose
    self.load_embeddings = args.parse_args().load_embeddings
    self.save_embeddings = args.parse_args().save_embeddings
    self.load_query_results = args.parse_args().load_query_results
    self.save_query_results = args.parse_args().save_query_results
    self.load_activities = args.parse_args().load_activities
    self.save_activites = args.parse_args().save_activites
    load_manip_data = args.parse_args().load_manip_data
    save_manip_data = args.parse_args().save_manip_data
    # self.ra = RobotActivities(load_data=load_manip_data, save_data=save_manip_data, verbose=self.verbose)
    ou = OntologyUtils()
    self.prolog = ou.prolog
    self.ns = ou.ns
    self.model = SentenceTransformer('bert-base-nli-mean-tokens')
    self.data_path = os.path.join(os.getcwd(), 'uji_butler_wokring_memory.txt')
    self.query_results_path = os.path.join(os.getcwd(), 'query_results.pkl')
    self.new_activities_path = os.path.join(os.getcwd(), 'new_activities.pkl')
    self.new_prompts_path = os.path.join(os.getcwd(), 'new_prompts.txt')
    self.new_activities = {}
    if self.load_query_results:
      with open(self.query_results_path, 'rb') as f:
        self.query_results = pickle.load(f)
    if self.load_activities:
      with open(self.new_activities_path, 'rb') as f:
        self.new_activities = pickle.load(f)
    rospy.Subscriber("/sign_command", String, self.sign_command_callback)
    self.sign_command = None
  
  def sign_command_callback(self, msg):
    self.sign_command = msg.data
    
  def gpt_string_commands_to_list(self, gpt_string, return_components=False, use_gpt_string=True):
    gpt_string = gpt_string.strip().strip().split("\n")
    print("gpt_string: ", gpt_string)    
    container = gpt_string[-1][10:-1]
    output_components = []
    rob_commands = []
    steps = []
    for i, step in enumerate(gpt_string):
      if i == len(gpt_string) - 1:
        break
      
      if not use_gpt_string:
        func_name = step.split(" to ")[0].split(" ")[0]
        input_args = [step.split(" to ")[0].split(" ")[1].strip()]
        input_args.append(step.split(" to ")[1].strip())
      else:
        step = step.split(".")[-1].strip()
        func_name = step.split("(")[0]
        input_args = step.split("(")[1].split(")")[0].split(",")
        input_args = [arg.strip() for arg in input_args]
        if len(input_args) == 2 and input_args[1] == container:
          steps.append(f"{func_name} {input_args[0]} to {input_args[1]}")
        elif len(input_args) == 1:
          steps.append(f"{func_name} {input_args[0]}")
      output_components.extend(input_args)
      rob_commands.append((func_name, input_args[0], input_args[1]))
      print(func_name, input_args)
    print(container)
    if return_components:
      return rob_commands, output_components, container, steps, func_name
    return rob_commands
  
  def add_new_activity(self, data_point, predict=False):
    output_name = data_point['output']
    act_name = "Making-"+output_name
    print("output = ",act_name)
    type_txt = data_point['type']
    
    # convert description to commands using previous experience
    act_description = data_point['steps_description']
    additional_prompts = ""
    for act_name, act in self.new_activities.items():
      if 'prompt' in act:
        if act['type'] == type_txt:
          additional_prompts += "\n".join(act['prompt'])
    res = gpt(act_description, prompt_to_use=f'text_to_commands_{type_txt.lower()}',new_prompt=additional_prompts, verbose=self.verbose)
    
    # convert description to commands without using previous experience
    res_no_exp = gpt(act_description, prompt_to_use=f'text_to_commands_{type_txt.lower()}', verbose=self.verbose)

    # predict commands using type and previous experience
    request = data_point['requests']
    additional_prompts = ""
    for act_name, act in self.new_activities.items():
      if 'prompt' in act:
        if act['type'] == type_txt:
          additional_prompts += "\n".join(act['prompt'])
    res_pred_type = gpt(request, prompt_to_use=f'request_to_commands_{type_txt.lower()}',new_prompt=additional_prompts, verbose=self.verbose)
    
    # predict commands using type without using previous experience
    res_pred_type_no_exp = gpt(request, prompt_to_use=f'request_to_commands_{type_txt.lower()}', verbose=self.verbose)
    
    # predict commands without using type, but using previous experience
    for act_name, act in self.new_activities.items():
      if 'prompt' in act:
        if act['type'] != type_txt:
          additional_prompts += "\n".join(act['prompt'])
    res_pred = gpt(request, prompt_to_use=f'request_to_commands',new_prompt=additional_prompts, verbose=self.verbose)
    
    # predict commands without using type or previous experience
    res_pred_no_exp = gpt(request, prompt_to_use=f'request_to_commands', verbose=self.verbose)
    
    output_components = []
    steps = []
    rob_commands = []
    # TODO: 1.container should be one of objects acted on, and logical reasoning should handle its addition and ignorance,
    #       2.logical reasoning should handle the problem of transport and pour,
    #       4.also the func_name needs to be checked for existence as a capability, if not, find a way to automatically define it,
    #         and ask user for any missing or needed information.
    #       5.Find how to add these to the ontology. (would help in the reasoning)
    rob_commands, output_components, container, steps, func_name = self.gpt_string_commands_to_list(res, return_components=True)
    output_components = list(set(output_components))
    print("to make ", output_name, "you need ", output_components)
    print("steps: ", steps)
    text_to_speech("Is this correct?, please tell me yes or no", verbose=self.verbose)
    txt = speach_to_text(verbose=self.verbose)
    new_rob_commands = None
    if 'no' in txt.lower():
      # text_to_speech("Ok, please correct the steps shown on the screen", verbose=self.verbose)
      with open("steps_tmp.txt", "a") as file_object:
          file_object.writelines("\n".join(steps))
          file_object.write(f"\ncontainer({container})")
      os.system("gedit steps_tmp.txt")
      with open("steps_tmp.txt", "r") as file_object:
          steps = file_object.readlines()
          if self.verbose:
            print("steps are ", steps)
          new_rob_commands = self.gpt_string_commands_to_list("".join(steps), use_gpt_string=False)
    elif 'yes' in txt.lower():
      # text_to_speech("Ok, I will do it right away", verbose=self.verbose)
      new_rob_commands = rob_commands
    new_prompt = ['Q:'+act_description+':']
    new_prompt.extend([f"{i+1}. {f}({arg1},{arg2})" for i, (f, arg1, arg2) in enumerate(new_rob_commands)])
    new_prompt.append(f"container({container})")
    request_to_cmds_prompt = ['Q:'+request+':']
    request_to_cmds_prompt.extend([f"{i+1}. {f}({arg1},{arg2})" for i, (f, arg1, arg2) in enumerate(new_rob_commands)])
    request_to_cmds_prompt.append(f"container({container})")
    print("new_prompt = ", new_prompt)
    sup_act = 'PreparingABeverage' if type_txt == 'Drink' else 'PreparingAFoodItem'
    self.new_activities[act_name] = {'name':act_name,
                                     'output':output_name,
                                       'objectActedOn':output_components,
                                       'steps':new_rob_commands,
                                         'level':'activity',
                                           'type':type_txt,
                                           'container':container,
                                           'description':act_description,
                                           'components':[],
                                           'votes':0,
                                           'sim':[],
                                           'super_activities':[sup_act],
                                           'super_objects':[type_txt],
                                           'prompt':new_prompt,
                                           'request':data_point['requests'],
                                           'request_to_cmds_prompt':request_to_cmds_prompt,}
    if self.save_activites:
      with open(self.new_activities_path, 'wb') as f:
        pickle.dump(self.new_activities, f)
    # self.perform_activity(self.new_activities[act_name])
  
  def perform_activity(self, chosen_activity):
    chosen_activity['objects_details'] = {}
    for obj in chosen_activity['objectActedOn']:
      chosen_activity['objects_details'][obj] = []
      query_string = "holds(" + self.ns + obj + "\", A, B)."
      query = self.prolog.query(query_string)
      for solution in query.solutions():
          if "Description" not in solution["B"]:
            chosen_activity['objects_details'][obj].append((solution["A"].split('#')[-1], solution["B"].split('#')[-1]))
    print("chosen activity is: ", chosen_activity)
    if 'steps' not in list(chosen_activity.keys()):
      chosen_activity['steps'] = []
      additional_prompts = ""
      for act_name, act in self.new_activities.items():
        if 'prompt' in act:
          if act['type'] == chosen_activity['type']:
            additional_prompts += "\n".join(act['prompt']) + "\n"
      commands_string = gpt(str(chosen_activity), prompt_to_use=f"ont_to_commands_{chosen_activity['type'].lower()}", new_prompt=additional_prompts, verbose=self.verbose)
      if self.verbose:
        print("commands_string: ", commands_string)
      rob_commands = self.gpt_string_commands_to_list(commands_string)
      chosen_activity['steps'] = rob_commands
    if self.verbose:
      print("rob_commands: ", chosen_activity['steps'])
    if chosen_activity['type'] == 'Drink':
      container = 'cup' if 'container' not in chosen_activity.keys() else chosen_activity['container']
    elif chosen_activity['type'] == 'Food':
      container = 'bowl' if 'container' not in chosen_activity.keys() else chosen_activity['container']
    self.ra.prepare_food_or_drink(chosen_activity, container=container)
  
  def main(self):
    commands_list_drinks = ["Make me tomato juice please",
                     "I would love a coffee machiato please",
                     "Please prepare me a hot-chocolate",
                     "I would like a cup of tea",
                     "Could you please prepare me a mug of warm milk",
                     "I want some coffee Latte",
                     "Is it possible to make me a carrot juice"]
    output_components_list_drinks = [['tomato', 'juice'],
                              ['coffee', 'machiato'],
                              ['hot-chocolate'],
                              ['tea'], 
                              ['cup','milk'],
                              ['coffee', 'Latte'],
                              ['carrot', 'juice']]
    output_name_list_drinks = ['TomatoJuice', 'CoffeeMachiato', 'HotChocolate', 'Tea', 'WarmMilk', 'CoffeeLatte', 'CarrotJuice']
    steps_list_drinks = [['transport tomato to cup','pour water to cup','container(cup)'],
                         ['transport coffee-powder to cup', 'pour water to cup', 'pour milk-foam to cup', 'container(cup)'],
                         ['transport chocolate-powder to cup', 'pour milk to cup','container(cup)'],
                         ['transport tea-packet to cup', 'pour water to cup','container(cup)'],
                         ['pour milk to mug', 'container(mug)'],
                         ['transport coffee-powder to cup', 'pour water to cup', 'pour milk to cup', 'container(cup)'],
                         ['transport carrot to cup', 'pour water to cup', 'container(cup)']]
    steps_description_list_drinks = ["first put tomato in a cup, then pour water into the cup",
                                     "put coffee powder in the cup, add water, then add some milk foam",
                                     "you have to use chocolate powder, and milk",
                                     "put a tea-packet in the cup, then pour some water",
                                     "pour milk into a mug",
                                     "add coffee powder, water, and milk to the cup",
                                     "mix carrots with water in a cup"]
    commands_list_foods = ["Make me a chicken sandwich please",
                           "I would appreciate it if you could prepare me a cheese sandwich",
                           "having a green salad would be great",
                           "I would like a bowl of pasta",
                           "Could you please prepare me a bowl of rice",
                           "I am craving for some chicken soup",
                           "I am in the mood for some beef steak"]
    output_components_list_foods = [['chicken', 'sandwich'],
                                    ['cheese', 'sandwich'],
                                    ['green', 'salad'],
                                    ['bowl', 'pasta'],
                                    ['bowl', 'rice'],
                                    ['chicken', 'soup'],
                                    ['beef', 'steak']]
    output_name_list_foods = ['ChickenSandwich', 'CheeseSandwich', 'GreenSalad', 'Pasta', 'Rice', 'ChickenSoup', 'BeefSteak']
    steps_list_foods = [['transport chicken to plate', 'transport bread to plate', 'transport cheese to plate', 'container(plate)'],
                        ['transport cheese to plate', 'transport bread to plate', 'container(plate)'],
                        ['transport lettuce to plate', 'transport tomato to plate', 'transport cucumber to plate', 'transport olive-oil to plate', 'container(plate)'],
                        ['transport pasta to bowl', 'transport tomato-sauce to bowl', 'container(bowl)'],
                        ['transport rice to bowl', 'container(bowl)'],
                        ['transport chicken to bowl', 'pour water to bowl', 'transport carrot to bowl', 'transport onion to bowl', 'transport tomato to bowl', 'container(bowl)'],
                        ['transport beef to plate', 'transport potato to plate', 'transport onion to plate', 'transport barbecue-sauce to plate', 'container(plate)']]
    steps_description_list_foods = ["first put chicken on the plate, then put bread on the plate, then put cheese on the plate",
                                    "put cheese and bread on the plate",
                                    "add lettuce, tomato, cucumber in one plate, then add some olive oil on top",
                                    "put some pasta in the bowl, then add some tomate sauce",
                                    "put some rice in the bowl",
                                    "boil some water with onions, carrots, and tomatoes, then put chicken in the bowl",
                                    "put the steak on the plate, the add some potatoes, and onions, with some barbecue sauce on the side"]
    
    drinks_data_list = list(zip(commands_list_drinks, output_components_list_drinks, output_name_list_drinks, steps_description_list_drinks, steps_list_drinks))
    drinks_data_sorted = sorted(drinks_data_list, key=lambda x: len(x[3]))
    drinks_data = [{'requests':x[0], 'components':x[1], 'output':x[2], 'steps_description':x[3], 'steps':x[4], 'type':'Drink'} for x in drinks_data_sorted]
    foods_data_list = list(zip(commands_list_foods, output_components_list_foods, output_name_list_foods, steps_description_list_foods, steps_list_foods))
    foods_data_sorted = sorted(foods_data_list, key=lambda x: len(x[3]))
    foods_data = [{'requests':x[0], 'components':x[1], 'output':x[2], 'steps_description':x[3], 'steps':x[4], 'type':'Food'} for x in foods_data_sorted]
    data = drinks_data + foods_data
    for data_point in data:    
      if rospy.is_shutdown():
        break
      input("Press Enter to continue...")
      output_components = data_point['components']
      
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
        query_results = self.prolog.all_solutions(query_string)
        if self.save_query_results:
          with open(self.query_results_path, 'wb') as f:
            pickle.dump(query_results, f)
      # votes represent the number of components from the output of GPT-3 that appear in the name of the objects or activities in the ontology.
      # the activity with the highest number of votes is the activity that we are looking for.
      activities = deepcopy(self.new_activities)
      if self.verbose:
        print(self.new_activities)
      super_activities = {}
      super_objects = {}
      other_objects = {}
      is_component_used = {component:0 for component in output_components}
      sim_thresh = 1.0
      data = {}
      encoded_before = {}
      if self.load_embeddings:
        trained_embeddings = vocab.Vectors(name = self.data_path,
                                      cache = 'custom_embeddings',
                                      unk_init = torch.Tensor.normal_)
     
      for act_name, act_data in activities.items():
        art_sol = {}
        for obj_acted_on in act_data['objectActedOn']:
          art_sol['B'] = '1#' + act_name
          art_sol['C'] = '1#' + act_data['output']
          art_sol['E'] = '1#' + obj_acted_on
          art_sol['Sc'] = '1#' + act_data['type']
          art_sol['Sb'] = '1#PreparingABeverage' if act_data['type'] == 'Drink' else '1#PreparingAFoodItem'
          art_sol['Other'] = '1#None'
          query_results.append(art_sol)
      
      activities = {}
      
      for solution in query_results:
        encodings = []
        # print(solution)
        # remove the namespace from the name of the activity
        B, C, E = solution['B'].split('#')[-1], solution['C'].split('#')[-1], solution['E'].split('#')[-1]
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
              try:
                token_idx = trained_embeddings.stoi[token]
                encodings.append(trained_embeddings.vectors[token_idx].numpy())
              except KeyError:
                to_encode = [token.lower()]
                encodings.append(self.model.encode(to_encode, device='cuda:0')[0])
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
          or (B.lower() in component) or (C.lower() in component) or (E.lower() in component) or\
           (act_sim > sim_thresh) or (output_sim > sim_thresh) or (acted_on_sim > sim_thresh):
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
          if (component in Sb.lower()) or (Sb.lower() in component) or (sup_act_sim > sim_thresh):
            is_component_used[component] = 1
            if Sb not in super_activities.keys():
              super_activities[Sb] = {
                  'level': 'superActivity', 'components': [component], 'votes': 1, 'sim': [sup_act_sim]}
            elif component not in super_activities[Sb]['components']:
              super_activities[Sb]['components'].append(component)
              super_activities[Sb]['votes'] += 1
              super_activities[Sb]['sim'].append(sup_act_sim)
          if (component in Sc.lower()) or (Sc.lower() in component) or (sup_obj_sim > sim_thresh):
            is_component_used[component] = 1
            if Sc not in super_objects.keys():
              super_objects[Sc] = {'level': 'superObject',
                                    'components': [component], 'votes': 1, 'sim': [sup_obj_sim]}
            elif component not in super_objects[Sc]['components']:
              super_objects[Sc]['components'].append(component)
              super_objects[Sc]['votes'] += 1
              super_objects[Sc]['sim'].append(sup_obj_sim)
            # print("Found activity {} that outputs {} and acts on {}".format(B, C, E))
          if (component in Other.lower()) or (Other.lower() in component) or (other_obj_sim > sim_thresh):
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
      
      activities_list = sorted([(key, val) for key, val in activities.items()], key=lambda x: x[1]['votes'], reverse=True)
      super_activities_list = sorted([(key, val) for key, val in super_activities.items()], key=lambda x: x[1]['votes'], reverse=True)
      super_objects_list = sorted([(key, val) for key, val in super_objects.items()], key=lambda x: x[1]['votes'], reverse=True)
      other_objects_list = sorted([(key, val) for key, val in other_objects.items()], key=lambda x: x[1]['votes'], reverse=True)
      sorted_candidates = activities_list + super_activities_list + super_objects_list + other_objects_list
      
      if self.verbose:
        print("Cadidates are: ", [(name, v['level']) for name, v in sorted_candidates])
      added_activity = False
      for component, used in is_component_used.items():
        if used == 0:
          # text_to_speech("I could not find a Food or Drink that matches your request for {}".format(component),verbose=self.verbose)
          # text_to_speech("Please tell me to add components or add new activity or tell me to skip it",verbose=self.verbose)
          while not rospy.is_shutdown():
            # txt = speach_to_text(verbose=self.verbose)
            txt = 'new'
            if 'new' in txt or 'activity' in txt or 'add' in txt:              
              self.add_new_activity(data_point=data_point)
              added_activity = True
              break
            elif 'skip' in txt:
              break
            elif 'component' in txt:
              text_to_speech("Please tell me name of activity to add the component into",verbose=self.verbose)
              while not rospy.is_shutdown():
                txt = speach_to_text(verbose=self.verbose)
                cand = [c for c, v in sorted_candidates]
                c_idx, m_idx, tr = get_top_matching_candidate(cand, txt, bert=True, bert_model=self.model, verbose=self.verbose)
                if tr >= 0.9:
                  sorted_candidates[c_idx][1]['objectActedOn'].append(component)
                  sorted_candidates[c_idx][1]['votes'] += 1
                  break
                else:
                  text_to_speech("Please say again, I didn't get it", verbose=self.verbose)
              break
            else:
                text_to_speech("Please say again, I didn't get it", verbose=self.verbose)
          break
      if added_activity:
        continue
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

      if chosen_activity['name'] in self.new_activities.keys():
        chosen_activity = self.new_activities[chosen_activity['name']]
      else:
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
      self.perform_activity(chosen_activity)

if __name__ == "__main__":
  butler_brain = ButlerBrain()
  butler_brain.main()