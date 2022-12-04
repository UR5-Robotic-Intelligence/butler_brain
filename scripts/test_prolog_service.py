#!/usr/bin/env python

import rospy
from rosprolog_client import Prolog
from owl_test.robot_activities import prepareADrink, prepareAMeal, bringObject


if __name__ == "__main__":
    rospy.init_node('test_rosprolog')
    prolog = Prolog()
    ns = "\"http://ias.cs.tum.edu/kb/knowrob.owl#"

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
        if (component.lower() in B.lower()) or (component.lower() in C.lower()) or (component.lower() in Sc.lower()) or (component.lower() in Sb.lower()):
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
      print("No activities or Food or Drink found to match your request")
      exit()
    print("Cadidates are: ", sorted_candidates)
    
    # Keep only the activities that have the highest number of votes.
    highest_vote = -1
    for key, val in sorted_candidates:
      if val['votes'] > highest_vote:
        highest_vote = val['votes']
        best_candidate = key
    sorted_candidates = list(filter(lambda x: x[1]['votes'] == highest_vote, sorted_candidates))

    if len(sorted_candidates) > 1:
      print("I am not sure which activity you want me to perform. Please choose one of the following activities:")
      for i, candidate in enumerate(sorted_candidates):
        print("{}. {}".format(i+1, candidate[0]))
      choice = int(input("Enter your choice: "))
      if choice == 1:
        sorted_candidates = sorted_candidates[:1]
      elif choice == 2:
        sorted_candidates = sorted_candidates[1:]
      else:
        print("Invalid choice. Please try again.")
        exit()
    
    chosen_activity = {sorted_candidates[0][0]: sorted_candidates[0][1]}
    
    # Find if it is a Drink or a Food
    for act in chosen_activity.values():
      found = False
      for val in ['Drink', 'Food']:
        objects = list(act['objectActedOn'])
        if 'output' in act.keys():
          objects.append(act['output'])
        for obj in objects:
          query = prolog.query("subclass_of(" + ns + obj + "\", " + ns + val + "\").")
          for solution in query.solutions():
            act['type'] = val
            found = True
            break
          query.finish()
          if found:
            break
        if found:
          break
    
    # Perform the activities
    filtered_objects = []
    for act, val in chosen_activity.items():
      for obj in val['objectActedOn']:
        t = "Drinking" if val['type'] == "Drink" else val['type']
        query_string = "subclass_of(" + ns + obj + "\", " + ns + t + "Ingredient" + "\")"
        query_string += "; subclass_of(" + ns + obj + "\", " + ns + t + "Vessel" + "\")."
        query = prolog.query(query_string)
        for solution in query.solutions():
          filtered_objects.append(obj)
        query.finish()
      val['objectActedOn'] = filtered_objects
      if val['type'] == 'Drink':
        prepareADrink(val['output'])
      elif val['type'] == 'Food':
        prepareAMeal(val['output'])
      else:
        bringObject(val['output'])