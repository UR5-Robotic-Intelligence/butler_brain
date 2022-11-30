#!/usr/bin/env python

import rospy
from rosprolog_client import Prolog


def prepareADrink(drink_name):
  print("Preparing a drink named {}".format(drink_name))
  # get components of the drink
  # if it is a hot drink, press the button on the kettle.
  # do I see a cup?
  # if not, ask for a cup
  # if yes, and I see many, ask which one to use
  # if yes, and I see one, use it
  # do I see the main drink component?
  # if not, ask for it
  # if yes, put it in the cup.
  # if sugar is one of the drink componets, ask how much sugar to put in.
  # if there's not enough sugar, ask for more sugar.
  # pour water from the kettle into the cup.
  pass

def prepareAMeal(meal_name):
  print("Preparing a meal named {}".format(meal_name))
  pass

def bringObject(object_name):
  print("Bringing an object named {}".format(object_name))
  pass

if __name__ == "__main__":
    rospy.init_node('test_rosprolog')
    prolog = Prolog()
    namespace = "\"http://ias.cs.tum.edu/kb/knowrob.owl#"

    # parse the output of GPT-3
    # output_of_gpt3 = "1.cup\n2.coffee"
    # output_of_gpt3 = "1.cup\n2.tea"
    output_of_gpt3 = input("Enter the output of GPT-3: ")
    output_components = output_of_gpt3.split("\n")
    output_components = [x.split(".")[-1] for x in output_components]
    
    # Find the activity that outputs the components, and the name of the components in the ontology.
    # The output of GPT-3 is not necessarily the same as the name of the components in the ontology.
    # For example, GPT-3 may output "coffee", but the name of the coffee in the ontology is "Coffee-Beverage".
    # so we need to find the activity that outputs an object the has the word "coffee" in its name.
    # we first find all the activities that create a final product.
    # then we search for the activity that outputs an object that has the word "coffee" in its name.
    query = prolog.query("is_restriction(A, some(" + namespace + "outputsCreated\", C)), subclass_of(B, A).")
    activities = {}
    for solution in query.solutions():
      # remove the namespace from the name of the activity
      A, B, C = solution['A'].split(
          '#')[-1], solution['B'].split('#')[-1], solution['C'].split('#')[-1]
      for component in output_components:
        if (component in C.lower()) or (component in B.lower()):
          activities[B] = {"output": C}
          # print("Found solution. A = {}, B = {}, C = {}".format(A, B, C))
    query.finish()
    
    # Find if it is a Drink or a Food
    for act in activities.values():
      for val in ['Drink', 'Food']:
        query = prolog.query("subclass_of(" + namespace + act['output'] + "\", " + namespace + val + "\").")
        for solution in query.solutions():
          act['type'] = val
        query.finish()

    print(activities)
    
    for act, val in activities.items():
      if val['type'] == 'Drink':
        prepareADrink(val['output'])
      elif val['type'] == 'Food':
        prepareAMeal(val['output'])
      else:
        bringObject(val['output'])