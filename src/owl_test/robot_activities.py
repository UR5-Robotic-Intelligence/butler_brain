from owl_test.robot_action_primitives import FindObject, PickObject, PlaceObject, PourObject, PushButton
from owl_test.utils import text_to_speech

def prepareADrink(drink_activity, verbose=True):
  text_to_speech("Preparing a drink named {}".format(drink_activity['name']), verbose=verbose)
  # get components of the drink
  components = drink_activity['components']
  # if it is a hot drink, press the button on the kettle.
  if 'temperature' in drink_activity.keys():
    if drink_activity['temperature'] == "hot":
      kettle_loc = FindObject("kettle")
      PushButton(kettle_loc)
  # do I see a cup?
  cup_loc = FindObject("cup")
  # if not, ask for a cup
  if cup_loc is None:
    text_to_speech("I don't see a cup. Please bring me a cup.", verbose=verbose)
  # if yes, and I see many, ask which one to use
  if len(cup_loc) > 1:
    text_to_speech("I see multiple cups. Please tell me which one to use.", verbose=verbose)
    idx = int(input("Enter the index of the cup to use: "))
    cup_loc = cup_loc[idx]
  # do I see all the drink components?
  components_loc = {}
  for component in components:
    components_loc[component] = FindObject(component)
    if components_loc[component] is None:
      text_to_speech("I don't see {}. Please bring me {}.".format(component, component), verbose=verbose)
      components_loc[component] = FindObject(component)
    # if sugar is one of the drink componets, ask how much sugar to put in.
    if component == "sugar":
      text_to_speech("How much sugar do you want?", verbose=verbose)
      n_sugar = int(input("Enter the number of sugar cubes: "))
      # if there's not enough sugar, ask for more sugar.
      if len(components_loc["sugar"]) < n_sugar:
        text_to_speech("I don't have enough sugar. Please bring me more sugar.", verbose=verbose)
        components_loc["sugar"] = FindObject("sugar")
      for i in range(n_sugar):
        PickObject(components_loc["sugar"][i])
        PlaceObject(cup_loc)
    elif component['type'] != "liquid":
      PickObject(components_loc[component])
      PlaceObject(cup_loc)
  
  # if there's a liquid component, pour it into the cup.
  for component in components:
    if component['type'] == "liquid":
      # pour water from the kettle into the cup.
      pour_from_loc = FindObject("bottle")
      if 'temperature' in drink_activity.keys():
        if drink_activity['temperature'] == "hot":
          pour_from_loc = FindObject("kettle")
      PourObject(pour_from_loc, cup_loc)


def prepareAMeal(meal_activity, verbose=True):
  text_to_speech("Preparing a meal named {}".format(meal_activity['name']),verbose=verbose)
  pass


def bringObject(object_name, verbose=True):
  text_to_speech("Bringing an object named {}".format(object_name), verbose=verbose)
  FindObject(object_name)
  PickObject(object_name)
  PlaceObject(object_name)
  pass
