from owl_test.robot_action_primitives import FindObject, PickObject, PlaceObject
from owl_test.utils import text_to_speech

def prepareADrink(drink_name, verbose=True):
  text_to_speech("Preparing a drink named {}".format(drink_name), verbose=verbose)
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


def prepareAMeal(meal_name, verbose=True):
  text_to_speech("Preparing a meal named {}".format(meal_name),verbose=verbose)
  pass


def bringObject(object_name, verbose=True):
  text_to_speech("Bringing an object named {}".format(object_name), verbose=verbose)
  FindObject(object_name)
  PickObject(object_name)
  PlaceObject(object_name)
  pass
