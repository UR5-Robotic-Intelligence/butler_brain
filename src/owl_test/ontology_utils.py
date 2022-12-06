from owl_test.utils import text_to_speech, speach_to_text, get_top_matching_candidate
from rosprolog_client import Prolog


class OntologyUtils:
  def __init__(self) -> None:
    self.prolog = Prolog()
    self.ns = "\"http://ias.cs.tum.edu/kb/knowrob.owl#"

  def handle_super_activity(self, chosen_activity_name, verbose=True):
    query_string = "subclass_of(B, " + self.ns + chosen_activity_name + "\")"
    query_string += ", is_restriction(A, some(" + self.ns + "outputsCreated\", C)), subclass_of(B, A), \\+((subclass_of(Sb, A), subclass_of(B, Sb)))"
    query = self.prolog.query(query_string)
    possible_activities = []
    possible_outputs = []
    for solution in query.solutions():
      possible_activities.append(solution['B'].split('#')[-1])
      possible_outputs.append(solution['C'].split('#')[-1])
    if len(possible_activities) == 0:
      text_to_speech("Currently there are no available options for {}".format(chosen_activity_name))
      return None, None
    else:
      text_to_speech("Your request is part of the {} activity, which has multiple options".format(chosen_activity_name))
      text_to_speech("Please specify which of the following options you want me to prepare")
    for i, candidate in enumerate(possible_outputs):
      text_to_speech("{}. {}".format(i+1, candidate), verbose=True)
    top_ratio = 0
    ratio_threshold = 100
    while top_ratio < ratio_threshold:
      choice_text = speach_to_text(verbose=verbose, show_all=True)
      if type(choice_text) == str:
        choice_text = [choice_text]
      choice, _ , top_ratio = get_top_matching_candidate(possible_activities, choice_text)
      if top_ratio < ratio_threshold:
        text_to_speech("I didn't understand. Please try again.", verbose=verbose)
    # choice = int(input(text_to_speech("Enter your choice: ", verbose=True)))
    return possible_activities[choice], possible_outputs[choice]