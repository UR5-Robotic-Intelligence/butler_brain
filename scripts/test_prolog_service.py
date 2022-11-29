#!/usr/bin/env python

from __future__ import print_function

import sys
import rospy
from rosprolog_client import Prolog


if __name__ == "__main__":
    rospy.init_node('test_rosprolog')
    prolog = Prolog()
    namespace = "\"http://ias.cs.tum.edu/kb/knowrob.owl#"

    # parse the output of GPT-3
    # output_of_gpt3 = "1.cup\n2.coffee"
    output_of_gpt3 = "1.cup\n2.tea"
    output_components = output_of_gpt3.split("\n")
    output_components = [x.split(".")[-1] for x in output_components]
    
    # Find the activity that outputs the components, and the name of the components in the ontology.
    # The output of GPT-3 is not necessarily the same as the name of the components in the ontology.
    # For example, GPT-3 may output "coffee", but the name of the coffee in the ontology is "Coffee-Beverage".
    # so we need to find the activity that outputs an object the has the word "coffee" in its name.
    # we first find all the activities that create a final product.
    # then we search for the activity that outputs an object that has the word "coffee" in its name.
    query = prolog.query("is_restriction(A, some(" + namespace + "outputsCreated\", C)), subclass_of(B, A).")
    components_related_activities = {output_components[0]: [], output_components[1]: []}
    for solution in query.solutions():
      # remove the namespace from the name of the activity
      A, B, C = solution['A'].split(
          '#')[-1], solution['B'].split('#')[-1], solution['C'].split('#')[-1]
      for component in output_components:
        if component in C.lower():
          components_related_activities[component].append({"name": C, "act": B})
          # print("Found solution. A = {}, B = {}, C = {}".format(A, B, C))
    query.finish()
    print("Found Activities and Their Components: ", components_related_activities)
