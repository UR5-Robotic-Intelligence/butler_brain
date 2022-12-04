from owlready2 import *

onto_path.append("/home/bass/ur5_ws/src/iai_maps/iai_kitchen/owl/iai-kitchen-objects.owl")
onto = get_ontology("http://knowrob.org/kb/knowrob.owl").load()
print(onto.get_parents_of(onto.Spoon))
print(onto.Spoon.instances())
Ontology.get_parents_of()
