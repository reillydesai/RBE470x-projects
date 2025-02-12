# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
import time
from game import Game
from monsters.selfpreserving_monster import SelfPreservingMonster

# TODO This is your code!
sys.path.insert(1, '../team02')
from testcharacter import TestCharacter
#from interactivecharacter import InteractiveCharacter

# Create the game
random.seed(int(time.time())) # TODO Change this if you want different random choices
g = Game.fromfile('map.txt')
g.add_monster(SelfPreservingMonster("selfpreserving", # name
                                    "S",              # avatar
                                    3, 9,             # position
                                    1                 # detection range
))

# TODO Add your character
g.add_character(TestCharacter("me", # name
                              "C",  # avatar
                              0, 0  # position
))
# g.add_character(InteractiveCharacter("me", # name
#                                      "C",  # avatar
#                                      0, 0  # position
# ))

# Run!
g.go()
