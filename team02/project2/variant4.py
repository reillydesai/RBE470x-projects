# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.selfpreserving_monster import SelfPreservingMonster
import time

# TODO This is your code!
sys.path.insert(1, '../team02')
from testcharacter2 import TestCharacter

# Create the game
#random.seed(123) # TODO Change this if you want different random choices
seed = int(time.time())  # Use current time as seed
random.seed(seed)


g = Game.fromfile('map.txt')
g.add_monster(SelfPreservingMonster("aggressive", # name
                                    "A",          # avatar
                                    3, 13,        # position
                                    2             # detection range
))

# TODO Add your character
g.add_character(TestCharacter("me", # name
                              "C",  # avatar
                              0, 0  # position
))

# Run!
g.go(1)
