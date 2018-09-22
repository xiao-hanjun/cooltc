import os
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []
print user_paths

from slim.nets import cifarnet
mynet = cifarnet.cifarnet
