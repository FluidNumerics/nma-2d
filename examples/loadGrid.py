#!/usr/bin/env python

import xnma.nma as nma
import inspect, os.path

# Get full path to examples/
# From https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python
filename = inspect.getframeinfo(inspect.currentframe()).filename
path     = os.path.dirname(os.path.abspath(filename))

model = nma.model()
model.loadGrid(f'{path}/data/')
print(model.ds)
print(model.grid)

print(model.ds['dxG'])
