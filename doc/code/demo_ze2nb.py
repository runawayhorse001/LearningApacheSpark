# import python library
import os, sys

# import zeppelin2nb module
from ze2nb import ze2nb


# scenario 1
# file and output at the current directory
# output path, the default output path will be the current directory
ze2nb('H2o_Sparking.json')

# scenario 2
output = os.path.abspath(os.path.join(sys.path[0])) +'/output'
ze2nb('H2o_Sparking.json', out_path=output, to_html=True, to_py=True)

# scenario 3
# with load and output path
load_path = '/Users/dt216661/Documents/MyJson/'
output = os.path.abspath(os.path.join(sys.path[0])) +'/output1'
ze2nb('H2o_GBM.json', load_path=load_path, out_path=output, to_html=True, to_py=True)

