# model controller
#
# 
# 
#
import argparse
import importlib as iml
import os
import sys


if __package__ is None or __package__ == "":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
def main():
    # make processor
    processors = {
        'train': iml.import_module('sources.train'),
        'predict': iml.import_module('sources.predict'),
        'visual': iml.import_module('sources.visualization')
    }

    # read main-parser
    parser = argparse.ArgumentParser(description='main execute script')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # read sub-parser
    for name, module in processors.items():
        subparsers.add_parser(name, parents=[module.get_parser()], add_help=False)

    # read arguments
    arg = parser.parse_args()       # get commandline arguments
    
    # start
    exep = processors[arg.mode]     # set processor mode
    exep.start(arg)                 # start execute file

if __name__ == '__main__':
    main()