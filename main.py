# model controller
#
# 
# 
#
import argparse
import importlib as iml
    
def main():
    # make processor
    processors = {
        'train': iml.import_module('notebooks.train'),
        'predict': iml.import_module('notebooks.predict'),
        'visual': iml.import_module('notebooks.visualization')
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