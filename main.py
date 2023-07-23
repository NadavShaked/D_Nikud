# from __future__ import print_function

import sys

from argparse import ArgumentParser

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('run')
    p.add_argument('services')
    p.add_argument('--config')
    p.add_argument('--local_constant')
    p.add_argument('--backdoor_port', type=int)
    p.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                   default="DEBUG", help="Set the logging level")
    p.add_argument("--log2file", dest="log2file", action='store_true', help="Set the logging level")
    args = p.parse_args()
    args.services = [args.services]
    main(args)
