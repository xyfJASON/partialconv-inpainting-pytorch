import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse

import engine
from configs.defaults import get_cfg_defaults


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--name',
        help='name of experiment directory, if None, use current time instead',
    )
    parser.add_argument(
        'func',
        choices=['train', 'train-finetune', 'evaluate', 'sample'],
        help='choose a function',
    )
    parser.add_argument(
        '-c', '--config-file',
        metavar='FILE',
        help='path to config file',
    )
    parser.add_argument(
        '-ni', '--no-interaction',
        action='store_true',
        help='do not interacting with the user',
    )
    parser.add_argument(
        '--opts',
        default=[],
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line 'KEY VALUE' pairs",
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = get_cfg_defaults()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.func == 'train':
        trainer = engine.Trainer(cfg, args)
        trainer.run_loop()
    elif args.func == 'train-finetune':
        trainer = engine.Trainer(cfg, args, finetune=True)
        trainer.run_loop()
    elif args.func == 'evaluate':
        tester = engine.Tester(cfg, args)
        tester.evaluate()
    elif args.func == 'sample':
        tester = engine.Tester(cfg, args)
        tester.sample()
