#!/usr/bin/env python
import argparse
from common import *
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','inventory','verify-baseline','offline','generate','aggregate','figures','test']);p.add_argument('--analysis',choices=['similarity','vote','got']);p.add_argument('--profile',default='subset_100',choices=['subset_100','pilot']);p.add_argument('--backbone',choices=MODELS);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--task',choices=TASKS);p.add_argument('--pair',choices=['got','reflexion']);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=100);args=p.parse_args()
 if args.command=='prepare':prepare()
 elif args.command=='inventory':
  from original import inventory
  inventory()
 elif args.command=='verify-baseline':
  from original import verify_baseline
  r=verify_baseline();assert r['status']=='PASS',r
 else:
  import importlib
  mod=importlib.import_module({'offline':'offline','generate':'generation','aggregate':'aggregate','figures':'figures','test':'checks'}[args.command]);mod.main(args)
if __name__=='__main__':main()
