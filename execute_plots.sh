#!/bin/bash

python params_to_metric.py -m 1 -K 3
python params_to_metric.py -m 1 -K 4

python params_to_metric.py -m 2 -K 4
python params_to_metric.py -m 2 -K 5

python params_to_metric.py -m 3 -K 3
python params_to_metric.py -m 4 -K 4

python plotting.py
