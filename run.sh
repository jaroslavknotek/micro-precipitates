#!/bin/bash

PYTHONPATH='.' python precipitates/ws.py --crop-size 256 --filter-size 0
PYTHONPATH='.' python precipitates/ws.py --crop-size 128 --filter-size 0
PYTHONPATH='.' python precipitates/ws.py --crop-size 64 --filter-size 0
PYTHONPATH='.' python precipitates/ws.py --crop-size 64 --filter-size 16

