#!/bin/bash
python syntactic_data.py --index 1 > mylogs/temp1.log 2>&1 &
python syntactic_data.py --index 2 > mylogs/temp2.log 2>&1 &
