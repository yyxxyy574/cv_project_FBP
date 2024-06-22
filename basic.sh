#!/bin/bash

# only regressor

python main.py --train --dataset=fbp5500 --save-name=basic_0_10 --weight-classifier=0.0 --weight-classifier=1.0
#python main.py --explain --dataset=fbp5500 --load-from=basic_0_10 --save-name=basic_0_10
python main.py --test --dataset=fbp5500 --load-from=basic_0_10 --save-name=basic_0_10

# adjust loss by adding classification

python main.py --train --dataset=fbp5500 --save-name=basic_2_8 --weight-classifier=0.2 --weight-classifier=0.8
#python main.py --explain --dataset=fbp5500 --load-from=basic_2_8 --save-name=basic_2_8
python main.py --test --dataset=fbp5500 --load-from=basic_2_8 --save-name=basic_2_8

python main.py --train --dataset=fbp5500 --save-name=basic_4_6 --weight-classifier=0.4 --weight-classifier=0.6
#python main.py --explain --dataset=fbp5500 --load-from=basic_4_6 --save-name=basic_4_6
python main.py --test --dataset=fbp5500 --load-from=basic_4_6 --save-name=basic_4_6