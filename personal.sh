#!/bin/bash

# train person one
python main.py --train --maml --dataset=fbp5500 --person=0 --save-name=user0 --weight-classifier=0.4 --weight-classifier=0.6
#python main.py --explain --maml --dataset=fbp5500 --person=0 --load-from=user0 --save-name=user0

# test on other person
user_list=(7 9 15 16 17 19 22 24 25 26 28 31 33 41 42 47 52 54 56 58)
for i in "${user_list[@]}"
do
    python main.py --test --maml --dataset=fbp5500 --person=${i} --load-from=user0 --save-name=user0-user${i}
done

# train person two
python main.py --train --maml --dataset=fbp5500 --person=2 --save-name=user2 --weight-classifier=0.4 --weight-classifier=0.6
#python main.py --explain --maml --dataset=fbp5500 --person=0 --load-from=user0 --save-name=user0

# test on other person
user_list=(7 9 15 16 17 19 22 24 25 26 28 31 33 41 42 47 52 54 56 58)
for i in "${user_list[@]}"
do
    python main.py --test --maml --dataset=fbp5500 --person=${i} --load-from=user2 --save-name=user0-user${i}
done