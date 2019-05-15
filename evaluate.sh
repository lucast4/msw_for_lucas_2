#!/bin/bash
a="`grep -oh "python.*" ./g-command.txt` --evaluate_checkpoint=./model.p"
echo $a
eval $a --evaluate_checkpoint=./model.p
