#!/bin/bash

while read requirements; do conda install --yes $requirement; done < requirements.txt
#while read requirements; echo $requirement; done < requirements.txt
#cd ..
