#!/bin/bash

for RUN in 0 1 2
do
	python runs/evolve_1_species_with_noise.py -d $1 -g 4 -r 1 --run-id $RUN --noise-channel 0 1 2 --noise-level 1.0 --noise-generation 0
done

python dataframe/combine.py ./data/$1
