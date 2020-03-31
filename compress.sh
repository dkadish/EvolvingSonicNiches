#!/bin/bash

tar czvf ./data/$1_data.tar.gz $(find ./data/$1/ -name *.joblib)
cp ./data/$1_data.tar.gz ~/Documents/ownCloud/Documents/projects/EvolvingNiches/
