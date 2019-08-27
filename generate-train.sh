#!/bin/bash

for FIRST in $(seq 4502 50 4952)
do

    LAST=$(($FIRST + 49))
    time for i in $(seq $FIRST $LAST); do echo $i; ./generate.py | gzip -c > /home/epsilon/chess/games/$i.gz; done

    (cd games; ls *.gz | sort -n | tail -50 | xargs cat) | ./train.py

done
