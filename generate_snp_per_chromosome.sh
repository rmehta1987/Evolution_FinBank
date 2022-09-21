#!/usr/bin/env bash

for chr in {1..22}; do 

awk -v awkchr=$chr '{if (awkchr == $1) print $2}' /mnt/sda/home/ludeep/Desktop/PopGen/eqtlGen/Reference/1kg.v3/EUR/EUR.bim > ReferenceData/data_chr$chr.txt; 

done

