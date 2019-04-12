#!/bin/sh

: '
mkdir -p potsdam

#Ortho_RGB
mkdir -p potsdam/2_Ortho_RGB
cp ../potsdam/2_Ortho_RGB.zip ./potsdam
unzip ./potsdam/2_Ortho_RGB -d ./potsdam/2_Ortho_RGB
cp ./potsdam/2_Ortho_RGB/2_Ortho_RGB/*.tif ./potsdam/2_Ortho_RGB
rm -r ./potsdam/2_Ortho_RGB/2_Ortho_RGB/ ./potsdam/2_Ortho_RGB.zip
#Labels
mkdir -p potsdam/5_Labels_all
cp ../potsdam/5_Labels_all.zip ./potsdam
unzip ./potsdam/5_Labels_all -d ./potsdam/5_Labels_all
rm ./potsdam/5_Labels_all.zip

#cp ./potsdam/5_Labels_all/5_Labels_all/*.tif ./potsdam/5_Labels_all
#rm -r ./potsdam/5_Labels_all/5_Labels_all/ ./potsdam/5_Labels_all.zip

OUTPUT="$(ls -1 ./potsdam/2_Ortho_RGB)"
for i in $OUTPUT
do
echo "$i"
done
echo "${#OUTPUT[@]}"
'

array=( $( ls ./potsdam/2_Ortho_RGB ) )
size=${#array[@]}

a=1
for i in "${array[@]}"
do
  i=a
  let a+=1
done
echo ${array[0]}
#for i in {01..38}
#do
#  echo $i
#done
