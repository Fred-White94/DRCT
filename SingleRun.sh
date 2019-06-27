#!/bin/bash



#Bins=(`ls *SimBin* | sort --version-sort -f`)
#Bins=randomSIM.txt

Bins=(`find 19_04_30_SimBin_{5,6,7,8,9,10}`)

for i in "${Bins[@]}"
do
  #startTime=$(date +%s)
  #endTime=$(date -d 15:30 +%s)
  #timeToWait=$(($endTime- $startTime))
  #sleep $timeToWait
  echo stacking
  python SDApretrain.py "$i" ~/"$i"_Pre_MODEL teX_preFT_Bin_"$i" teY_preFT_Bin_"$i" trX_preFT_Bin_"$i" trY_preFT_Bin_"$i" 
  echo "$i" "$i"_DONE
  python Fine_Tune.py teX_preFT_Bin_"$i" teY_preFT_Bin_"$i" "$i"_Pre_MODEL.meta teX_FT_Bin_"$i" teY_FT_Bin_"$i" teX_pred_FT"$i" BIN_"$i"_HiddenL_FOR_R


done


