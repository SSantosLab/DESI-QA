#!/usr/bin
# Script to execute 50 arcsequence moves, given that the user provides the config file
DEFAULTDIR=/home/msdos/DESI-QA/conf/

echo -e "Please enter the name of the config file: "
read filename
echo "Filename $filename read"

fulldir=$DEFAULTDIR$filename

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
do
  echo "Testing arcsequence ... iteration $i/50"
  python3 /home/msdos/DESI-QA/run_ang.py -c $fulldir
done
