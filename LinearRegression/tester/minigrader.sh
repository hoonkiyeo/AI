#
python3 hw5.py toy.csv |tr -d [], > test.txt
sed -n -e '/^Q4/,/^Q6/{//!d;};' -e '/Q6b/q;p' test.txt > test2.txt
#cat test.txt
sed -E -e 's/^Q[1-9][a-z]?\://g;' test2.txt | tr '\n' ' ' | sed -E '$s/ $/\n/' |sed -E -e 's/^[ \t]*//' -e 's/  */ /g' > test.txt
rm test2.txt
python3 tester.py
#python3 tester.py>>