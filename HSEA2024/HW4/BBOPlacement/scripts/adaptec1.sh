for ((seed=2023; seed<=2025; seed++))
do
    python ../src/bbo_placer.py --dataset adaptec1 --seed $seed --timestamp &
done