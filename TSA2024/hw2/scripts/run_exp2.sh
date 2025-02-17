python code/main.py \
    --model TsfKNN \
    --embedding lag \
    --distance manhattan \
    --decomposition ma

python code/main.py \
    --model TsfKNN \
    --embedding lag \
    --distance manhattan \
    --decomposition differential

python code/main.py \
    --model DLinear \
    --decomposition ma

python code/main.py \
    --model DLinear \
    --decomposition differential