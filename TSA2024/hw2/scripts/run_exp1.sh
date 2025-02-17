python code/main.py \
    --model TsfKNN \
    --embedding lag \
    --distance euclidean \
    --decomposition no

python code/main.py \
    --model TsfKNN \
    --embedding lag \
    --distance manhattan \
    --decomposition no

python code/main.py \
    --model TsfKNN \
    --embedding lag \
    --distance chebyshev \
    --decomposition no

python code/main.py \
    --model TsfKNN \
    --embedding fft \
    --distance euclidean \
    --decomposition no

python code/main.py \
    --model TsfKNN \
    --embedding fft \
    --distance manhattan \
    --decomposition no

python code/main.py \
    --model TsfKNN \
    --embedding fft \
    --distance chebyshev \
    --decomposition no