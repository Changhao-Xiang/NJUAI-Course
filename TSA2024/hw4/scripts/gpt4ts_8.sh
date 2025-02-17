export CUDA_VISIBLE_DEVICES=3

python main.py \
    --model GPT4TS \
    --e_layers 8 \
    --data_path ./dataset/ETT/ETTh1.csv \

python main.py \
    --model GPT4TS \
    --e_layers 8 \
    --data_path ./dataset/ETT/ETTh2.csv \

python main.py \
    --model GPT4TS \
    --e_layers 8 \
    --dataset Custom \
    --data_path dataset/weather/weather.csv \

python main.py \
    --model GPT4TS \
    --e_layers 8 \
    --dataset Custom \
    --data_path dataset/exchange_rate/exchange_rate.csv \

python main.py \
    --model GPT4TS \
    --e_layers 8 \
    --dataset Custom \
    --data_path dataset/illness/national_illness.csv