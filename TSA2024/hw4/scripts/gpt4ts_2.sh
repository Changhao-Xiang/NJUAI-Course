export CUDA_VISIBLE_DEVICES=1

python main.py \
    --model GPT4TS \
    --data_path ./dataset/ETT/ETTh1.csv \

python main.py \
    --model GPT4TS \
    --data_path ./dataset/ETT/ETTh2.csv \

python main.py \
    --model GPT4TS \
    --data_path ./dataset/ETT/ETTm1.csv \

python main.py \
    --model GPT4TS \
    --data_path ./dataset/ETT/ETTm2.csv \

python main.py \
    --model GPT4TS \
    --dataset Custom \
    --data_path dataset/electricity/electricity.csv \

python main.py \
    --model GPT4TS \
    --dataset Custom \
    --data_path dataset/traffic/traffic.csv \

python main.py \
    --model GPT4TS \
    --dataset Custom \
    --data_path dataset/weather/weather.csv \

python main.py \
    --model GPT4TS \
    --dataset Custom \
    --data_path dataset/exchange_rate/exchange_rate.csv \

python main.py \
    --model GPT4TS \
    --dataset Custom \
    --data_path dataset/illness/national_illness.csv