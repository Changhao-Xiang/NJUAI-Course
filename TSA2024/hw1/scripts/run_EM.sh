python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model ExponentialSmoothingForecast \
    --transform IdentityTransform

python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model ExponentialSmoothingForecast \
    --transform NormalizationTransform

python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model ExponentialSmoothingForecast \
    --transform StandardizationTransform

python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model ExponentialSmoothingForecast \
    --transform MeanNormalizationTransform

python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model ExponentialSmoothingForecast \
    --transform BoxCoxTransform \
    --lamda 0.1

python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model ExponentialSmoothingForecast \
    --transform BoxCoxTransform \
    --lamda 2.0
