python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model LinearRegressionForecast \
    --transform IdentityTransform

python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model LinearRegressionForecast \
    --transform NormalizationTransform

python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model LinearRegressionForecast \
    --transform StandardizationTransform

python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model LinearRegressionForecast \
    --transform MeanNormalizationTransform

python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model LinearRegressionForecast \
    --transform BoxCoxTransform \
    --lamda 0.1 

python code/main.py \
    --data_path ./dataset/electricity/electricity.csv \
    --model LinearRegressionForecast \
    --transform BoxCoxTransform \
    --lamda 2.0
