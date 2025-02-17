- Necessary dependencies: argparse, numpy, torch, reformer_pytorch, transformers, pandas, matplotlib
- To reproduce the reported results, you can run the following scripts under `./HW4/` directory, for evaluation of ARIMA and ThetaMethod forecasting models:

    `bash scripts/patchtst.sh`

    `bash scripts/gpt4ts_2.sh`

    for ablation on number of layers:

    `bash scripts/gpt4ts_4.sh`

    `bash scripts/gpt4ts_8.sh`

- The datasets used in HW3 are `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Weather`, `Exchange`, `ILI`.