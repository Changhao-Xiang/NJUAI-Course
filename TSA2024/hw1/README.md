- Necessary dependencies: argparse, numpy
- To reproduce the reported results, you can run the following scripts under `./HW1/` directory:

    `bash scripts/visualize_dataset.sh`

    `bash scripts/run_LR.sh`

    `bash scripts/run_EM.sh`

    You can modify line 41 in `trainer.py` to `test=False` to obtain results with separately applied transformation during training and evaluation.

- The dataset used in my homework is electricity.csv