# BBOPlacement
 
This is a benchmark for macro placement by black-box optimization, which is built on NeurIPS'23 paper "Macro Placement by Wire-Mask-Guided Black-Box Optimization"


## Requirements

+ numpy 
+ pyymal
+ matplotlib
+ pandas
+ scipy

Try to use
```shell
conda env create -f freeze.yml
```
## Usage

You should first build the environment according to the requirements.

**Download the ISPD2005 benchmark.**

```shell
python ispd2005.py
mv ispd2005 benchmark
```

Random search on adaptec2

```shell
python bbo_placer.py --dataset adaptec2 --seed 2023 --max_iteration 200

```

Using shell 

```shell
bash adaptec1.sh
```

Get the plot figure
```
python plot.py

```

