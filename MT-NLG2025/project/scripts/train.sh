export CUDA_VISIBLE_DEVICES=0,1,2,3
export MAIN_PROCESS_PORT=29666

accelerate launch --main_process_port $MAIN_PROCESS_PORT \
    --config_file accelerate_configs/deepspeed_zero2.yaml \
    src/train.py