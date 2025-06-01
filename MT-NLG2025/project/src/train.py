import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


def main():
    # Load dataset
    train_dataset = load_dataset("data/OpenMathReasoning", split="genselect")
    train_dataset = train_dataset.rename_column("problem", "prompt")
    train_dataset = train_dataset.rename_column("generated_solution", "completion")

    # Load model
    model_path = "/home/nfs02/model/Qwen2.5/Qwen2.5-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )

    # Train model
    training_args = SFTConfig(
        output_dir=f"qwen25-3b-sft",
        logging_steps=5,
        bf16=True,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=8192,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        dataset_num_proc=64,
        num_train_epochs=0.1,
        report_to="none",
    )
    trainer = SFTTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,  # type: ignore
    )
    trainer.train()


if __name__ == "__main__":
    main()
