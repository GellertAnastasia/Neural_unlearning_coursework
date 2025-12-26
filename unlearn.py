import torch
from transformers import TrainingArguments
from methods.grad_ascent import GradAscentTrainer
from methods.grad_diff import GradDiffTrainer
from methods.wga import WeightedGradAscentTrainer
from src.prepare_config import (parse_arguments, load_config)
from src.seed import set_seed
from src.model_loader import (load_finetuned_model, attach_unlearning_lora)
from src.data_loader import load_data
from src.generation import (format_chat_prompt, generate_test)
from src.data_collator import UnlearningDataCollator


def main(config):
    
    set_seed(config["training"]["seed"])

    print("Load fine-tuned model")
    model_dir = config['model']['lora_path']

    model, tokenizer = load_finetuned_model(model_dir)
    model = attach_unlearning_lora(model)
    model.eval()
    code = '''
    def median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    if n % 2 == 0:
        return (sorted_lst[n//2 - 1] + sorted_lst[n//2]) / 2
    else:
        return sorted_lst[n//2]
    '''
    prompt = format_chat_prompt(tokenizer, code)
    test = generate_test(model, tokenizer, prompt)
    print(test)
    model.train()

    print("Load  datasets...")
    train_dataset = load_data(
        forget_path=config['data']['forget_path'],
        retain_path=config['data']['retain_path'],
        tokenizer=tokenizer,
    )

    data_collator = UnlearningDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        logging_dir=f"{config['training']['output_dir']}/logs",
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=100,
        report_to="none",
        remove_unused_columns=False,
        seed=config["training"]["seed"],
        max_grad_norm=1.0,
        warmup_steps=config["training"].get("warmup_steps", 100),
        weight_decay=config["training"].get("weight_decay", 0.01),
        fp16=False,
        bf16=False,
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
    )

    if config['training']['type']=='GradAscentTrainer':
        trainer = GradAscentTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif config['training']['type']=='GradDiffTrainer':
        trainer = GradDiffTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif config['training']['type']=='WeightedGradAscentTrainer':
        trainer = WeightedGradAscentTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    train_result = trainer.train()

    print(f"\n End unlearning")
    print(f"  Total steps: {train_result.global_step}")
    print(f"  Total time: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"  Final loss: {train_result.metrics['train_loss']:.4f}")
    unlearned_model = trainer.model 
    
    unlearned_model.eval()
    with torch.no_grad():
        code = ''' 
        def median(lst):
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        if n % 2 == 0:
            return (sorted_lst[n//2 - 1] + sorted_lst[n//2]) / 2
        else:
            return sorted_lst[n//2]
        '''
        prompt = format_chat_prompt(tokenizer, code)
        test = generate_test(unlearned_model, tokenizer, prompt)
        print(test)

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    main(config)