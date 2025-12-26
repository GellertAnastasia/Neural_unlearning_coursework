import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.prepare_config import (parse_arguments, load_config)
from src.generation import (format_chat_prompt, generate_test)
from src.evaluator import UnlearningEvaluator

def main(config):
    model_dir = config['model']
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    dataset_path = config['data']
    with open(dataset_path, 'r', encoding='utf-8') as f:
        eval_codes = [json.loads(line)['code'] for line in f if line.strip()]

    prompts = [
        format_chat_prompt(tokenizer, code) for code in eval_codes
    ]

    evaluator = UnlearningEvaluator(
        model=model,
        tokenizer=tokenizer,
        generator_fn=generate_test,
        k=3
    )

    result = evaluator.evaluate_all_at_k(prompts)

    print(f"Leak rate: {result['leak_at_k']}")
    print(f"Prompts with leak: {result['prompts_with_leak']}/{result['total_prompts']}")
    print("\nPattern statistics:")
    for pattern, count in result['pattern_stats'].items():
        pattern_name = pattern.split('\\s')[0].replace(':', '')
        print(f"  - {pattern_name}: {count} occurrences")

    if result['leak_at_k'] < 0.05:
        print("\n High leak")
    elif result['leak_at_k'] < 0.2:
        print("\n Medium leak")
    else:
        print("\n Low leak")

    print(f"Syntax Correctness Rate: {result['syntax_success_at_k']}")
    print(f"  (Correct: {result['successful_prompts']}/{result['total_prompts']})")
    if result['syntax_success_at_k'] > 0.95:
        print("High pass")
    else:
        print("Low pass")
        print("\n" + "="*60)

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    main(config) 