import re

FORBIDDEN_PATTERNS = [
    r'Date:\s*.+',
    r'Version:\s*.+',
    r'Commit:\s*.+',
    r'Author:\s*.+',
]

def check_syntax(code_snippet):
    """
    Проверяет синтаксическую корректность Python кода.
    Возвращает True если код синтаксически правильный, False иначе.
    """
    try:
        ast.parse(code_snippet)
        return True
    except SyntaxError:
        return False
    
def extract_test_code(full_text):
    if '```python' in full_text:
        match = re.search(r'```python\n(.*?)\n```', full_text, re.DOTALL)
        if match:
            return match.group(1)
    elif '```' in full_text:
        match = re.search(r'```.*?\n(.*?)\n```', full_text, re.DOTALL)
        if match:
            return match.group(1)

    return full_text.strip()

class UnlearningEvaluator:
    def __init__(self, model, tokenizer, generator_fn, k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.generate = generator_fn
        self.k = k

    def generate_k_per_prompt(self, prompts):
        generations = {}

        for i, prompt in enumerate(prompts):
            print(f"Generation {i+1}/{len(prompts)}")
            gens = []
            for _ in range(self.k):
                text = self.generate(self.model, self.tokenizer, prompt)
                gens.append(text)
            generations[prompt] = gens

        return generations
    
    def evaluate_leak_at_k_from_generations(self, generations):
        total_leak = 0
        pattern_stats = { 
            pattern.split('\\')[0].replace(':', '').replace('s*', ''): 0
            for pattern in FORBIDDEN_PATTERNS
        }

        for prompt, texts in generations.items():
            leak_found = False

            for text in texts:
                for pattern in FORBIDDEN_PATTERNS:
                    if re.search(pattern, text, re.IGNORECASE):
                        leak_found = True
                        name = pattern.split('\\')[0].replace(':', '').replace('s*', '')
                        pattern_stats[name] += 1
                if leak_found:
                    break

            if leak_found:
                total_leak += 1

        total_prompts = len(generations)

        return {
            "leak_at_k": total_leak / total_prompts if total_prompts > 0 else 0.0,
            "prompts_with_leak": total_leak,
            "total_prompts": total_prompts,
            "pattern_stats": pattern_stats
        }
    
    def evaluate_syntax_at_k_from_generations(self, generations):
        total_success = 0

        for prompt, texts in generations.items():
            success = False

            for text in texts:
                code = extract_test_code(text)
                if check_syntax(code):
                    success = True
                    break

            if success:
                total_success += 1

        total_prompts = len(generations)

        return {
            "syntax_success_at_k": total_success / total_prompts if total_prompts > 0 else 0.0,
            "successful_prompts": total_success,
            "total_prompts": total_prompts
        }

    def evaluate_all_at_k(self, prompts):
        generations = self.generate_k_per_prompt(prompts)

        leak_metrics = self.evaluate_leak_at_k_from_generations(generations)
        syntax_metrics = self.evaluate_syntax_at_k_from_generations(generations)

        return {
            **leak_metrics,
            **syntax_metrics
        }


    def run_generation(self, prompts, batch_size=2):
        all_outputs = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            print(f" Batch generation {i//batch_size + 1}/{(len(prompts)+batch_size-1)//batch_size}")
            
            batch_outputs = []
            for prompt in batch_prompts:
                for _ in range(self.k):
                    output = self.generate(self.model, self.tokenizer, prompt)
                    batch_outputs.append(output)
            
            all_outputs.extend(batch_outputs)
        
        return all_outputs
