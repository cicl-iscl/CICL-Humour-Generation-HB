from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import gc
import torch
import json


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

class Scorer:
    """ Scorer that uses a language model to evaluate the funniness of jokes from multiple personas. """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", personas: list | None = None):
        self.personas = personas or ["self-defeating", "affiliative", "self-enhancing", "aggressive"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )

    def _generate_response(self, prompt, temperature=0.6, top_p=0.9, max_new_tokens=512):
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                **inputs,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=True
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _construct_prompt(self, persona, joke):
        return f"""
    You are a person who enjoys {persona} humour. 
    Do you think the following joke is funny: \n{joke}\n
    Reply with a valid JSON object that contains `final_answer` (either "yes" or "no") and `reason`.
    """.strip()

    def _extract_json(self, text: str):
        text = text.strip().removeprefix("```json").removesuffix("```")
        try:
            return json.loads(text)
        except:
            # crude fallback
            text = text[text.find("{"):text.rfind("}")+1]
            return json.loads(text)

    def _get_crowd_score(self, joke: str):
        """Get crowd score for a joke by querying multiple personas in parallel."""

        def ask(persona):
            prompt = self._construct_prompt(persona, joke)
            resp = self._generate_response(prompt)
            try:
                data = self._extract_json(resp)
                return 1.0 if data["final_answer"].lower() == "yes" else 0.0
            except Exception:
                return 0.0

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(ask, self.personas))
        
        torch.cuda.empty_cache()
        return sum(results)

    def crowd_score_rewards(self, completions, **kwargs):
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(self._get_crowd_score, completions))