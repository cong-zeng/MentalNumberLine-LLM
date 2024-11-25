from .Task import BaseTask
import json
import re

class AdditionTask(BaseTask):
    task_name = "addition"
    task_data_file = "integer_addition.jsonl"
    llm_eval = False

    def _single_eval_postprocess(self, instance):
        response = str(instance["response"])
        response = response.strip()
        # split by newline and whitespace
        response = re.split(r'[.\n\s]+', response)
        if response[0].isdigit():
            answer = response[0]
            if str(instance["label"]) == answer:
                return 1
            else:
                return 0
        else:
            print("Warning: Invalid response format")
            return 0

    def _compute_score(self):
        return self.data_df["single_score"].mean()