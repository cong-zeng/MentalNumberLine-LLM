from .Task import BaseTask
import json




class ComparisonTask(BaseTask):
    task_name = "comparison"
    task_data_file = "integer_pair_comparison.jsonl"
    llm_eval = False

    def _single_eval_postprocess(self, instance):
        response = str(instance["response"])
        response = response.strip()
        # split by newline and whitespace
        response = response.split()
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