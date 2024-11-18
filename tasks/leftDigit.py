from .Task import BaseTask
import json
from scipy.stats import norm


class LeftDigit(BaseTask):
    task_name = "left_digit"
    # task_data_file = "left_digit_diff_comparison.jsonl"
    task_data_file = "left_digit_diff_comparison_easy.jsonl"
    llm_eval = False

    def _single_eval_postprocess(self, instance):
        try:
            response = instance["response"]
            is_wrong_answer_left_digit = instance["is_wrong_answer_left_digit"]
            response = response.strip()[0]
            assert response in ["A", "B"]
            true_label = instance["label"]
            if response == true_label:
                return 0
            return 1 if is_wrong_answer_left_digit else -1
        except:
            print("Warning: Invalid response format")
            return 0
    
    def _compute_score(self):
        single_score = self.data_df["single_score"]
        group_A = len(single_score[single_score == 1])
        group_B = len(single_score[single_score == -1])
        print(group_A, group_B)
        n = group_A + group_B
        p_hat = group_A / n
        p_0 = 0.5
        z = (p_hat - p_0) / ((p_0 * (1 - p_0) / n) ** 0.5)
        alpha = 0.05
        z_critical = norm.ppf(1 - alpha)
        p_value = 1 - norm.cdf(z)
        result = int(z > z_critical)
        
        return ({
            "group_A": group_A,
            "group_B": group_B,
            "n": n,
            "p_hat": p_hat,
            "z": z,
            "p_value": p_value,
            "result": result
        })


        
        