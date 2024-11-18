from .Task import BaseTask
import json
from scipy.stats import norm


class ClassifyNumTask(BaseTask):
    task_name = "classifyNum"
    task_data_file = "classify_number_five_categories.jsonl"
    # task_data_file = "test_classify.jsonl"
    llm_eval = False

    def _single_eval_postprocess(self, instance):
        try:
            response = instance["response"]
            response = response.strip()[0]
            assert response in ["1", "2", "3", "4", "5"]
            true_label = instance["label"]
            response = int(response)
            if response == true_label:
                return 0
            elif response > true_label:
                return 1
            else:
                return (-1)
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


        
        