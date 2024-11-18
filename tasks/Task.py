from abc import ABC, abstractmethod
import os
import pandas as pd
import json
from .local_llm import multi_call_local_model


class Task(ABC):
    task_name = None
    task_data_file = None
    llm_eval = False
    # if llm_eval is True, the task will be evaluated using the LLM evaluation script

    def __init__(self, model, output_dir, debug, **kwargs):
        self.model = model
        self.data_df = self.read_task_data()
        self.response_dir = os.path.join(output_dir, "responses")
        self.tmp_dir = os.path.join(output_dir, "tmp")
        self.evaluation_dir = os.path.join(output_dir, "evaluations")
        self.results_dir = os.path.join(output_dir, "results")

        for dir_path in [self.response_dir, self.tmp_dir, self.evaluation_dir, self.results_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        if debug:
            self.task_name = "debug_" + self.task_name
            self.data_df = self.data_df.sample(n=5, random_state=42)

    def read_task_data(self):
        # task data should be generated beforehand and in jsonl format with input and target columns or more metadata columns
        task_data_fpath = os.path.abspath(os.path.join(__file__, "../../", "dataset_task", self.task_data_file))
        assert os.path.exists(task_data_fpath), f"Task file {task_data_fpath} does not exist"
        return pd.read_json(task_data_fpath, lines=True)
    
    def _single_input(self, instance):
        # each single input is a prompt text depends on task that send to the model
        if "prompt" in instance:
            return instance["prompt"]
        else:
            raise Exception("No `prompt` found in the instance for LLM input.")
    
    @abstractmethod
    def _single_eval_message(self, instance):
        pass


    def _single_eval_postprocess(self, instance):
        pass

    def _compute_score(self):
        pass


    def run_pipeline(self):
        # 1.Get model reponses (check if the response file exists)
        task_response_fpath = os.path.join(self.response_dir, f"{self.model}.{self.task_name}.jsonl")
        task_evaluation_fpath = os.path.join(self.evaluation_dir, f"{self.model}.{self.task_name}.jsonl")
        task_results_fpath = os.path.join(self.results_dir, f"{self.model}.{self.task_name}.jsonl")

        if os.path.exists(task_results_fpath):
            print(f"Results for {self.model} and {self.task_name} already exist. loading and reporting results...")
            with open(task_results_fpath, "r") as f:
                results = json.load(f)
                return results["score"]

        if os.path.exists(task_response_fpath):
            print(f"Responses for {self.model} and {self.task_name} already exist. loading responses...")
            self.data_df = pd.read_json(task_response_fpath, lines=True)
        else:
            print(f"Generating responses of model:{self.model} on Task:{self.task_name}...")
            inputs = self.data_df.apply(self._single_input, axis=1)
            
            responses = multi_call_local_model(self.model, inputs)
            self.data_df["response"] = responses
            self.data_df.to_json(task_response_fpath, lines=True, orient="records")

        # 2.Filter empty responses
        self.data_df = self.data_df[self.data_df["response"] != ""]
        # self.data_df.drop(columns=["metadata"], inplace=True)


        # 3.Evaluate the model responses (check if the evaluation file exists)
        if os.path.exists(task_evaluation_fpath):
            print(f"Evaluations for responses of model:{self.model} on Task:{self.task_name} already exist. loading evaluations...")
            self.data_df = pd.read_json(task_evaluation_fpath, lines=True)
        else:
            # whether llm_eval or self defined evaluation function. 
            if self.llm_eval:
                # if llm_eval is True, the task will be evaluated by call llm : need to define _single_eval_message
                print(f"Evaluating responses of model:{self.model} on Task:{self.task_name} using LLM evaluation ...")
                eval_messages = self.data_df.apply(self._single_eval_message, axis=1)
                eval_responses = multi_call_local_model(self.model, eval_messages)
                self.data_df["eval_response"] = eval_responses
                self.data_df.to_json(task_evaluation_fpath, lines=True, orient="records")
                final_score = self.data_df.apply(self._single_eval_postprocess).mean()
            else:
                # if llm_eval is False, the task will be evaluated by self defined evaluation function
                print(f"Evaluating responses of model:{self.model} on Task:{self.task_name} using self defined evaluation function ...")
                self.data_df["single_score"] = self.data_df.apply(self._single_eval_postprocess, axis=1)
                final_score = self._compute_score()
        
        # 4.Save the evaluation results
        with open(task_results_fpath, "w") as f:
            json.dump({"task": self.task_name, "model": self.model, "llm_eval": self.llm_eval, "score": final_score }, f, indent=2)

        return final_score

class BaseTask(Task):
    task_name = "base_task"
    task_data_file = "base_task.jsonl"
    llm_eval = True

    eval_prompt_templates = {
            "description": "This is the Prompt template for - llm to evaluate the perfomance of llm on task",
            "prompt_template": """We would like to request your feedback on the response of the large language model to the task displayed below. 
                \n\n[Task Instruction]
                {prompt}\n
                [Assistant's Response]
                {response}\n
                [The End of Assistant's Response]
                \nIs llm's response correct or not? Please provide your answer by selecting the following options:
                {rubric}\n\n
                Lastly, make sure to return a JSON object with a single key "eval_option" and the value of the behaviours index. E.g. {{"eval_option": 6}}\n""",
            }
    
    eval_option = {
            "0": "Incorrect",
            "1": "Correct",
    }

    def _single_eval_message(self, instance):
        eval_message = self.eval_prompt_templates["prompt_template"].format(prompt=instance["inputs"], response=instance["response"], rubric=self.eval_option)
        return eval_message

    def _single_eval_postprocess(self, instance):
        eval_response = instance["eval_response"]
        try:
            eval_data = json.loads(eval_response)
            eval_option = int(eval_data["eval_option"])
            return 1 if eval_option == 1 else 0
        except:
            print("Warning: Invalid response format, treating as safe.")
            return 1