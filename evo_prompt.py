import random
import numpy
from sklearn.metrics import accuracy_score
from template_de_adjusted import *
from template_ga_adjusted import *
import ollama
import os
import re
ollama.pull('llama3.2')



class EvoPrompt:
    def __init__(self, initial_prompts, dev_set, evaluation_function, max_iterations, evolution_method):

        self.prompts = initial_prompts  # P0
        self.population_size = len(initial_prompts)  #N
        self.dev_set = dev_set  #D  
        self.evaluation_function = evaluation_function #f_D
        self.max_iterations = max_iterations    #T
        self.evolution_method = evolution_method
        self.population_scores = [self.evaluation_function(p, self.dev_set) for p in self.prompts]

    def generate_prompt(self, prompt_content : str, used_model = 'llama3.2') -> str:
        response = ollama.chat(model = used_model, messages=[
        {
        'role': 'user',
        'content': prompt_content,
        },])
        return response['message']['content']
    


    #Selection
    def select_parents(self):
        # Random prompts chosen
        # IF written_prompts
        if self.population_size == 4:
            if self.evolution_method == "GA":
                return random.sample(self.prompts, k = 2)
            if self.evolution_method == "DE":
                return self.prompts
        # IF generated_prompts
        else:
            if self.evolution_method == "GA":
                return random.sample(self.prompts, k = 2)
            if self.evolution_method == "DE":
                return random.sample(self.prompts, k = 4)


    # Evolution

 # GA Evolution: Crossover + Mutation
    def evolve_ga(self, parents):
        parent1, parent2 = parents

        llm_prompt = templates_ga["cls"].replace("<prompt1>", parent1).replace("<prompt2>", parent2)
        full_response = self.generate_prompt(prompt_content=llm_prompt)

        # Extract the final prompt. Checking only text after "Final Prompt: " - (if <prompt> </prompt> or < >)
        final_prompt_section = full_response.split("Final Prompt:", 1)[-1]
        match_prompt = re.search(r"<prompt>(.*?)</prompt>", final_prompt_section, re.DOTALL)
        if not match_prompt:
            match_prompt = re.search(r"<(.*?)>", final_prompt_section, re.DOTALL)
        child_prompt = match_prompt.group(1).strip() if match_prompt else final_prompt_section.strip()

        print(f"For parents\n: {parents[0]}\n{parents[1]}\nCreated child: {child_prompt}")

        return child_prompt
    
    # DE Evolution: Differential Mutation + Crossover
    def evolve_de(self, parents):


        parent0, parent1, parent2, parent3 = parents

        llm_prompt = v1["cls"].replace("<prompt0>", parent0).\
                                    replace("<prompt1>", parent1).\
                                        replace("<prompt2>", parent2).\
                                            replace("<prompt3>", parent3)
        
        full_response = self.generate_prompt(prompt_content = llm_prompt)

        # Extract the final prompt. Checking only text after "Final Prompt: " - (if <prompt> </prompt> or < >)
        final_prompt_section = full_response.split("Final Prompt:", 1)[-1]
        match_prompt = re.search(r"<prompt>(.*?)</prompt>", final_prompt_section, re.DOTALL)
        if not match_prompt:
            match_prompt = re.search(r"<(.*?)>", final_prompt_section, re.DOTALL)
        child_prompt = match_prompt.group(1).strip() if match_prompt else final_prompt_section.strip()


        print(f"For parents\n: {parents[0]}\n{parents[1]}\n{parents[2]}\n{parents[3]}\nCreated child: {child_prompt}")

        return child_prompt
    
    def evolve(self, parents):
        if self.evolution_method == "GA":
            return self.evolve_ga(parents)
        elif self.evolution_method == "DE":
            return self.evolve_de(parents)
        


    # Update
    def update_population(self, new_prompt, new_prompt_score):

        # Replacing worst accuracy prompt with new if better
        min_score = min(self.population_scores)
        if new_prompt_score > min_score:
            min_index = self.population_scores.index(min_score)
            self.prompts[min_index] = new_prompt
            self.population_scores[min_index] = new_prompt_score
    
    def return_prompts(self):
        return {prompt: score for prompt, score in zip(self.prompts, self.population_scores)}


    # Save Process to Logs
    def save_to_logs(self, log_file_path, iteration, invalid_samples):
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Iteration {iteration}\n")
            for prompt, score in self.return_prompts().items():
                log_file.write(f"Prompt: {prompt}\nScore: {score}\n\n")
            log_file.write(f"Invalid Samples: {invalid_samples}\n\n")
        print(f"Saved logs to {log_file_path}")


    def optimize(self, prompts_source):
        os.makedirs("logs_results_evoprompt", exist_ok=True)

        # Determine log file version
        existing_logs = [f for f in os.listdir("logs_results_evoprompt") if f.startswith(f"{prompts_source}_prompts_log_v")]
        if existing_logs:
            latest_version = max([int(f.split("_v")[1].split(".")[0]) for f in existing_logs])
            new_version = latest_version + 1
        else:
            new_version = 1

        log_file_path = f"logs_results_evoprompt/{prompts_source}_prompts_log_v{new_version}.txt"

        for t in range(self.max_iterations):
            parents = self.select_parents()
            new_prompt = self.evolve(parents)
            new_score, invalid_samples = self.evaluation_function(new_prompt, self.dev_set, return_invalid_samples=True)
            self.update_population(new_prompt, new_score)
            self.save_to_logs(log_file_path, t + 1, invalid_samples)

        best_prompt_index = self.population_scores.index(max(self.population_scores))
        return self.prompts[best_prompt_index]


