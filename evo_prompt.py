import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from template_de_adjusted import *
from template_ga_adjusted import *
import ollama
import os
import re
import seaborn as sns
from sklearn.metrics import confusion_matrix



ollama.pull('phi3.5')



class EvoPrompt:
    def __init__(self, initial_prompts, dev_set, evaluation_function, max_iterations, evolution_method):

        self.prompts = initial_prompts  # P0
        self.population_size = len(initial_prompts)  #N
        self.dev_set = dev_set  #D  
        self.evaluation_function = evaluation_function #f_D
        self.max_iterations = max_iterations    #T
        self.evolution_method = evolution_method
        self.population_scores = [self.evaluation_function(p, self.dev_set) for p in self.prompts]
        print(self.population_scores)
        self.best_scores = []

        self.de_version = ""

        #If DE, make user specify which version of the template_de he uses (1/2/3)
        if self.evolution_method == "DE":
            self.de_version = input("Which DE version do you want to use? Enter 1, 2, or 3: ").strip()
            if self.de_version not in ["1", "2", "3"]:
                raise ValueError("Invalid DE version. Please enter 1, 2, or 3.")

    def generate_prompt(self, prompt_content : str, used_model = 'phi3.5') -> str:
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
            self.prompts_source = "written"
            if self.evolution_method == "GA":
                return random.sample(self.prompts, k = 2)
            if self.evolution_method == "DE":
                if self.de_version in ["1", "3"]:
                    return self.prompts
                else:
                    return random.sample(self.prompts, k = 3)
        # IF generated_prompts
        else:
            self.prompts_source = "generated"
            if self.evolution_method == "GA":
                return random.sample(self.prompts, k = 2)
            if self.evolution_method == "DE":
                if self.de_version in ["1", "3"]:
                    return random.sample(self.prompts, k = 4)
                else:
                    return random.sample(self.prompts, k = 3)


    # Evolution

 # GA Evolution: Crossover + Mutation
    def evolve_ga(self, parents):
        parent1, parent2 = parents

        llm_prompt = templates_ga["cls"].replace("<prompt1>", parent1).replace("<prompt2>", parent2)
        full_response = self.generate_prompt(prompt_content=llm_prompt)
        #print("LLM PROMPT GIVEN: \n", llm_prompt)
        #print("FULL RESPONSE: \n", full_response)

        #safety procedure for situation where model rewrite the instruction given before the final prompt is presented
        matches = re.findall(r"<prompt>(.*?)</prompt>", full_response, re.DOTALL)
        child_prompt = ""
        for match in matches:
            if len(match.split()) > 5:
                child_prompt = match.strip()
                break
        if not child_prompt:

            raise ValueError("No valid prompt found.")

        #print(f"For parents:\n{parents[0]}\n{parents[1]}\nCreated child: {child_prompt}")
        if self.prompts_source == "written":
            child_prompt = child_prompt + " Tweet: [TWEET]"
        else:
            child_prompt = child_prompt + " Tweet: [INSERT TWEET HERE]"
        return child_prompt
        
    # DE Evolution: Differential Mutation + Crossover
    def evolve_de(self, parents):
        
        if self.de_version == "1":
            llm_prompt = v1["cls"]['sst-5']
        elif self.de_version == "2":
            llm_prompt = v2["cls"]['sst-5']
        else:
            llm_prompt = v3["cls"]['sst-5']

        if self.de_version in ["1", "3"]:
            parent0, parent1, parent2, parent3 = parents
            llm_prompt = llm_prompt.replace("<prompt0>", parent0).\
                            replace("<prompt1>", parent1).\
                            replace("<prompt2>", parent2).\
                            replace("<prompt3>", parent3)
        else:
            parent0, parent1, parent2 = parents
            llm_prompt = llm_prompt.replace("<prompt0>", parent0).\
                            replace("<prompt1>", parent1).\
                            replace("<prompt2>", parent2)
            
        full_response = self.generate_prompt(prompt_content=llm_prompt)

        #print("LLM PROMPT GIVEN: \n", llm_prompt)
        #print("FULL RESPONSE: \n", full_response)

        #safety procedure for situation where model rewrite the instruction given before the final prompt is presented
        matches = re.findall(r"<prompt>(.*?)</prompt>", full_response, re.DOTALL)
        child_prompt = ""
        for match in matches:
            if len(match.split()) > 5:
                child_prompt = match.strip()
                break
        if not child_prompt:
            # Check if there's <prompt> without closed </prompt>
            open_prompt_match = re.search(r"<prompt>(.*)", full_response, re.DOTALL)
            if open_prompt_match:
                child_prompt = open_prompt_match.group(1).strip()
            else:
                child_prompt = full_response.strip()
        
        if "[TWEET]" in child_prompt or "[INSERT TWEET HERE]" in child_prompt:
            return child_prompt
        else:           
            if self.prompts_source == "written":
                child_prompt = child_prompt + " Tweet: [TWEET]"
            else:
                child_prompt = child_prompt + " Tweet: [INSERT TWEET HERE]"
            return child_prompt

        #print(f"For parents:\n {parents[0]}\n{parents[1]}\n{parents[2]}\n{parents[3]}\nCreated child:\n {child_prompt}")
    

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
    def save_to_logs(self, log_file_path, iteration, new_prompt, new_score, invalid_samples):

        with open(log_file_path, "a") as log_file:
            log_file.write(f"Iteration {iteration}\n")
            log_file.write(f"Prompt: {new_prompt}\n")
            log_file.write(f"Accuracy Score: {new_score}\n")
            log_file.write(f"Invalid Samples: {invalid_samples}\n\n")
        print(f"Saved logs to {log_file_path}")


    def optimize(self, prompts_source, print_results = False):

        os.makedirs("logs_results_evoprompt", exist_ok=True)
        # Determine log file version
        existing_logs = [f for f in os.listdir("logs_results_evoprompt") if f.startswith(f"{prompts_source}_prompts_{self.evolution_method}_log_v")]
        if existing_logs:
            latest_version = max([int(f.split("_v")[1].split(".")[0]) for f in existing_logs])
            new_version = latest_version + 1
        else:
            new_version = 1

        if self.evolution_method == "GA":
            log_file_path = f"logs_results_evoprompt/{prompts_source}_prompts_{self.evolution_method}_log_v{new_version}.txt"
        else:
            log_file_path = f"logs_results_evoprompt/{prompts_source}_prompts_{self.evolution_method}_{self.de_version}_log_v{new_version}.txt"

        for t in range(self.max_iterations):
            parents = self.select_parents()

            new_prompt = self.evolve(parents)
            
            new_score, invalid_samples = self.evaluation_function(new_prompt, self.dev_set, return_invalid_samples=True, prompt_source = self.prompts_source)
            self.best_scores.append(max(self.population_scores))

            self.update_population(new_prompt, new_score)
            self.save_to_logs(log_file_path, t + 1, new_prompt, new_score, invalid_samples)

            if print_results:
                print("ITERATION NUMBER: ", t, "\n")
                print("PARENTS: ", parents)
                print("\nNEW PROMPT: ",new_prompt)
                print("\nACCURACY SCORE: ", new_score)
                print("\nINVALID SAMPLES: ",invalid_samples, "\n"*3)


        best_prompt_index = self.population_scores.index(max(self.population_scores))
        return self.prompts[best_prompt_index]

    #Progression over iterations
    def plot_score_progression(self):

        results_path = f"result_graphs/evoprompt_results"
        os.makedirs(results_path, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(self.best_scores, marker='o', linestyle='-', color='b')
        plt.title('Accuracy Score Progression Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Accuracy Score')
        plt.grid(True)

        #Saving as png
        if self.evolution_method == "DE":
            file_path = os.path.join(results_path, f"conf_matrix_{self.prompts_source}_prompt_{self.evolution_method}_v{self.de_version}_{self.max_iterations}.png")
        else:
            file_path = os.path.join(results_path, f"conf_matrix_{self.prompts_source}_prompt_{self.evolution_method}_{self.max_iterations}.png")
        print(f"Accuracy progression plot saved to {file_path}")
        plt.show()


    #Creating Confusion Matrix
    def plot_conf_matrix(self, y_true, y_pred):

        results_path = f"result_graphs/evoprompt_results"
        os.makedirs(results_path, exist_ok=True)

        conf_matrix = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive'], 
                    yticklabels=['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Sentiment')
        plt.ylabel('True Sentiment')
        plt.tight_layout()

        #Saving as png
        if self.evolution_method == "DE":
            file_path = os.path.join(results_path, f"conf_matrix_{self.prompts_source}_prompt_{self.evolution_method}_v{self.de_version}_{self.max_iterations}.png")
        else:
            file_path = os.path.join(results_path, f"conf_matrix_{self.prompts_source}_prompt_{self.evolution_method}_{self.max_iterations}.png")
        plt.savefig(file_path)
        print(f"Confusion matrix plot saved to {file_path}")
        plt.show()