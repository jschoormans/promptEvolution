import shutil
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import random
from transformers import CLIPModel, CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV1
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from transformers import AutoTokenizer

TEMPERATURE = 0.95


# use blend and weighting from here: https://huggingface.co/docs/diffusers/en/using-diffusers/weighted_prompts


# Load a small LLM for generating text
# llm_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device=0 if torch.cuda.is_available() else -1)
llm_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device=0 if torch.cuda.is_available() else -1)

# Load SD1.5 for image generation (for evaluation purposes)
sd_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None)
sd_pipeline = sd_pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# Load the aesthetics predictor
model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"
predictor = AestheticsPredictorV1.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
predictor = predictor.to("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer (just for checking the number of unique tokens)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def generate_prompt(prompt="Generate a description of a random image. Include a detailed description of the image, including the main subject, background, details, style, etc. New description:"):
    generated_text = llm_pipeline(prompt, max_new_tokens=77, do_sample=True, temperature=TEMPERATURE)[0]['generated_text']

    generated_text = generated_text[len(prompt):].strip()
    # remove newlines and slashes and spaces and tabs
    generated_text = generated_text.replace('\n', '').replace('/', '').replace('\t', '')
    # Remove the prompt from the beginning of the generated text
    return generated_text

def evaluate_fitness(prompt, generation, population_nr, current_run):
    # Generate an image based on the prompt
    
    image = sd_pipeline(prompt, num_inference_steps=30).images[0]
    
    # Preprocess the image for aesthetics prediction
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(predictor.device) for k, v in inputs.items()}
    # Get aesthetics score
    with torch.no_grad():
        outputs = predictor(**inputs)
    aesthetic_score = outputs.logits.item()

    image_name = f"generation_{generation}_{population_nr}_{aesthetic_score}_{prompt[:50].replace(' ', '_')}.png"
    image.save(f"images/{current_run}/{image_name}")

    # print(f"Aesthetic score for prompt: \n '{prompt}...\n': {aesthetic_score}\n\n")
    return {
        "aesthetic_score": aesthetic_score,
        "image_name": image_name
    }

def select_parents(population, fitness_scores, exponent=2):
    # Remove bottom 25%
    num_parents = 10*len(population)
    num_elite = int(0.75 * len(population))
    sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)
    elite_indices = sorted_indices[:num_elite]
    elite_population = [population[i] for i in elite_indices]
    
    # Rank-based weights: best gets highest weight
    n = len(elite_population)
    weights = [(n - rank) ** exponent for rank in range(n)]
    
    # Weighted sampling with replacement
    return random.choices(elite_population, weights=weights, k=num_parents)


def llm_crossover(parent1, parent2):
    prompt = f"Combine these two descriptions into one, creating a new, unique description: \n- {parent1}\n- {parent2}, New description:"
    return generate_prompt(prompt)


def mutate(individual):
    if random.random() < 0.2:  # 20% chance of mutation
        return generate_prompt(f"Modify this description: {individual}. Modified description:")
    return individual


def population_variation(population):
    # calculate the number of unique tokens in the population.
    population_tokens = set()
    ntokens = 0
    for prompt in population:
        tokens = tokenizer.encode(prompt)
        # Filter out padding token (10247) since it doesn't contribute to meaningful variation between prompts
        tokens = [token for token in tokens if token != 10247]
        # Convert tokens list to tuple so it can be added to set
        population_tokens.add(tuple(tokens))
        ntokens += len(tokens)
    # Count total unique tokens across all prompts
    all_tokens = set()
    for tokens in population_tokens:
        all_tokens.update(tokens)
    
    return len(all_tokens) / ntokens




def genetic_algorithm_llm(population_size, generations):

    current_run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # create a folder for the current run
    os.makedirs(f"images/{current_run}", exist_ok=True)
    os.makedirs(f"images/{current_run}/best", exist_ok=True)

    population = [generate_prompt() for _ in range(population_size)]
    
    print(f"Population variation: {population_variation(population)}")

    fitness_stats = []

    for generation in range(generations):
        print(f"Generation {generation + 1}")
        f = [evaluate_fitness(prompt, generation, population_nr, current_run) for population_nr, prompt in enumerate(population)]
        
        # Selection
        fitness_scores = [fitness_score["aesthetic_score"] for fitness_score in f]
        
        # Create new population
        new_population = []

        keep_best_size = int(np.ceil(0.1*population_size))
        new_population_size = population_size - keep_best_size
        print(f"Keep best size: {keep_best_size}, New population size: {new_population_size}")

        parents = select_parents(population, fitness_scores)  # Select top 75%
        for _ in range(new_population_size):
            parent1, parent2 = random.sample(parents, 2)
            child = llm_crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        

        # keep best 10 % of the population
        best_population = sorted(population, key=lambda x: fitness_scores[population.index(x)], reverse=True)[:keep_best_size]

        # keep 90 % of the best population
        population = best_population + new_population[:new_population_size]    
        
        # Print the best individual of this generation based on the fitness scores
        best_prompt = population[np.argmax(fitness_scores)]
        best_image_name = f[np.argmax(fitness_scores)]["image_name"]
        print(f"Generation {generation + 1}: Best Prompt - {best_prompt}")
        print(f"Fitness Score: {max(fitness_scores)}")

        fitness_stats.append({"generation": generation, "prompt": best_prompt, "mean_fitness": np.mean(fitness_scores), "best_fitness": max(fitness_scores), "population_variation": population_variation(population)})
        print(fitness_stats)

        # Plot fitness statistics over generations
        plt.figure(figsize=(10, 6))
        mean_fitness = [stat["mean_fitness"] for stat in fitness_stats]
        best_fitness = [stat["best_fitness"] for stat in fitness_stats]
        
        plt.plot(mean_fitness, label="Mean Fitness", linestyle='-', marker='o', markersize=4)
        plt.plot(best_fitness, label="Best Fitness", linestyle='-', marker='s', markersize=4)
        
        plt.title("Fitness Scores Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(f"fitness_stats_{current_run}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Plot population variation over generations
        plt.figure(figsize=(10, 6))
        population_variation_stats = [stat["population_variation"] for stat in fitness_stats]
        plt.plot(population_variation_stats, label="Population Variation", linestyle='-', marker='o', markersize=4)
        plt.title("Population Variation Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Population Variation")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        plt.savefig(f"population_variation_{current_run}.png", dpi=300, bbox_inches='tight')
        plt.close() 


        # copy the best image to the current run folder
        shutil.copy(f"images/{current_run}/{best_image_name}", f"images/{current_run}/best/{generation}_{np.round(max(fitness_scores), 3)}.jpg")
    return population

# Run the genetic algorithm
if __name__ == "__main__":
    final_population = genetic_algorithm_llm(population_size=200, generations=100)
    # final_population = genetic_algorithm_llm(population_size=5, generations=100)
    print("Done.")