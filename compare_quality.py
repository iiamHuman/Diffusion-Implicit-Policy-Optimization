import torch
import pandas as pd
import numpy as np
from PIL import Image
import time
import os
import argparse
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, DDPMScheduler
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from multiprocessing import Pool, Manager, cpu_count
import torch.multiprocessing as mp
from functools import partial

class Config:
    MODEL_PATH = "/your/model/path/here"
    
    PICKSCORE_PROCESSOR = "./CLIP-ViT-H-14-laion2B-s32B-b79K"
    PICKSCORE_MODEL = "./PickScore_v1"
    
    CLIP_MODEL = "./clip-vit-base-patch32"
    
    DDIM_STEPS = [20, 30, 50]
    DDPM_STEPS = [300, 500, 800]

    NUM_PROMPTS = 1632 
    IMAGE_SIZE = (1024, 1024) 
    MAX_TOKEN_LENGTH = 77  
    
    OUTPUT_DIR = "./compare_quality"
    IMAGES_DIR = "./compare_quality-images"
    
    DEVICE = None

config=Config()

class ModelManager:
    def __init__(self, device):
        self.device = device
        self.setup_models()
    
    def setup_models(self):
        print(f"Loading models on device: {self.device}...")
        
        self.sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
            config.MODEL_PATH,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            use_safetensors=True,
        ).to(self.device)
        
        self.ddim_scheduler = DDIMScheduler.from_config(self.sdxl_pipeline.scheduler.config)
        self.ddpm_scheduler = DDPMScheduler.from_config(self.sdxl_pipeline.scheduler.config)

        self.pickscore_processor = AutoProcessor.from_pretrained(config.PICKSCORE_PROCESSOR)
        self.pickscore_model = AutoModel.from_pretrained(config.PICKSCORE_MODEL).eval().to(self.device)
        
        self.clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        self.clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(self.device)
        
        print(f"Models loaded successfully on device: {self.device}!")

def truncate_prompt(prompt, max_length=75):
    words = prompt.split()
    if len(words) <= 10:
        return prompt
    
    estimated_safe_words = max_length // 2
    if len(words) > estimated_safe_words:
        truncated = ' '.join(words[:estimated_safe_words])
        print(f"Warning: Truncated prompt from {len(words)} to {estimated_safe_words} words")
        return truncated
    
    return prompt

def safe_tokenize_text(processor, text, max_length=77):
    return processor(
        text=[text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

class Evaluator:
    def __init__(self, model_manager, worker_id=0):
        self.model_manager = model_manager
        self.results = []
        self.worker_id = worker_id

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.IMAGES_DIR, exist_ok=True)
    
    def calculate_pickscore(self, prompt, image):
        try:
            safe_prompt = truncate_prompt(prompt)
            
            image_inputs = self.model_manager.pickscore_processor(
                images=[image],
                padding=True,
                truncation=True,
                max_length=config.MAX_TOKEN_LENGTH,
                return_tensors="pt",
            ).to(self.model_manager.device)
            
            text_inputs = safe_tokenize_text(
                self.model_manager.pickscore_processor,
                safe_prompt,
                max_length=config.MAX_TOKEN_LENGTH
            ).to(self.model_manager.device)
            
            with torch.no_grad():
                image_embs = self.model_manager.pickscore_model.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
                
                text_embs = self.model_manager.pickscore_model.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
                
                score = self.model_manager.pickscore_model.logit_scale.exp() * (text_embs @ image_embs.T)[0, 0]
                
            return score.cpu().item()
            
        except Exception as e:
            print(f"Error calculating PickScore: {e}")
            return 0.0
    
    def calculate_clip_score(self, prompt, image):
        try:
            safe_prompt = truncate_prompt(prompt)
            
            inputs = self.model_manager.clip_processor(
                text=[safe_prompt], 
                images=[image], 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=config.MAX_TOKEN_LENGTH
            ).to(self.model_manager.device)
            
            with torch.no_grad():
                outputs = self.model_manager.clip_model(**inputs)
                
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                
                cosine_sim = (text_embeds @ image_embeds.T).squeeze()
                clip_score = (cosine_sim + 1) / 2
                
            return clip_score.cpu().item()
            
        except Exception as e:
            print(f"Error calculating CLIP score: {e}")
            return 0.0
    
    def generate_image(self, prompt, scheduler_type, num_steps):
        safe_prompt = truncate_prompt(prompt, max_length=75)
        
        if scheduler_type == "DDIM":
            self.model_manager.sdxl_pipeline.scheduler = self.model_manager.ddim_scheduler
        else:
            self.model_manager.sdxl_pipeline.scheduler = self.model_manager.ddpm_scheduler
        
        start_time = time.time()
        try:
            with torch.no_grad():
                image = self.model_manager.sdxl_pipeline(
                    prompt=safe_prompt,
                    num_inference_steps=num_steps,
                    height=config.IMAGE_SIZE[0],
                    width=config.IMAGE_SIZE[1],
                    generator=torch.Generator(device=self.model_manager.device).manual_seed(42)
                ).images[0]
            generation_time = time.time() - start_time
            return image, generation_time
            
        except Exception as e:
            print(f"Error generating image: {e}")
            blank_image = Image.new('RGB', config.IMAGE_SIZE, color='white')
            generation_time = time.time() - start_time
            return blank_image, generation_time
    
    def evaluate_configuration(self, prompts, scheduler_type, num_steps):
        """Evaluate specific configuration on a subset of prompts"""
        print(f"Worker {self.worker_id}: Evaluating {scheduler_type} - {num_steps} steps on {len(prompts)} prompts...")
        
        pickscore_scores = []
        clip_scores = []
        generation_times = []
        
        for i, prompt in enumerate(prompts):
            if i % 5 == 0:
                print(f"Worker {self.worker_id}: Progress {i}/{len(prompts)}")
            
            try:
                image, gen_time = self.generate_image(prompt, scheduler_type, num_steps)
                generation_times.append(gen_time)
                
                image_path = os.path.join(
                    config.IMAGES_DIR, 
                    f"{scheduler_type}_{num_steps}_{self.worker_id}_{i}.png"
                )
                image.save(image_path)
                
                pickscore = self.calculate_pickscore(prompt, image)
                clip_score = self.calculate_clip_score(prompt, image)
                
                pickscore_scores.append(pickscore)
                clip_scores.append(clip_score)
                
                print(f"Worker {self.worker_id}: Prompt {i}: PickScore={pickscore:.3f}, CLIP={clip_score:.3f}, Time={gen_time:.1f}s")
                
            except Exception as e:
                print(f"Worker {self.worker_id}: Error processing prompt {i}: {e}")
                pickscore_scores.append(0.0)
                clip_scores.append(0.0)
                generation_times.append(0.0)
        
        return {
            'scheduler': scheduler_type,
            'steps': num_steps,
            'worker_id': self.worker_id,
            'pickscore_scores': pickscore_scores,
            'clip_scores': clip_scores,
            'generation_times': generation_times
        }

def load_parti_prompts(num_prompts=100):
    print(f"Loading PartiPrompts dataset, getting {num_prompts} prompts...")
    
    try:
        dataset = load_dataset("./nateraw/parti-prompts", split="train")
        prompts = [item["Prompt"] for item in dataset.select(range(min(num_prompts, len(dataset))))]
        
        filtered_prompts = []
        for prompt in prompts:
            word_count = len(prompt.split())
            if word_count > 50:
                print(f"Skipping very long prompt ({word_count} words): {prompt[:100]}...")
                continue
            filtered_prompts.append(prompt)
        
        prompts = filtered_prompts[:num_prompts]
        
    except Exception as e:
        print(f"Cannot load PartiPrompts dataset ({e}), using example prompts...")
        example_prompts = [
            "A beautiful sunset over a mountain landscape",
            "A cute cat sitting on a windowsill",
            "A futuristic city with flying cars",
            "A peaceful garden with colorful flowers",
            "A majestic eagle soaring through clouds",
            "A vintage car on a country road",
            "A modern architectural building",
            "A forest path in autumn",
            "A lighthouse on a rocky coast",
            "A space station orbiting Earth"
        ]
        prompts = (example_prompts * (num_prompts // len(example_prompts) + 1))[:num_prompts]
    
    print(f"Successfully loaded {len(prompts)} prompts")
    return prompts

def evaluate_worker(worker_id, gpu_id, prompts_subset, all_configs):
    """Worker function for multi-GPU evaluation"""
    try:
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        config.DEVICE = device
        
        model_manager = ModelManager(device)
        evaluator = Evaluator(model_manager, worker_id)
        
        worker_results = []
        
        for scheduler_type, num_steps in all_configs:
            result = evaluator.evaluate_configuration(prompts_subset, scheduler_type, num_steps)
            worker_results.append(result)
        
        return worker_results
    
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        return []

def create_comparison_table(results):
    """Create comparison table"""
    df_data = []
    for result in results:
        df_data.append({
            'Sampling Method': result['scheduler'],
            'Sampling Steps': result['steps'],
            'PickScore Mean': f"{result['avg_pickscore']:.4f}",
            'PickScore Std': f"{result['std_pickscore']:.4f}",
            'CLIP Score Mean': f"{result['avg_clip_score']:.4f}",
            'CLIP Score Std': f"{result['std_clip_score']:.4f}",
            'Generation Time (s)': f"{result['avg_generation_time']:.2f}",
            'Speedup (vs 500-step)': f"{result['speedup']:.2f}x"
        })
    
    df = pd.DataFrame(df_data)
    return df

def create_visualizations(results):
    """Create visualization charts with English labels and value annotations"""
    plt.style.use('default')
    sns.set_palette("Set2")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # Increased figure size
    fig.suptitle('SDXL Sampling Methods Comparison Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    schedulers = [r['scheduler'] for r in results]
    steps = [r['steps'] for r in results]
    labels = [f"{s}-{st}" for s, st in zip(schedulers, steps)]
    
    pickscores = [r['avg_pickscore'] for r in results]
    clip_scores = [r['avg_clip_score'] for r in results]
    gen_times = [r['avg_generation_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    colors = ['#4A90E2' if 'DIPO' in l else '#E24A4A' for l in labels]

    bars1 = axes[0, 0].bar(labels, pickscores, color=colors, alpha=0.8)
    axes[0, 0].set_title('PickScore Comparison', fontsize=16, fontweight='bold', pad=20)
    axes[0, 0].set_ylabel('PickScore', fontsize=14)
    axes[0, 0].tick_params(axis='x', rotation=45, labelsize=12)
    axes[0, 0].tick_params(axis='y', labelsize=12)
    for bar, val in zip(bars1, pickscores):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pickscores)*0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    bars2 = axes[0, 1].bar(labels, clip_scores, color=['#2ECC71' if 'DIPO' in l else '#F39C12' for l in labels], alpha=0.8)
    axes[0, 1].set_title('CLIP Score Comparison', fontsize=16, fontweight='bold', pad=20)
    axes[0, 1].set_ylabel('CLIP Score', fontsize=14)
    axes[0, 1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[0, 1].tick_params(axis='y', labelsize=12)
    for bar, val in zip(bars2, clip_scores):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(clip_scores)*0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    bars3 = axes[1, 0].bar(labels, gen_times, color=['#F1C40F' if 'DIPO' in l else '#9B59B6' for l in labels], alpha=0.8)
    axes[1, 0].set_title('Average Generation Time Comparison', fontsize=16, fontweight='bold', pad=20)
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=14)
    axes[1, 0].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1, 0].tick_params(axis='y', labelsize=12)
    for bar, val in zip(bars3, gen_times):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gen_times)*0.02,
                       f'{val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    bars4 = axes[1, 1].bar(labels, speedups, color=['#1ABC9C' if 'DIPO' in l else '#E67E22' for l in labels], alpha=0.8)
    axes[1, 1].set_title('Speedup Comparison (vs 500-step baseline)', fontsize=16, fontweight='bold', pad=20)
    axes[1, 1].set_ylabel('Speedup Ratio', fontsize=14)
    axes[1, 1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1, 1].tick_params(axis='y', labelsize=12)
    for bar, val in zip(bars4, speedups):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups)*0.02,
                       f'{val:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'comparison_charts.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

def create_comparison_table_image(results):
    """Create a high-quality table image with better spacing and no font overlap"""
    df = create_comparison_table(results)
    
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, 
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)  
    table.scale(1.0, 3.0) 
    
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
        table[(0, i)].set_height(0.15)
        if i in [2, 3, 4, 5]:  
            table[(0, i)].set_width(0.12)
        elif i == 7: 
            table[(0, i)].set_width(0.15)
    
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F8F9FA')
            else:
                table[(i, j)].set_facecolor('white')
            table[(i, j)].set_height(0.12) 
            table[(i, j)].set_text_props(fontsize=10)
            if j in [2, 3, 4, 5]: 
                table[(i, j)].set_width(0.12)
            elif j == 7:  
                table[(i, j)].set_width(0.15)
    
    plt.title('SDXL Sampling Methods Performance Comparison\n(Speedup relative to 500-step baseline)', 
              fontsize=18, fontweight='bold', pad=40)
    
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'comparison_table.png'), 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

def save_detailed_results(results):
    """Save detailed results"""
    df = create_comparison_table(results)
    df.to_csv(os.path.join(config.OUTPUT_DIR, 'comparison_table.csv'), index=False)
    
    detailed_data = []
    for result in results:
        for i, (pick, clip) in enumerate(zip(result['pickscore_scores'], result['clip_scores'])):
            detailed_data.append({
                'scheduler': result['scheduler'],
                'steps': result['steps'],
                'prompt_id': i,
                'pickscore': pick,
                'clip_score': clip
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(config.OUTPUT_DIR, 'detailed_scores.csv'), index=False)

def main():
    """Main evaluation workflow with multi-GPU support"""
    print("Starting SDXL sampling methods comparison evaluation...")
    
    parser = argparse.ArgumentParser(description='SDXL Multi-GPU Evaluation')
    parser.add_argument('--gpus', type=str, default="0", 
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3")')
    args = parser.parse_args()

    gpu_ids = [int(id) for id in args.gpus.split(",")]
    num_workers = len(gpu_ids)
    print(f"Using GPUs: {gpu_ids}, Total workers: {num_workers}")

    prompts = load_parti_prompts(config.NUM_PROMPTS)

    prompt_chunks = np.array_split(prompts, num_workers)
    print(f"Split prompts into {len(prompt_chunks)} chunks with sizes: {[len(chunk) for chunk in prompt_chunks]}")
    
    all_configs = []
    for steps in config.DDIM_STEPS:
        all_configs.append(("DIPO", steps))
    for steps in config.DDPM_STEPS:
        all_configs.append(("Diffusion-DPO", steps))

    with Pool(processes=num_workers) as pool:
        worker_args = [(i, gpu_ids[i], list(prompt_chunks[i]), all_configs) 
                      for i in range(num_workers)]
        
        print("Starting workers...")
        all_worker_results = pool.starmap(evaluate_worker, worker_args)

    flat_results = [item for sublist in all_worker_results for item in sublist]
    
    baseline_time = None
    for r in flat_results:
        if r['steps'] == 500:  
            baseline_time = np.mean(r['generation_times'])
            print(f"Using {r['scheduler']}-500 as baseline: {baseline_time:.2f}s")
            break
    
    if baseline_time is None:
        print("Warning: 500 steps configuration not found, using estimated baseline of 127.55s")
        baseline_time = 127.55

    combined_results = []
    for scheduler, steps in all_configs:
        config_results = [r for r in flat_results if r['scheduler'] == scheduler and r['steps'] == steps]
        
        if not config_results:
            continue
        
        all_pickscores = []
        all_clip_scores = []
        all_gen_times = []
        
        for worker_result in config_results:
            all_pickscores.extend(worker_result['pickscore_scores'])
            all_clip_scores.extend(worker_result['clip_scores'])
            all_gen_times.extend(worker_result['generation_times'])
        
        avg_pickscore = np.mean(all_pickscores)
        std_pickscore = np.std(all_pickscores)
        avg_clip_score = np.mean(all_clip_scores)
        std_clip_score = np.std(all_clip_scores)
        avg_generation_time = np.mean(all_gen_times)

        if steps == 500:
            speedup =1.00
        else:
            speedup = baseline_time / avg_generation_time if avg_generation_time > 0 else 1.0
        
        combined_results.append({
            'scheduler': scheduler,
            'steps': steps,
            'avg_pickscore': avg_pickscore,
            'std_pickscore': std_pickscore,
            'avg_clip_score': avg_clip_score,
            'std_clip_score': std_clip_score,
            'avg_generation_time': avg_generation_time,
            'speedup': speedup,
            'pickscore_scores': all_pickscores,
            'clip_scores': all_clip_scores
        })
    
    comparison_table = create_comparison_table(combined_results)
    print("\n=== Combined Results Table ===")
    print(comparison_table.to_string(index=False))
    
    print("\nGenerating visualizations...")
    create_visualizations(combined_results)
    create_comparison_table_image(combined_results)
    
    save_detailed_results(combined_results)
    
    print(f"\nEvaluation completed! Results saved to {config.OUTPUT_DIR}")
    
    print("\n=== Summary ===")
    best_quality_pick = max(combined_results, key=lambda x: x['avg_pickscore'])
    best_quality_clip = max(combined_results, key=lambda x: x['avg_clip_score'])
    fastest = min(combined_results, key=lambda x: x['avg_generation_time'])
    
    print(f"Best quality (PickScore): {best_quality_pick['scheduler']}-{best_quality_pick['steps']} "
          f"(PickScore: {best_quality_pick['avg_pickscore']:.3f})")
    print(f"Best quality (CLIP): {best_quality_clip['scheduler']}-{best_quality_clip['steps']} "
          f"(CLIP: {best_quality_clip['avg_clip_score']:.3f})")
    print(f"Fastest generation: {fastest['scheduler']}-{fastest['steps']} "
          f"({fastest['avg_generation_time']:.1f}s, {fastest['speedup']:.1f}x speedup)")
    print(f"Baseline (500-step): {baseline_time:.2f}s")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
