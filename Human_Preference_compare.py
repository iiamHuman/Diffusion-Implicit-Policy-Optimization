import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, DDPMScheduler
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import json
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Config:
    BASE_MODEL_PATH = "/your/base/model/path/"
    TRAINED_MODEL_PATH = "/your/trained/model/path/"  
    HPSV2_MODEL_PATH = "./HPSv2-hf"  
    
    DATASET_NAME = "./nateraw/parti-prompts" 
    NUM_PROMPTS = 1632  
    
    DDIM_STEPS = 50 
    DDPM_STEPS = 500  
    GUIDANCE_SCALE = 7.5  
    HEIGHT = 1024 
    WIDTH = 1024 
    
    GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7] 
    BATCH_SIZE = 2  
    
    OUTPUT_DIR = "./Human_Preference_compare" 
    SAVE_IMAGES = True  
    
    MAX_TEXT_LENGTH = 75  
    
    ENABLE_MEMORY_EFFICIENT_ATTENTION = True 
    ENABLE_CPU_OFFLOAD = False  
    LOW_MEM_MODE = True  

def setup_distributed(rank, world_size):
    """Setup distributed training"""
    actual_gpu_id = Config.GPU_IDS[rank]
    
    torch.cuda.set_device(actual_gpu_id)
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    print(f"Process {rank} using GPU {actual_gpu_id}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def clear_gpu_memory():
    torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def truncate_text(text, max_length=None):
    """Truncate text to prevent sequence length error"""
    if max_length is None:
        max_length = Config.MAX_TEXT_LENGTH
    
    words = text.split()
    if len(words) <= max_length:
        return text
    return ' '.join(words[:max_length])

def load_parti_prompts(num_prompts=100):
    """Load PartiPrompts dataset"""
    try:
        dataset = load_dataset(Config.DATASET_NAME, split='train')
        prompts = [truncate_text(item['Prompt']) for item in dataset.select(range(min(num_prompts, len(dataset))))]
        return prompts
    except Exception as e:
        print(f"Error loading dataset: {e}")
        fallback_prompts = [
            "A beautiful landscape with mountains and rivers",
            "A cat sitting on a windowsill",
            "A futuristic city with flying cars",
            "A portrait of an elderly man with a beard",
            "A colorful flower garden in spring",
            "A robot walking in a forest",
            "A sunset over the ocean",
            "A medieval castle on a hill",
            "A busy street in Tokyo",
            "A cozy coffee shop interior"
        ]
        extended_prompts = []
        while len(extended_prompts) < num_prompts:
            extended_prompts.extend(fallback_prompts)
        return extended_prompts[:num_prompts]

def create_pipeline(model_path, scheduler_type="ddim", gpu_id=0):
    """Create diffusion pipeline with specified scheduler and memory optimization"""
    try:
        torch.cuda.set_device(gpu_id)
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            device_map=None,
        )
        
        if Config.ENABLE_MEMORY_EFFICIENT_ATTENTION:
            pipeline.enable_attention_slicing()
            pipeline.enable_vae_slicing()
            pipeline.enable_vae_tiling()
        
        if scheduler_type == "ddim":
            scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            pipeline.scheduler = scheduler
        elif scheduler_type == "ddpm":
            scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
            pipeline.scheduler = scheduler

        pipeline = pipeline.to(f'cuda:{gpu_id}')
        
        if Config.ENABLE_CPU_OFFLOAD:
            pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
        
        print(f"Pipeline created on GPU {gpu_id}")
        print(f"UNet device: {next(pipeline.unet.parameters()).device}")
        print(f"VAE device: {next(pipeline.vae.parameters()).device}")
        
        return pipeline
    
    except Exception as e:
        print(f"Error creating pipeline on GPU {gpu_id}: {e}")
        raise e

def generate_images_batch(pipeline, prompts, num_steps, gpu_id=0):
    """Generate images in smaller batches to save memory"""
    images = []
    
    batch_size = 1 
    
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating images on GPU {gpu_id}"):
        batch_prompts = prompts[i:i+batch_size]
        
        try:
            clear_gpu_memory()
            
            for prompt in batch_prompts:
                image = pipeline(
                    prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=Config.GUIDANCE_SCALE,
                    height=Config.HEIGHT,
                    width=Config.WIDTH,
                    generator=torch.Generator(device=f'cuda:{gpu_id}').manual_seed(77)
                ).images[0]
                images.append(image)
                
        except Exception as e:
            print(f"Error generating batch on GPU {gpu_id}: {e}")
            for _ in batch_prompts:
                images.append(Image.new('RGB', (Config.WIDTH, Config.HEIGHT), color='black'))
            
        clear_gpu_memory()
    
    return images

def calculate_hps_scores_batch(images, prompts, gpu_id=0):
    """Calculate HPSv2 scores in batches to save memory"""
    try:
        clear_gpu_memory()
        
        model = CLIPModel.from_pretrained(Config.HPSV2_MODEL_PATH).to(f'cuda:{gpu_id}')
        processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32")
        
        scores = []
        batch_size = 2  
        
        for i in tqdm(range(0, len(images), batch_size), desc=f"Calculating HPS scores on GPU {gpu_id}"):
            batch_images = images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            
            try:
                for image, prompt in zip(batch_images, batch_prompts):
                    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
                    inputs = {k: v.to(f'cuda:{gpu_id}') for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        score = torch.cosine_similarity(
                            outputs.text_embeds, 
                            outputs.image_embeds, 
                            dim=1
                        ).cpu().item()
                    scores.append(score)
                    
                    del inputs, outputs
                    
            except Exception as e:
                print(f"Error calculating scores for batch on GPU {gpu_id}: {e}")
                scores.extend([0.0] * len(batch_images))
            
            clear_gpu_memory()
        
        return scores
        
    except Exception as e:
        print(f"Error loading HPSv2 model on GPU {gpu_id}: {e}")
        return [0.5] * len(images) 

def compare_human_preferences_batch(images_before, images_after, prompts, gpu_id=0):
    """Compare human preferences in batches to save memory"""
    try:
        clear_gpu_memory()
        
        model = CLIPModel.from_pretrained(Config.HPSV2_MODEL_PATH).to(f'cuda:{gpu_id}')
        processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32")
        
        preferences = []
        batch_size = 1 
        
        for i in tqdm(range(0, len(images_before), batch_size), 
                     desc=f"Comparing preferences on GPU {gpu_id}"):
            
            batch_before = images_before[i:i+batch_size]
            batch_after = images_after[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            
            try:
                for img_before, img_after, prompt in zip(batch_before, batch_after, batch_prompts):
                    inputs_before = processor(text=[prompt], images=[img_before], return_tensors="pt", padding=True)
                    inputs_after = processor(text=[prompt], images=[img_after], return_tensors="pt", padding=True)
                    
                    inputs_before = {k: v.to(f'cuda:{gpu_id}') for k, v in inputs_before.items()}
                    inputs_after = {k: v.to(f'cuda:{gpu_id}') for k, v in inputs_after.items()}
                    
                    with torch.no_grad():
                        outputs_before = model(**inputs_before)
                        outputs_after = model(**inputs_after)
                        
                        score_before = torch.cosine_similarity(
                            outputs_before.text_embeds, 
                            outputs_before.image_embeds, 
                            dim=1
                        ).cpu().item()
                        
                        score_after = torch.cosine_similarity(
                            outputs_after.text_embeds, 
                            outputs_after.image_embeds, 
                            dim=1
                        ).cpu().item()
                    
                    if score_after > score_before:
                        preferences.append("after")
                    else:
                        preferences.append("before")
                    
                    del inputs_before, inputs_after, outputs_before, outputs_after
                    
            except Exception as e:
                print(f"Error comparing preferences for batch on GPU {gpu_id}: {e}")
                preferences.extend(["after"] * len(batch_before)) 
            
            clear_gpu_memory()
        
        return preferences
        
    except Exception as e:
        print(f"Error in preference comparison on GPU {gpu_id}: {e}")
        return ["after"] * len(images_before) 

def create_visualizations(results, output_dir):
    """Create visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    categories = ['DIPO Before', 'DIPO After', 'Difusion-DPO Before', 'Difusion-DPO After']
    scores = [
        np.mean(results['ddim_before_scores']),
        np.mean(results['ddim_after_scores']),
        np.mean(results['ddpm_before_scores']),
        np.mean(results['ddpm_after_scores'])
    ]
    bars = plt.bar(categories, scores, color=['#ff7f7f', '#87ceeb', '#ff7f7f', '#87ceeb'])
    plt.title('HPSv2 Score Comparison (Higher is Better)', fontsize=16, fontweight='bold')
    plt.ylabel('Average HPSv2 Score', fontsize=12)
    plt.xlabel('Model and Sampling Method', fontsize=12)
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        if i == 1:  
            improvement = ((scores[1] - scores[0]) / scores[0]) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'+{improvement:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', color='green' if improvement > 0 else 'red')
        elif i == 3:  
            improvement = ((scores[3] - scores[2]) / scores[2]) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'+{improvement:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', color='green' if improvement > 0 else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hpsv2_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    def pie_subplot(ax, imgA_scores, imgB_scores, labelA, labelB, title):
        betterA = sum(a > b for a, b in zip(imgA_scores, imgB_scores))
        betterB = sum(a < b for a, b in zip(imgA_scores, imgB_scores))
        equal = len(imgA_scores) - betterA - betterB
        ax.pie([betterA, betterB, equal],
               labels=[f'Prefer {labelA}', f'Prefer {labelB}', 'Equal'],
               colors=['#87ceeb', '#ff7f7f', '#cccccc'],
               autopct='%1.1f%%', startangle=90)
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    pie_subplot(axes[0,0], results['ddim_after_scores'], results['ddim_before_scores'],
                'After-DIPO', 'Before-DIPO', 'After-DIPO vs Before-DIPO')
    
    pie_subplot(axes[0,1], results['ddpm_after_scores'], results['ddpm_before_scores'],
                'After-Difusion-DPO', 'Before-Difusion-DPO', 'After-Difusion-DPO vs Before-Difusion-DPO')
    
    pie_subplot(axes[1,0], results['ddim_after_scores'], results['ddpm_after_scores'],
                'After-DIPO', 'After-Difusion-DPO', 'After-DIPO vs After-Difusion-DPO')
    
    pie_subplot(axes[1,1], results['ddim_before_scores'], results['ddpm_before_scores'],
                'Before-DIPO', 'Before-Difusion-DPO', 'Before-DIPO vs Before-Difusion-DPO')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_pie_comparisons.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ddim_prefs = results['ddim_preferences']
    ddim_before_count = ddim_prefs.count('before')
    ddim_after_count = ddim_prefs.count('after')
    ax1.pie([ddim_before_count, ddim_after_count], 
            labels=['Prefer Before', 'Prefer After'],
            colors=['#ff7f7f', '#87ceeb'],
            autopct='%1.1f%%',
            startangle=90)
    ax1.set_title(f'DIPO - Human Preference Comparison\n(After-training win rate: {ddim_after_count}/{len(ddim_prefs)})', 
                  fontsize=14, fontweight='bold')
    ddpm_prefs = results['ddpm_preferences']
    ddpm_before_count = ddpm_prefs.count('before')
    ddpm_after_count = ddpm_prefs.count('after')
    ax2.pie([ddpm_before_count, ddpm_after_count], 
            labels=['Prefer Before', 'Prefer After'],
            colors=['#ff7f7f', '#87ceeb'],
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title(f'DDifusion-DPO - Human Preference Comparison\n(After-training win rate: {ddpm_after_count}/{len(ddpm_prefs)})', 
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'preference_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualizations saved to {output_dir}")

def save_sample_images(images, prompts, output_dir, prefix):
    """Save sample generated images"""
    if not Config.SAVE_IMAGES:
        return
        
    sample_dir = os.path.join(output_dir, f"{prefix}_samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    for i, (image, prompt) in enumerate(zip(images[:5], prompts[:5])):
        filename = f"{prefix}_sample_{i+1}.png"
        image.save(os.path.join(sample_dir, filename))
        
        with open(os.path.join(sample_dir, f"{prefix}_sample_{i+1}_prompt.txt"), 'w') as f:
            f.write(prompt)

def run_evaluation(rank, world_size):
    """Main evaluation function - 优化内存使用"""
    actual_gpu_id = Config.GPU_IDS[rank]
    torch.cuda.set_device(actual_gpu_id)
    print(f"Process {rank} initialized with GPU {actual_gpu_id}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    torch.cuda.set_device(actual_gpu_id)
    print(f"Process {rank} initialized with GPU {actual_gpu_id}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    print(f"Starting evaluation on GPU {actual_gpu_id} (process rank {rank})")
    
    prompts = load_parti_prompts(Config.NUM_PROMPTS)
    
    prompts_per_gpu = len(prompts) // world_size
    remainder = len(prompts) % world_size
    
    if rank < remainder:
        start_idx = rank * (prompts_per_gpu + 1)
        end_idx = start_idx + prompts_per_gpu + 1
    else:
        start_idx = remainder * (prompts_per_gpu + 1) + (rank - remainder) * prompts_per_gpu
        end_idx = start_idx + prompts_per_gpu
    
    local_prompts = prompts[start_idx:end_idx]
    
    print(f"GPU {actual_gpu_id} processing {len(local_prompts)} prompts (indices {start_idx}-{end_idx-1})")
    
    results = {}
    
    try:
        
        print(f"GPU {actual_gpu_id}: Processing DIPO before training...")
        torch.cuda.set_device(actual_gpu_id) 
        clear_gpu_memory()
        
        base_pipeline_ddim = create_pipeline(Config.BASE_MODEL_PATH, "ddim", actual_gpu_id)
        ddim_before_images = generate_images_batch(base_pipeline_ddim, local_prompts, Config.DDIM_STEPS, actual_gpu_id)
        results['ddim_before_scores'] = calculate_hps_scores_batch(ddim_before_images, local_prompts, actual_gpu_id)
        
        if Config.SAVE_IMAGES:
            local_output_dir = os.path.join(Config.OUTPUT_DIR, f"gpu_{actual_gpu_id}")
            save_sample_images(ddim_before_images, local_prompts, local_output_dir, "ddim_before")
        
        del base_pipeline_ddim
        clear_gpu_memory()
        
        torch.cuda.set_device(actual_gpu_id)
        
        trained_pipeline_ddim = create_pipeline(Config.TRAINED_MODEL_PATH, "ddim", actual_gpu_id)
        ddim_after_images = generate_images_batch(trained_pipeline_ddim, local_prompts, Config.DDIM_STEPS, actual_gpu_id)
        results['ddim_after_scores'] = calculate_hps_scores_batch(ddim_after_images, local_prompts, actual_gpu_id)
        
        results['ddim_preferences'] = compare_human_preferences_batch(ddim_before_images, ddim_after_images, local_prompts, actual_gpu_id)
        
       
        if Config.SAVE_IMAGES:
            save_sample_images(ddim_after_images, local_prompts, local_output_dir, "ddim_after")
        
        
        del trained_pipeline_ddim
        clear_gpu_memory()
        
        torch.cuda.set_device(actual_gpu_id) 
        
        base_pipeline_ddpm = create_pipeline(Config.BASE_MODEL_PATH, "ddpm", actual_gpu_id)
        ddpm_before_images = generate_images_batch(base_pipeline_ddpm, local_prompts, Config.DDPM_STEPS, actual_gpu_id)
        results['ddpm_before_scores'] = calculate_hps_scores_batch(ddpm_before_images, local_prompts, actual_gpu_id)
        
        if Config.SAVE_IMAGES:
            save_sample_images(ddpm_before_images, local_prompts, local_output_dir, "ddpm_before")
        
        del base_pipeline_ddpm
        clear_gpu_memory()
        
        torch.cuda.set_device(actual_gpu_id)  
        
        trained_pipeline_ddpm = create_pipeline(Config.TRAINED_MODEL_PATH, "ddpm", actual_gpu_id)
        ddpm_after_images = generate_images_batch(trained_pipeline_ddpm, local_prompts, Config.DDPM_STEPS, actual_gpu_id)
        results['ddpm_after_scores'] = calculate_hps_scores_batch(ddpm_after_images, local_prompts, actual_gpu_id)
        
        results['ddpm_preferences'] = compare_human_preferences_batch(ddpm_before_images, ddpm_after_images, local_prompts, actual_gpu_id)
        
        if Config.SAVE_IMAGES:
            save_sample_images(ddpm_after_images, local_prompts, local_output_dir, "ddpm_after")
        
        del trained_pipeline_ddpm
        clear_gpu_memory()
        
    except Exception as e:
        print(f"Error during evaluation on GPU {actual_gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        results = {
            'ddim_before_scores': [0.5] * len(local_prompts),
            'ddim_after_scores': [0.5] * len(local_prompts),
            'ddpm_before_scores': [0.5] * len(local_prompts),
            'ddpm_after_scores': [0.5] * len(local_prompts),
            'ddim_preferences': ["after"] * len(local_prompts),
            'ddpm_preferences': ["after"] * len(local_prompts)
        }
    
    local_output_dir = os.path.join(Config.OUTPUT_DIR, f"gpu_{actual_gpu_id}")
    os.makedirs(local_output_dir, exist_ok=True)
    
    with open(os.path.join(local_output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(local_output_dir, 'prompts.json'), 'w') as f:
        json.dump({
            'prompts': local_prompts,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'total_prompts': len(local_prompts),
            'gpu_id': actual_gpu_id,
            'process_rank': rank
        }, f, indent=2)
    
    print(f"GPU {actual_gpu_id} completed processing {len(local_prompts)} prompts")
    
    if world_size > 1:
        cleanup_distributed()
    
    return results

def aggregate_results():
    """Aggregate results from all GPUs"""
    all_results = {
        'ddim_before_scores': [],
        'ddim_after_scores': [],
        'ddpm_before_scores': [],
        'ddpm_after_scores': [],
        'ddim_preferences': [],
        'ddpm_preferences': []
    }
    
    total_processed_prompts = 0
    for gpu_id in Config.GPU_IDS:
        gpu_results_path = os.path.join(Config.OUTPUT_DIR, f"gpu_{gpu_id}", 'results.json')
        gpu_prompts_path = os.path.join(Config.OUTPUT_DIR, f"gpu_{gpu_id}", 'prompts.json')
        
        if os.path.exists(gpu_results_path):
            with open(gpu_results_path, 'r') as f:
                gpu_results = json.load(f)
                for key in all_results.keys():
                    all_results[key].extend(gpu_results[key])
            
            if os.path.exists(gpu_prompts_path):
                with open(gpu_prompts_path, 'r') as f:
                    prompts_info = json.load(f)
                    total_processed_prompts += prompts_info['total_prompts']
                    print(f"GPU {gpu_id} processed {prompts_info['total_prompts']} prompts (indices {prompts_info['start_idx']}-{prompts_info['end_idx']-1})")
    
    print(f"\nTotal prompts processed across all GPUs: {total_processed_prompts}")
    
    create_visualizations(all_results, Config.OUTPUT_DIR)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    
    print(f"\nHPSv2 Scores (Higher is Better):")
    print(f"DIPO Before:  {np.mean(all_results['ddim_before_scores']):.4f} ± {np.std(all_results['ddim_before_scores']):.4f}")
    print(f"DIPO After:   {np.mean(all_results['ddim_after_scores']):.4f} ± {np.std(all_results['ddim_after_scores']):.4f}")
    print(f"Difusion-DPO Before:  {np.mean(all_results['ddpm_before_scores']):.4f} ± {np.std(all_results['ddpm_before_scores']):.4f}")
    print(f"Difusion-DPO After:   {np.mean(all_results['ddpm_after_scores']):.4f} ± {np.std(all_results['ddpm_after_scores']):.4f}")
    
    print(f"\nHuman Preference Results:")
    ddim_after_wins = all_results['ddim_preferences'].count('after')
    ddpm_after_wins = all_results['ddpm_preferences'].count('after')
    
    print(f"DIPO: After-training preferred in {ddim_after_wins}/{len(all_results['ddim_preferences'])} cases ({ddim_after_wins/len(all_results['ddim_preferences'])*100:.1f}%)")
    print(f"Difusion-DPO: After-training preferred in {ddpm_after_wins}/{len(all_results['ddpm_preferences'])} cases ({ddpm_after_wins/len(all_results['ddpm_preferences'])*100:.1f}%)")

    with open(os.path.join(Config.OUTPUT_DIR, 'final_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to: {Config.OUTPUT_DIR}")

def main():
    """Main function"""
    print("SDXL DiffusionDPO Evaluation - 4 GPU Distributed")
    print("="*50)
    print(f"Base Model: {Config.BASE_MODEL_PATH}")
    print(f"Trained Model: {Config.TRAINED_MODEL_PATH}")
    print(f"Number of Prompts: {Config.NUM_PROMPTS}")
    print(f"Using GPUs: {Config.GPU_IDS}")
    print(f"Output Directory: {Config.OUTPUT_DIR}")
    print(f"DIPO Steps: {Config.DDIM_STEPS}")
    print(f"Difusion-DPO Steps: {Config.DDPM_STEPS}")
    print(f"Image Size: {Config.WIDTH}x{Config.HEIGHT}")
    print(f"Memory Optimization: {Config.ENABLE_MEMORY_EFFICIENT_ATTENTION}")
    print(f"CPU Offload: {Config.ENABLE_CPU_OFFLOAD}")
    print(f"Low Memory Mode: {Config.LOW_MEM_MODE}")
    print(f"Save Images: {Config.SAVE_IMAGES}")
    print("="*50)

    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if not Config.GPU_IDS:
        print("ERROR: No GPU IDs specified in Config.GPU_IDS")
        return
    
    for gpu_id in Config.GPU_IDS:
        if gpu_id >= available_gpus:
            print(f"ERROR: GPU {gpu_id} not available. Available GPUs: 0-{available_gpus-1}")
            return
    
    if len(Config.GPU_IDS) != 4:
        print(f"WARNING: This script is optimized for 4 GPUs, but {len(Config.GPU_IDS)} GPUs specified")
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        print(f"Running distributed evaluation on GPUs: {Config.GPU_IDS}")
        mp.spawn(run_evaluation, args=(len(Config.GPU_IDS),), nprocs=len(Config.GPU_IDS), join=True)
        
        print("\nAggregating results and creating visualizations...")
        aggregate_results()
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()