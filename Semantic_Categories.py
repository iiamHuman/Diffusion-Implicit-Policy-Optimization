import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from diffusers import StableDiffusionXLPipeline, DDPMScheduler, DDIMScheduler
from datasets import load_dataset
import os
import json
from tqdm import tqdm
import warnings
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from multiprocessing import Process, Queue
import time
import random

warnings.filterwarnings("ignore")

def set_global_seed(seed):
    random.seed(seed)            
    np.random.seed(seed)         
    torch.manual_seed(seed)      
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class DiffusionEvaluator:
    def __init__(self, model_path, output_dir="./evaluation_results", devices=["cuda:0"]):
        """
        Initialize the evaluator with model path and output directory
        
        Args:
            model_path: Path to your DiffusionDPO-trained SDXL model
            output_dir: Directory to save results and visualizations
            devices: List of GPU devices to use (e.g., ["cuda:0", "cuda:1"])
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.devices = devices
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading HPSv2 model for evaluation...")
        self.hps_model = CLIPModel.from_pretrained("./HPSv2-hf")
        self.hps_processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32")
        
        print("Loading PickScore model for evaluation...")
        self.pickscore_processor = AutoProcessor.from_pretrained("./CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.pickscore_model = AutoModel.from_pretrained("/./PickScore_v1").eval()
        
        self.categorized_prompts = {"Portrait": [], "Landscape": [], "Abstract": []}
    
    def setup_pipeline_for_device(self, device):
        """Setup pipeline for a specific device"""
        print(f"Setting up pipeline on {device}...")
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)
        ddim_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        ddim_scheduler.config.eta = 0.2  
        pipeline.scheduler = ddim_scheduler

        return pipeline
    
    def worker(self, device, categories, device_prompts, base_seed, queue):
        try:
            torch.cuda.set_device(device)
            print(f"Worker started on {device} for categories: {categories}")
            device_seed = base_seed + hash(device) % 100000
            torch.manual_seed(device_seed)
            torch.cuda.manual_seed_all(device_seed)
            np.random.seed(device_seed % (2**32 - 1))
            random.seed(device_seed)
            print(f"Device {device} using seed: {device_seed}")

            base_pipeline = self.setup_pipeline_for_device(device)
            ddpm_pipeline = base_pipeline
            ddpm_pipeline.scheduler = DDPMScheduler.from_config(
                base_pipeline.scheduler.config
            )
            ddim_pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            ).to(device)
            ddim_pipeline.scheduler = DDIMScheduler.from_config(
                base_pipeline.scheduler.config
            )

            device_results = {
                "Difusion-DPO_500": {cat: [] for cat in categories},
                "DIPO_30": {cat: [] for cat in categories}
            }

            for category in categories:
                selected_prompts = device_prompts[category]
                for i, prompt in enumerate(selected_prompts):
                    try:
                        prompt_seed = device_seed + i
                        generator = torch.Generator(device=device).manual_seed(prompt_seed)
                        ddpm_image = ddpm_pipeline(
                            prompt=prompt,
                            num_inference_steps=500,
                            generator=generator,
                            guidance_scale=7.5,
                            width=1024,
                            height=1024,
                            eta=0.2
                        ).images[0]
                        ddim_generator = torch.Generator(device=device).manual_seed(prompt_seed)
                        ddim_image = ddim_pipeline(
                            prompt=prompt,
                            num_inference_steps=30,
                            generator=ddim_generator,
                            guidance_scale=7.5,
                            width=1024,
                            height=1024
                        ).images[0]
                        ddpm_path = f"{self.output_dir}/ddpm_{category.lower()}_{i}_{device.replace(':', '_')}.png"
                        ddim_path = f"{self.output_dir}/ddim_{category.lower()}_{i}_{device.replace(':', '_')}.png"
                        ddpm_image.save(ddpm_path)
                        ddim_image.save(ddim_path)
                        device_results["Difusion-DPO_500"][category].append({
                            "prompt": prompt,
                            "image_path": ddpm_path,
                            "image": ddpm_image,
                            "seed": prompt_seed
                        })
                        device_results["DIPO_30"][category].append({
                            "prompt": prompt,
                            "image_path": ddim_path,
                            "image": ddim_image,
                            "seed": prompt_seed
                        })
                    except Exception as e:
                        print(f"Error generating image on {device} for prompt '{prompt}': {str(e)}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        torch.cuda.empty_cache()
            print(f"[{device}] Worker finished normally.")
            queue.put(device_results)
        except Exception as e:
            print(f"[{device}] Worker crashed with exception: {e}")
            import traceback
            traceback.print_exc()
            queue.put({
                "Difusion-DPO_500": {cat: [] for cat in categories},
                "DIPO_30": {cat: [] for cat in categories}
            })
    
    def generate_images(self, num_samples=5, base_seed=42):
        """Generate images using multiple GPUs with global seed control (均匀分配prompt到每张卡)"""
        print(f"Generating images with base seed: {base_seed}")
        self.results = {
            "Difusion-DPO_500": {"Portrait": [], "Landscape": [], "Abstract": []},
            "DIPO_30": {"Portrait": [], "Landscape": [], "Abstract": []}
        }
        categories = list(self.categorized_prompts.keys())
        per_device_prompts = []
        for i in range(len(self.devices)):
            device_prompts = {}
            for cat in categories:
                all_prompts = self.categorized_prompts[cat]
                split = np.array_split(all_prompts, len(self.devices))
                device_prompts[cat] = list(split[i])
            per_device_prompts.append(device_prompts)
        processes = []
        queue = Queue()
        for i, device in enumerate(self.devices):
            p = Process(
                target=self.worker,
                args=(device, categories, per_device_prompts[i], base_seed, queue)
            )
            p.start()
            processes.append(p)
        for _ in range(len(processes)):
            device_results = queue.get()
            for method in ["Difusion-DPO_500", "DIPO_30"]:
                for category in device_results[method].keys():
                    self.results[method][category].extend(device_results[method][category])
        for p in processes:
            p.join()
    
    def calculate_scores(self):
        """Calculate both HPSv2 and PickScore scores for all generated images"""
        print("Calculating HPSv2 and PickScore scores...")
        
        self.hps_scores = {
            "Difusion-DPO_500": {"Portrait": [], "Landscape": [], "Abstract": []},
            "DIPO_30": {"Portrait": [], "Landscape": [], "Abstract": []}
        }
        
        self.pick_scores = {
            "Difusion-DPO_500": {"Portrait": [], "Landscape": [], "Abstract": []},
            "DIPO_30": {"Portrait": [], "Landscape": [], "Abstract": []}
        }
        
        for method in ["Difusion-DPO_500", "DIPO_30"]:
            for category in ["Portrait", "Landscape", "Abstract"]:
                for item in tqdm(self.results[method][category], desc=f"{method} {category}"):
                    try:
                        hps_score = self.compute_hps_score(item["prompt"], item["image"])
                        pick_score = self.compute_pickscore(item["prompt"], item["image"])
                        
                        self.hps_scores[method][category].append(hps_score)
                        self.pick_scores[method][category].append(pick_score)
                    except Exception as e:
                        print(f"Error computing scores for prompt: {item['prompt']}, error: {str(e)}")
                        self.hps_scores[method][category].append(0.0)
                        self.pick_scores[method][category].append(0.0)
                    finally:
                        torch.cuda.empty_cache()
    
    def compute_hps_score(self, prompt, image):
        """Compute HPSv2 score for a single image-prompt pair"""
        inputs = self.hps_processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        device = self.hps_model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.hps_model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.item()
        
        return score
    
    def compute_pickscore(self, prompt, image):
        """Compute PickScore for a single image-prompt pair"""
        device = next(self.pickscore_model.parameters()).device

        image_inputs = self.pickscore_processor(
            images=[image],
            return_tensors="pt",
        ).to(device)
        
        text_inputs = self.pickscore_processor.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            image_embs = self.pickscore_model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = self.pickscore_model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            score = self.pickscore_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
        return score.item()
    
    def create_comparison_grid(self):
        """Create a visual comparison grid showing sample results with both scores"""
        print("Creating comparison grid...")

        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        fig.suptitle('DiffusionDPO: Difusion-DPO vs DIPO Sampling Comparison', fontsize=28, fontweight='bold')
    
        categories = ["Portrait", "Landscape", "Abstract"]
        methods = ["Difusion-DPO_500", "DIPO_30"]
        
        for row, category in enumerate(categories):

            axes[row, 0].text(0.5, 0.5, category, ha='center', va='center', 
                        fontsize=24, fontweight='bold', transform=axes[row, 0].transAxes)
            axes[row, 0].axis('off')
            

            for col_offset, method in enumerate(methods):
                for sample_idx in range(2): 
                    col = 1 + col_offset * 2 + sample_idx
                    
                    if sample_idx < len(self.results[method][category]):
                        img = self.results[method][category][sample_idx]["image"]
                        prompt = self.results[method][category][sample_idx]["prompt"]
                        
                        axes[row, col].imshow(img)


                        method_name = "Difusion-DPO (500)" if method == "Difusion-DPO_500" else "DIPO (30)"
                        axes[row, col].set_title(f"{method_name}", fontsize=16, fontweight='bold')
                        axes[row, col].text(1.02, 0.5, prompt, ha='left', va='center',
                                      fontsize=12, fontweight='bold', wrap=True,
                                      transform=axes[row, col].transAxes)
                    
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/comparison_grid.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_dir}/comparison_grid.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_score_comparison(self):
        """Create horizontal bar chart comparing both HPS and PickScore scores"""
        print("Creating score comparison chart...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        categories = ["Portrait", "Landscape", "Abstract"]
        methods = ["Difusion-DPO (500 steps)", "DIPO (30 steps)"]

        hps_data = []
        for category in categories:
            ddpm_mean = np.mean(self.hps_scores["Difusion-DPO_500"][category])
            ddim_mean = np.mean(self.hps_scores["DIPO_30"][category])
            ddpm_std = np.std(self.hps_scores["Difusion-DPO_500"][category])
            ddim_std = np.std(self.hps_scores["DIPO_30"][category])
            
            hps_data.extend([
                {"Category": category, "Method": "Difusion-DPO (500 steps)", "Score": ddpm_mean, "Std": ddpm_std},
                {"Category": category, "Method": "DIPO (30 steps)", "Score": ddim_mean, "Std": ddim_std}
            ])

        pick_data = []
        for category in categories:
            ddpm_mean = np.mean(self.pick_scores["Difusion-DPO_500"][category])
            ddim_mean = np.mean(self.pick_scores["DIPO_30"][category])
            ddpm_std = np.std(self.pick_scores["Difusion-DPO_500"][category])
            ddim_std = np.std(self.pick_scores["DIPO_30"][category])
            
            pick_data.extend([
                {"Category": category, "Method": "Difusion-DPO (500 steps)", "Score": ddpm_mean, "Std": ddpm_std},
                {"Category": category, "Method": "DIPO (30 steps)", "Score": ddim_mean, "Std": ddim_std}
            ])

        self._plot_score_comparison(ax1, hps_data, "HPSv2 Score Comparison: Difusion-DPO vs DIPO", "HPSv2 Score")

        self._plot_score_comparison(ax2, pick_data, "PickScore Comparison: Difusion-DPO vs DIPO", "PickScore")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/score_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_dir}/score_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_score_comparison(self, ax, data, title, ylabel):
        """Helper function to plot score comparison"""
        df = pd.DataFrame(data)
        categories = ["Portrait", "Landscape", "Abstract"]
        x = np.arange(len(categories))
        width = 0.35
        
        ddpm_scores = [df[(df['Category'] == cat) & (df['Method'] == 'Difusion-DPO (500 steps)')]['Score'].iloc[0] 
                      for cat in categories]
        ddim_scores = [df[(df['Category'] == cat) & (df['Method'] == 'DIPO (30 steps)')]['Score'].iloc[0] 
                      for cat in categories]
        
        ddpm_stds = [df[(df['Category'] == cat) & (df['Method'] == 'Difusion-DPO (500 steps)')]['Std'].iloc[0] 
                    for cat in categories]
        ddim_stds = [df[(df['Category'] == cat) & (df['Method'] == 'DIPO (30 steps)')]['Std'].iloc[0] 
                    for cat in categories]
        
        bars1 = ax.barh(x - width/2, ddpm_scores, width, label='Difusion-DPO (500 steps)', 
                       color='#2E86AB', alpha=0.8, xerr=ddpm_stds)
        bars2 = ax.barh(x + width/2, ddim_scores, width, label='DIPO (30 steps)', 
                       color='#A23B72', alpha=0.8, xerr=ddim_stds)

        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax.text(bar1.get_width() + ddpm_stds[i] + 0.01, bar1.get_y() + bar1.get_height()/2, 
                   f'{ddpm_scores[i]:.3f}', ha='left', va='center', fontweight='bold')
            ax.text(bar2.get_width() + ddim_stds[i] + 0.01, bar2.get_y() + bar2.get_height()/2, 
                   f'{ddim_scores[i]:.3f}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_ylabel('Prompt Category', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_yticks(x)
        ax.set_yticklabels(categories)
        ax.legend(fontsize=12)
        ax.grid(axis='x', alpha=0.3)
    
    def create_win_rate_visualization(self):
        """Create win rate visualization for both metrics"""
        print("Creating win rate visualization...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        hps_win_rates = {}
        for category in ["Portrait", "Landscape", "Abstract"]:
            ddpm_scores = self.hps_scores["Difusion-DPO_500"][category]
            ddim_scores = self.hps_scores["DIPO_30"][category]
            
            ddpm_wins = sum(1 for d, i in zip(ddpm_scores, ddim_scores) if d > i)
            ties = sum(1 for d, i in zip(ddpm_scores, ddim_scores) if d == i)
            ddim_wins = len(ddpm_scores) - ddpm_wins - ties
            
            total = len(ddpm_scores)
            hps_win_rates[category] = {
                "Difusion-DPO": ddpm_wins / total * 100,
                "DIPO": ddim_wins / total * 100,
                "Tie": ties / total * 100
            }

        pick_win_rates = {}
        for category in ["Portrait", "Landscape", "Abstract"]:
            ddpm_scores = self.pick_scores["Difusion-DPO_500"][category]
            ddim_scores = self.pick_scores["DIPO_30"][category]
            
            ddpm_wins = sum(1 for d, i in zip(ddpm_scores, ddim_scores) if d > i)
            ties = sum(1 for d, i in zip(ddpm_scores, ddim_scores) if d == i)
            ddim_wins = len(ddpm_scores) - ddpm_wins - ties
            
            total = len(ddpm_scores)
            pick_win_rates[category] = {
                "Difusion-DPO": ddpm_wins / total * 100,
                "DIPO": ddim_wins / total * 100,
                "Tie": ties / total * 100
            }

        self._plot_win_rates(ax1, hps_win_rates, "HPSv2 Win Rate Comparison")

        self._plot_win_rates(ax2, pick_win_rates, "PickScore Win Rate Comparison")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/win_rates.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_dir}/win_rates.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_win_rates(self, ax, win_rates, title):
        """Helper function to plot win rates"""
        categories = list(win_rates.keys())
        ddpm_rates = [win_rates[cat]["Difusion-DPO"] for cat in categories]
        ddim_rates = [win_rates[cat]["DIPO"] for cat in categories]
        tie_rates = [win_rates[cat]["Tie"] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.6
        
        bars1 = ax.bar(x, ddpm_rates, width, label='Difusion-DPO Wins', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x, ddim_rates, width, bottom=ddpm_rates, label='DIPO Wins', color='#A23B72', alpha=0.8)
        bars3 = ax.bar(x, tie_rates, width, bottom=[d+i for d,i in zip(ddpm_rates, ddim_rates)], 
                      label='Ties', color='#F18F01', alpha=0.8)

        for i, (ddpm, ddim, tie) in enumerate(zip(ddpm_rates, ddim_rates, tie_rates)):
            if ddpm > 5: 
                ax.text(i, ddpm/2, f'{ddpm:.1f}%', ha='center', va='center', fontweight='bold', color='white')
            if ddim > 5:
                ax.text(i, ddpm + ddim/2, f'{ddim:.1f}%', ha='center', va='center', fontweight='bold', color='white')
            if tie > 5:
                ax.text(i, ddpm + ddim + tie/2, f'{tie:.1f}%', ha='center', va='center', fontweight='bold', color='white')
        
        ax.set_xlabel('Prompt Category', fontsize=14, fontweight='bold')
        ax.set_ylabel('Win Rate (%)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 100)
    
    def create_performance_table(self):
        """Create detailed performance table with both metrics"""
        print("Creating performance table...")
        
        table_data = []
        for category in ["Portrait", "Landscape", "Abstract"]:
            ddpm_hps_scores = self.hps_scores["Difusion-DPO_500"][category]
            ddim_hps_scores = self.hps_scores["DIPO_30"][category]
            ddpm_pick_scores = self.pick_scores["Difusion-DPO_500"][category]
            ddim_pick_scores = self.pick_scores["DIPO_30"][category]

            ddpm_hps_stats = {
                "mean": np.mean(ddpm_hps_scores),
                "std": np.std(ddpm_hps_scores),
            }
            
            ddim_hps_stats = {
                "mean": np.mean(ddim_hps_scores),
                "std": np.std(ddim_hps_scores),
            }

            ddpm_pick_stats = {
                "mean": np.mean(ddpm_pick_scores),
                "std": np.std(ddpm_pick_scores),
            }
            
            ddim_pick_stats = {
                "mean": np.mean(ddim_pick_scores),
                "std": np.std(ddim_pick_scores),
            }

            hps_wins = sum(1 for d, i in zip(ddpm_hps_scores, ddim_hps_scores) if d > i)
            pick_wins = sum(1 for d, i in zip(ddpm_pick_scores, ddim_pick_scores) if d > i)
            total = len(ddpm_hps_scores)
            hps_win_rate = hps_wins / total * 100
            pick_win_rate = pick_wins / total * 100
            
            table_data.append({
                "Category": category,
                "Difusion-DPO HPS": f"{ddpm_hps_stats['mean']:.3f}±{ddpm_hps_stats['std']:.3f}",
                "DIPO HPS": f"{ddim_hps_stats['mean']:.3f}±{ddim_hps_stats['std']:.3f}",
                "Difusion-DPO Pick": f"{ddpm_pick_stats['mean']:.3f}±{ddpm_pick_stats['std']:.3f}",
                "DIPO Pick": f"{ddim_pick_stats['mean']:.3f}±{ddim_pick_stats['std']:.3f}",
                "HPS Win Rate": f"{hps_win_rate:.1f}%",
                "Pick Win Rate": f"{pick_win_rate:.1f}%",
                "HPS Improvement": f"{((ddpm_hps_stats['mean'] - ddim_hps_stats['mean']) / ddim_hps_stats['mean'] * 100):+.1f}%",
                "Pick Improvement": f"{((ddpm_pick_stats['mean'] - ddim_pick_stats['mean']) / ddim_pick_stats['mean'] * 100):+.1f}%"
            })
        
        df = pd.DataFrame(table_data)

        fig, ax = plt.subplots(figsize=(18, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)

        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Detailed Performance Comparison (HPS & PickScore)', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f"{self.output_dir}/performance_table.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.output_dir}/performance_table.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        df.to_csv(f"{self.output_dir}/performance_metrics.csv", index=False)
    
    def run_full_evaluation(self, num_samples=5, seed=42):
        """Run the complete evaluation pipeline with seed control"""
        print(f"Starting full evaluation with seed: {seed}")
        start_time = time.time()

        set_global_seed(seed)
        
        self.generate_images(num_samples, base_seed=seed)

        self.calculate_scores()

        self.create_comparison_grid()
        self.create_score_comparison()
        self.create_win_rate_visualization()
        self.create_performance_table()

        with open(f"{self.output_dir}/raw_hps_scores.json", "w") as f:
            json.dump(self.hps_scores, f, indent=2)
        
        with open(f"{self.output_dir}/raw_pick_scores.json", "w") as f:
            json.dump(self.pick_scores, f, indent=2)
        
        elapsed = time.time() - start_time
        print(f"Evaluation complete in {elapsed:.2f} seconds! Results saved to {self.output_dir}")
        print("\nGenerated files:")
        print("- comparison_grid.png/pdf: Visual comparison grid with both scores")
        print("- score_comparison.png/pdf: Score comparison chart for HPS & PickScore")  
        print("- win_rates.png/pdf: Win rate visualization for both metrics")
        print("- performance_table.png/pdf: Detailed metrics table")
        print("- performance_metrics.csv: Raw metrics data")
        print("- raw_hps_scores.json: All HPS scores")
        print("- raw_pick_scores.json: All PickScore scores")
    
    def run_full_evaluation_from_txt(self, portrait_txt, landscape_txt, abstract_txt, num_samples=5, seed=42):
        print(f"Starting full evaluation from txt with seed: {seed}")
        start_time = time.time()
        set_global_seed(seed)
        self.load_prompts_from_txt(portrait_txt, landscape_txt, abstract_txt, num_samples)
        self.generate_images(num_samples, base_seed=seed)
        self.calculate_scores()
        self.create_comparison_grid()
        self.create_score_comparison()
        self.create_win_rate_visualization()
        self.create_performance_table()
        with open(f"{self.output_dir}/raw_hps_scores.json", "w") as f:
            json.dump(self.hps_scores, f, indent=2)
        with open(f"{self.output_dir}/raw_pick_scores.json", "w") as f:
            json.dump(self.pick_scores, f, indent=2)
        elapsed = time.time() - start_time
        print(f"Evaluation complete in {elapsed:.2f} seconds! Results saved to {self.output_dir}")
    
    def load_prompts_from_txt(self, portrait_txt, landscape_txt, abstract_txt, num_samples=5):
        def read_txt_prompts(txt_path, n):
            prompts = []
            if not os.path.exists(txt_path):
                print(f"Warning: {txt_path} not found.")
                return prompts
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line[0].isdigit():
                        line = line.lstrip('0123456789. ')
                    prompts.append(line)
                    if len(prompts) >= n:
                        break
            return prompts

        self.categorized_prompts = {
            "Portrait": read_txt_prompts(portrait_txt, num_samples),
            "Landscape": read_txt_prompts(landscape_txt, num_samples),
            "Abstract": read_txt_prompts(abstract_txt, num_samples)
        }
        for cat, plist in self.categorized_prompts.items():
            print(f"Loaded {len(plist)} prompts for category: {cat}")

if __name__ == "__main__":
    MODEL_PATH = "path/to/your/model" 
    OUTPUT_DIR = "./Semantic_Categories"
    NUM_SAMPLES = 4
    DEVICES = ["cuda:4", "cuda:5"]
    SEED = 0

    PORTRAIT_TXT = "./Portrait.txt"
    LANDSCAPE_TXT = "./Landscape.txt"
    ABSTRACT_TXT = "./Abstract.txt"

    mp.set_start_method('spawn', force=True)
    evaluator = DiffusionEvaluator(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        devices=DEVICES
    )
    evaluator.run_full_evaluation_from_txt(
        portrait_txt=PORTRAIT_TXT,
        landscape_txt=LANDSCAPE_TXT,
        abstract_txt=ABSTRACT_TXT,
        num_samples=NUM_SAMPLES,
        seed=SEED
    )
