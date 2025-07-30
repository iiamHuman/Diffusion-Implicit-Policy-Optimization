 # Diffusion Implicit Policy Optimization (DIPO)

![DIPO vs Diffusion-DPO Sampling]![alt text](assets/figure3.png)  
*Figure 1: Visual comparison between DIPO (30 steps, top) and Diffusion-DPO (800 steps, bottom) - From Paper Figure 3*

> **Key Breakthrough**: **10-15× sampling acceleration** while preserving generation quality and human preference alignment  
✅ **No training objective modification** ✅ **Compatible with existing diffusion models** ✅ **Stochasticity control**

---

## 📌 Core Contributions
DIPO seamlessly integrates **DDIM accelerated sampling** into the Diffusion-DPO framework, with theoretical proof:
```math
L_{\text{DIPO}} = L_{\text{Diffusion-DPO}} + C \quad \text{(Gradient Invariance)}

Key Advantages:

⚡ 10-15× Speedup: Matches 500-step Diffusion-DPO quality in 30-50 steps (Table 1,2)

🎯 Preserved Preference Alignment: 59.9\% human preference win rate vs. base model (Figure 4)

🧩 Plug-and-Play: Reuses Diffusion-DPO training pipelines and model weights

🔧 Quick Start

## Environment Installation
Execute the following command:
```
conda create -n env python=3.10 -y
conda activate env
pip install -r requirements.txt

```
Training Configuration

base_model: "stabilityai/stable-diffusion-xl-1.0"
batch_size: 2048      # Effective distributed batch size
learning_rate: 8.192e-9
loss:
  dpo_coef: 5000      # β_{\text{DPO}} parameter
dataset: "Pick-a-Pic" # Human preference dataset

Note: Training code is fully compatible with Diffusion-DPO - only sampler replacement needed -Just put the trained model path into the model path of the python file

## File Description

- **nateraw**:parti-prompts dataset
- **hasty_sample.py:**: Quickly sample and generate images and save them to the generated_images folder.
- **Ablation_experiment.py**: The effect of the η parameter (randomness control) in DIPO sampling on generation quality is studied, and PickScore/HPSv2 scoring tables and visualization charts are generated through 50-step sampling..
- **compare_quality.py**: Test the efficiency and quality of different sampling configurations (DIPO 30/50/100 steps vs. Diffusion-DPO 500/800 steps), and generate a comparison table including sampling time, speedup, and PickScore/CLIP scores.
- **Semantic_Categories.py**: We compared the generation quality of 500 steps and 30 steps for three categories of themes (portraits, landscapes, and abstract art), and analyzed the difference in success rate using a visual grid and quantitative metrics (HPSv2 score).
- **Human_Preference_compare.py**: Compare the performance of the pre- and post-training models at 50 steps of DIPO and 500 steps of Diffusion-DPO. Use HPSv2 for single-image scoring and preference comparison, generate bar charts and pie charts, and support multi-GPU execution.
- **download_models.py**:Download the required evaluation model.

Download the evaluation model locally and modify the path in the python file.

📊 Experimental Results
1. Sampling Efficiency vs. Generation Quality
Method	Steps	PickScore	Time (s)	Speedup
DIPO	20	22.0609	5.34	23.80×
DIPO	50	22.3247	12.87	9.87×
Diffusion-DPO	500	22.6312	126.28	1.00×
*Table 1: DIPO achieves comparable quality to 500-step Diffusion-DPO in 50 steps (CLIPScore ≈0.663)*

2. Human Preference Alignment
(assets/figure4.png)
*Figure 2: Fine-tuned model maintains preference advantage under DIPO sampling (59.9% win rate) - From Paper Figure 4*

3. Stochasticity Parameter (η) Analysis
(assets/figure5.png)
*Figure 3: Minimal quality impact across η∈[0,1] (fluctuations <0.15%) - From Paper Figure 5*

4. Cross-Category Performance
Category	HPSv2 Win Rate	PickScore Win Rate
Portrait	74.0% DDPO	50.5% DDPO
Landscape	68.3% DDPO	54.0% DIPO
Abstract Art	63.7% DDPO	51.2% DDPO
DIPO maintains text-image consistency while DDPO excels in aesthetic preference

📖 Citation

@article{anonymous2024dipo,
  title={Diffusion Implicit Policy Optimization},
  author={Anonymous},
  journal={Submitted to arXiv},
  year={2024}
}

🙏 Acknowledgements
Codebase builds upon Diffusion-DPO. Evaluated using Pick-a-Pic dataset.