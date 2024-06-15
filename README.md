# RWKU: Real-World Knowledge Unlearning Benchmark



### Installation

```bash
git clone https://github.com/jinzhuoran/RWKU.git
conda create -n rwku python=3.10
conda activate rwku
cd RWKU
pip install -r requirements.txt
```

### Dataset Download and Processing


One way is to load the dataset from [Huggingface](https://huggingface.co/datasets/jinzhuoran/RWKU) and preprocess it:
```bash
cd data
python data_process.py
```
```python
from datasets import load_dataset
forget_target = load_dataset("jinzhuoran/RWKU", 'forget_target')['train'] # 200 unlearning targets
forget_level1 = load_dataset("jinzhuoran/RWKU", 'forget_level1')['test'] # forget knowledge memorization probes
forget_level2 = load_dataset("jinzhuoran/RWKU", 'forget_level2')['test'] # forget knowledge manipulation probes
forget_level3 = load_dataset("jinzhuoran/RWKU", 'forget_level3')['test'] # forget adversarial attack probes
neighbor_level1 = load_dataset("jinzhuoran/RWKU", 'neighbor_level1')['test'] # neighbor knowledge memorization probes
neighbor_level2 = load_dataset("jinzhuoran/RWKU", 'neighbor_level2')['test'] # neighbor knowledge manipulation probes
```

Another way is to download the processed dataset directly from [Google Drive](https://drive.google.com/file/d/1ukWg-T3GPvqpyW7058vNyRWdXuQHRJPb/view?usp=sharing):
```bash
cd LLaMA-Factory/data
bash download.sh
```

### Evaluating Models 

To evaluate the model original performance before unlearning:
```bash
cd LLaMA-Factory/scripts
bash run_original.sh
```


### Unlearning Models

We adapt [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to train the model. 
We provide several scripts to run various unlearning methods.
To run the In-Context Unlearning (ICU) method on Llama-3-8B-Instruct:
```bash
cd LLaMA-Factory/scripts
bash run_icu.sh
```
To run the Gradient Ascent (GA) method on Llama-3-8B-Instruct:
```bash
cd LLaMA-Factory/scripts
bash run_ga.sh
```
To run the Direct Preference Optimization (DPO) method on Llama-3-8B-Instruct:
```bash
cd LLaMA-Factory/scripts
bash run_dpo.sh
```
To run the Negative Preference Optimization (NPO) method on Llama-3-8B-Instruct:
```bash
cd LLaMA-Factory/scripts
bash run_npo.sh
```
To run the Rejection Tuning (RT) method on Llama-3-8B-Instruct:
```bash
cd LLaMA-Factory/scripts
bash run_rt.sh
```