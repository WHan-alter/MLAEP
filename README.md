# Machine Learning-guided Antigenic Evolution Prediction (MLAEP)

Here we introduce the Machine Learning-guided Antigenic Evolution Prediction (MLAEP), which combines structure modeling, multi-task learning, and genetic algorithm to model the viral fitness landscape and explore the antigenic evolution via in silico directed evolution.

## Components
- `data` :  Required data
- `scripts` : Bash scripts
- `src` : python code 
- `analysis` : jupyter notebooks for all the analysis and plots
- `result` : directory for the results


## System requirements
### Hardware requirements
Trained and tested on one NVIDIA Tesla V100 with 32GB GPU memory  

For storing all intermediate files for all methods and all datasets, approximately 100G of disk space will be needed.

### Software requirements

The codes have been tested on CentOS Linux release 7.9.2009 with conda 4.13.0 and python 3.8.5. The list of software dependencies are provided in the `environment.yml` file.


## Installation

1. Create the conda environment from the environment.yaml file:
```
    conda env create -f environment.yml
```

2. Activate the new conda environment:
```
    conda activate covid_predict
```
3. Update huggingface_hub package
```
    conda install huggingface_hub=0.2.1 --force
```
## Data Download
GISAD dataset repuires authentication, and registration is needed to access the data. Therefore, we can't provide the data directly. You can download the data from their web: https://www.gisaid.org. 

## Model Download

The model could be download through the link https://drive.google.com/file/d/1em8015ooDVihvyKbcva9ty70mzoBFvgS/view?usp=sharing

The model could be put under the folder trained_model 

## Usage
Here, we provide an example of model inference and variants synthetic using the selected high-risk variant(HRV) RBD sequences `data/pVNT_seq.csv`(Which is also used in the Fig. 2b). You will find the our model embeddings, predicted escape/binding potentials, and the possible successors along the antigenic evolution direction in `result` directory after running the corresponding commands.

We also provide code to perform the model training with the deep mutational scanning data. The training of the entire model takes around 10 hours with a V100 GPU. The inference can be completed in less than one minute. For the synthetic process, we used a toolkit OpenAttack to perform and visualize the process. It was designed for the natural language and we extended it to protein sequences, the original repository could be find here: https://github.com/thunlp/OpenAttack/tree/bfedfa74f37c69db6d7092d9cc61822ee324919d. It takes approximately one minute to synthesize one variant. 

To get the predictions and embeddings for variants :

```
    bash scripts/run_infer.sh
```


To sythesize the high-risk variants:
```
    bash scripts/run_synthetic.sh
```

