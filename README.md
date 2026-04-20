# KickCap first tries on existing pipelines

## How to use
Follow the `main.ipynb` notebook at the root of the repository. Preprocessing founds the `data/raw_data` directory and outputs in `data/clean_data` as well as the usable data folders `data/train_data`, `data/test_data`. The OpenGo preprocessed data in the `Insole` folders and the Awinda preprocessed data in the `skeleton` folders. You will find a folder `Other files` in each subfolder of those folders, to easily manipulate training and testing data if needed.

To train the model, predict skeleton joints dimensions from OpenGo data and visualize the resulting output compared to the ground-truth data, we normally use the terminal on the computer. However, I simplified the procedure by putting the commandlines in `main.ipynb`. Just follow the notebook.

**First, you NEED to install the kickcap environment: `environment.yml`.** Then select this kernel when running `main.ipynb`.



## How it is organised

**environment.yml**: packages used.

**main.ipynb**: to run preprocessing, (pre-)training, prediction and visualization.

**sources/main.py**: code controller.

**main.Rmd**: to run statistical analyses about model quality.

**PRETRAINING_GUIDE.md**: to understand auxiliaries pretraining for SoleFormer.

**TODO.txt**: next things to do, in short- and medium- terms.

**Data**: gathers all source data, from raw unprocessed to training and test datasets:
 - `data/ablation_commands.csv`: information about ablation configurations of SoleFormer
 - `data/subject_anthropo.txt`: anonymised anthropometric data per subject
 - `data/raw_data/Awinda`: raw data of Awinda IMMU system (full-body no-hands)
 - `data/raw_data/Insoles`: raw data of OpenGo insole sensors
 - `data/clean_data`: preprocessed data of both systems
 - `data/clean_data/Awinda_targets_soleformer`: skeleton data for the SoleFormer pipeline
 - `data/training_data/Insole`: training datasets of preprocessed insole data
 - `data/training_data/skeleton`: training datasets of preprocessed Awinda data
 - `data/test_data/Insole`: test datasets of preprocessed insole data
 - `data/test_data/skeleton`: test datasets of preprocessed Awinda data.

**Notebooks**: notebook-only workflow files:
 - `notebooks/usefull_tools`: data preprocessing notebook

**Sources**: all Python source code and model configuration files:
 - `sources/config/transformer_encoder`: configuration files (model, hyperparameters, paths)
 - `sources/loader.py`: functions and classes to load and arrange data for the model
 - `sources/model.py`: deep learning models
 - `sources/train.py`: to train the model with training data
 - `sources/predict.py`: to predict joints' positions from test insole data and trained model
 - `sources/visualization.py`: code to create an animation comparing ground-truth and predicted skeletons
 - `sources/usefull_tools`: weight checking tool and insole-extracted tabs synchroniser

**Sources**
Conceptually, all the code should be in this file, because we are using source Python files. However, they distinguish themselves by having a "start" function that organizes functions between them, before sending to sources/main.py (model controller), and then being called by the commandlines in main.ipynb. We put only the util.py, defining functions used in main.ipynb for the main pipeline,.
 - `sources/util.py`: functions for robust path resolving, file names creation, preprocessing calling, robust command running, ablation management

**Report**: R markdown, references and images for HTML project report.

**Results**: all analyses' results:
 - `results/animation`: HTML skeletons' animation
 - `results/learning_results`: training and validation loss per epoch
 - `results/output`: predicted skeleton(s)
 - `results/pretrained_aux`: auxiliary networks models for SoleFormer model
 - `results/weight`: trained model(s)
 - `results/R`: statistical analyses output



## Logic
Project of the Python-R-Git course, focused on the technical aspect of the research question I try to answer (so far) during my M1 internship.

This repository tries to incorporate the data we collected into already existing code from the literature and replicate its results. We found the code of one study. Our work mainly focused on:
- understanding the code structure;
- familiarizing with PyTorch library;
- adapting/improving the code;
- being able to use the same structure for other models;
- testing different hyperparameters configurations for the best results.

We made major modifications to the original functions and structure to: increase the reliability of the code, have detailed information about the training process, optimize the training, prediction and visualization processes and outputs. Finally, R code produces the statistics about the performance of the Transformer model.

In particular, I work from a publicly available code of the P2P-Insole experiment ([Watanabe, Aisuwarya and Jing, 2025](https://doi.org/10.48550/arXiv.2505.00755), [GitHub link](https://github.com/onya31-git/P2P-Insole)) in which authors designed a custom device that replicates the one I am using for the data collection (see materials' list below). The aim of this code is to provide a 3D skeleton prediction using insole pressure sensors and accelerometers at the feet level. It is a deep learning algorithm based on a transformer architecture (see original publication for more details). I am essentially trying to replicate their results with my data, and extending to other models, like SoleFormer ([Wu *et al.*, 2024a](https://doi.org/10.1145/3654777.3676418)).

The originality of my work compared to other publications on pose estimation with pressure sensors is the nature of the data that I am using: participants are performing typical Kickboxing situations while wearing the OpenGo-Awinda-Delsys-video materials. Those first analyses will allow me to assess the feasibility of pose estimation using pressure sensors and 1 or 2 wrist' IMU in this complex physical activity, as well as giving directions on what to try next.


## Data collection 
### Materials
Data collected:
- **soles' pressure**: connected insoles featuring 16 capacitive pressure sensors, and a 6-DoF IMU in each (100 Hz, OpenGo, Moticon).

- **Whole-body accelerometry**: 9-DoF IMMU (60 Hz, Awinda, Xsens).

- **Electromyography** (not used here): 10 sEMG (1259.259 Hz, Trigno, Delsys) on relevant muscle groups.

- **Video** (not used here): 1 triggered synchronised webcam (720 p, 30 Hz) and 1 manually synchronised camera (720 p, 25 Hz).

Other:
- Kickboxing practice material: shin pads, groin guard, mouthguard, bands, gloves.

- Kickboxing clothes: shorts, t-shirt.

- Boxing shoes.

- Kicking pads.

### Task

Five Kickboxing situations of 3 min each, with 3 min of rest between them:

1. *tech_vide*: 10 repetitions of single techniques alternating between left and right side: straight, hook, uppercut punches (head level), front, side kicks (stomach level), high-, middle-, low-kicks.
2. *tech_paos*: same task on the kicking pads.
3. *shadow*: shadow-boxing (imaginary fight in the air).
4. *leçon*: coach guiding an imaginary fight with the kicking pads.
5. *sparring*: light training fight.


## Other similar works
SolePoser ([Wu *et al.*, 2024a](https://doi.org/10.1145/3654777.3676418); [Wu *et al.*, 2024b](https://doi.org/10.1109%2FISMAR-Adjunct64951.2024.00036)) -> fundational work (code not available).

Smart Insole ([Han *et al.*, 2024](https://openreview.net/forum?id=DX8C7rAi7O)) -> 600+ insole pressure sensors, detailed pipeline and architecture (code not available).

MotionPRO ([Ren *et al.*, 2025](https://doi.org/10.1109%2FCVPR52734.2025.02585)) -> pressure collected by a sensing mat, large-scale dataset, FRAPPE algorithm fusing pressure and RGB (surely a next try to compare pressure-RGB and pressure-IMU fusions).

PressInPose ([Gao *et al.*, 2024](https://doi.org/10.1145/3699773)) -> very similar study as ours (insole pressure sensors, wrist' IMU), but with 7 activities performed, in which boxing movements. Computationnaly expensive pipeline with synthetic data generation. Richness of pose estimation strategies. Synthetic data improves by around 5 to 20 % the MPJPE score. Very complete results, detailed paper. Highlights the need for a specialized architecture/preprocessing configuration for acyclic and upper-limb needing activities like boxing. Higher sampling rate induces higher accurracy. (code not available)

KineticsSense ([Zhang et al., 2025](https://dl.acm.org/doi/10.1145/3749462)) -> predicting lower-limb EMG from plantar pressure and IMU.

Ground Reaction Inertial Poser ([Hori et al., March 2026](https://doi.org/10.48550/arXiv.2603.16233)) -> OpenGo + 2 smart watches, refined feature extractors, control of a humanoid model in physics-aware virtual environment (code will be made available, e-mail confirmation).

## Workflow

1. Understand code behaviour and links.
2. Adjust entries:
    - change paths and names,
    - adapt/add sample rate synchronising (60 Hz, Awinda) and other preprocessing steps (change df col names, ...),
    - adapt scales and dimensionnality (including transformer hyperparameters related to input and output dimensions).
3. Adjust classes, functions and variables:
    - especially what is related to pressure, IMU and joints' features,
    - also the gradient features' computation,
    - also the model feed-forward behaviour,
    - also the features' scalers,
    - also the prediction dataframe,
    - also the visualization procedure,
    - also display useful information during processes.
4. Test different hyperparameters' configurations.
5. Add a robust structure to test multiple models with the same code.
6. Test other models.
