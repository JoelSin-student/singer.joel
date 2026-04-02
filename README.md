# KickCap first tries on existing pipelines

## How to use
To preprocess the data, we use the `notebooks/usefull_tools/data_preprocesing.ipynb` notebook, just run it all (it founds the `data/raw_data` directory and outputs in `data/clean_data`). Then, we choose the training and testing data in the `data/clean_data` directory and copy-paste them in `data/training_data` and `data/test_data`, with the OpenGo preprocessed data in the `Insole` folders and the Awinda preprocessed data in the `skeleton` folders. You will find a folder `Other testing files` in each subfolder of `data/test_data`, because each prediction is on unique files, so this is a place to store the other data you will test after the current ones.

To train the Transformer model, predict skeleton joints dimensions from OpenGo data and visualize the resulting output compared to the ground-truth data, we use the terminal on the computer (replace <...> by the actual names):

if you have a kickcap environment (packages available in the `environment.yml` file at the repository's root) in `C:/Users/<username>/anaconda3/envs` to run python and packages:
```
conda activate kickcap
```
Then set the working directory to the repository root:
```
cd <repository root on the computer>/singer.joel
```
Then launch training mode with:
```
python main.py train
```
Here, you can use either the original Transformer-Encoder model or a simple sequence-to-sequence Transformer model by changing model_mode in train.yaml, or override it in your terminal with the line:
```
python main.py train --model_mode <original|simple_seq2seq>
```
prediction mode with:
```
python main.py predict
```
Here, you can use either the original Transformer-Encoder outputed model or the simple sequence-to-sequence Transformer outputed model by changing model_mode and checkpoint path in predict.yaml, or override it in your terminal with the line:
```
python main.py predict --model_mode <original|simple_seq2seq> --checkpoint_file <.\results\weight\best_skeleton_model_original_mse.pth|.\results\weight\best_skeleton_model_simple_seq2seq_mse.pth>
```
visualization mode with:
```
python main.py visual
```
"train": model training (output model in `results/weight`)

"predict": prediction over the test data in `data/test_data/Insole`, outputs the predicted joints' positions in `results/output`

"visual": creates html file in `results/animation` to visualize prediction (file in `results/output`) and ground-truth (file in `data/test_data/skeleton`) skeletons in a web navigator


## How it is organised
Data: gathers all source data, from raw unprocessed to training and test datasets:
 - `data/raw_data/Awinda`: raw data of Awinda IMMU system (full-body no-hands)
 - `data/raw_data/Insoles`: raw data of OpenGo insole sensors
 - `data/clean_data`: preprocessed data of both systems
 - `data/training_data/Insole`: training datasets of preprocessed insole data
 - `data/training_data/skeleton`: training datasets of preprocessed Awinda data
 - `data/test_data/Insole`: test datasets of preprocessed insole data
 - `data/test_data/skeleton`: test datasets of preprocessed Awinda data.

Notebooks: all the code for model training, prediction and visualization:
 - `notebooks/config`: configuration files (what model to use, with what hyperparameters, where to find files, ...)
 - `notebooks/old`: old analyses
 - `notebooks/usefull_tools`: weight checking tool and data preprocessing notebook
 - `notebooks/loader.py`: functions and classes to load and arrange data for the model
 - `notebooks/model.py`: deep learning models
 - `notebooks/predict.py`: code to predict joints' positions from test insole data and trained model
 - `notebooks/train.py`: code to train the model with training data
 - `notebooks/util.py`: code to print logs information in the terminal
 - `notebooks/visualization.py`: code to create an animation comparing ground-truth and predicted skeletons.

Report: R markdown, references and images for HTML project report.

Results: all analyses' results:
 - `results/animation`: HTML skeletons' animation
 - `results/output`: predicted skeleton(s)
 - `notebooks/weight`: trained model(s)

main.py: 

## Logic
Project of the Python-R-Git course, focused on the technical aspect of the research question I try to answer (so far) during my M1 internship.

This repository tries to incorporate the data we collected into already existing code from the literature and replicate its results. This requires to understand the existing code so as to adapt it to the structure of my data. Also, we made major modifications to existing functions to increase the reliability of the code and optimize the training, predicting and visualizing processes. Finally, R code produces the statistics about the performance of the Transformer model.

In particular, I work on a publicly available code of the P2P-Insole experiment ([Watanabe, Aisuwarya and Jing, 2025](https://doi.org/10.48550/arXiv.2505.00755), [GitHub link](https://github.com/onya31-git/P2P-Insole)) in which authors designed a custom device that replicates the one I am using for the data collection (see materials' list below). The aim of this code is to provide a 3D skeleton prediction using insole pressure sensors and accelerometers on the feet. It is a deep learning algorithm based on a transformer architecture (see original publication for more details). I am essentially trying to replicate their results with my data.

The originality of my work compared to other similar publications on pose estimation with pressure sensors is the nature of the data that I am using: typical Kickboxing situations and the OpenGo-Awinda-Delsys materials used. I want to assess the feasibility of pose estimation using pressure sensors and 1 or 2 wrist' IMU in this complex physical activity. 

This repository tries to replicate a former work, adapting the existing code to make it work with other measurement devices and modality addressed. The results will give insights about what preprocessing steps and algorithm architecture would be better suited to the type of data I am using. 

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

1. *tech_vide*: 10 repetitions of single techniques alternating between left and right side: straight, hook, uppercut punches (head level), front, side kicks (stomach level), high, middle, low kicks.
2. *tech_paos*: same task on the kicking pads.
3. *shadow*: shadow-boxing (imaginary fight in the air).
4. *leçon*: coach guiding an imaginary fight with the kicking pads.
5. *sparring*: light training fight.


## Other similar works
SolePoser ([Wu *et al.*, 2024a](https://doi.org/10.1145/3654777.3676418); [Wu *et al.*, 2024b](https://doi.org/10.1109%2FISMAR-Adjunct64951.2024.00036)) -> fundational work (code not available).

Smart Insole ([Han *et al.*, 2024](https://openreview.net/forum?id=DX8C7rAi7O)) -> 600+ insole pressure sensors, detailed pipeline and architecture (code not available).

MotionPRO ([Ren *et al.*, 2025](https://doi.org/10.1109%2FCVPR52734.2025.02585)) -> pressure collected by a sensing mat, large-scale dataset, FRAPPE algorithm fusing pressure and RGB (surely a next try to compare pressure-RGB and pressure-IMU fusions).

PressInPose ([Gao *et al.*, 2024](https://doi.org/10.1145/3699773)) -> very similar study as ours (insole pressure sensors, wrist' IMU), but with 7 activities performed, in which boxing movements. Computationnaly expensive pipeline with synthetic data generation. Richness of pose estimation strategies. Synthetic data improves by around 5 to 20 % the MPJPE score. Very complete results, detailed paper. Highlights the need for a specialized architecture/preprocessing configuration for acyclic and upper-limb needing activities like boxing. Higher sampling rate induces higher accurracy. **Most interesting work**, but (code not available).

## Workflow

1. Understand code behaviour and links.
2. Adjust entry:
    - change paths and names,
    - adapt/add sample rate synchronising (60 Hz, Awinda) and other preprocessing steps (change df col names, ...),
    - adapt scales and dimensionnality (including transformer hyperparameters related to input and output dimensions).
3. Adjust classes, functions and variables if needed.
    - especially what is related to pressure, IMU and joints' features,
    - also the gradient features' computation,
    - also the model feed-forward behaviour,
    - also the features' scalers.
4. Test different hyperparameters' configurations.
