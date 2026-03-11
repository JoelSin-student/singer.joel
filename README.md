# KickCap first tries on existing pipelines

## How to use


## How it is organised


## Logic
Project of the Python-R-Git course, focused on the technical aspect of the research question I try to answer (so far) during my M1 internship.

This repository aims at trying to incorporate the data we collected into already existing code from the literature. This requires to understand the code so as to adapt it to the structure of my data.

In particular, I work on a publicly available code of the P2P-Insole experiment ([Watanabe, Aisuwarya and Jing, 2025](https://doi.org/10.48550/arXiv.2505.00755)) in which authors designed a custom device that replicates the one I am using for the data collection (see materials' list below). The aim of this code is to provide a 3D skeleton prediction using insole pressure sensors and accelerometers on the feet. It is a deep learning algorithm based on a transformer architecture (see original publication for more details). I am essentially trying to replicate their results with my data.

The originality of my work compared to other similar publications on pose estimation with pressure sensors is the nature of the data that I am using: typical Kickboxing situations. I want to assess the feasibility of pose estimation using pressure sensors and 1 or 2 wrist' IMU in this complex physical activity. 

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
    - especially the sensors'  and skeleton's coordinates tables
4. Test different hyperparameters' configurations.
