# GLAB-dataset
GLAB-VOD: Global L-band AI-Based Vegetation Optical Depth Dataset 


The dataset is available at https://zenodo.org/doi/10.5281/zenodo.10306094

GLAB VOD is a Global L-band Ai-Based vegetation optical depth dataset with 18-day temporal and 25 km spatial resolution, covering 2002 to 2020. The dataset is created using a neural network with SMOS-SMAP-INRAE-BORDEAUX (SMOSMAP-IB) VOD product as a target (over 2015-2020) and brightness temperatures (TB) from the SMOS, AMSR-E, and AMSR-2 spaceborne missions alongside with a novel soil moisture dataset (CASM) as inputs. The GLAB-VOD dataset was created using a recently developed methodology previously used to create a long-term consistent soil moisture dataset CASM, adapted to the  VOD retrievals. First, the TB and VOD signals were divided into fixed seasonal cycle and residuals, where the residual part of the signal contains sub-seasonal periodic signals, trends, extremes, and noise. Then, a multi-staged neural network training scheme was used to achieve internally consistent predictions by merging data from different sources without introducing biases or compromising data distribution. A side-product of this project is GLAB TB - a global long-term brightness temperature dataset that matches SMOS TB quality and spawns back to 2002. GLAB TB has daily temporal resolution and 25 km spatial resolution. 


The dataset was created with the following steps. Computationally heavy and/or memory intensive steps are highlighed with * and were performed om Columbia University HPC cluster.

1. Download SMAP-IB VOD (https://doi.org/10.1016/j.rse.2022.112921), SMOS TB (https://doi.org/10.5194/essd-9-293-2017), AMSR-2 (10.5067/IKQ0G7ODMLC7), E (10.5067/AMSR-E/AE_LAND3.002) data, validation datasets (https://doi.org/10.3390/rs9050457, https://doi.org/10.1073/pnas.1019576108, https://climate.esa.int/en/projects/biomass/, https://doi.org/10.1016/j.rse.2021.112760, https://doi.org/10.5194/bg-15-5779-2018).

2.* Regrid datasets to the target grid (see example regridding script).

3.* Compute seasonal cycle per location for SMAP-IB VOD, SMOS TB and AMSRE/2 TB (see example seasonal cycle script).

4.* Train SMOS_TB -> SMAP_VOD NN; train AMSR2_TB -> SMOS_TB NN, perform transfer learning (see exanple NN training script). 

5.* Repeat 4 to get structural uncertainty.

6.* Finalize the resulting dataset for analysis and comarison to biomass measurements

7. Analyze the GLAB-VOD dataset (spatial and temporal consistency, see Jupyter Notebook)

8. Compare to validation datasets (Jupiter Notebook)
