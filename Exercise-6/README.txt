The raw data is stored in /Corridor_Data and /Bottleneck_Data.
The clean data is stored in /Corridor_Clean and /Bottleneck_Clean.

The implementation of the knowledge-based model in Task 3 is in /kb.
The implementation of the neural network model in Task 2.2 is in /nn.
Some utility functions for data processing and visualization are in /utils.

In the main folder there are the notebooks used for the analysis and the report:
- `Task 1 - data_visualization.ipynb` contains the analysis of the data and the plots used in the report.
- `Task 2.1 - 2.1 experiment.ipynb` contains the experiment from Task 2.1.
- `Task 2.2 - network.ipynb` contains the train and test of the neural network model and the plots used in the report.
- `Task 3 - knowledge.ipynb` contains the analysis of the knowledge-based model and the plots used in the report.
- data_preprocessing.ipynb contains the data preprocessing and cleaning for the neural network model.
- kb_preprocessing.ipynb contains the data preprocessing and cleaning for the knowledge-based model.
- pedestrian_trajectory_video_generation.ipynb contains the code to generate the videos of the pedestrian trajectories. (This is not a task but it can be used as an utility to understand in motion the trajectories of the pedestrians)


The data description is as follows:

Corridor data are trajectories of pedestrians in a closed corridor of lenght 30m and width 1.8m. 
The trajectories are measured on a section of length 6m. 
Experiments are carried out with N=15, 30, 60, 85, 95, 110, 140 and 230 participants. 

Bottleneck data are trajectories of pedestrian in a bottleneck of lenght 8m and width 1.8m. 
Experiments are carried out with 150 participants for bottleneck widths w=0.7, 0.95 1.2 and 1.8m. 

See http://ped.fz-juelich.de/experiments/2009.05.12_Duesseldorf_Messe_Hermes/docu/VersuchsdokumentationHERMES.pdf page 20 and 28 for details. Column names of the file are: ID FRAME X Y Z. ID is the pedestrian ID. FRAME is the frame number (frame rate is 1/16s). X Y and Z pedestrian position in 3D. The data are part of the online database http://ped.fz-juelich.de/database.

Column names of the file are: ID FRAME X Y Z. 
ID is the pedestrian ID. 
FRAME is the frame number (frame rate is 1/16s). 
X Y and Z pedestrian position in 3D. 
