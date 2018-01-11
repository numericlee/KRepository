 # Project Overview #
 
This project includes a series of programs used to predict the likelihood an airline passenger is carrying contraband in each of 17 body zones. The methodology is an ensemble based on several tensor flow machine learning models combined with some heuristic algorithms utilizing morphological transformations

The project is based on a contest sponsored by Homeland security with $1.5 million of prize money and hosted by Kaggle. https://www.kaggle.com/c/passenger-screening-algorithm-challenge
Our solution, based on an ensemble of machine learning models using the tensor flow library, placed 54th out of 518 entrants

## Data Source ##

The TSA supplies data in several formats. Our algorithms reply on the A3DAPS images. 64 images are offered, each corresponding to a camera angle 1/64th of a degree apart. We utilize 8 images, corresponding to the front, rear, and 6 side views spaced at equal intervals.
Stage1 of the contest involved 1247 passengers (of which 100 initially had hidden labels.) Stage2 had 1388 passengers. Data was provided by the TSA via Google Cloud Buckets and exceeded 4 Terabytes

Note that the raw data supplied by Homeland Security is confidential and cannot be shared

## Workflow ##

The workflow is a series of batch processes.  The workflow was developed after a month of analysis and a variety of exploratory studies. The major steps are:

1)Preprocess the images (crop, normalize, grayscale) followed by extraction of 17 zone images. The extraction was based on a single set of coordinates for the approximate position and depth of each zone relative to the overall subject in each camera angle. This was followed by re-cropping.  Some zones were flipped (Horizontal symmetry.)  There is code to straighten forearms and upper arms to a vertical position to facilitate machine learning, but in the final version these methods were abandoned.  The process generated over 130,000 ZoneView files, (roughly 106 per passenger) in .csv format for further analysis. The relevant code can be found in code block 4 of DHHS (except for arms code block 5)

2) Perform a series of computations on the ZoneView files.  Some of these computations were intended to detect specific patterns (e.g.  a lighter shape of specified size enclosed by a darker shape of specified size.) Others looked for unusual and unnatural variations. In stage 1, the results of these computations were compared against the known passenger labels using a logistic regression and the resulting model was saved in a .pkl file. The relevant code can be found in code block 10 of DHHS 

3) Organize the ZoneView files in minibatches which would serve as inputs to the machine learning models. The 17 body zones were combined using symmetry into 10 regional models; for example, a single model predicted both left knee and right knee. The minibatch process reflected the machine learning parameters of each model including image dimensions and batch sizes.  Passengers with hidden labels are batched separately from those with known labels. The relevant code can be found in code block 15 of DHHS 

4) Train (and sometimes retrain) the 10 machine learning models.  In Stage II, contestants had the opportunity to retrain their models to reflect 100 labels which had previously been hidden. However, the tight turnaround afforded us only a limited amount of time for retraining. Our model generated a prediction for each individual image in each region.  A separate logistic regression combined these individual predictions and the results of Step II into an overall prediction of the likelihood a given passenger carried contraband in each zone. The ML used a 7-layer Convolutional Neural Net from the TFLearn library.  The ML parameters (including image size, number of iterations, learning rate and decay, etc. varied by region.  The logistic regression relied on the three images with the highest probability along with cross-product terms.  (The relevant code can be found in Code block 16 of DHHS as well as Reloader and (in Stage II) BatchRunner

The python code generated a series of csv files, one for each region.  Combiner.bat combines these. Some minor editing is needed to make this submission-ready (e.g. remove redundant “ID, prediction” and remove EOL character

## Data Files ##

Vertices5 – coordinates of each zone in front view; views to be utilized
Phase Moons -geometric inputs used to calculate coordinates in other views
Inventories of Passengers and PZ files

Model Files: The main model files were not uploaded due to GitHub space limitations. The model outputs were generally 500MB – 1GB per region

Logistic Regression models for each of 10 regions in pkl format

Logistic regression of the 48 metrics

Python Scripts:
- DHHS (Exploration, ZoneView Generation, MiniBatch generation,)
- Reloader (Retraining ML)
- BatchRunner (ML Execution and Ensemble Predictions)
