Produce Quality Evaluation
This project focuses on evaluating the quality of produce using advanced image processing and machine learning techniques. The system evaluates three types of produce:

Apple Defect Detection
Orange Defect Detection
Green Pepper Curvature Evaluation
By providing accurate quality assessments, this project aids in automating the quality control process in agricultural and supply chain industries.


Detailed Modules
1. Apple Defect Detection
Description: This module uses YOLO (You Only Look Once) to detect surface defects on apples, including bruises, scars, and discolorations.
Inputs: High-resolution images of apples.
Outputs: A binary classification for each apple (defective or non-defective) with defect localization.
Key Features:
Trained on a diverse dataset of apple defects.
Provides bounding boxes for defect regions.
2. Orange Defect Detection
Description: Focuses on identifying peel defects in oranges, such as rotting spots, cuts, or abrasions.
Inputs: Images of oranges under uniform lighting.
Outputs: A defect score for each orange (0–1 scale, where 1 indicates severe defects).
Key Features:
Implements preprocessing to handle variations in peel texture.
Outputs both defect segmentation masks and scores.
3. Green Pepper Curvature Evaluation
Description: This module evaluates the curvature of green peppers, classifying them into categories (e.g., straight, moderately curved, and highly curved).
Inputs: Images of green peppers placed on a uniform background.
Outputs: A curvature score and classification label.
Key Features:
Uses contour extraction and curvature calculation techniques.
Generates a visualization of the pepper contour with curvature measurements.
