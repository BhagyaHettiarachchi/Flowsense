# Machine Learning-Based Road Traffic Volume Analysis



## Project Overview
This project applies machine learning techniques to predict road traffic volume using historical traffic count data. It supports Intelligent Transportation Systems (ITS) by modeling traffic behavior based on temporal and directional features. The implementation is optimized for large datasets and constrained lab environments.

## Objectives
- Predict traffic count using supervised learning models
- Evaluate model performance using RMSE and R² metrics
- Visualize prediction accuracy against actual traffic data
- Demonstrate ITS relevance and software reliability

## Dataset
- Source: [Insert dataset source or citation]
- Format: CSV file with fields including `startDate`, `siteID`, `regionName`, `siteReference`, `classWeight`, `laneNumber`, `flowDirection`, and `trafficCount`
- Preprocessing includes date parsing, feature extraction, encoding, and memory optimization

## Features Used
- Temporal: Day, Month, Year
- Directional: Lane Number, Flow Direction
- Vehicle Class: Light/Heavy (encoded)
- Target Variable: `trafficCount`

## Models Compared
- Linear Regression
- Decision Tree Regressor

## Results
| Model              | RMSE     | R² Score |
|--------------------|----------|----------|
| Linear Regression  | 3956.11  | 0.37     |
| Decision Tree      | 2508.23  | 0.75     |

Decision Tree outperformed Linear Regression, indicating non-linear traffic patterns.

## Visualization
The output graph compares actual vs. predicted traffic counts over a sample of 100 observations, demonstrating the model’s accuracy and responsiveness to traffic fluctuations.

## File Structure
Machine_Learning_Traffic_Analysis/

│
├── traffic_volume_analysis.py       # Main script
├── traffic_volume_dataset.csv       # Dataset
├── README.md                        # Project documentation
├── Project_Report.docx              # Final APA-style report
├── Presentation_Slides.pptx         # Final presentation
├── Turnitin_Report.pdf              # Originality check


## How to Run
1. Install dependencies:
   ```bash
   pip install pandas matplotlib scikit-learn
   
2. Place traffic_volume_dataset.csv in the same directory

 ```bash
python traffic_volume_analysis.py




