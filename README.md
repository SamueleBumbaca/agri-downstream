# Project Title: Agri-Downstream Object Detection

## Overview
The Agri-Downstream project focuses on implementing object detection techniques using the Faster R-CNN model. This repository contains the necessary configurations, datasets, and scripts to train and evaluate the model on agricultural datasets.

## Project Structure
```
agri-dowstream
├── src
│   ├── object_detection_agrinet
│   │   ├── config
│   │   │   └── split.yaml          # Configuration for dataset splitting
│   │   ├── dataset                  # Directory for dataset files
│   │   └── object_detection_agrinet
│   │       └── config_Faster_RCNN_TPN_SAGIT22_C_4175_2022_05_27.yaml  # Model training configuration
├── experiments
│   └── Faster_RCNN_TPN_SAGIT22_C_4175_2022_05_27
│       └── Handcrafted_dataset
│           └── bboxes.csv           # Bounding box annotations for the dataset
├── dataset                            # Main dataset files
├── .gitignore                         # Files and directories to ignore by Git
└── README.md                          # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd agri-dowstream
   ```

2. Install the required dependencies. Ensure you have Python and pip installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset according to the specifications in `src/object_detection_agrinet/config/split.yaml`.

## Usage Guidelines
- Modify the configuration files in `src/object_detection_agrinet/config` to suit your dataset and training preferences.
- Use the provided scripts to train the Faster R-CNN model and evaluate its performance on the dataset.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
