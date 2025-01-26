# Food Object Detection on CPU with YOLOv8

## Table of Contents
1. [Business Use Case](#business-use-case)
2. [Project Overview](#project-overview)  
3. [Data](#data)  
4. [Setup & Installation](#setup--installation)  
5. [Training the Model](#training-the-model)  
6. [Inference on CPU](#inference-on-cpu)  
7. [Deployment](#deployment)
8. [Results & Evaluation](#results--evaluation)  
9. [Project Structure](#project-structure)  
10. [License](#license)  

---

## Business Use Case
This project addresses several key business needs in the food industry:
- **Restaurant Menu Digitization**: Automatically identify and catalog food items from images
- **Food Delivery Apps**: Verify food item accuracy and presentation
- **Quality Control**: Ensure consistency in food presentation across chain restaurants
- **Customer Experience**: Enable visual search for food items
- **Inventory Management**: Track food items through visual identification
- **Nutritional Analysis**: Support automated food logging and dietary tracking

## Project Overview
This repository provides a complete solution for food object detection using YOLOv8, optimized for CPU deployment. The system can detect 10 food classes:
- Pizza 🍕
- Burger 🍔
- Sandwich 🥪
- Salad 🥗
- Pasta 🍝
- Sushi 🍱
- Steak 🥩
- Soup 🥣
- Taco 🌮
- Curry 🍛

**Key Features**:
- CPU-optimized implementation
- Production-ready Gradio web interface
- Docker containerization
- AWS deployment support
- Comprehensive evaluation metrics

**Performance Metrics**:
- mAP50: 0.630
- mAP50-95: 0.562
- Precision: 0.636
- Recall: 0.566

## Data Collection and Labeling

### Data Sources
- Images collected from various sources:
  - Yelp Dataset Challenge
  - Food101 Dataset
  - Custom web scraping (using `scripts/scrape_images.py`)
  - Manual collection for underrepresented classes

### Data Labeling Process
1. **Initial Automated Labeling**:
   - Used GroundingDINO for zero-shot detection (`scripts/grounding_dino_sam_infer.py`)
   - Generated initial bounding boxes for common food items
   - Confidence threshold set to 0.25 to capture more potential objects

2. **Manual Verification and Correction**:
   - Used CVAT (Computer Vision Annotation Tool) for verification
   - Corrected bounding boxes and class labels
   - Added missing annotations
   - Removed false positives

3. **Label Format Conversion**:
   - Converted CVAT exports to YOLO format (`scripts/convert_labels.py`)
   - Each label file contains: `<class_id> <x_center> <y_center> <width> <height>`
   - Normalized coordinates (0-1) for better model generalization

### Labeling Challenges and Solutions

1. **Class Ambiguity**:
   - **Challenge**: Similar-looking foods (e.g., sandwich vs burger)
   - **Solution**: Created detailed annotation guidelines (`docs/annotation_guidelines.md`)
   - Example: Burgers must show the bun and patty clearly

2. **Inconsistent Annotations**:
   - **Challenge**: Different annotators, different standards
   - **Solution**: 
     - Implemented label verification script (`scripts/verify_labels.py`)
     - Double-review process for ambiguous cases
     - Regular team calibration sessions

3. **Scale and Viewpoint Variations**:
   - **Challenge**: Food items photographed from different angles
   - **Solution**: 
     - Augmented training data with rotations and scales
     - Included multi-scale training in YOLOv8 configuration

4. **Label Quality Issues**:
   - **Challenge**: Initial automated labels had many errors
   - **Solution**: 
     - Developed label cleaning script (`scripts/fix_labels.py`)
     - Added validation checks for label coordinates
     - Implemented class mapping corrections

### Data Storage and Organization

```
data/
├─ raw/                      # Original unprocessed images
│  ├─ yelp_photos/          # Images from Yelp dataset
│  ├─ food101/              # Images from Food101
│  └─ custom/               # Custom collected images
├─ train/                   # Training dataset (70%)
│  ├─ images/              # JPG format
│  └─ labels/              # YOLO format txt files
├─ val/                    # Validation dataset (20%)
│  ├─ images/             
│  └─ labels/              
├─ test/                   # Test dataset (10%)
│  ├─ images/
│  └─ labels/
└─ data.yaml               # Dataset configuration
```

### Label Statistics
- Total Images: 15,000
- Distribution per class:
  - Pizza: 2,500
  - Burger: 2,000
  - Sandwich: 1,800
  - Salad: 1,700
  - Sushi: 1,500
  - Pasta: 1,400
  - Steak: 1,300
  - Soup: 1,200
  - Taco: 900
  - Curry: 700

### Scripts for Label Management
- `scripts/grounding_dino_sam_infer.py`: Initial automated labeling
- `scripts/fix_labels.py`: Clean and correct label issues
- `scripts/verify_labels.py`: Validate label format and content
- `scripts/convert_labels.py`: Convert between annotation formats
- `scripts/analyze_labels.py`: Generate label statistics
- `scripts/split_data.py`: Split dataset into train/val/test

## Setup & Installation
1. Clone the repository
2. Create a conda environment:
```bash
conda create -n food_detection python=3.11
conda activate food_detection
```
3. Install requirements:
```bash
pip install -r requirements.txt
```

## Training the Model
1. Prepare your data in YOLO format
2. Update data.yaml with your paths
3. Train the model:
```bash
yolo train model=yolov8n.pt data=data/data.yaml epochs=50 imgsz=640 batch=16 device=cpu
```

## Inference on CPU
Run the Gradio web interface:
```bash
python app.py
```
The interface provides:
- Image upload capability
- Real-time detection
- Confidence scores
- Model performance metrics

## Deployment
For detailed deployment instructions, see [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md).

Quick deployment steps:
1. Build Docker image:
```bash
docker build -t food-detection-app .
```

2. Test locally:
```bash
docker run -p 7860:7860 food-detection-app
```

3. Deploy to AWS:
- Push to Amazon ECR
- Deploy using AWS App Runner
- Monitor and scale as needed

**Live Demo**:  
The application is deployed and accessible at:  
[https://4wtec9nstj.us-east-1.awsapprunner.com](https://4wtec9nstj.us-east-1.awsapprunner.com)

## Project Structure
```
food_detection_cpu/
├─ app.py                    # Gradio web interface
├─ data/                     # Dataset directory
├─ requirements.txt          # Python dependencies
├─ Dockerfile               # Container configuration
├─ AWS_DEPLOYMENT.md        # AWS deployment guide
├─ runs/                    # Training outputs
│  └─ detect/
│     └─ train/
│        └─ weights/        # Model weights
└─ examples/                # Example images
```

**Key Components**:
- `app.py`: Main application with Gradio interface
- `data/`: Dataset and annotations
- `runs/detect/train/`: Training outputs and model weights
- `Dockerfile`: Container configuration for deployment
- `AWS_DEPLOYMENT.md`: Detailed AWS deployment instructions

## Results & Evaluation
The model achieves strong performance across different food categories:
- Best performing: Sushi (mAP50: 85.8%)
- Most challenging: Pasta (mAP50: 33.2%)

### Performance Visualization
![Model Performance Metrics](performance_metrics.png)

### Example Detections
Here are some example detections from our model:

#### Pizza Detection
![Pizza Detection](examples/pizza.jpg)

#### Burger Detection
![Burger Detection](examples/burger.jpg)

#### Sushi Detection
![Sushi Detection](examples/sushi.jpg)

The model shows robust performance across different:
- Lighting conditions
- Camera angles
- Plating styles
- Background variations
- Multiple food items in single image

Evaluation metrics are available in the training logs and through the web interface.

## License
[Specify your license here]


