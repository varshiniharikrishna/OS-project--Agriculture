# PlantVillage Dataset & Model Training Instructions

This directory is designated for uploading the **PlantVillage** dataset for training and evaluating crop disease classification models.

## Dataset Structure

Place your dataset images in `ml/dataset/PlantVillage/` organized by class folders:

```
ml/dataset/PlantVillage/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
├── Blueberry___healthy/
├── Cherry_(including_sour)___Powdery_mildew/
├── Cherry_(including_sour)___healthy/
├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
├── Corn_(maize)___Common_rust_/
├── Corn_(maize)___Northern_Leaf_Blight/
├── Corn_(maize)___healthy/
├── Cotton___Bacterial_blight/
├── Cotton___Leaf_curl_virus/
├── Cotton___healthy/
├── Grape___Black_rot/
├── Grape___Esca_(Black_Measles)/
├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
├── Grape___healthy/
├── Peach___Bacterial_spot/
├── Peach___healthy/
├── Pepper,_bell___Bacterial_spot/
├── Pepper,_bell___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Raspberry___healthy/
├── Rice___Bacterial_leaf_blight/
├── Rice___Brown_spot/
├── Rice___Leaf_smut/
├── Rice___healthy/
├── Soybean___healthy/
├── Squash___Powdery_mildew/
├── Strawberry___Leaf_scorch/
├── Strawberry___healthy/
├── Tomato___Bacterial_spot/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___Leaf_Mold/
├── Tomato___Septoria_leaf_spot/
├── Tomato___Spider_mites Two-spotted_spider_mite/
├── Tomato___Target_Spot/
├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
├── Tomato___Tomato_mosaic_virus/
└── Tomato___healthy/
```

## How to Train

Run the training script from the project root:

```bash
python3 ml/train.py --data_dir ml/dataset/PlantVillage --epochs 10 --batch_size 32 --model efficientnet_b0
```

The trained weights will be saved automatically to `ml/models/efficientnet_b0_plantvillage.pth`.
