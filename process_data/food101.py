import glob
import os
from pathlib import Path
import pandas as pd
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image

from process_data.food_categories import FOOD_CATEGORY_MAPPING

# --- Step 1: Load model ---
num_classes = 101
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

# --- Step 2: Food-101 class names ---
class_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

# --- Step 3: Image transform ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Step 4: Folder to save new CSVs ---
output_folder = Path("../data/CGMacros/cgm")
output_folder.mkdir(exist_ok=True)

cgm_folder_path = "../data/CGMacros/cgm"
file_pattern = os.path.join(cgm_folder_path, "*.csv")
log_dfs = [(pd.read_csv(file), int(os.path.basename(file).split('-')[1].split('.')[0])) for file in glob.glob(file_pattern)]

def get_food_category(food_name: str) -> str:
    for category, foods in FOOD_CATEGORY_MAPPING.items():
        if food_name in foods:
            return category
    raise ValueError(food_name, "ABORT")

# --- Step 5: Loop over all dataframes ---
for df, df_id in log_dfs:
    # Make a copy to avoid modifying original directly
    df_updated = df.copy()

    # Keep only rows with images
    mask = df_updated["Image path"].notna()
    df_with_images = df_updated[mask].copy()

    predicted_foods = []
    prediction_scores = []

    for img_rel_path in df_with_images["Image path"]:
        # Construct full path
        full_path = Path(f"../../CGMacros/CGMacros-{df_id:03d}") / img_rel_path
        if not full_path.exists():
            predicted_foods.append("MISSING")
            prediction_scores.append(0.0)
            continue

        # Open and transform image
        image = Image.open(full_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            score, idx = torch.max(probs, 1)
            predicted_foods.append(class_names[idx.item()])
            prediction_scores.append(score.item())

    # Compute food types
    food_types = [
        get_food_category(food) if score > 0.6 else "Undefined"
        for food, score in zip(predicted_foods, prediction_scores)
    ]

    # Update only the rows with images in the original DataFrame
    df_updated.loc[mask, "Predicted Food"] = predicted_foods
    df_updated.loc[mask, "Prediction Score"] = prediction_scores
    df_updated.loc[mask, "Food Types"] = food_types

    # Save to CSV
    output_path = output_folder / f"CGMacros-{df_id:03d}.csv"
    df_updated.to_csv(output_path, index=False)
    print(f"Saved predictions for CGMacros-{df_id:03d} → {output_path}")

