"""
Controlled Agricultural Knowledge Base for PlantVillage Disease Diagnosis & Action Plan.
Prevents LLM hallucination by supplying verified agricultural management guidance.
"""

KNOWLEDGE_BASE = {
    "Pepper__bell___Bacterial_spot": {
        "crop": "Pepper (Bell)",
        "disease": "Bacterial Spot (Xanthomonas spp.)",
        "is_healthy": False,
        "severity": "High",
        "explanation": "Bacterial spot causes small, dark, water-soaked spots on bell pepper leaves and fruit, leading to leaf drop and fruit lesions.",
        "actions": [
            "Apply copper-based bactericides combined with Mancozeb at 7-10 day intervals.",
            "Remove and destroy severely infected plant leaves.",
            "Avoid overhead irrigation to keep foliage dry."
        ],
        "prevention": [
            "Use certified disease-free seeds and transplants.",
            "Rotate bell pepper crops with non-solanaceous crops for 2-3 years.",
            "Maintain proper plant spacing for air circulation."
        ]
    },
    "Pepper__bell___healthy": {
        "crop": "Pepper (Bell)",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "explanation": "Bell pepper foliage is vibrant green, firm, and showing no signs of bacterial or fungal infection.",
        "actions": [
            "Continue regular watering and balanced N-P-K fertilization.",
            "Perform weekly routine field inspections."
        ],
        "prevention": [
            "Maintain drip irrigation directly to root zones.",
            "Apply organic mulch around bases."
        ]
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight (Alternaria solani)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Early blight produces brown concentric target-board spots on mature lower potato leaves, causing yellowing and defoliation.",
        "actions": [
            "Prune infected lower foliage to reduce spore splash.",
            "Apply Mancozeb or Chlorothalonil fungicide spray.",
            "Ensure foliage dries quickly after rain."
        ],
        "prevention": [
            "Rotate potato fields with corn or grain crops.",
            "Maintain optimal nitrogen fertility."
        ]
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight (Phytophthora infestans)",
        "is_healthy": False,
        "severity": "Critical",
        "explanation": "Late blight is a severe water-mold disease causing dark, water-soaked spots with white fungal mold underneath in humid weather.",
        "actions": [
            "Destroy infected potato vines immediately to protect tubers.",
            "Apply systemic Metalaxyl or Dimethomorph fungicide.",
            "Suspend overhead sprinkler irrigation."
        ],
        "prevention": [
            "Plant certified late-blight resistant potato seed tubers.",
            "Monitor regional late blight forecasting advisories."
        ]
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "explanation": "Potato foliage is healthy with uniform green leaf canopy.",
        "actions": [
            "Maintain balanced hill cultivation and soil moisture.",
            "Inspect weekly for early pest signs."
        ],
        "prevention": [
            "Ensure proper soil drainage.",
            "Avoid excessive nitrogen late in season."
        ]
    },
    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot (Xanthomonas spp.)",
        "is_healthy": False,
        "severity": "High",
        "explanation": "Bacterial spot produces small, dark brown necrotic spots surrounded by yellow halos on tomato foliage.",
        "actions": [
            "Apply fixed copper mixed with Mancozeb to restrict bacterial spread.",
            "Sanitize pruning shears between plants.",
            "Avoid touching foliage while plants are wet."
        ],
        "prevention": [
            "Use certified pathogen-free seeds.",
            "Rotate crops annually with non-solanaceous plants."
        ]
    },
    "Tomato_Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight (Alternaria solani)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Early blight produces dark target-like concentric rings on mature leaves, causing yellowing and premature leaf drop.",
        "actions": [
            "Prune infected lower leaves to restrict fungal spore splash.",
            "Apply copper-based fungicide or Mancozeb at 7-10 day intervals.",
            "Water directly near plant root zone."
        ],
        "prevention": [
            "Rotate tomato crops with non-solanaceous crops every 2-3 years.",
            "Apply organic mulch around bases."
        ]
    },
    "Tomato_Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight (Phytophthora infestans)",
        "is_healthy": False,
        "severity": "Critical",
        "explanation": "Late blight is a destructive water-mold disease causing rapid brown water-soaked lesions on leaves and stems.",
        "actions": [
            "Isolate infected plants and destroy heavily diseased tissue.",
            "Spray systemic fungicide such as Metalaxyl or Dimethomorph.",
            "Suspend overhead sprinkler watering."
        ],
        "prevention": [
            "Plant resistant tomato varieties.",
            "Ensure field drainage is adequate."
        ]
    },
    "Tomato_Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold (Passalora fulva)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Leaf mold produces pale green/yellow spots on upper leaf surfaces with olive-green velvety mold underneath.",
        "actions": [
            "Increase canopy ventilation and air movement.",
            "Apply sulfur or copper fungicides.",
            "Remove lower foliage showing severe mold."
        ],
        "prevention": [
            "Maintain humidity below 85% with proper pruning.",
            "Use drip irrigation."
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot (Septoria lycopersici)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Septoria leaf spot causes numerous small circular spots with grey centers and dark borders.",
        "actions": [
            "Remove infected lower leaves promptly.",
            "Apply Chlorothalonil or copper spray.",
            "Keep foliage dry during irrigation."
        ],
        "prevention": [
            "Clear crop residue post-harvest.",
            "Rotate crops annually."
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Two-Spotted Spider Mites (Tetranychus urticae)",
        "is_healthy": False,
        "severity": "High",
        "explanation": "Spider mites cause yellow speckling/stippling on leaf surfaces accompanied by fine silken webbing under hot dry conditions.",
        "actions": [
            "Spray insecticidal soap, neem oil, or miticide (Abamectin).",
            "Increase ambient humidity around plants.",
            "Remove heavily infested leaves."
        ],
        "prevention": [
            "Avoid over-fertilizing with high nitrogen.",
            "Introduce predatory mites (Phytoseiulus persimilis)."
        ]
    },
    "Tomato__Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot (Corynespora cassiicola)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Target spot produces circular brown lesions with light brown centers and dark yellow halos.",
        "actions": [
            "Apply Azoxystrobin or Chlorothalonil fungicide.",
            "Prune canopy for sunlight penetration."
        ],
        "prevention": [
            "Eliminate solanaceous weed hosts.",
            "Rotate tomato crops regularly."
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "is_healthy": False,
        "severity": "Critical",
        "explanation": "TYLCV is transmitted by whiteflies, causing severe upward leaf curling, yellowing margins, and stunted growth.",
        "actions": [
            "Control whitefly vectors using yellow sticky traps and neem oil.",
            "Apply insecticidal soap or Imidacloprid if whitefly density is high.",
            "Remove virus-infected plants to stop vector spreading."
        ],
        "prevention": [
            "Plant TYLCV-resistant tomato hybrids.",
            "Use fine insect mesh nets over nursery beds."
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus (ToMV)",
        "is_healthy": False,
        "severity": "High",
        "explanation": "Tomato mosaic virus causes light and dark green mottled leaf patterning, leaf blistering, and stunting.",
        "actions": [
            "Isolate and discard infected plants immediately.",
            "Disinfect farm tools and hands with milk/trisodium phosphate solution.",
            "Do not smoke or use tobacco products near plants."
        ],
        "prevention": [
            "Plant mosaic-resistant tomato varieties.",
            "Sanitize seed beds before planting."
        ]
    },
    "Tomato_healthy": {
        "crop": "Tomato",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "explanation": "Tomato foliage is healthy, showing vibrant green leaf surface without lesions or viral mottle.",
        "actions": [
            "Continue regular drip irrigation and balanced fertilization.",
            "Monitor crop weekly."
        ],
        "prevention": [
            "Maintain clean field practices.",
            "Mulch around plant bases."
        ]
    }
}

DEFAULT_DIAGNOSIS = {
    "crop": "General Crop",
    "disease": "Unspecified Leaf Spot / Healthy",
    "is_healthy": True,
    "severity": "None",
    "explanation": "Leaf image processed successfully. No severe pathogen pattern detected.",
    "actions": [
        "Continue monitoring leaf condition every 5-7 days.",
        "Maintain clean field practices and proper watering."
    ],
    "prevention": [
        "Use certified seeds.",
        "Ensure adequate sunlight and aeration."
    ]
}


def get_diagnosis(class_name: str) -> dict:
    """Retrieve controlled diagnosis and action plan for a given class label."""
    if class_name in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[class_name]

    # Clean fallback based on label parsing
    clean_label = class_name.replace("___", "_").replace("__", "_")
    parts = clean_label.split("_")
    crop = parts[0] if len(parts) > 0 else "Crop"
    dis = " ".join(parts[1:]) if len(parts) > 1 else "Condition"
    is_healthy = "healthy" in dis.lower()

    return {
        "crop": crop.capitalize(),
        "disease": dis.title(),
        "is_healthy": is_healthy,
        "severity": "None" if is_healthy else "Moderate",
        "explanation": f"Detected {dis} on {crop} foliage.",
        "actions": [
            "Monitor affected plants regularly.",
            "Consult local agricultural extension officer if symptoms spread."
        ],
        "prevention": [
            "Maintain clean agricultural practices.",
            "Ensure proper field drainage and crop rotation."
        ]
    }
