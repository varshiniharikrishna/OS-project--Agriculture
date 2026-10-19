"""
Controlled Agricultural Knowledge Base for PlantVillage Disease Diagnosis & Action Plan.
Prevents LLM hallucination by supplying verified agricultural management guidance.
"""

KNOWLEDGE_BASE = {
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight (Alternaria solani)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Early blight is caused by the fungus Alternaria solani. It produces target-like dark concentric rings on mature leaves, causing yellowing and premature leaf loss.",
        "actions": [
            "Prune infected lower leaves to restrict fungal spore splash.",
            "Apply copper-based fungicide or Mancozeb at 7-10 day intervals.",
            "Avoid overhead irrigation; water directly near the plant root zone."
        ],
        "prevention": [
            "Rotate tomato crops with non-solanaceous crops every 2-3 years.",
            "Apply organic mulch to reduce soil splashing onto foliage.",
            "Ensure wide plant spacing (45-60cm) for optimal air circulation."
        ]
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight (Phytophthora infestans)",
        "is_healthy": False,
        "severity": "Critical",
        "explanation": "Late blight is a destructive water-mold disease causing rapid brown water-soaked lesions on leaves and stems, often accompanied by white fungal growth in high humidity.",
        "actions": [
            "Immediately isolate infected plants and destroy heavily diseased tissue.",
            "Spray systemic fungicide such as Metalaxyl or Dimethomorph combined with Mancozeb.",
            "Strictly suspend all overhead sprinkler watering."
        ],
        "prevention": [
            "Plant late-blight resistant tomato varieties (e.g., Mountain Magic, Defiant).",
            "Monitor field humidity closely during rainy spells.",
            "Ensure field drainage is adequate to prevent water stagnation."
        ]
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold (Passalora fulva)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Leaf mold thrives in high relative humidity (>85%). Pale green/yellow spots appear on the upper leaf surface with velvety olive-green spore masses underneath.",
        "actions": [
            "Increase ventilation and air movement around plants.",
            "Apply preventative sulfur or copper fungicides.",
            "Remove lower foliage showing severe velvet mold growth."
        ],
        "prevention": [
            "Maintain canopy humidity below 85% with proper pruning.",
            "Use drip irrigation instead of sprinkler systems.",
            "Space plants widely in rows."
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot (Septoria lycopersici)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Septoria leaf spot causes numerous small circular spots with dark borders and grey/white centers, leading to leaf drop from the bottom upward.",
        "actions": [
            "Remove infected lower leaves promptly.",
            "Apply Chlorothalonil or copper-based fungicide spray.",
            "Keep foliage dry during watering."
        ],
        "prevention": [
            "Clear crop residue immediately post-harvest.",
            "Rotate crops annually.",
            "Mulch around plant bases."
        ]
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot (Xanthomonas spp.)",
        "is_healthy": False,
        "severity": "High",
        "explanation": "Bacterial spot forms dark water-soaked leaf spots that turn brown and necrotic. It spreads rapidly through rain splash and farm tools.",
        "actions": [
            "Spray fixed copper mixed with Mancozeb to control bacterial spread.",
            "Sanitize tools and hands with alcohol solution between plants.",
            "Avoid handling plants while foliage is wet."
        ],
        "prevention": [
            "Use certified pathogen-free seeds.",
            "Implement a 2-year crop rotation.",
            "Avoid sprinkler irrigation."
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "disease": "Yellow Leaf Curl Virus (TYLCV)",
        "is_healthy": False,
        "severity": "Critical",
        "explanation": "TYLCV is transmitted by whiteflies (Bemisia tabaci). Infected plants exhibit severe leaf curling, yellowing margins, stunting, and reduced fruit yield.",
        "actions": [
            "Control whitefly vectors using yellow sticky traps and neem oil spray.",
            "Apply imidacloprid or insecticidal soap if whitefly density is high.",
            "Rogue out and bury severely stunted viral plants."
        ],
        "prevention": [
            "Use fine insect mesh nets over nursery beds.",
            "Plant TYLCV-tolerant hybrid varieties.",
            "Keep fields free from weed hosts."
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus (ToMV)",
        "is_healthy": False,
        "severity": "High",
        "explanation": "ToMV causes mottling, light and dark green mosaic patterns, and distorted fern-like foliage. It spreads easily via mechanical contact.",
        "actions": [
            "Disinfect tools, hands, and stakes with trisodium phosphate or milk solution.",
            "Remove infected plants immediately to prevent mechanical transmission.",
            "Prohibit tobacco use near plants (tobacco can harbor mosaic virus)."
        ],
        "prevention": [
            "Select ToMV-resistant seeds.",
            "Sterilize seed trays and farm implements.",
            "Maintain clean field borders."
        ]
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "explanation": "Foliage is green, vigorous, and free from dark lesions, yellowing, or viral distortion.",
        "actions": [
            "Maintain balanced N-P-K fertilization and consistent soil moisture.",
            "Conduct routine leaf scans every 5-7 days."
        ],
        "prevention": [
            "Keep soil healthy with organic compost.",
            "Ensure good drainage and air flow."
        ]
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight (Alternaria solani)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Early blight forms dark brown concentric rings on older potato leaves. It reduces tuber size if defoliation occurs before crop maturity.",
        "actions": [
            "Spray Mancozeb or Chlorothalonil fungicide.",
            "Maintain optimal plant nutrition (nitrogen & potassium levels)."
        ],
        "prevention": [
            "Use certified disease-free seed tubers.",
            "Destroy potato vine residue after harvest.",
            "Rotate fields with maize or wheat."
        ]
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight (Phytophthora infestans)",
        "is_healthy": False,
        "severity": "Critical",
        "explanation": "A destructive water-mold disease causing fast-spreading dark water-soaked patches. It can wipe out potato canopies and infect tubers in wet weather.",
        "actions": [
            "Spray systemic fungicides (Cymoxanil + Mancozeb or Metalaxyl) immediately.",
            "Hill up soil around potato stems to protect developing tubers from spores.",
            "Cut and remove infected haulms if harvest is near."
        ],
        "prevention": [
            "Plant certified late-blight resistant seed tubers.",
            "Avoid excessive nitrogen fertilization which creates dense humid canopies.",
            "Monitor weather alerts for high humidity and rainfall."
        ]
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "explanation": "Leaf tissue is healthy, dark green, and free from blights or bacterial wilt.",
        "actions": [
            "Maintain regular irrigation during tuber initiation.",
            "Continue periodic field monitoring."
        ],
        "prevention": [
            "Hill soil properly around tubers.",
            "Practice proper crop rotation."
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Maize",
        "disease": "Common Rust (Puccinia sorghi)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Common rust creates reddish-brown powdery pustules on upper and lower leaf surfaces. It thrives in cool, moist weather.",
        "actions": [
            "Apply foliar fungicide (Propiconazole or Azoxystrobin) if pustules cover >10% of leaf area before flowering.",
            "Ensure balanced fertilization."
        ],
        "prevention": [
            "Plant resistant maize hybrids.",
            "Sow early in the season to avoid peak rust spore counts."
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Maize",
        "disease": "Northern Corn Leaf Blight (Exserohilum turcicum)",
        "is_healthy": False,
        "severity": "High",
        "explanation": "Produces large, elliptical cigar-shaped grayish-green lesions on leaves, reducing photosynthetic area significantly during grain filling.",
        "actions": [
            "Apply recommended strobilurin or triazole fungicides at onset of symptoms.",
            "Incorporate crop residue post-harvest."
        ],
        "prevention": [
            "Use resistant maize cultivars.",
            "Rotate with non-host crops like legumes or cotton."
        ]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Maize",
        "disease": "Gray Leaf Spot (Cercospora zeae-maydis)",
        "is_healthy": False,
        "severity": "High",
        "explanation": "Forms rectangular tan/gray leaf spots bounded by leaf veins. Can cause extensive leaf death under warm humid conditions.",
        "actions": [
            "Spray foliar fungicides at silking if disease pressure is high.",
            "Promote field residue decomposition."
        ],
        "prevention": [
            "Select gray-leaf-spot resistant hybrids.",
            "Practice 2-year crop rotation."
        ]
    },
    "Corn_(maize)___healthy": {
        "crop": "Maize",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "explanation": "Maize canopy is green, healthy, and vigorous.",
        "actions": [
            "Maintain nitrogen side-dressing during vegetative stage.",
            "Monitor soil moisture."
        ],
        "prevention": [
            "Use good quality hybrid seeds.",
            "Keep fields weed-free."
        ]
    },
    "Cotton___Bacterial_blight": {
        "crop": "Cotton",
        "disease": "Bacterial Blight (Xanthomonas citri pv. malvacearum)",
        "is_healthy": False,
        "severity": "High",
        "explanation": "Bacterial blight causes angular, water-soaked dark leaf spots bounded by veins, and can lead to blackarm lesions on stems and boll rot.",
        "actions": [
            "Spray Copper Oxychloride + Streptocycline (100 ppm).",
            "Avoid field operations while plants are wet with dew."
        ],
        "prevention": [
            "Delint seed with acid before sowing.",
            "Use resistant cotton cultivars (e.g., Bt cotton hybrids).",
            "Destroy infected crop residue."
        ]
    },
    "Cotton___Leaf_curl_virus": {
        "crop": "Cotton",
        "disease": "Cotton Leaf Curl Virus (CLCuV)",
        "is_healthy": False,
        "severity": "Critical",
        "explanation": "Transmitted by whitefly vector (Bemisia tabaci). Causes upward curling of leaves, leaf enations (thickening of veins underneath), and stunting.",
        "actions": [
            "Spray systemic insecticides (Diafenthiuron or Thiamethoxam) to reduce whiteflies.",
            "Eradicate weed hosts like Abutilon and Solanum around fields.",
            "Uproot severely deformed viral plants."
        ],
        "prevention": [
            "Grow CLCuV-resistant cotton varieties.",
            "Maintain yellow sticky traps in fields.",
            "Avoid excessive nitrogen application."
        ]
    },
    "Cotton___healthy": {
        "crop": "Cotton",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "explanation": "Cotton leaves are green, broad, and free from vein thickening or angular lesions.",
        "actions": [
            "Provide timely irrigation during flowering and boll formation.",
            "Scout weekly for sucking pests."
        ],
        "prevention": [
            "Maintain balanced potash and nitrogen levels.",
            "Ensure proper field drainage."
        ]
    },
    "Rice___Bacterial_leaf_blight": {
        "crop": "Rice",
        "disease": "Bacterial Leaf Blight (Xanthomonas oryzae pv. oryzae)",
        "is_healthy": False,
        "severity": "Critical",
        "explanation": "Causes yellowing and drying of leaves starting from tips and margins, producing wavy, water-soaked streaks. Can cause 'kresek' wilt in seedlings.",
        "actions": [
            "Drain the paddy field for 3-4 days to reduce humidity.",
            "Spray Copper Hydroxide or Streptomycin sulfate.",
            "Avoid high nitrogen top-dressing during disease outbreak."
        ],
        "prevention": [
            "Plant resistant rice varieties (e.g., IR64, Swarna Sub1).",
            "Apply balanced potassium fertilization.",
            "Keep field bunds free from weed hosts."
        ]
    },
    "Rice___Brown_spot": {
        "crop": "Rice",
        "disease": "Brown Spot (Bipolaris oryzae)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Produces oval, reddish-brown spots with yellow halos across leaves. Common in nutrient-deficient or water-stressed paddy soils.",
        "actions": [
            "Apply potassium and zinc fertilizers to soil.",
            "Spray Mancozeb or Edifenphos at early tillering."
        ],
        "prevention": [
            "Correct soil nutrient deficiencies.",
            "Treat seed with Thiram or Carbendazim before sowing."
        ]
    },
    "Rice___Leaf_smut": {
        "crop": "Rice",
        "disease": "Leaf Smut (Entyloma oryzae)",
        "is_healthy": False,
        "severity": "Low",
        "explanation": "Forms tiny, black linear spots (sori) on leaf blades. Generally a minor disease unless infection is very severe late in the season.",
        "actions": [
            "Usually no fungicide needed unless infection spreads to upper leaves.",
            "Apply recommended foliar copper spray if severe."
        ],
        "prevention": [
            "Avoid over-fertilization with nitrogen.",
            "Clear crop stubble after harvest."
        ]
    },
    "Rice___healthy": {
        "crop": "Rice",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "explanation": "Paddy blades are bright green, upright, and free from bacterial streaks or brown spots.",
        "actions": [
            "Maintain optimal standing water depth (2-5cm).",
            "Apply scheduled split nitrogen application."
        ],
        "prevention": [
            "Use certified seed.",
            "Maintain proper plant density."
        ]
    },
    "Wheat___Leaf_rust": {
        "crop": "Wheat",
        "disease": "Leaf Rust (Puccinia triticina)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Causes small, round orange-brown pustules randomly scattered on the upper leaf surface of wheat.",
        "actions": [
            "Apply Tebuconazole or Propiconazole foliar fungicide spray.",
            "Monitor flag leaf infection closely."
        ],
        "prevention": [
            "Plant rust-resistant wheat varieties.",
            "Avoid late sowing."
        ]
    },
    "Wheat___Powdery_mildew": {
        "crop": "Wheat",
        "disease": "Powdery Mildew (Blumeria graminis)",
        "is_healthy": False,
        "severity": "Moderate",
        "explanation": "Forms white to gray powdery patches on leaf sheaths and blades under humid, dense canopy conditions.",
        "actions": [
            "Spray Triadimefon or Sulfur-based fungicide.",
            "Reduce canopy humidity if possible."
        ],
        "prevention": [
            "Maintain proper seed rate to prevent overcrowded stands.",
            "Select resistant cultivars."
        ]
    },
    "Wheat___healthy": {
        "crop": "Wheat",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "explanation": "Wheat foliage is green and healthy.",
        "actions": [
            "Irrigate at critical stages (Crown Root Initiation, Booting, Grain Filling).",
            "Routine monitoring."
        ],
        "prevention": [
            "Practice balanced soil fertilization.",
            "Use clean seeds."
        ]
    }
}

DEFAULT_DIAGNOSIS = {
    "crop": "General Crop",
    "disease": "Unspecified Leaf Spot / Healthy",
    "is_healthy": True,
    "severity": "Low",
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
    
    # Generic fallback based on label parsing
    parts = class_name.split("___")
    crop = parts[0].replace("_", " ") if len(parts) > 0 else "Crop"
    dis = parts[1].replace("_", " ") if len(parts) > 1 else "Condition"
    is_healthy = "healthy" in dis.lower()
    
    return {
        "crop": crop,
        "disease": dis,
        "is_healthy": is_healthy,
        "severity": "None" if is_healthy else "Moderate",
        "explanation": f"Detected {dis} on {crop} leaf.",
        "actions": [
            "Monitor affected plants regularly.",
            "Consult local agricultural extension officer if symptoms spread."
        ],
        "prevention": [
            "Maintain clean agricultural practices.",
            "Ensure proper field drainage and crop rotation."
        ]
    }
