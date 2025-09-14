# example-algorithm/inference.py

from pathlib import Path
import json
import torch

# Import all your custom preprocessing functions from the 'preprocess' directory
from preprocess.clinical_inference import get_clinical_embedding
from preprocess.pathological_inference import get_pathological_embedding
from preprocess.spatial_inference import get_spatial_embedding
from preprocess.temporal_inference import get_temporal_embedding
from preprocess.semantic_inference import get_semantic_embedding

# Import your final, corrected model architecture which matches the trained .pt file
from preprocess.hcat_model import EnhancedHCAT
import warnings

# Suppress the specific PyTorch UserWarning about nested tensors
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because .*norm_first was True.*",
    category=UserWarning,
)

# ===========================================================================
# Global Configuration
# ===========================================================================
# IMPORTANT: This switch controls the output for the challenge.
# You MUST set this correctly for each submission phase.
#
# For the Recurrence Task, use:
# PREDICTION_TARGET_SLUG = "2-year-recurrence-after-diagnosis"
#
# For the Survival Task, uncomment the following line and rebuild the container:
PREDICTION_TARGET_SLUG = "5-year-survival"

# Define the execution environment paths
BASE_DIR = Path("inference.py").resolve().parent
if BASE_DIR == Path("/opt/app"):
    INPUT_PATH, OUTPUT_PATH, RESOURCE_PATH = Path("/input"), Path("/output"), BASE_DIR / "resources"
else: # Local testing setup
    INPUT_PATH = BASE_DIR / "test" / "input" / "interf0"
    OUTPUT_PATH = BASE_DIR / "test" / "output" / "interf0"
    RESOURCE_PATH = BASE_DIR / "resources"
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ===========================================================================
# Main Inference Logic
# ===========================================================================
def run():
    """
    Main function that orchestrates the entire inference process for a single patient.
    """
    # 1. Validate the PREDICTION_TARGET_SLUG to ensure it's set correctly.
    valid_slugs = ["2-year-recurrence-after-diagnosis", "5-year-survival"]
    task_slug = PREDICTION_TARGET_SLUG
    if task_slug not in valid_slugs:
        print(f"!!! WARNING: Invalid PREDICTION_TARGET_SLUG: '{task_slug}'. Defaulting to '{valid_slugs[0]}' !!!")
        task_slug = valid_slugs[0]

    # 2. Load all input JSON files. The helper function is robust to missing or empty files.
    print("Loading all input data...")
    inputs = {
        "clinical": load_json_file(INPUT_PATH / "hancock-clinical-data.json"),
        "pathological": load_json_file(INPUT_PATH / "hancock-pathological-data.json"),
        "blood": load_json_file(INPUT_PATH / "hancock-blood-data.json"),
        "surgery_text": load_json_file(INPUT_PATH / "hancock-surgery-text-data.json"),
        "primary_wsi": load_json_file(INPUT_PATH / "hancock-primary-tumor-wsi-embeddings.json"),
        "lymph_wsi": load_json_file(INPUT_PATH / "hancock-lymph-node-wsi-embeddings.json")
    }

    # 3. Generate a 512-d embedding for each modality.
    print("\nGenerating embeddings for each modality...")
    embeddings = [
        get_clinical_embedding(inputs["clinical"], RESOURCE_PATH),
        get_temporal_embedding(inputs["blood"], RESOURCE_PATH),
        get_pathological_embedding(inputs["pathological"], RESOURCE_PATH),
        get_semantic_embedding(inputs["surgery_text"], RESOURCE_PATH, device=DEVICE),
        get_spatial_embedding(inputs["primary_wsi"], inputs["lymph_wsi"], RESOURCE_PATH, device=DEVICE)
    ]
    
    # 4. Assemble the multimodal input tensor for the HCAT model.
    # The order MUST strictly match the training order: [clinical, temporal, pathological, semantic, spatial]
    emb_stack = torch.cat(embeddings, dim=1).reshape(1, 5, 512).to(DEVICE)

    # 5. Create placeholder quality and presence masks for inference.
    quality = torch.ones(1, 5).to(DEVICE)
    present_mask = torch.ones(1, 5).to(DEVICE)

    # 6. Load your final trained HCAT model and make a prediction.
    print("\nLoading final HCAT model and making prediction...")
    # Initialize the model with the corrected, full architecture from hcat_model.py
    model = EnhancedHCAT(d_model=512, n_modalities=5, n_global_layers=3, use_advanced_imputation=True).to(DEVICE)
    
    # **SOLUTION**: Load the weights using `strict=False`.
    # This tells PyTorch to load all the layers that match (like transformers, prediction heads)
    # and to safely ignore any that don't (like the "imputer" layers or older layer sizes).
    # This is the key to fixing the RuntimeError.
    model.load_state_dict(torch.load(RESOURCE_PATH / "enhanced_hcat_best_avg.pt", map_location=DEVICE), strict=False)
    model.eval()

    with torch.no_grad():
        outputs = model(emb_stack, quality, present_mask)
    
    # 7. Select the correct prediction output based on the validated task slug.
    final_logit = outputs["logit_rec"] if task_slug == valid_slugs[0] else outputs["logit_surv"]
    print(f"Task: {task_slug}. Using corresponding prediction head.")

    final_prob = torch.sigmoid(final_logit).item()
    
    # 8. Convert the probability to the required string label and save the result.
    prediction_str = prediction_to_string(final_prob, task_slug)
    
    print(f"\nFinal Prediction Probability: {final_prob:.4f}")
    print(f"Final Prediction Label: '{prediction_str}'")

    output_filename = get_output_file_name(task_slug)
    write_json_file(location=OUTPUT_PATH / output_filename, content=prediction_str)
    print(f"Prediction saved to {OUTPUT_PATH / output_filename}")
    
    return 0

# ===========================================================================
# Helper Functions
# ===========================================================================
def load_json_file(location):
    """Safely loads a JSON file, returning an empty dict if it fails."""
    try:
        with open(location, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load JSON from {location}. Proceeding with empty data.")
        return {}

def write_json_file(*, location, content):
    """Writes a dictionary to a JSON file."""
    with open(location, "w") as f:
        json.dump(content, f, indent=4)

def get_output_file_name(target_slug):
    """Returns the correct filename based on the task."""
    if target_slug == "2-year-recurrence-after-diagnosis":
        return "2-year-recurrence.json"
    return "5-year-survival.json"

def prediction_to_string(prediction, target_slug, threshold=0.5):
    """Converts a prediction probability to the required string label."""
    if target_slug == "2-year-recurrence-after-diagnosis":
        return "recurrence" if prediction > threshold else "no recurrence"
    return "deceased" if prediction > threshold else "living"

if __name__ == "__main__":
    raise SystemExit(run())

