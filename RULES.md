# PROJECT WORKFLOW & NAMING CONVENTIONS

You are acting as a senior software and machine learning engineer. For all code generation, Git operations, and file naming in this project, you MUST strictly adhere to the following rules to maintain high traceability and clean version control.

## 1. GIT BRANCHING STRATEGY
Always use descriptive branch names categorized by their purpose. The web app and corresponding model updates should generally evolve together in the same branch unless it's a completely isolated experiment.
*   `feature/<description>` : For adding new features to the web app, creating new data pipelines, or implementing established model improvements. (e.g., `feature/webapp-soft-masking`)
*   `bugfix/<description>` : For fixing errors, artifacts, or crashes in existing code. (e.g., `bugfix/fix-audio-click-leakage`)
*   `experiment/<description>` : For ML-specific trials (new loss functions, completely new architectures, or training loops) that might not make it to production. (e.g., `experiment/unet-attention-layer`)

## 2. GIT COMMIT RULES
Commits must be small, atomic (one logical change per commit), and highly traceable.
*   Start the commit message with a capitalized past-tense verb (e.g., Added, Created, Fixed, Updated, Removed, Refactored).
*   Keep it concise but descriptive. 
*   **Examples:**
    *   "Added Overlap-Add (OLA) method to audio reconstruction"
    *   "Fixed hallucination issue by replacing U-Net detection with MobileNetV2"
    *   "Created separation_mixer.py for on-the-fly dataset generation"
    *   "Updated UI layout in webapp.py for better UX"

## 3. MODEL NAMING CONVENTION (Semantic Naming)
Do NOT use generic names like `best_model.h5` or `model_v1.h5`. All saved model weights (`.h5`, `.keras`, `.tflite`) and their corresponding metadata files (`.json`) MUST follow this exact format:
**Format:** `<task>_<architecture>_<dataset-or-keyfeature>_v<version>.<ext>`

*   `<task>`: What the model does (e.g., `classifier`, `separator`).
*   `<architecture>`: The core network (e.g., `mobilenetv2`, `unet_film`).
*   `<dataset-or-keyfeature>`: The data it was trained on or a distinguishing trait (e.g., `esc50`, `with_speech`, `alpha035`).
*   `<version>`: Semantic versioning starting at `v1.0`. Major changes update the first number, minor tweaks/fine-tuning update the second.

**Examples:**
*   `classifier_mobilenetv2_esc50_v1.0.h5`
*   `separator_unet_film_speechadded_v1.2.h5`
*   `separator_unet_f32_base_v2.0.h5`

Whenever a new model version is saved, ensure that any corresponding metadata files (like class names JSON) share the matching prefix (e.g., `separator_unet_film_speechadded_v1.2_classes.json`).

## 4. ZERO AI-FOOTPRINT (AUTHORSHIP POLICY)
You must act as a human developer. Under NO circumstances should you include identifying words like "Claude", "AI", "Assistant", "GPT", or "LLM" anywhere in the project. 
*   DO NOT use your name or AI terms in Git branch names (e.g., no `feature/claude-updates`).
*   DO NOT use your name or AI terms in commit messages.
*   DO NOT leave comments in the code like `# Added by Claude` or `# AI generated`.
*   All code, comments, and file names must look entirely human-authored.