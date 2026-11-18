# Coral Bleaching Detector

**21CEO310T – Global Warming and Climate Change | LLT / CLA-2 machine-learning project**

A deep-learning model that classifies a coral photo as **healthy** or **bleached** and highlights which parts of the photo drove the decision (Grad-CAM). A web app lets anyone upload or take a photo and see the result live.

**Syllabus link:** Unit III (climate impacts on fisheries and tourism, ocean warming evidence), Unit I (greenhouse effect, ocean heat uptake) and Unit V (monitoring for adaptation and public awareness).

## Live demo

**Try the deployed app:** [coral-bleaching-detector on Streamlit Cloud](https://coral-bleaching-detector-g2bqqqk2ifzmemxfjnkig2.streamlit.app/)

No installation needed: open the link, upload or take a coral photo, and see the result and heatmap. (If the app has been idle, it may take a few seconds to wake up.)

## Project files

| File | What it is |
|---|---|
| `coral_bleaching_detector.ipynb` | Google Colab notebook: downloads the data, trains the model, evaluates it and makes every figure for your report |
| `app.py` | Streamlit web app for the live demonstration |
| `requirements.txt` | Python packages the app needs |

## Step 1: Train the model in Google Colab (about 10–15 minutes)

1. Go to [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → choose `coral_bleaching_detector.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.**
4. When it finishes, `coral_outputs.zip` downloads automatically. It contains:
   - `coral_model.keras`: the trained model
   - `model_info.json`: every number for your report (accuracy, precision, recall, ROC-AUC, baseline comparison)
   - `figures/`: seven ready-to-use figures (see below)
   - `samples/`: six test photos for the demo

The dataset is public, so it normally downloads without a Kaggle login.

**If Kaggle asks you to log in:** create a free Kaggle account → *Settings → API → Generate New Token* and copy the token. In Colab, click the key icon (**Secrets**) in the left sidebar, add a secret named `KAGGLE_API_TOKEN` with the token as its value, switch on *Notebook access*, then run again.

**Or download the data yourself:** download the zip from the [dataset page](https://www.kaggle.com/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification), upload it to Colab, unzip it (`!unzip archive.zip -d /content/corals`) and set `LOCAL_DATA_DIR = "/content/corals"` in the first code cell.

## Step 2: Run the demo app on your laptop

You need Python 3.10–3.12.

1. Unzip `coral_outputs.zip` and place `coral_model.keras`, `model_info.json` and the `samples` folder **in the same folder as `app.py`**.
2. Open a terminal in that folder and run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. The app opens in your browser. Choose **Upload a photo**, **Take a photo** (webcam) or **Try a sample photo**. The result card shows the verdict and a bar with the model's decision line; under the heatmap you can adjust its strength and download it for your report.

**If the model fails to load**, your laptop's TensorFlow version probably differs from Colab's. Install the same one: open `model_info.json`, find `tensorflow_version`, then run `pip install tensorflow==<that version>`.

**Backup demo with no installation:** in the notebook, set `RUN_GRADIO_DEMO = True` in the settings cell and run the last cell. Colab prints a public link that works on any phone or laptop.

**Already trained a model with the previous version of the notebook?** The new app still works with it (it falls back to a 0.5 threshold without mirror-image averaging). Retrain with the new notebook to get the model improvements below.

## What's new in this version

**Model (notebook)**
- **Decision threshold tuned on the validation set** (best balanced accuracy, kept between 0.30 and 0.70) instead of a fixed 0.5. It is saved in `model_info.json` and used by the app, Gradio demo and Grad-CAM figures, so everything agrees. The test accuracy at a plain 0.5 is also saved (`accuracy_at_0.5_threshold`) for comparison.
- **Test-time augmentation:** each photo is scored as it is and mirrored, and the two probabilities are averaged.
- **Training tweaks:** light label smoothing (0.05), a learning-rate cut when validation loss stalls (ReduceLROnPlateau), a small random shift added to augmentation, early-stopping patience 5, and up to 15 fine-tuning epochs (early stopping usually ends sooner).
- **Fixes:** photos are now rotated upright from their EXIF tag during cleaning, as the app already did (training and the app previously saw sideways phone photos differently); Grad-CAM figures now use exactly the same prediction as the evaluation; the colour-only baseline now standardises its features for a fairer comparison; the ROC figure marks the chosen threshold.

**App**
- Cleaner layout: header banner, colour-coded result card, probability bar showing the decision line, photo tips before an image is chosen, readable sample names, and a sidebar with the scorecard on top and the background reading in collapsible sections.
- Heatmap-strength slider and a **Download heatmap** button.
- Faster: each photo is analysed once and cached, so moving the slider doesn't rerun the model.
- **Fixes:** a corrupt or unsupported file now shows a friendly message instead of crashing; a model that fails to load (usually a TensorFlow version mismatch) shows the exact `pip install` command to fix it; a missing or partial `model_info.json` no longer crashes the sidebar; Grad-CAM explains the class predicted at the saved threshold, not always 0.5; "unsure" is now measured from the decision line.

## Using the outputs in your report

| Figure | Use it for |
|---|---|
| `01_class_balance.png`, `02_sample_images.jpg` | Dataset description |
| `03_training_curves.png` | How accuracy and loss changed during training (training and validation curves close together = little overfitting) |
| `04_confusion_matrix_and_roc.png` | Main results: your "observations and measurements" (the red dot on the ROC curve is the chosen threshold) |
| `05_baseline_vs_cnn.png` | Comparing deep learning with a simple "count the white pixels" rule |
| `06_gradcam_examples.jpg` | Whether the model looks at the coral or at the background (water, divers) |
| `07_misclassified.jpg` | Limitations section |

Take the numbers from `model_info.json`. They come from photos the model never saw during training, so they are genuine results. Your numbers will differ slightly from your classmates' if they change any settings.

**How the project meets the brief:**
- **Working feature:** the live app classifies new photos on the spot.
- **Real measurements:** test-set accuracy, precision, recall, ROC-AUC and the confusion matrix.
- **Scientific principle:** heat stress breaks the coral–algae symbiosis and the coral turns white. The CNN learns the visual signs of this colour and texture loss.
- **Applications:** divers and citizen scientists could upload reef photos to map bleaching quickly. Reef managers could prioritise surveys and protection.
- **Limitations:** a small dataset of internet photos; underwater lighting and the blue colour cast vary; the model can't distinguish bleached from dead coral or naturally pale species; it gives a label even when the photo contains no coral.
- **Scaling up:** train on thousands of expert-labelled survey images (for example from CoralNet), add "dead" and "not coral" classes, run on drone or underwater-robot footage, and combine with satellite sea-surface-temperature heat-stress alerts such as NOAA Coral Reef Watch.

## Dataset credit

*Healthy and Bleached Corals Image Classification*, Kaggle (vencerlanz09): 923 photos collected from Flickr (438 healthy, 485 bleached).
