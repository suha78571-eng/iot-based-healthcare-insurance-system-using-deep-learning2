# iot-based-healthcare-insurance-system-using-deep-learning2
An IoT-based healthcare insurance management system using deep learning (TabNet) for accurate, transparent risk and cost prediction.

<b>Importance of the Project</b>
This work is of importance to modern health insurance systems as a pressing requirement to ensure that good, transparent, and intelligent management decisions are made. However, traditional analytical approaches are inadequate for predicting insurance costs and risks, and the scale of the healthcare data explosion and the growth of insurance offerings are growing rapidly. By leveraging IoT (Internet of Things) and deep learning, the proposed system can continuously analyse demographic and behavioural characteristics that influence healthcare expenditures. The TabNet model is a relevant and meaningful class of models because it achieves high prediction accuracy and can be interpreted as an attention-based feature selection mechanism that complies with rules and trust insurance players. And the project also enables better premium pricing, risk stratification, and the evaluation of agent performance towards fairer and smarter insurance policies. In summary, this project presents an improved smart healthcare insurance management system that combines accuracy, transparency, and scalability within a single integrated framework.

<h1>Milestone 1 (M1): IoT Data Collection</h1>
UC ID	Actor	Description	Trigger	Main Flow
UC-M1-1	IoT Device / App	Collect healthcare and lifestyle data	Data generation event	An IoT device captures data and sends it to gateway
UC-M1-2	System	Receive IoT data	Incoming IoT request	Data is accepted and queued for ingestion
 Milestone 1 — IoT Data Collection (UC-M1-1)
@startuml
actor "IoT Device" as IoT
participant "IoT Gateway" as Gateway
database "Raw Data Store" as RawDB

IoT -> Gateway : Send health & lifestyle data
Gateway -> RawDB : Store raw data
RawDB -> Gateway : ACK
@enduml

Milestone 2 (M2): Data Validation & Ingestion
UC ID	Actor	Description	Trigger	Main Flow
UC-M2-1	System	Validate incoming data	Data received	Check format, ranges, and consistency
UC-M2-2	System	Ingest validated data	Validation success	Store validated data in raw database

Milestone 2 — Data Validation & Ingestion (UC-M2-1)
@startuml
actor System
database "Raw Data Store" as RawDB
participant "Validation Module" as Validate
database "Validated Data Store" as ValidDB

System -> RawDB : Load raw data
RawDB -> Validate : Provide data
Validate -> Validate : Validate format & ranges
Validate -> ValidDB : Store validated data
@enduml


Milestone 3 (M3): Data Preprocessing
UC ID	Actor	Description	Trigger	Main Flow
UC-M3-1	System	Handle missing values	Preprocessing request	Remove or impute missing values
UC-M3-2	System	Encode categorical features	Preprocessing stage	Apply label encoding
UC-M3-3	System	Normalize numerical features	Scaling stage	Apply StandardScaler
Milestone 3 — Data Preprocessing (UC-M3-1)
@startuml
actor System
database "Validated Data Store" as ValidDB
participant "Preprocessing Module" as Prep
database "Processed Data Store" as ProcDB

System -> ValidDB : Load validated data
ValidDB -> Prep : Provide dataset
Prep -> Prep : Clean missing values
Prep -> ProcDB : Save cleaned data
@enduml


Milestone 4 (M4): Feature Engineering
UC ID	Actor	Description	Trigger	Main Flow
UC-M4-1	System	Generate engineered features	Clean data available	Create model-ready features
UC-M4-2	System	Store feature set	Feature creation completed	Save features to feature store
Milestone 4 — Feature Engineering (UC-M4-1)
@startuml
actor System
database "Processed Data Store" as ProcDB
participant "Feature Engineering" as FE
database "Feature Store" as FeatureDB

System -> ProcDB : Load processed data
ProcDB -> FE : Provide features
FE -> FE : Encode & scale features
FE -> FeatureDB : Save engineered features
@enduml




Milestone 5 (M5): Dataset Partitioning
UC ID	Actor	Description	Trigger	Main Flow
UC-M5-1	System	Split dataset	Training request	Stratified train-test split
UC-M5-2	System	Verify class balance	Dataset split completed	Ensure balanced distribution
Milestone 5 — Dataset Partitioning (UC-M5-1)
@startuml
actor System
database "Feature Store" as FeatureDB
participant "Data Splitter" as Split
database "Train/Test Sets" as Sets

System -> FeatureDB : Load feature dataset
FeatureDB -> Split : Provide data
Split -> Sets : Create stratified train/test sets
@enduml


Milestone 6 (M6): Model Initialization
UC ID	Actor	Description	Trigger	Main Flow
UC-M6-1	System	Configure TabNet hyperparameters	Model setup request	Load architecture and parameters
UC-M6-2	System	Initialize TabNet model	Configuration completed	Instantiate model
 Milestone 6 — Model Initialization (UC-M6-1)
@startuml
actor System
participant "Model Configurator" as Config
participant "TabNet Model" as TabNet

System -> Config : Load hyperparameters
Config -> TabNet : Initialize TabNet architecture
@enduml



Milestone 7 (M7): Model Training
UC ID	Actor	Description	Trigger	Main Flow
UC-M7-1	System	Load training dataset	Training initiated	Fetch training data
UC-M7-2	System	Train TabNet classifier	Training command	Train model with class weights
UC-M7-3	System	Save trained model	Training completed	Store model in registry
Milestone 7 — Model Training (UC-M7-2)
@startuml
actor System
database "Train Set" as Train
participant "Trainer" as Trainer
participant "TabNet Model" as TabNet
database "Model Registry" as Registry

System -> Train : Load training data
Train -> Trainer : Features & labels
Trainer -> TabNet : Train with class weights
TabNet -> Registry : Save trained model
@enduml



Milestone 8 (M8): Model Evaluation
UC ID	Actor	Description	Trigger	Main Flow
UC-M8-1	System	Generate predictions	Test phase	Predict on test data
UC-M8-2	System	Compute performance metrics	Predictions available	Calculate accuracy, precision, recall, etc.
UC-M8-3	System	Visualize results	Evaluation completed	Generate ROC and confusion matrix
 Milestone 8 — Model Evaluation (UC-M8-1)
@startuml
actor System
database "Test Set" as Test
database "Model Registry" as Registry
participant "Evaluation Module" as Eval
database "Results Store" as Results

System -> Registry : Load trained model
System -> Test : Load test data
Eval -> Eval : Compute metrics & plots
Eval -> Results : Save evaluation results
@enduml

Milestone 9 (M9): Model Interpretability
UC ID	Actor	Description	Trigger	Main Flow
UC-M9-1	Analyst	Extract feature importance	Model trained	Retrieve attention-based importance
UC-M9-2	Analyst	Interpret key predictors	Importance available	Identify influential features
Milestone 9 — Model Interpretability (UC-M9-1)
@startuml
actor Analyst
database "Model Registry" as Registry
participant "Interpretability Module" as Interp
database "Insights Store" as Insights

Analyst -> Registry : Request model
Registry -> Interp : Provide TabNet model
Interp -> Interp : Extract feature importance
Interp -> Insights : Store interpretability results
@enduml


Milestone 10 (M10): Insurance Cost Prediction
UC ID	Actor	Description	Trigger	Main Flow
UC-M10-1	System	Predict insurance cost class	New client data	Classify as high/low cost
UC-M10-2	System	Store prediction results	Prediction completed	Save outputs for analysis
Milestone 10 — Insurance Cost Prediction (UC-M10-1)
@startuml
actor System
database "Client Data" as ClientDB
database "Model Registry" as Registry
participant "Prediction Engine" as Predict

System -> ClientDB : Load client features
System -> Registry : Load trained model
Predict -> Predict : Predict High/Low cost
@enduml

Milestone 11 (M11): Decision Support & Agent Segmentation
UC ID	Actor	Description	Trigger	Main Flow
UC-M11-1	Insurance Analyst	Review cost predictions	Prediction results ready	Inspect risk categories
UC-M11-2	Insurance Company	Support premium pricing	High-risk identified	Adjust pricing strategies
UC-M11-3	Insurance Company	Evaluate agent performance	Segmentation request	Rank agents using model insights

Segmentation (UC-M11-1)
@startuml
actor "Insurance Analyst" as Analyst
participant "Decision Support System" as DSS
database "Prediction Results" as Results

Analyst -> DSS : Request insights
DSS -> Results : Retrieve predictions
DSS -> Analyst : Show risk & agent segmentation
@enduml





Milestone 12 (M12): Deployment, Monitoring & Update
UC ID	Actor	Description	Trigger	Main Flow
UC-M12-1	Developer	Deploy system	Deployment request	Launch system in production
UC-M12-2	System	Monitor system performance	Runtime monitoring	Track logs and metrics
UC-M12-3	System	Retrain model with new data	New data arrival	Update and redeploy model



*************************************************

The Code of the Project:
 # Bulletproof TabNet with plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from pytorch_tabnet.tab_model import TabNetClassifier
from google.colab import drive
import os
import torch

# --------------------------
1. Mount Google Drive
# --------------------------
try:
    drive.mount('/content/drive')
except:
    print("Google Drive already mounted.")

# --------------------------
2. Load Dataset
# --------------------------
DATA_PATH = "/content/drive/MyDrive/insurance.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f" Loaded  {df.shape}")


# --------------------------
3. Create Binary Target
# --------------------------
median_charge = df['charges'].median()
df['high_cost'] = (df['charges'] > median_charge).astype(int)
print(f"\n Target distribution:")
print(f"Low cost: {np.sum(df['high_cost'] == 0)}")
print(f"High cost: {np.sum(df['high_cost'] == 1)}")

# --------------------------
4. Prepare Features
# --------------------------
feature_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']

X = df[feature_columns].copy()
y = df['high_cost'].values

# Encode categoricals
categorical_dims = []
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    categorical_dims.append(len(le.classes_))

# --------------------------
5. Split and Scale
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.astype(np.float32)
X_test_scaled = X_test.astype(np.float32)

num_indices = [i for i, f in enumerate(feature_columns) if f in numerical_features]
X_train_scaled[:, num_indices] = scaler.fit_transform(X_train[:, num_indices])
X_test_scaled[:, num_indices] = scaler.transform(X_test[:, num_indices])

# --------------------------
6. Train TabNet
# --------------------------
print("\n Training TabNet Classifier...")

# Class weights
_, counts = np.unique(y_train, return_counts=True)
class_weights = len(y_train) / (len(np.unique(y_train)) * counts)
weight_map = {i: float(w) for i, w in enumerate(class_weights)}

clf = TabNetClassifier(
    n_d=32, n_a=32, n_steps=3,
    gamma=1.3, n_independent=2, n_shared=2,
    momentum=0.02, lambda_sparse=1e-3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr": 2e-2},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"step_size": 10, "gamma": 0.9},
    mask_type="sparsemax",
    device_name="auto",
    seed=42
)

clf.fit(
    X_train=X_train_scaled,
    y_train=y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    eval_name=['train', 'test'],
    eval_metric=['logloss', 'accuracy'],
    max_epochs=100,
    patience=20,
    batch_size=256,
    virtual_batch_size=128,
    weights=weight_map,
    drop_last=False
)

# --------------------------
7. BULLETPROOF HISTORY EXTRACTION
# --------------------------
def extract_history_safe(clf):
    """Works with ANY TabNet version"""
    try:
        # Try to get history length from loss
        if hasattr(clf, 'history') and hasattr(clf.history, '__getitem__'):
            try:
                n_epochs = len(clf.history['loss'])
            except:
                n_epochs = 50
        else:
            n_epochs = 50

        # Create dummy history (we'll use evaluation metrics instead)
        return ([0.8]*n_epochs, [0.8]*n_epochs, [0.5]*n_epochs, [0.5]*n_epochs)
    except:
        return ([0.8]*50, [0.8]*50, [0.5]*50, [0.5]*50)

train_acc, test_acc, train_loss, test_loss = extract_history_safe(clf)

# --------------------------
8. Evaluate
# --------------------------
y_pred = clf.predict(X_test_scaled)
y_pred_prob = clf.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
roc_auc = auc(*roc_curve(y_test, y_pred_prob)[:2])

print("\n" + "="*50)
print("    TABNET CLASSIFIER RESULTS")
print("="*50)
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-Score:    {f1:.4f}")
print(f"ROC-AUC:     {roc_auc:.4f}")
print("="*50)

# --------------------------
9. PLOTS 
# --------------------------
plt.rcParams.update({'font.size': 12})

# ROC Curve 
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/tabnet_roc.png", dpi=150, bbox_inches='tight')
plt.show()

# Confusion Matrix 
plt.figure(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True, fmt='d', cmap='Blues',
    xticklabels=['Low Cost', 'High Cost'],
    yticklabels=['Low Cost', 'High Cost']
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/tabnet_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# Feature Importance (ALWAYS works)
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(9, 6))
plt.title("TabNet Feature Importance")
plt.bar(range(len(feature_importance)), feature_importance[sorted_idx], align="center")
plt.xticks(range(len(feature_importance)), [feature_columns[i] for i in sorted_idx], rotation=45, ha='right')
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/tabnet_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nROC curve, confusion matrix, and feature importance saved to Google Drive!")
print(" Note: Training accuracy/loss curves unavailable due to TabNet version differences")
print(f" Your model achieved {accuracy:.4f} accuracy with {roc_auc:.4f} AUC!")








