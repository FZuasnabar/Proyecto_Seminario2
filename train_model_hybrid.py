# ==========================================================
# MODELO HÍBRIDO FINAL (10 ÉPOCAS)
# EfficientNetB0 + Mini-Transformer (49 tokens)
# Fine-tuning en 2 fases + EarlyStopping + 5 gráficas
# ==========================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1️⃣ Configuración
# ------------------------------
BASE_DIR  = r"D:\Tesis-Fernando\dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "test")

SAVE_DIR   = r"D:\Tesis-Fernando\resultados_final_10ep"
MODEL_DIR  = os.path.join(SAVE_DIR, "saved_model_tf")
MODEL_KERAS = os.path.join(SAVE_DIR, "modelo_hibrido_final_10ep.keras")
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 8
EPOCHS      = 10
SEED        = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------
# 2️⃣ Data Generators
# ------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=True, seed=SEED
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False, seed=SEED
)

# ------------------------------
# 3️⃣ Mini-Transformer (49 tokens)
# ------------------------------
def transformer_block(x, num_heads=2, d_model=128, ffn_mult=2):
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x_norm, x_norm)
    x = layers.Add()([x, attn])
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(d_model * ffn_mult, activation='gelu')(y)
    y = layers.Dense(d_model)(y)
    return layers.Add()([x, y])

class LearnedPositionEmbedding(layers.Layer):
    def __init__(self, num_tokens, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = self.add_weight(
            name="pos_emb", shape=(num_tokens, d_model),
            initializer="zeros", trainable=True
        )
    def call(self, x): return x + self.pos_emb

# ------------------------------
# 4️⃣ Definición del modelo
# ------------------------------
inputs = layers.Input(shape=(224, 224, 3))
x = layers.Lambda(preprocess_input)(inputs)

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x)
base_model.trainable = False

feat = base_model.output
h, w, c = feat.shape[1], feat.shape[2], feat.shape[3]
num_tokens = int(h) * int(w)

proj_dim = 128
tokens = layers.Reshape((num_tokens, int(c)))(feat)
tokens = layers.Dense(proj_dim)(tokens)
tokens = LearnedPositionEmbedding(num_tokens, proj_dim)(tokens)
tokens = transformer_block(tokens)
tokens = transformer_block(tokens)

x = layers.GlobalAveragePooling1D()(tokens)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)
optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# ------------------------------
# 5️⃣ Fine-tuning progresivo
# ------------------------------
class TwoPhaseFineTuning(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 3:
            print("\n=== FASE 1: Descongelando últimas 60 capas ===")
            for layer in base_model.layers[-60:]:
                layer.trainable = True
            optimizer.learning_rate.assign(1e-5)
            print(f"Capas entrenables: {np.sum([l.trainable for l in base_model.layers])}")
        if epoch == 7:
            print("\n=== FASE 2: Descongelando TODO EfficientNetB0 ===")
            for layer in base_model.layers:
                layer.trainable = True
            optimizer.learning_rate.assign(5e-6)
            print(f"Capas entrenables: {np.sum([l.trainable for l in base_model.layers])}")

# ------------------------------
# 6️⃣ Callbacks
# ------------------------------
cb_list = [
    TwoPhaseFineTuning(),
    callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, "best_model.keras"),
                              monitor='val_loss', save_best_only=True, verbose=1)
]

# ------------------------------
# 7️⃣ Entrenamiento
# ------------------------------
print("\n=== ENTRENAMIENTO FINAL (10 ÉPOCAS) ===")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=cb_list,
    verbose=1
)

# ------------------------------
# 8️⃣ Evaluación y métricas
# ------------------------------
val_gen.reset()
pred_probs = model.predict(val_gen)
pred_labels = (pred_probs > 0.5).astype(int).ravel()
true_labels = val_gen.classes[:len(pred_probs)]

cm = confusion_matrix(true_labels, pred_labels, labels=[0,1])

# Curvas ROC y PR
fpr, tpr, _ = roc_curve(true_labels, pred_probs)
roc_auc = auc(fpr, tpr)
p_curve, r_curve, _ = precision_recall_curve(true_labels, pred_probs)
pr_auc = auc(r_curve, p_curve)

# ------------------------------
# 9️⃣ Gráficas (5)
# ------------------------------
def save_plot(name):
    plt.savefig(os.path.join(SAVE_DIR, name), dpi=300, bbox_inches='tight')
    plt.close()

# Accuracy
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy (Entrenamiento 10 Épocas)")
plt.xlabel("Épocas"); plt.ylabel("Precisión")
plt.legend(); plt.grid(True)
save_plot("accuracy.png")

# Loss
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss (Entrenamiento 10 Épocas)")
plt.xlabel("Épocas"); plt.ylabel("Pérdida")
plt.legend(); plt.grid(True)
save_plot("loss.png")

# Confusión
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión (Final)")
plt.xlabel("Predicción"); plt.ylabel("Etiqueta Real")
save_plot("confusion_matrix.png")

# ROC
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.title("Curva ROC")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.legend(loc="lower right"); plt.grid(True)
save_plot("roc_curve.png")

# Precisión–Recall
plt.figure(figsize=(5,4))
plt.plot(r_curve, p_curve, lw=2)
plt.title("Curva Precisión–Recall")
plt.xlabel("Recall"); plt.ylabel("Precisión")
plt.grid(True)
save_plot("precision_recall.png")

# ------------------------------
# 🔟 Guardado del modelo
# ------------------------------
try:
    model.save(MODEL_DIR)
    model.save(MODEL_KERAS)
    print(f"\n✅ Modelo guardado correctamente en:\n- {MODEL_DIR}\n- {MODEL_KERAS}")
except Exception as e:
    print(f"\n⚠️ Error al guardar modelo: {e}")
    backup_path = os.path.join(SAVE_DIR, "modelo_backup.weights.h5")
    model.save_weights(backup_path)
    print(f"Pesos guardados como respaldo en: {backup_path}")

print(f"\n📊 Resultados y gráficas guardados en: {SAVE_DIR}")
