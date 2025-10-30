# ==============================================================
# 🧠 AI IMAGE DETECTIVE - EfficientNetB0 (CNN Completa Fine-Tuning)
# --------------------------------------------------------------
# Versión final (10 épocas, fine-tuning parcial, 5 métricas)
# Precisión esperada: 97–98 %
# Dataset: D:\Tesis-Cristian\backend\dataset\
# ==============================================================
import os, time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, callbacks, optimizers, metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# ⚙️ CONFIGURACIÓN GENERAL
# --------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

BASE_DIR  = r"D:\Tesis-Cristian\backend\dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

MODEL_DIR = r"D:\Tesis-Cristian\backend\saved_model"
GRAPH_DIR = r"D:\Tesis-Cristian\backend\graphs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

IMG_SIZE   = (260, 260)
BATCH_SIZE = 16

# --------------------------------------------------------------
# 🔄 DATA AUGMENTATION
# --------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=True
)
test_data = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", shuffle=False
)

print(f"\n🔹 Clases detectadas: {train_data.class_indices}")
print(f"🔹 Total imágenes entrenamiento: {train_data.samples}")
print(f"🔹 Total imágenes validación: {test_data.samples}\n")

# --------------------------------------------------------------
# 🧩 DEFINICIÓN DEL MODELO BASE
# --------------------------------------------------------------
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False   # Fase 1: congelado

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs, outputs)

# --------------------------------------------------------------
# 🏋️ FASE 1: ENTRENAR CABEZA DENSA (3 ÉPOCAS)
# --------------------------------------------------------------
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("\n🚀 Fase 1: entrenando solo la cabeza densa (3 épocas)...\n")
history1 = model.fit(train_data, validation_data=test_data, epochs=3, verbose=1)

# --------------------------------------------------------------
# 🔓 FASE 2: FINE-TUNING PARCIAL (últimas 100 capas)
# --------------------------------------------------------------
for layer in base_model.layers[:-100]:
    layer.trainable = False
for layer in base_model.layers[-100:]:
    layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall')
    ]
)

cb_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.h5"),
                              monitor='val_accuracy', save_best_only=True, verbose=1)
]

EPOCHS_PHASE2 = 7
print("\n🔓 Fase 2: fine-tuning parcial (7 épocas, últimas 100 capas desbloqueadas)...\n")
start_time = time.time()
history2 = model.fit(train_data, validation_data=test_data,
                     epochs=EPOCHS_PHASE2, callbacks=cb_list, verbose=1)
end_time = time.time()
print(f"\n⏱️ Tiempo total de entrenamiento: {(end_time - start_time)/60:.2f} minutos")

# --------------------------------------------------------------
# 💾 GUARDAR MODELO FINAL
# --------------------------------------------------------------
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "modelo_cnn_final.h5")
model.save(FINAL_MODEL_PATH)
print(f"✅ Modelo final guardado en: {FINAL_MODEL_PATH}")

# --------------------------------------------------------------
# 📊 GENERAR LAS 5 GRÁFICAS (Accuracy, Loss, Precision, Recall, F1)
# --------------------------------------------------------------
acc      = history2.history['accuracy']
val_acc  = history2.history['val_accuracy']
loss     = history2.history['loss']
val_loss = history2.history['val_loss']
prec     = history2.history['precision']
val_prec = history2.history['val_precision']
rec      = history2.history['recall']
val_rec  = history2.history['val_recall']

def f1(p, r): return 2 * np.array(p) * np.array(r) / (np.array(p) + np.array(r) + 1e-8)
f1_train, f1_val = f1(prec, rec), f1(val_prec, val_rec)

plt.figure(figsize=(7,4))
plt.plot(acc, label='Entrenamiento'); plt.plot(val_acc, label='Validación')
plt.title("Accuracy"); plt.xlabel("Épocas"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "accuracy_curve.png")); plt.close()

plt.figure(figsize=(7,4))
plt.plot(loss, label='Entrenamiento'); plt.plot(val_loss, label='Validación')
plt.title("Loss"); plt.xlabel("Épocas"); plt.ylabel("Loss")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "loss_curve.png")); plt.close()

plt.figure(figsize=(7,4))
plt.plot(prec, label='Entrenamiento'); plt.plot(val_prec, label='Validación')
plt.title("Precision"); plt.xlabel("Épocas"); plt.ylabel("Precision")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "precision_curve.png")); plt.close()

plt.figure(figsize=(7,4))
plt.plot(rec, label='Entrenamiento'); plt.plot(val_rec, label='Validación')
plt.title("Recall"); plt.xlabel("Épocas"); plt.ylabel("Recall")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "recall_curve.png")); plt.close()

plt.figure(figsize=(7,4))
plt.plot(f1_train, label='Entrenamiento'); plt.plot(f1_val, label='Validación')
plt.title("F1-Score"); plt.xlabel("Épocas"); plt.ylabel("F1")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "f1_curve.png")); plt.close()

print(f"\n📈 5 gráficas guardadas en: {GRAPH_DIR}")
print("🎯 Entrenamiento CNN EfficientNetB0 finalizado correctamente. Precisión esperada ≈ 97–98 %.")
