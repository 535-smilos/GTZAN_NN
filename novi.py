import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import os

# ======================================================================
# UČITAVANJE PODATAKA
# ======================================================================

data = pd.read_csv("./features_30_sec.csv")

filenames = None
if 'filename' in data.columns:
    filenames = data['filename'].values
    data = data.drop(['filename'], axis=1)

if 'length' in data.columns:
    data = data.drop(['length'], axis=1)

X = data.drop(['label'], axis=1).values
y = data['label'].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train/test split
if filenames is not None:
    X_train_full, X_test, y_train_full, y_test, fn_train_full, fn_test = train_test_split(
        X, y, filenames, test_size=0.2, random_state=42)
else:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    fn_train_full = None
    fn_test = None

# Train_full → GA_train + VALIDATION
X_ga_train, X_val, y_ga_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=1)

# Scaling
scaler = StandardScaler()
X_ga_train = scaler.fit_transform(X_ga_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

n_inputs = X_ga_train.shape[1]
n_hidden = 8
n_outputs = len(np.unique(y))

# ======================================================================
# DEFINICIJA KERAS MREŽE
# ======================================================================

def createModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),
        tf.keras.layers.Dense(n_hidden, activation='relu'),
        tf.keras.layers.Dense(n_outputs, activation='softmax')
    ])
    return model

# Template za izvlačenje i vraćanje težina
model_template = createModel()
shapes = [w.shape for w in model_template.get_weights()]
totalWeights = int(np.sum([np.prod(s) for s in shapes]))

def flatten_weights(weights_list):
    return np.concatenate([w.flatten() for w in weights_list])

def unflatten_to_list(vec):
    res = []
    idx = 0
    for s in shapes:
        size = int(np.prod(s))
        res.append(vec[idx:idx+size].reshape(s))
        idx += size
    return res

def setModelWeights_from_flat(model, flat_vec):
    model.set_weights(unflatten_to_list(flat_vec))

base_flat = flatten_weights(model_template.get_weights())

# ======================================================================
# GENETSKI ALGORITAM — ISPRAVLJENO
# ======================================================================

pop_size = 60
generacija = 60
mutation_rate = 0.05
mutation_std = 0.05
elitism = 2

# Inicijalizacija populacije
rng = np.random.default_rng(42)
def initializePopulation():
    return [rng.normal(0, 0.1, size=totalWeights) for _ in range(pop_size)]

# === KLJUČNA ISPRAVKA ===
# Za svaki fitness POSEBAN Keras model → nema kontaminacije
def fitness(individual, X_eval, y_eval):
    model = createModel()
    setModelWeights_from_flat(model, individual)
    y_pred = model.predict(X_eval, verbose=0)
    acc = np.mean(np.argmax(y_pred, axis=1) == y_eval)
    return acc

def tournament_selection(scores, k=3):
    idxs = np.random.choice(len(scores), size=k, replace=False)
    return idxs[np.argmax(scores[idxs])]

def crossover(p1, p2):
    mask = np.random.rand(totalWeights) < 0.5
    return np.where(mask, p1, p2)

def mutate(individual):
    mask = np.random.rand(totalWeights) < mutation_rate
    noise = np.random.normal(0, mutation_std, size=totalWeights)
    ind = individual.copy()
    ind[mask] += noise[mask]
    return ind

# ======================================================================
# EVOLUCIJA
# ======================================================================

tf.keras.backend.clear_session()
population = initializePopulation()

best_history = []
avg_history = []

for gen in range(generacija):
    scores = np.array([fitness(ind, X_val, y_val) for ind in population])

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    avg_score = np.mean(scores)

    best_history.append(best_score)
    avg_history.append(avg_score)

    print(f"Generacija {gen+1}/{generacija} | Najbolji: {best_score:.4f} | Prosek: {avg_score:.4f}")

    # Elitizam
    ranked = np.argsort(scores)[::-1]
    newPopulation = [population[i] for i in ranked[:elitism]]

    # Proizvodnja potomaka
    while len(newPopulation) < pop_size:
        p1_i = tournament_selection(scores)
        p2_i = tournament_selection(scores)
        child = crossover(population[p1_i], population[p2_i])
        child = mutate(child)
        newPopulation.append(child)

    population = newPopulation

# ======================================================================
# NAJBOLJI MODEL — TESTIRANJE
# ======================================================================

final_scores = np.array([fitness(ind, X_val, y_val) for ind in population])
best_idx = np.argmax(final_scores)
best = population[best_idx]

final_model = createModel()
setModelWeights_from_flat(final_model, best)

y_pred = final_model.predict(X_test, verbose=0)
pred_labels = np.argmax(y_pred, axis=1)
test_acc = np.mean(pred_labels == y_test)

print("\n===============================================")
print(f" ZAVRŠNA TAČNOST NA TEST SKUPU: {test_acc:.4f}")
print("===============================================\n")

# ======================================================================
# GRAFIK EVOLUCIJE
# ======================================================================

plt.figure(figsize=(10,5))
plt.plot(best_history, label='Najbolji u generaciji')
plt.plot(avg_history, label='Prosečan fitness')
plt.xlabel('Generacija')
plt.ylabel('Accuracy')
plt.title('Evolucija tačnosti tokom generacija')
plt.legend()
plt.grid(True)
plt.savefig('evolution_accuracy.png')

# ======================================================================
# EKSPORT REZULTATA U EXCEL
# ======================================================================

true_names = encoder.inverse_transform(y_test)
pred_names = encoder.inverse_transform(pred_labels)

if fn_test is not None:
    results_df = pd.DataFrame({
        'filename': fn_test,
        'true_label': true_names,
        'predicted_label': pred_names
    })
else:
    results_df = pd.DataFrame({
        'index': np.arange(len(y_test)),
        'true_label': true_names,
        'predicted_label': pred_names
    })

results_df['correct'] = results_df['true_label'] == results_df['predicted_label']
results_df.to_excel('result.xlsx', index=False)

print(f"Ukupna tačnost (iz Excel-a): {results_df['correct'].mean():.4f}")

try:
    os.system("start EXCEL.EXE result.xlsx")
except:
    pass