import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#----------ucitavanje podataka
data = pd.read_csv("./features_30_sec.csv")

# Ako postoji kolona 'filename', sačuvaj je da bismo mogli mapirati predikcije na redove/fajlove
filenames = None
if 'filename' in data.columns:
    filenames = data['filename'].values
    data = data.drop(['filename'], axis=1)
if 'length' in data.columns:
    data = data.drop(['length'], axis=1)

X = data.drop(['label'], axis=1).values
y = data['label'].values

encoder=LabelEncoder()
y=encoder.fit_transform(y)
scaler=StandardScaler()
X=scaler.fit_transform(X)

# Podijelimo i filenames (ako postoje) tako da znamo koji red je u testnom skupu
if filenames is not None:
    X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
        X, y, filenames, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    fn_train = None
    fn_test = None

n_inputs=X_train.shape[1]
n_hidden=4
n_outputs=len(np.unique(y))

#----------definicija neuronske mreze
def createModel():
    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),
        tf.keras.layers.Dense(n_hidden, activation='relu'),
        tf.keras.layers.Dense(n_outputs, activation='softmax')
    ])
    return model

model=createModel()

totalWeights=np.sum([np.prod(w.shape) for w in model.get_weights()])

#genetski algoritam
pop_size=50 #velicina populacije
generacija=100 #broj generacija
mutation_rate=0.1 #stopa mutacije
elitism=4 #za selekciju

def initializePopulation():
    return [np.random.randn(totalWeights) for _ in range(pop_size)]

def setModelWeights(model, weights_vector):
    shapes=[w.shape for w in model.get_weights()]
    newWeights=[]
    idx=0
    for shape in shapes:
        size=np.prod(shape)
        newWeights.append(weights_vector[idx:idx+size].reshape(shape))
        idx+=size
    model.set_weights(newWeights)

def fitness(individual):
    setModelWeights(model, individual)
    y_pred=model.predict(X_train, verbose=0)
    acc=np.mean(np.argmax(y_pred, axis=1)==y_train)
    return acc

def crossover(p1, p2):
    point=np.random.randint(0, len(p1))
    child=np.concatenate([p1[:point], p2[point:]])
    return child

def mutate(individual):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.randn()*0.2
    return individual

#evolucija
population=initializePopulation()

for gen in range(generacija):
    scores=np.array([fitness(ind) for ind in population])
    best_indexes=np.argsort(scores)[::-1]
    best_score=scores[best_indexes[0]]
    print(f'Generacija {gen+1}, Najbolji fitness: {best_score:.4f}')

    newPopulation=[population[i] for i in best_indexes[:elitism]]

    while(len(newPopulation) < pop_size):
        parents=np.random.choice(best_indexes[:10], size=2, replace=False)
        child=crossover(population[parents[0]], population[parents[1]])
        child=mutate(child)
        newPopulation.append(child)

    population=newPopulation

    # Nakon evolucije, ponovno izračunaj fitness-e za konačnu populaciju i odaberi najbolju jedinku
    final_scores = np.array([fitness(ind) for ind in population])
    best_idx = np.argmax(final_scores)
    best = population[best_idx]
    setModelWeights(model, best)

    # Predikcija na testnom skupu
    y_pred = model.predict(X_test)
    pred_labels = np.argmax(y_pred, axis=1)
    test_acc = np.mean(pred_labels == y_test)
    print(f'Tacnost na testnom skupu: {test_acc:.4f}')

    # Mapiranje natrag na originalne nazive klasa (ako su bile tekstualne)
    try:
        true_names = encoder.inverse_transform(y_test)
        pred_names = encoder.inverse_transform(pred_labels)
    except Exception:
        # fallback: ako encoder ne može (ne bi trebao), koristimo numeričke oznake
        true_names = y_test
        pred_names = pred_labels

    # Kreiraj DataFrame s per-row rezultatima i spremi u CSV
    if fn_test is not None:
        results_df = pd.DataFrame({
            'filename': fn_test,
            'true_label': true_names,
            'predicted_label': pred_names,
        })
    else:
        results_df = pd.DataFrame({
            'index': np.arange(len(y_test)),
            'true_label': true_names,
            'predicted_label': pred_names,
        })

    results_df['correct'] = results_df['true_label'] == results_df['predicted_label']
    results_df.to_csv('predictions.csv', index=False)
    # print('\nPrimjer predikcija (prvih 10):')
    # print(results_df.head(10))
    print(f"\nUkupna tacnost (provjerena iz DataFrame): {results_df['correct'].mean():.4f}")