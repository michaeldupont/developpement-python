# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:49:24 2018

@author: fmdu
"""

# Imports
import numpy as np
import tensorflow as tf

# Longueur de la représentation binaire
NUM_DIGITS = 10

# Converti un nombre en sa décomposition binaire
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])
    
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])

# Les poids seront initialisés au hasard
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Perceptron à 1 couche utilisant ReLu comme fonction d'activation
# Softmax est appliqué en sortie pour la fonction de coût (pour convertir la sortie en probabilité)
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

# Nos variables
X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 4])

# Nombre de neurones dans la couche cachée
NUM_HIDDEN = 100

# Génération des poids
w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

# La prédiction de y sera donnée en fonction de x grâce au modèle
py_x = model(X, w_h, w_o)

# Fonction de coût et stratégie de mise à jour des poids
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

predict_op = tf.argmax(py_x, 1)

# Conversion en sens inverse de fizzbuzz : la prédiction finale devient le mot à afficher
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

BATCH_SIZE = 128
with tf.Session() as sess:
    tf.initialize_all_variables().run()
        
    for epoch in range(10000):
        # Mélange des données aléatoire
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        # Entraînement par batch de 128 lignes
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        # Affichage au fur et à mesure de la précision
        print(epoch, np.mean(np.argmax(trY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: trX, Y: trY})))
        
    numbers = np.arange(1, 101)
    teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
    teY = sess.run(predict_op, feed_dict={X: teX})
    output = np.vectorize(fizz_buzz)(numbers, teY)

    print(output)       
    
actuals = [fizz_buzz(i, fizz_buzz_encode(i).argmax()) for i in numbers]

for i, (predicted, actual) in enumerate(zip(output, actuals)):
    if predicted != actual:
        print("{0} {1} {2}".format(i+1, predicted, actual))