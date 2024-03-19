import sys
print("tensor flow fonctione avec python 3.9, voici la version utilis√©e :",sys.version)
import struct; print(struct.calcsize("P") * 8)
import tensorflow as tf
from tensorflow                    import keras
from tensorflow.keras              import layers
from tensorflow.keras.models       import Sequential
from tensorflow.keras.layers       import Dense, Activation
from tensorflow.keras.optimizers   import SGD
from tensorflow.keras.optimizers   import Adam
from tensorflow.keras              import initializers
from tensorflow.keras.initializers import Constant, zeros
from tensorflow.keras.models       import load_model

import numpy               as np 
import matplotlib.pyplot   as plt
from transcripteur                 import transcrire_symbol_liste, transcrire_liste_symbol,traduire_ech_en_carte
print("version de tensorflow: ",tf.__version__)


import numpy as np
from keras.layers import Input, Dense, Lambda, Concatenate
from keras.models import Model

def initialisation_model() : 
    # Définition du modèle : nous avons 24 entrées
    input_layer = Input(shape=(24,))
    
    #---------------meldunek
    # Utilisez une couche Lambda pour sélectionner les Roi et Dame de coeur  et connexion aux entrées sélectionnées
    selected_inputs_2H = Lambda(lambda x: tf.gather(x, [2,3], axis=1))(input_layer)
    hidden_layer_2_0H = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-1))(selected_inputs_2H)
    # Ajoutez une couche Lambda pour multiplier la sortie par 100
    meldunek_2H = Lambda(lambda x: x * 100)(hidden_layer_2_0H)
    #--
    # Utilisez une couche Lambda pour sélectionner les Roi et Dame de coeur  et connexion aux entrées sélectionnées
    selected_inputs_2D = Lambda(lambda x: tf.gather(x, [8,9], axis=1))(input_layer)
    hidden_layer_2_0D = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-1))(selected_inputs_2D)
    # Ajoutez une couche Lambda pour multiplier la sortie par 80
    meldunek_2D = Lambda(lambda x: x * 80)(hidden_layer_2_0D)
    #--
    # Utilisez une couche Lambda pour sélectionner les Roi et Dame de coeur  et connexion aux entrées sélectionnées
    selected_inputs_2C = Lambda(lambda x: tf.gather(x, [14,15], axis=1))(input_layer)
    hidden_layer_2_0C = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-1))(selected_inputs_2C)
    # Ajoutez une couche Lambda pour multiplier la sortie par 60
    meldunek_2C = Lambda(lambda x: x * 60)(hidden_layer_2_0C)
    #--
    # Utilisez une couche Lambda pour sélectionner les Roi et Dame de coeur  et connexion aux entrées sélectionnées
    selected_inputs_2S = Lambda(lambda x: tf.gather(x, [20,21], axis=1))(input_layer)
    hidden_layer_2_0S = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-1))(selected_inputs_2S)
    # Ajoutez une couche Lambda pour multiplier la sortie par 40
    meldunek_2S = Lambda(lambda x: x * 40)(hidden_layer_2_0S)
    
    #------------------ nombre de series AS
    # Utilisez une couche Lambda pour sélectionner les AS : entrées 1,7,13,19
    input_A     = Lambda(lambda x: tf.gather(x, [0,6,12,18], axis=1))(input_layer)
    # Connectez la première couche cachée uniquement aux entrées sélectionnées, 4 neurones
    nb_plis_1   = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(0) )(input_A)
    pnts_A      = Lambda(lambda x: x * 11)(nb_plis_1)
    #---------------- nombre de series AS et 10
    input_A10_H = Lambda(lambda x: tf.gather(x, [0 , 1], axis=1))(input_layer)
    input_A10_D = Lambda(lambda x: tf.gather(x, [6 , 7], axis=1))(input_layer)
    input_A10_C = Lambda(lambda x: tf.gather(x, [12,13], axis=1))(input_layer)
    input_A10_S = Lambda(lambda x: tf.gather(x, [18,19], axis=1))(input_layer)
    layer_A10_H = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-1))(input_A10_H)
    layer_A10_D = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-1))(input_A10_D)
    layer_A10_C = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-1))(input_A10_C)
    layer_A10_S = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-1))(input_A10_S)
    serie_A10   = Concatenate()([ layer_A10_H, layer_A10_D, layer_A10_C , layer_A10_S])
    nb_plis_2   = Dense(1, activation='linear', kernel_initializer=initializers.Constant(1))(serie_A10)
    pnts_A10    = Dense(1, activation='linear', kernel_initializer=initializers.Constant(12))(serie_A10)
    #---------------- nombre de series AS et 10 et Roi
    input_A10R_H = Lambda(lambda x: tf.gather(x, [0 , 1, 2], axis=1))(input_layer)
    input_A10R_D = Lambda(lambda x: tf.gather(x, [6 , 7, 8], axis=1))(input_layer)
    input_A10R_C = Lambda(lambda x: tf.gather(x, [12,13,14], axis=1))(input_layer)
    input_A10R_S = Lambda(lambda x: tf.gather(x, [18,19,20], axis=1))(input_layer)
    layer_A10R_H = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-2))(input_A10R_H)
    layer_A10R_D = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-2))(input_A10R_D)
    layer_A10R_C = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-2))(input_A10R_C)
    layer_A10R_S = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-2))(input_A10R_S)
    serie_A10R   = Concatenate()([ layer_A10R_H, layer_A10R_D, layer_A10R_C , layer_A10R_S])
    nb_plis_3    = Dense(1, activation='linear', kernel_initializer=initializers.Constant(1))(serie_A10R)
    pnts_A10R    = Dense(1, activation='linear', kernel_initializer=initializers.Constant(17))(serie_A10R)
    #---------------- nombre de series AS et 10 et Roi et Dame
    input_A10RD_H = Lambda(lambda x: tf.gather(x, [0 , 1, 2, 3], axis=1))(input_layer)
    input_A10RD_D = Lambda(lambda x: tf.gather(x, [6 , 7, 8, 9], axis=1))(input_layer)
    input_A10RD_C = Lambda(lambda x: tf.gather(x, [12,13,14,15], axis=1))(input_layer)
    input_A10RD_S = Lambda(lambda x: tf.gather(x, [18,19,20,21], axis=1))(input_layer)
    layer_A10RD_H = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-3))(input_A10RD_H)
    layer_A10RD_D = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-3))(input_A10RD_D)
    layer_A10RD_C = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-3))(input_A10RD_C)
    layer_A10RD_S = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-3))(input_A10RD_S)
    serie_A10RD   = Concatenate()([ layer_A10RD_H, layer_A10RD_D, layer_A10RD_C , layer_A10RD_S])
    nb_plis_4     = Dense(1, activation='linear', kernel_initializer=initializers.Constant(1))(serie_A10RD)
    pnts_A10RD    = Dense(1, activation='linear', kernel_initializer=initializers.Constant(7))(serie_A10RD)
    #---------------- nombre de series AS et 10 et Roi et Dame et Valet
    input_A10RDV_H = Lambda(lambda x: tf.gather(x, [0 , 1, 2, 3, 4], axis=1))(input_layer)
    input_A10RDV_D = Lambda(lambda x: tf.gather(x, [6 , 7, 8, 9,10], axis=1))(input_layer)
    input_A10RDV_C = Lambda(lambda x: tf.gather(x, [12,13,14,15,16], axis=1))(input_layer)
    input_A10RDV_S = Lambda(lambda x: tf.gather(x, [18,19,20,21,22], axis=1))(input_layer)
    layer_A10RDV_H = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-4))(input_A10RDV_H)
    layer_A10RDV_D = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-4))(input_A10RDV_D)
    layer_A10RDV_C = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-4))(input_A10RDV_C)
    layer_A10RDV_S = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-4))(input_A10RDV_S)
    serie_A10RDV   = Concatenate()([ layer_A10RDV_H, layer_A10RDV_D, layer_A10RDV_C , layer_A10RDV_S])
    nb_plis_5      = Dense(1, activation='linear', kernel_initializer=initializers.Constant(1))(serie_A10RDV)
    pnts_A10RDV    = Dense(1, activation='linear', kernel_initializer=initializers.Constant(8))(serie_A10RDV)
    #---------------- nombre de series AS et 10 et Roi et Dame et Valet et 9
    input_A10RDV9_H = Lambda(lambda x: tf.gather(x, [0 , 1, 2, 3, 4, 5], axis=1))(input_layer)
    input_A10RDV9_D = Lambda(lambda x: tf.gather(x, [6 , 7, 8, 9,10,11], axis=1))(input_layer)
    input_A10RDV9_C = Lambda(lambda x: tf.gather(x, [12,13,14,15,16,17], axis=1))(input_layer)
    input_A10RDV9_S = Lambda(lambda x: tf.gather(x, [18,19,20,21,22,23], axis=1))(input_layer)
    layer_A10RDV9_H = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-5))(input_A10RDV9_H)
    layer_A10RDV9_D = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-5))(input_A10RDV9_D)
    layer_A10RDV9_C = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-5))(input_A10RDV9_C)
    layer_A10RDV9_S = Dense(1, activation='relu', kernel_initializer=Constant(1),bias_initializer=Constant(-5))(input_A10RDV9_S)
    serie_A10RDV9   = Concatenate()([ layer_A10RDV9_H, layer_A10RDV9_D, layer_A10RDV9_C , layer_A10RDV9_S])
    nb_plis_6       = Dense(1, activation='linear', kernel_initializer=initializers.Constant(1))(serie_A10RDV9)
    pnts_A10RDV9    = Dense(1, activation='linear', kernel_initializer=initializers.Constant(8))(serie_A10RDV9)
    #----------------------------------------------------------------------------------------------------
    #concaténation des couches de calcul de plis
    concat_plis    = Concatenate()([nb_plis_1,nb_plis_2,nb_plis_3,nb_plis_4,nb_plis_5,nb_plis_6])
    points_plis7   = Dense(1, activation='relu', kernel_initializer=initializers.Constant(8), bias_initializer=Constant(-6*8))(concat_plis) #rajoute 8 points par pli supplémentaire si fait plus de 7 plis et plus
    points_plis8   = Dense(1, activation='relu', kernel_initializer=initializers.Constant(12),bias_initializer=Constant(-7*12))(concat_plis) #rajoute 12 points par pli supplémentaire si fait plus de 8 plis et plus
    #---------------------------------------------------------------------------------------------------
    #concaténation des  couches de calcul de points et divers blocs de reseaux de neurones (meldunek, séries, plis)
    concat_points = Concatenate()([meldunek_2H,meldunek_2D,meldunek_2C,meldunek_2S,
                                   pnts_A,pnts_A10,pnts_A10R,pnts_A10RD,pnts_A10RDV,pnts_A10RDV9,
                                   points_plis7 ,points_plis8 
                                   ])
    output_layer  = Dense(1, activation='softplus', kernel_initializer=initializers.Constant(1))(concat_points)
    #========================================================================================================
    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) # Compilation du modèle
    return model

#-----------------------------------------------------------------------------------------
def affichage_graphe_evol_entrainenement(history, 
                                         titre='Evolution de la fonction loss au fur et à mesure de l\'apprentissage',
                                         color='orange'):
    #--------- affiche l'évlution de la fonction loss  au fut et à mesure entrinemetn -----
    # Plot the loss values
    plt.plot(history.history['loss'], color=color)
    plt.title(titre)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
#-------------------------------------------------------------------------------
model = initialisation_model()
model.save(        "model_0_mille.h5")   # sauvegarde du modèle initial #Lorsque vous sauvegardez un modèle Keras au format HDF5, non seulement l'architecture du modèle est sauvegardée, mais aussi les poids du modèle, la configuration de l'entraînement (loss, optimizer), et l'état de l'optimizer, ce qui vous permet de reprendre l'entraînement exactement là où vous l'avez laissé.


#--------définitions des valeurs d'entrainement  entrées / sorties---------------
if True: #=====> entrainement avec 11 jeux  d'entrainement -------------------------- 
  #------------------------- génération  des entrée sortiée ----------
    Xsymbols = [['♥-A', '♥-9', '♠-R', '♠-9'],['♥-A', '♥-V', '♥-R', '♠-9'],['♥-A','♠-A'],['♥-A','♠-A',"♣-A",'♦10'],['♥-A', '♥-9', '♥-R', '♥-D'],['♥-V', '♥-9', '♣-R', '♠-9', '♦-R'],['♥-A', '♥10',"♣-9"],['♥-A', '♥10','♥-R',"♣-9",'♦-R' ],['♥-A', '♥10','♥-R','♥-D',"♣-9",'♦-R','♠-9' ],['♥-A', '♥10','♥-R','♥-D','♥-V',"♣-9",'♦-R','♠-9' ],
                ['♣-R','♣-D']]
       
    X = np.array([transcrire_symbol_liste(jeu ) for jeu in Xsymbols])  # X = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], ... etc ..])
    
    y = np.array([[11],[11],[22],[33],[111],[0],[23],[30],[137],[145],
                  [60]])
  #----------------------------------------------
     # ----------- Entraînement initial du modèle
    model   = load_model("model_0_mille.h5")   # chargement du modèle 0 (modèle intial, non entrainé, poids initialisés à la main)
    history =model.fit(X, y, epochs=30, batch_size=11)
    model.save(        "model_A_mille.h5")
    affichage_graphe_evol_entrainenement(history, titre ="A) Apprentissage initial, 11 échantillons, 30 epochs, taille batch = 11")

if True:  #=====> entrainement avec un nombre arbitraire de jeux  d'entrainement (en jouant les jeux et mesurant les points obtenus)
  #------------------------- génération  des entrée sortiée ----------
    from  generer_set_de_jeux_et_points import genere_jeux_et_points
    nbr_series_de_3 = 200
    X0,y0 = genere_jeux_et_points(nbr_series_de_3 =  nbr_series_de_3, verbose= False)
    
    X = np.array(X0)
    y = np.array([[y1] for y1  in y0])
  #-----------------------------------------------
    # ----------- Ré-Entraînement du modèle ------------------------------------
    model   = load_model( "model_A_mille.h5")   # chargement du modèle A, entrainé sur 11 échantillons
    epochs , batch_size = 300, 51
    history =model.fit(X, y, epochs=30, batch_size=11)
    model.save(        "model_B_mille.h5")
    affichage_graphe_evol_entrainenement(history,  titre ="B) Apprentissage complementaire, "+str(nbr_series_de_3)+" échantillons, "+str(epochs)+" epochs, taille batch ="+str(batch_size))

if True:  #=====> entrainement avec un nombre arbitraire de jeux  d'entrainement (en jouant les jeux et mesurant les points obtenus)
  #------------------------- génération  des entrée sortiée ----------
    from  generer_set_de_jeux_et_points import genere_jeux_et_points
    nbr_series_de_3 = 3000
    X0,y0 = genere_jeux_et_points(nbr_series_de_3 =  nbr_series_de_3, verbose= False)
    
    X = np.array(X0)
    y = np.array([[y1] for y1  in y0])
  #-----------------------------------------------
    # ----------- Ré-Entraînement du modèle ------------------------------------
    model   = load_model( "model_B_mille.h5")   # chargement du modèle A, entrainé sur 11 échantillons
    epochs , batch_size = 30, 100
    history =model.fit(X, y, epochs=30, batch_size=11)
    model.save(        "model_C_mille.h5")
    affichage_graphe_evol_entrainenement(history,color='green',  titre ="C) Apprentissage complementaire, "+str(nbr_series_de_3)+" échantillons, "+str(epochs)+" epochs, taille batch ="+str(batch_size))

# sgd = SGD(learning_rate=0.1)
# adam = Adam(learning_rate=0.1,clipnorm=1)
# model.compile(loss='mean_squared_error', optimizer=adam)
# # Fit the model and save the history
# history = model.fit(X, y, batch_size=17, epochs=100,verbose=2) #on calcule la moyenne du gradient de 4 cartes, on revoit les paramètres, et on repète l'opération 50 fois


#-------- affiche les résultats  : prédictions -------
print("="*34, " Prédictions  pour les 10 premiers échantillons ", "="*30)
Y = model.predict(X)
#     --- affichages sous divers formats  ---
for n,x in enumerate(X) :
    print( "jeu n° ",n,"\t composé de :",x," résultat attendu :",y[n]," prédiction:",Y[n])
    if n>10 : break
print()
for n,x in enumerate(X) :
    print( "jeu n° ",n,"\t composé de :",traduire_ech_en_carte([x])," résultat attendu :",y[n]," prédiction:",Y[n])
    if n>10 : break
print()
for n,x in enumerate(X) :
    print( "jeu n° ",n,"\t composé de :",transcrire_liste_symbol(x)," résultat attendu :",y[n]," prédiction:",Y[n])
    if n>10 : break
print()
#----------- 
Y1     = [10*int(z/10) for z in Y]            # predictions arrondies
#Paris0 = [0 if z < 100 else z for z in Y1]
# Créez les intervalles
bins = np.arange(0, 260, 10)
# Calculez les probabilités pour chaque intervalle
probs = []
for i in range(len(bins) - 1):
    mask = (Y1 >= bins[i]) & (Y1 < bins[i + 1])
    prob = np.mean((y[mask] >= bins[i]) & (y[mask] < bins[i + 1]))
    probs.append(prob)

# Créez le graphique
plt.bar(bins[:-1], probs, width=9, color='green')
plt.xlabel('Valeur de la prédiction Y1')
plt.ylabel('Probabilité que y réel soit dans le même intervalle')
plt.show()
#----------------Pour une prédiction de 100 , quelle sont les valeurs réelles de y
# Convertissez les listes en tableaux numpy
Y2 = np.array(Y1)
y = np.array(y)

# Sélectionnez les prédictions dans l'intervalle 100 à 109
mask = (Y2 >= 100) & (Y2 < 110)
selected_y = y[mask]

# Créez les intervalles pour y
bins = np.arange(0, 260, 10)

# Calculez les probabilités pour chaque intervalle de y
probs = []
for i in range(len(bins) - 1):
    prob = np.mean((selected_y >= bins[i]) & (selected_y < bins[i + 1]))
    probs.append(prob)

# Créez le graphique
plt.bar(bins[:-1], probs, width=9)
plt.xlabel('Valeur de y')
plt.ylabel('Probabilité')
plt.title('Pour une prédiction de 100, quelles sont les valeurs réelles de points y obtenus ?')

plt.show()


#--------------- Pour une prédiction de 110 , quelle sont les valeurs réelles de y
# Convertissez les listes en tableaux numpy
Y2 = np.array(Y1)
y = np.array(y)

# Sélectionnez les prédictions dans l'intervalle 100 à 109
mask = (Y2 >= 110) & (Y2 < 120)
selected_y = y[mask]

# Créez les intervalles pour y
bins = np.arange(0, 260, 10)

# Calculez les probabilités pour chaque intervalle de y
probs = []
for i in range(len(bins) - 1):
    prob = np.mean((selected_y >= bins[i]) & (selected_y < bins[i + 1]))
    probs.append(prob)

# Créez le graphique
plt.bar(bins[:-1], probs, width=9)
plt.xlabel('Valeur de y')
plt.ylabel('Probabilité')
plt.title('Pour une prédiction de 110, quelles sont les valeurs réelles de points y obtenus ?')

plt.show()


