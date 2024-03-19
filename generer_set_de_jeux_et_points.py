import random
#from estimateur_03        import compte_le_nombre_mini_de_plis_et_valeur
from transcripteur        import transcrire_liste_symbol
from simulation_jeu       import   point_entrainement, jouercartes,convertir_bin_en_carte

def traduire_ech_en_carte (listedecartes):
        dico_carte = {0:'AH',   1:'0H',  2:'RH',  3:'DH',  4:'VH',  5:'9H',
                      6:'AD',   7:'0D',  8:'RD',  9:'DD', 10:'VD', 11:'9D',
                      12:'AC', 13:'0C', 14:'RC', 15:'DC', 16:'VC', 17:'9C',
                      18:'AS', 19:'0S', 20:'RS', 21:'DS', 22:'VS', 23:'9S'}
        trad =[]
        for i in range(len(listedecartes)):
            trad.append([])
            for j in range(len(listedecartes[i])):
                if listedecartes[i][j]== 1 :
                    trad[-1].append(dico_carte[j])
        return trad

def echantillions(nbr_ech,nbr_cartes, nbr_joueurs = 3):
    # crée  nbrech   échantillons tirés au hasard, comportant chacun un nombre de cartes = cartes
    # nombre_joueurs  := est le nombre de joueurs,   Si c'est 3 ,  les cartes numérotés 0,1,2  sont toutes différentes
    
    listedecartes =[]
    
    for i in range(nbr_ech // nbr_joueurs):
        main = [0]*24

        # tirage aléatoire de nbr_joueurs * nbr_cartes
        def remplacer_zero_par_un(liste, nombre):
            indices = random.sample(range(len(liste)), nombre)
            for i in indices:
                liste[i] = 1
            return liste
        
        main =  remplacer_zero_par_un(main, nbr_joueurs * nbr_cartes )

        def diviser_liste(liste, nbr_joueurs):
            # Trouver tous les indices des 1
            indices = [i for i, x in enumerate(liste) if x == 1]
            
            # Mélanger les indices
            random.shuffle(indices)
            
            # Diviser les indices en nbr_joueurs listes
            taille = len(indices) // nbr_joueurs
            indices_divises = [indices[i * taille:(i + 1) * taille] for i in range(nbr_joueurs)]
            
            # Créer nbr_joueurs listes de zéros et remplacer les zéros par des uns aux indices divisés
            listes = []
            for indices_liste in indices_divises:
                nouvelle_liste = [0] * len(liste)
                for i in indices_liste:
                    nouvelle_liste[i] = 1
                listes.append(nouvelle_liste)
            
            return listes

        jeux_pour_nbr_joueurs =  diviser_liste(main, nbr_joueurs)
        
        listedecartes.extend(jeux_pour_nbr_joueurs)
        
    #retourner aussi les valeures
    #listetrad = traduire_ech_en_carte(listedecartes)

    
    return listedecartes   #,listetrad 


#print(echantillions(2, 3))
# crée 2 jeux, comportant chacun 3 cartes , et les estimations du nombre de poitns associées
# exemple :  ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0], [11]])
liste_ech = echantillions(3, 7)
print(echantillions(3, 7))
for l in liste_ech : print(transcrire_liste_symbol(l))

#----------------------------------
def genere_jeux_et_points(nbr_series_de_3 = 100, verbose = True):
    print(" ----------- génération des échantillons en jouant  "+str(nbr_series_de_3)+" jeux ---------")
    #nbr_series_de_3 = 100
    nbr_ech         = 3*nbr_series_de_3
    nbr_cartes      = 7
    listedecartes   = echantillions(nbr_ech,nbr_cartes, nbr_joueurs = 3)
    
    X = []
    y = []
    
    for i in range(0, len(listedecartes), 3):
        # Obtenir la tranche de 3 éléments
        slice_of_three = listedecartes[i:i+3]
        for k in slice_of_three :
            if verbose :  print("  ", k)
        s = [sum(x) for x in zip(slice_of_three[0], slice_of_three[1], slice_of_three[2])]
        if verbose :  print(" =",s,sum([x > 1 for x in s]), " nombre cartes:", sum(s) )
    
        main1,main2,main3 = slice_of_three
        
        pts1,pts2,pts3    = jouercartes(convertir_bin_en_carte(main1),
                                        convertir_bin_en_carte(main2),
                                        convertir_bin_en_carte(main3),
                                        distribue_3des_cartes_restantes = True, verbose = verbose )
    
        X.append(main1)
        y.append(pts1)
        print("jeu "+ str(i)+"/"+str(nbr_series_de_3)+" -",end="")
    return X,y
    

if False:  # pour test
     X,y = genere_jeux_et_points(nbr_series_de_3 = 100)







































