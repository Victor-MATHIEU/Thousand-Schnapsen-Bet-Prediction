dic_cart =dict()
dic_cart = {"A":11,"0":10,"R":4,"D":3,"V":2,"9":0}
ord_cart = [11,10,4,3,2,0]
dic_col  =dict()
dic_col  = {"H":100,"D":80,"C":60,"S":40}
dic_couleur = {"H":"♥","D":"♦","C":"♣","S":"♠" }
dic_lacarte = {"A":"As   ","0":"10   ","R":"Roi  ","D":"Dame ","V":"Valet","9":"9    "}

dic_symbol_indice = {"♥-A": 0, "♥10": 1,"♥-R": 2,"♥-D": 3,"♥-V" :4,"♥-9": 5,
                     "♦-A": 6, "♦10": 7,"♦-R": 8,"♦-D": 9,"♦-V":10,"♥-9":11,
                     "♣-A":12, "♣10":13,"♣-R":14,"♣-D":15,"♣-V":16,"♣-9":17,
                     "♠-A":18, "♠10":19,"♠-R":20,"♠-D":21,"♠-V":22,"♠-9":23,}
liste_symbols = ["♥-A","♥10","♥-R","♥-D","♥-V","♥-9",
                 "♦-A","♦10","♦-R","♦-D","♦-V","♥-9",
                 "♣-A","♣10","♣-R","♣-D","♣-V","♣-9",
                 "♠-A","♠10","♠-R","♠-D","♠-V","♠-9"]

def transcrire_cartes_chiffres(liste):
    cartes = []
    for i in liste :
        cartes.append(dic_cart[i[0]])
        cartes.append(dic_col[ i[1]])
    return cartes
        
def transcrire_cartes_symbole(liste):
    cartes = []
    for i in liste :
        cartes.append(dic_lacarte[i[0]]+dic_couleur[i[1]])
    return cartes 

def sort_list_based_on_dicts(lst):
    return sorted(lst, key=lambda x: (dic_col[x[1]], dic_cart[x[0]]), reverse=True)

def transcrire_symbol_liste(liste_de_symbols):
    cartes = 24*[0]
    for i in liste_de_symbols : cartes[dic_symbol_indice[i]] = 1
    return cartes

def transcrire_liste_symbol(liste):
    cartes = []
    for n,i in enumerate(liste) : 
        if i==1 : cartes.append(liste_symbols[n])
    return cartes

def traduire_ech_en_carte(listedecartes):
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




if False :  # pour tests    
    a = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    cartes = transcrire_liste_symbol(a)
    print(a, cartes)
    cartes = ["♠-R", "♥-R", "♠-9", "♥-A","♦10","♦-R"]
    b = transcrire_symbol_liste(cartes)
    print("\n",cartes, b,"\n",transcrire_liste_symbol(b ))




