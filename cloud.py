Cle = [1,2,3,4,5]
LongCle = len(Cle)
TableTranscodage=[6,38,32,4,8,30,36,34,39,31,78,72,70,76,9,79,71,58,2,0,52,50,56,54,1,59]
MessageCrypt = ""
PhraseEnClair = input("entrer le message a crypter ")
LongPhraseEnClair = len(PhraseEnClair)

for i in range(LongPhraseEnClair):
    CaractereClair = PhraseEnClair[i]
    if CaractereClair == " ":
        CaractereCode = "  "
    else:
        j = i % LongCle
        # On retire 97 : les indices de la table de transcodage doivent être compris entre 0 et 25
        CaractereCode = (TableTranscodage[ord(CaractereClair) - 97] + Cle[int(j)]) % 100
        if CaractereCode < 10:
            CaractereCode = "0" + str(CaractereCode)
        else:
            CaractereCode = str(CaractereCode)
    MessageCrypt += CaractereCode

print ("           Votre message est  :")  # non modifié
print (PhraseEnClair)
print
print ("           Le message crypté est:")
print (MessageCrypt)

#dechiffrement
Cle = [1,2,3,4,5]
LongCle = len(Cle)
TableTranscodage={6:'a',38:'b',32:'c',4:'d',8:'e',30:'f',36:'g',34:'h',39:'i',31:'j',78:'k',\
                  72:'l',70:'m',76:'n',9:'o',79:'p',71:'q',58:'r',2:'s',0:'t',52:'u',\
                  50:'v',56:'w',54:'x',1:'y',59:'z'}
MessageCrypt= input("entrer le code si vous souhaité décrypter le message ")
LongMessageCrypt=len(MessageCrypt)
MessageClair=""

for i in range(0,LongMessageCrypt,2):                    # Parcourt les caractères par pas de deux
    Caractere=MessageCrypt[i:i+2]
    if Caractere == "  ":                                          # Remplacement de deux espaces consécutifs par un seul
        Caractere=" "
    else:
        if LongCle >= LongMessageCrypt:
            j=i/2
        else:
            j=(i/2)%LongCle                                   # Indice de position de la clé calculée modulo longueur de clé
            CaractereCode = (int(Caractere)-Cle[int(j)]+100)%100

            Caractere=TableTranscodage[CaractereCode]
    MessageClair+=Caractere

print ("           vous avez entré comme code :")  # non modifié
print (MessageCrypt)
print
print ("           Le message décrypté est :")
print (MessageClair)