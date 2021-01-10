
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
import cv2

#On défini les valeurs initiales des variables que l'on utilisera ensuite

nbr_ni=100 #nombre de neurones de la couche intermédiaire.
learning_rate=0.0001 #coefficient d'apprentissage
taille_batch=100 #taille des lots d'images que l'on va donner au réseau
nbr_entrainement=200 #nombre d'entrainement du réseau


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#On lit les fichiers à l'aide de la fonction fromfile de numpy :

mnist_train_images=np.fromfile("train-images.idx3-ubyte", dtype=np.uint8)[16:].reshape(-1,784)/255
mnist_train_labels=np.eye(10)[np.fromfile("train-labels.idx1-ubyte", dtype=np.uint8)[8:]]
mnist_test_images=np.fromfile("t10k-images.idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 784)/255
mnist_test_labels=np.eye(10)[np.fromfile("t10k-labels.idx1-ubyte", dtype=np.uint8)[8:]]

#On rentre dans la fonction le (fichier, le type dans lequel est codé le fichier).
#Ici le fichier est codé sur des entiers non signés sur 8 bits soit uint8. On l'indique à numpy pour qu'il puisse l'ouvrir.
#Les fichiers possèdent des en-tête de 8 octets pour les labels ou 16 octet pour les images que l'on enlève avec l'écriture [8:] et [16:] (on commence à partir du 8ème et du 16ème octet)
#Les fonctions nous renvoient des matrices de dim 28*28 que nous n'allons pas traiter ainsi. On utilise la commande reshape pour transformer ces matrices en vecteurs de dim 28*28=784
#Le -1 permet de laisser la fonction découper vecteurs de dim 784 sans savoir combien il y en a de base.
#On divise par 255 afin d'obtenir des nombres entre 0 et 1 car le fichier est codé sur des entiers non signés sur 8 bits.
#On obtient un tableau de nombre. Mais on souhaite des vecteurs de dimension 10 avec toutes les coordonnées du vecteur qui sont égales à 0 sauf la coordonnée correspondant à la classe de l'image
#A l'aide de la commande np.eye(10), on peut changer par exemple le nombre 3 en un vecteur de dimension 10 dont toutes les composantes sont nulles sauf la 3ème égale à 1

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()#on utilise cette commande pour pouvoir faire fonctionner le code qui suit marchant seulement avec Tensorflow 1.X alors qu'ici on possède la version Tensorflow 2.4.0

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Nous allons maintenant déclarer des placeholders qui sont des interfaces entre l'IA et nous.
#Pour s'adresser à l'IA, on ne fait pas directement référence aux images que l'on a créées mais aux placeholders dans lesquels on les a placées.

ph_images=tf.compat.v1.placeholder(shape=(None,784), dtype=tf.float32) #placeholder pour les images
ph_labels=tf.compat.v1.placeholder(shape=(None,10), dtype=tf.float32)  #placeholder pour les labels

#Pour le placeholder image (resp. label), on lui donne une géométrie de 784 (resp. 10) pour qu'il puisse recueillir les vecteurs images (resp. label) de dimension 784 (resp. 10)
#Le None à la même utilité que le -1, il permet de ne pas donner un nombre défini d'image ou de label qu'il peut recevoir.
#Le float32 représente le type des placeholders.


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#On explique à tensorflow comment effectuer les calculs

#On commence d'abord par exprimer les calculs entre l'entrée et la couche intermédiaire du réseau de neurones.

wci=tf.Variable(tf.compat.v1.truncated_normal(shape=(784, nbr_ni)), dtype=tf.float32) #on crée une variable tenderflow qui tire des poids au hasard selon une loi normale (centrée sur 0 et de variance 1) tronquée (à laquelle on enlève les valeurs extrèmes hors de l'intervalle [-2,2] afin de les éviter)
#La géométrie de la wci accepte des vecteurs de dim 784 et ressort des vecteurs de dim nbr_ni

bci=tf.Variable(np.zeros(shape=(nbr_ni)), dtype=tf.float32) #on crée un tableau de 100 neurones dont les biais seront initialisées à 0

sci=tf.matmul(ph_images, wci)+bci #sortie de la couche intermédiaire. Ici on pondère les informations présentes dans le placeholder venant de la couche d'entrée par les poids.

sci=tf.nn.sigmoid(sci) #on rentre nos informations pondérées dans une fonction d'activation sigmoïde 


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#On explique maintenant les calculs entre la couche intermédiaire et celle de sortie

wcs=tf.Variable(tf.compat.v1.truncated_normal(shape=(nbr_ni, 10)), dtype=tf.float32) #On applique le poids aux informations de la même façon qu'entre la couche d'entrée et celle intermédiaire sauf qu'ici les neurones de sortie reçoivent des vecteurs de dim nbr_ni et qu'il en sort des vecteurs de de dim 10.

bcs=tf.Variable(np.zeros(shape=(10)), dtype=tf.float32)#On crée un tableau de 10 neurones dont les biais seront initialisés à 0

scs=tf.matmul(sci, wcs)+bcs #sortie de la couche de sortie. Ici on pondère les informations venant de la couche intermédiaire par les poids.

scso=tf.nn.softmax(scs)#On passe les informations dans la fonction softmax permettant de transformer un vecteur en un vecteur de probabilité (dont la somme de toutes les composantes est égale à 1)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#On va maintenant expliquer à Tensorflow comment on souhaite calculer notre erreur entre ce que donne le réseau de neurones et ce qu'on souahite avoir.

#Entre deux vecteurs de probabilitées, on fait le calcul d'erreur avec l'entropie croisée donc ici en utilisant la fonction 'softmax_cross_entropy_with_logits_v2'

loss=tf.nn.softmax_cross_entropy_with_logits(labels=ph_labels, logits=scs)#on donne 2 paramètres à la fonction : le placeholder dans lequel on viendra placer les labels et la sortie du réseau de neurones AVANT son entrée dans la fonction softmax soit scs 

train=tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)#On dit à Tensorflow qu'on utilise l'algorithme d'apprentissage GradientDescentOptimizer auquel on donne le coefficient d'apprentissage learning_rate définie précédemment.
#On lui demande de minimiser l'erreur, ce qui a pour conséquence que durant l'apprentissage, le réseau de neurones aura pour but de posséder l'erreur la plus petite possible entre le nombre attendu et celui qu'il trouve. C'est comme ça que le réseau de neurones apprend.

accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(scso, 1), tf.argmax(ph_labels, 1)), dtype=tf.float32)) #On évalue la précision du réseau de neurones en utilisant la commande tf.argmax (étant la fonction inverse de np.eye) permettant de transformer un vecteur en un nombre (en renvoyant le numéro de la coordonnées possèdant la valeur la plus élevée
#On récupère ainsi la sortie la plus élevée soit la classe dans laquelle le réseau de neurones met l'image et on la compare à la bonne réponse présente dans ph_labels en faisant la moyenne des 2.


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#On a maintenant terminé de créer nos fonctions. On va maintenant les utiliser.

#On utilise la commande Session.run afin de lancer le réseau de neurones.


with tf.compat.v1.Session() as s:
    
    s.run(tf.compat.v1.global_variables_initializer()) #On initialise les variables

#On commence l'entrainement qui sera répeté 200 fois
    
    tab_acc_train=[]
    tab_acc_test=[]
    
    for id_entrainement in range(nbr_entrainement):
        print("ID entrainement", id_entrainement)
        
        for batch in range(0, len(mnist_train_images), taille_batch): #Sur une range entre 0 et la taille totale de la base en avançant par pas de taille_batch soit ici de 100
            s.run(train,feed_dict={ #on fait appelle à la fonction run et on lui donne la variable train étant la variable d'entrainement.
                ph_images: mnist_train_images[batch:batch+taille_batch],#On demande, à l'aide du paramètre feed_dict, de placer dans le placeholder image les informations de la base de données allant du rang batch au rang batch+taille_batch soit ici par lot de 100.
                ph_labels: mnist_train_labels[batch:batch+taille_batch]#Idem pour les labels
            })
#On calcule la précision après l'entrainement
            
        tab_acc=[]
        
        for batch in range(0, len(mnist_train_images), taille_batch):
            acc=s.run(accuracy, feed_dict={
                ph_images:mnist_train_images[batch:batch+taille_batch],
                ph_labels:mnist_train_labels[batch:batch+taille_batch]
            })
            
        tab_acc.append(acc) #on place les précisions des différents lots dans un tableau

        print("Accuracy train :", np.mean(tab_acc))

    #On calcule l'erreur

        tab_acc_test.append(1-np.mean(tab_acc)) #On stocke dans le tableau l'erreur

    #On effectue la même chose avec les données de test
    
        tab_acc=[]
        
        for batch in range(0, len(mnist_test_images), taille_batch):
            acc=s.run(accuracy, feed_dict={
                ph_images: mnist_test_images[batch:batch+taille_batch],
                ph_labels: mnist_test_labels[batch:batch+taille_batch]
            })

            tab_acc.append(acc)

        print("Accuracy test :", np.mean(tab_acc))
        tab_acc_test.append(1-np.mean(tab_acc))
    

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#On va maintenant représenter l'erreur dans un graphe

#On crée le graphe

    plot.ylim(0, 1)
    plot.grid()
    plot.plot(tab_acc_train, label="Train error")
    plot.plot(tab_acc_test, label="Test error")
    plot.legend(loc="upper right")
    plot.show()

    resulat=s.run(scso,feed_dict={ph_images: mnist_test_images[0:taille_batch]})#On demande au réseau de faire un calcul avec s.run en utilisant le vecteur de probabilité de sortie de réseau de neurones scso sur une partie des images de test (les 100 premières) 
    np.set_printoptions(formatter={'float':'{:0.3f}'.format})#on précise le format des tableaux

    for image in range(taille_batch): #pour l'ensemble du lot de 100 on affiche ce que donne le réseau de neurones et le bon label (celui attendu)
    
        print("image", image)
        print("sortie du réseau", resulat[image], np.argmax(resulat[image]))
        print("sortie attendue", mnist_test_labels[image], np.argmax(mnist_test_labels[image]))
    
        cv2.imshow('image', mnist_test_images[image].reshape(28,28)) #on affiche l'image redimensionnée en 28*28
        if cv2.waitKey()==ord('q'):
            break

    
