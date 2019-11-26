import numpy as np
import IPython as ipy 
import os
import time

def get_confusion_matrix(reais, preditos, labels):

    if len(reais) != len(preditos):
        return None
    
    

    # valores preditos corretamente
    tp = 0
    tn = 0
    
    # valores preditos incorretamente
    fp = 0
    fn = 0
    
    for (indice, v_real) in enumerate(reais):
        # considerando a primeira classe como a positiva, e a segunda a negativa
        true_class = v_real
        negative_class = labels[1]
        
        v_predito = preditos[indice]

        # se trata de um valor real da classe positiva
        if v_real == true_class:
            tp += 1 if v_predito == v_real else 0
            fp += 1 if v_predito != v_real else 0
        else:
            tn += 1 if v_predito == v_real else 0
            fn += 1 if v_predito != v_real else 0
    
    matrix_confusion = np.array([
        # valores da classe positiva
        [ tp, fp ],
        # valores da classe negativa
        [ fn, tn ]
    ])

    print(matrix_confusion)


def init_matrix():
    ipy.embed()
    get_confusion_matrix(reais=valores_reais, preditos=valores_preditos, labels=[1,0])
    
    # array([[3, 1], [2, 4]])


if __name__ == '__main__':
    directory =  os.path.abspath('.') +  '/classes/'

    for filename in os.listdir(directory):
        
        if filename.endswith(".txt"): 
            with open(directory + filename) as file:  
                data = file.read()
            
            valores_reais = []
            valores_preditivos = []
            data = data.replace('[','')  
            data = data.replace("'",'') 
            data = data.replace(']','') 
            data = data.replace(',','') 
            data = data.replace(' ','') 
            data = data.replace('-','') 

            list_vals = list(data)
            
            for index, i in enumerate(list_vals): 
                if index%2 == 0:
                    valores_reais.append(i)
                else:
                    valores_preditivos.append(i)
            
            get_confusion_matrix(reais=valores_reais, preditos=valores_preditivos, labels=[1,0])

              

            
       

