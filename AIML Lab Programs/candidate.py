import pandas as pd
import numpy as np

data = pd.DataFrame(pd.read_csv('td.csv'))

concepts = np.array(data.iloc[:, :-1])
targets = np.array(data.iloc[:,-1])


print("\n\n")

def learn(concepts, targets):
    specific_h = concepts[0].copy()
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]

    only_q = general_h[0].copy()

    for i, h in enumerate(concepts):

        if str(targets[i]) != ' yes':
            #target is no
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'            
        else:
            for j in range(len(specific_h)):
              
                if h[j] != specific_h[j]:
                    
                    specific_h[j] = '?'
                    general_h[j][j] = '?'
                    
                else:
                    continue
    
    while only_q in general_h:
        general_h.remove(only_q)
    
    print("General Hypothesis", general_h)
    print("Specific Hypothesis", specific_h)
    print("\n\n")

learn(concepts=concepts, targets=targets)

