import numpy as np
import pandas as pd

dataset = pd.read_csv('enjoysport.csv')

def entropy(target_col):
    elements,counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def infogain(data, split_att, target_name = "Decision"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_att], return_counts=True)

    weighted_entropy = np.sum([ (counts[i]/np.sum(counts)) * entropy(data.where(data[split_att] == vals[i]).dropna()[target_name]) for i in range(len(vals))])

    infogain = total_entropy - weighted_entropy
    return infogain

def id3(data, original_data, features, target_att="Decision", parent_node_class=None):
    if len(np.unique(data[target_att])) <= 1:
        return np.unique(data[target_att])[0]
    elif len(data)==0:
        return np.unique(original_data[target_att])[np.argmax(np.unique(original_data[target_att], return_counts=True)[1])]
    
    elif len(features) == 0:
        return parent_node_class 
    else:
        parent_node_class = np.unique(data[target_att])[np.argmax(np.unique(data[target_att], return_counts=True)[1])]
        items_values = [infogain(data, feature, target_att) for feature in features]

        best_ftr_index = np.argmax(items_values)
        best_feature = features[best_ftr_index]

        tree = {best_feature: {}}

        for value in np.unique(data[best_feature]):
            subdata = data.where(data[best_feature]==value).dropna()
            subtree = id3(subdata, dataset, features, target_att, parent_node_class)
            tree[best_feature][value] = subtree
        
        return tree


tree = id3(dataset, dataset, dataset.columns[:-1])

print(dataset.head())
print("The Tree: \n\t", tree)
