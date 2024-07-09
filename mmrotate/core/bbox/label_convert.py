import torch

def convert_label(labels):
    def label_map(x):
        if 0<=x<=10:
            return 0
        elif 11<=x<=19:
            return 1
        elif 20<=x<=29:
            return 2
        elif 30<=x<=33:
            return 3
        elif 34<=x<=36:
            return 4
        else:
            return 5
    new_labels = torch.tensor(list(map(label_map,labels)))
    return new_labels

def convert_label_inverse(cls):
    if cls == 0:
        return range(0,11)
    elif cls == 1:
        return range(11,20)
    elif cls == 2:
        return range(20,30)
    elif cls == 3:
        return range(30,34)
    elif cls == 4:
        return range(33,37)
    else:
        return 37