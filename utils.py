def num_flat_features(x):
    sizes = x.size()[1: ]
    num_flat_features = 1
    for size in sizes:
        num_flat_features *= size
    
    return num_flat_features