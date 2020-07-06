import os

def model_paths():
    """
    Returns models (dict): Best models' paths for all label sizes. 
    """
    
    models = {
        "100":[
            ["best_models/60/C_epoch_23_label_100.pt","best_models/60/D_epoch_23_label_100.pt","best_models/60/G_epoch_23_label_100.pt"],
            ["best_models/61/C_epoch_15_label_100.pt","best_models/61/D_epoch_15_label_100.pt","best_models/61/G_epoch_15_label_100.pt"],
            ["best_models/62/C_epoch_29_label_100.pt","best_models/62/D_epoch_29_label_100.pt","best_models/62/G_epoch_29_label_100.pt"],
            ["best_models/63/C_epoch_23_label_100.pt","best_models/63/D_epoch_23_label_100.pt","best_models/63/G_epoch_23_label_100.pt"],
            ["best_models/64/C_epoch_24_label_100.pt","best_models/64/D_epoch_24_label_100.pt","best_models/64/G_epoch_24_label_100.pt"]
         ],
     
         "600":[
             ["best_models/60/C_epoch_7_label_600.pt","best_models/60/D_epoch_7_label_600.pt","best_models/60/G_epoch_7_label_600.pt"],
             ["best_models/61/C_epoch_7_label_600.pt","best_models/61/D_epoch_7_label_600.pt","best_models/61/G_epoch_7_label_600.pt"],
             ["best_models/62/C_epoch_21_label_600.pt","best_models/62/D_epoch_21_label_600.pt","best_models/62/G_epoch_21_label_600.pt"],
             ["best_models/63/C_epoch_10_label_600.pt","best_models/63/D_epoch_10_label_600.pt","best_models/63/G_epoch_10_label_600.pt"],
             ["best_models/64/C_epoch_40_label_600.pt","best_models/64/D_epoch_40_label_600.pt","best_models/64/G_epoch_40_label_600.pt"],
         ],
     
         "1000":[
             ["best_models/60/C_epoch_15_label_1000.pt","best_models/60/D_epoch_15_label_1000.pt","best_models/60/G_epoch_15_label_1000.pt"],
             ["best_models/61/C_epoch_10_label_1000.pt","best_models/61/D_epoch_10_label_1000.pt","best_models/61/G_epoch_10_label_1000.pt"],
             ["best_models/62/C_epoch_15_label_1000.pt","best_models/62/D_epoch_15_label_1000.pt","best_models/62/G_epoch_15_label_1000.pt"],
             ["best_models/63/C_epoch_31_label_1000.pt","best_models/63/D_epoch_31_label_1000.pt","best_models/63/G_epoch_31_label_1000.pt"],
             ["best_models/64/C_epoch_11_label_1000.pt","best_models/64/D_epoch_11_label_1000.pt","best_models/64/G_epoch_11_label_1000.pt"]
         ],
     
         "3000":[
             ["best_models/60/C_epoch_21_label_3000.pt","best_models/60/D_epoch_21_label_3000.pt","best_models/60/G_epoch_21_label_3000.pt"],
             ["best_models/61/C_epoch_39_label_3000.pt","best_models/61/D_epoch_39_label_3000.pt","best_models/61/G_epoch_39_label_3000.pt"],
             ["best_models/62/C_epoch_5_label_3000.pt","best_models/62/D_epoch_5_label_3000.pt","best_models/62/G_epoch_5_label_3000.pt"],
             ["best_models/63/C_epoch_15_label_3000.pt","best_models/63/D_epoch_15_label_3000.pt","best_models/63/G_epoch_15_label_3000.pt"],
             ["best_models/64/C_epoch_1_label_3000.pt","best_models/64/D_epoch_1_label_3000.pt","best_models/64/G_epoch_1_label_3000.pt"]
         ]
    }
    
    return models
