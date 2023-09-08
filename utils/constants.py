"""
Title: constants.py

Description:
    Constant values used across many scripts 
    
    Note: this is now slightly outdated
    
"""

# Paths
base_dir = "C:\\Users\\abzeh\\Documents\\ETH Zurich\\Masters Thesis\\main"
data_folder = base_dir + "\data"
code_folder = base_dir + "\code"



# Feature Names
time_index = "DateTimeUTC"


# Midpoint Diameters List (may change)

diams = [1.6849000e-02, 1.7466000e-02, 1.8106000e-02, 1.8769000e-02,
        1.9456000e-02, 2.0169000e-02, 2.0908000e-02, 2.1674000e-02,
        2.2468000e-02, 2.3291000e-02, 2.4144000e-02, 2.5029000e-02,
        2.5946000e-02, 2.6896000e-02, 2.7881000e-02, 2.8903000e-02,
        2.9961000e-02, 3.1059000e-02, 3.2197000e-02, 3.3376000e-02,
        3.4599000e-02, 3.5866000e-02, 3.7180000e-02, 3.8542000e-02,
        3.9954000e-02, 4.1418000e-02, 4.2935000e-02, 4.4508000e-02,
        4.6138000e-02, 4.7829000e-02, 4.9581000e-02, 5.1397000e-02,
        5.3280000e-02, 5.5232000e-02, 5.7255000e-02, 5.9352000e-02,
        6.1527000e-02, 6.3780000e-02, 6.6117000e-02, 6.8539000e-02,
        7.1050000e-02, 7.3653000e-02, 7.6351000e-02, 7.9148000e-02,
        8.2047000e-02, 8.5053000e-02, 8.8168000e-02, 9.1398000e-02,
        9.4746000e-02, 9.8217000e-02, 1.0181500e-01, 1.0554500e-01,
        1.0941100e-01, 1.1341900e-01, 1.1757400e-01, 1.2188100e-01,
        1.2634600e-01, 1.3097500e-01, 1.3577300e-01, 1.4074600e-01,
        1.4590200e-01, 1.5124700e-01, 1.5678800e-01, 1.6253100e-01,
        1.6848500e-01, 1.7465800e-01, 1.8105600e-01, 1.8768800e-01,
        1.9456400e-01, 2.0169100e-01, 2.0908000e-01, 2.1673900e-01,
        2.2467900e-01, 2.3291000e-01, 2.4144200e-01, 2.5028700e-01,
        2.5945500e-01, 2.6896000e-01, 2.7881300e-01, 2.8902600e-01,
        2.9961400e-01, 3.1059000e-01, 3.2196800e-01, 3.3376200e-01,
        3.4598900e-01, 3.5866400e-01, 3.7180300e-01, 3.8542300e-01,
        3.9954200e-01, 4.1417800e-01, 4.2935100e-01, 4.4507900e-01,
        4.6138400e-01, 4.7828600e-01, 4.9580700e-01, 5.0513300e-01,
        5.4282000e-01, 5.8331900e-01, 6.2683900e-01, 6.7360600e-01,
        7.2386200e-01, 7.7786800e-01, 8.3590300e-01, 8.9826800e-01,
        9.6528600e-01, 1.0373040e+00, 1.1146950e+00, 1.1978600e+00,
        1.2872300e+00, 1.3832670e+00, 1.4864700e+00, 1.5973720e+00,
        1.7165480e+00, 1.8446160e+00, 1.9822390e+00, 2.1301300e+00,
        2.2890540e+00, 2.4598350e+00, 2.6433580e+00, 2.8405730e+00,
        3.0525020e+00, 3.2802430e+00, 3.5249750e+00, 3.7879660e+00,
        4.0705780e+00, 4.3742740e+00, 4.7006290e+00, 5.0513330e+00,
        5.4282020e+00, 5.8331890e+00, 6.2683900e+00, 6.7360610e+00,
        7.2386240e+00, 7.7786820e+00, 8.3590330e+00, 8.9826820e+00,
        9.6528610e+00, 1.0373039e+01, 1.1146949e+01, 1.1978599e+01,
        1.2872296e+01, 1.3832670e+01, 1.4864696e+01, 1.5973718e+01,
        1.7165483e+01, 1.8446161e+01, 1.9822390e+01, 2.1301296e+01,
        2.2890539e+01, 2.4598352e+01, 2.6433582e+01]

# Empirical Transition Matrix:
transitions = [[0.99516818, 0.00483182],[0.02478551, 0.97521449]]