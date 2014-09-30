from itertools import product


unif_single_cell(K, radius, r_ref=1, path_loss_exp=3):
    angles = np.r_[0:K] * 2*np.pi/K

    locs = radius * (np.cos(angles) + 1j*np.sin(angles));

    dist_BS = [np.abs(locs[k]) for k in range(K)]

    dist_UE = [[np.abs(locs[i]-locs[j])
                for i in range(K)] for j in range(K)]

    gains = {}
    gains{'D2D'} = np.zeros((K, K))

    for (i, j) in product(range(K), range(K)):
        gains[i, j] = (r_ref/dist_UE[i][j])**path_loss_exp

        if i == j:
            gains[i, j] = 1
