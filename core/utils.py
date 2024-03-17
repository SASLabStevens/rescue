""" Copyright 2023, Rayan Bahrami, 
    Safe Autonomous Systems Lab (SAS Lab)
    Stevens Institute of Technology
    See LICENSE file for the license information.
"""

import numpy as np
from scipy import linalg
from scipy.interpolate import interp1d


def Adj_to_Lap(Adj):
    return np.diag(np.sum(Adj, 1)) - Adj


def Lap_to_Adj(Lap):
    return np.diag(np.diag(Lap)) - Lap


def adjacency_distance_based(P, comm_radios=np.inf, undirected=True):
    # P.shape == (N, d), N agents with d as the state dim =[1,2,3] == (x,y,z)
    if P.ndim == 2:
        N, d = P.shape
    elif P.ndim == 1:
        N = P.shape[0]  # numAgents
        P = P.reshape(N, 1)  # vectorize
        d = 1
    else:
        raise ValueError(
            "P (agents_pos) must be a 2D array of N agents of dimension d!"
        )

    Adj_matrix = np.zeros((N, N))

    # undirected graph <==> symmetric adj. matrix
    for i in range(N):
        for j in range(i + 1, N):
            if linalg.norm(P[i, :] - P[j, :]) <= comm_radios:
                Adj_matrix[i, j] = 1
                Adj_matrix[j, i] = 1

    return Adj_matrix


def adjacency_nullify_connections(matrix, index_set: set = {}, undirected=True):
    """
    Update the matrix by setting (j, i) and (i, j) elements to zero for all j's in the given set.

    Parameters:
    - matrix: Input Adjacency matrix
    - index_set: Set of indices for which to set (j, i) and (i, j) elements to zero
     (elements must be int and in range (0, N=dimMatrix))

    Returns:
    - Updated matrix
    """
    N = matrix.shape[0]  # numAgents
    set_B = np.array(list(index_set))
    set_A = np.setdiff1d(np.arange(N), set_B)
    # Set (j, i) and (i, j) elements to zero
    matrix[set_B, set_A[:, None]] = 0
    matrix[set_A[:, None], set_B] = 0
    # set_A = np.array(list(set(range(len(matrix))).difference(index_set)))
    # set_B = np.array(list(index_set))
    # print(set_A[:, None], set_B)
    # matrix[set_B, set_A[:, None]] = 0
    # matrix[set_A[:, None], set_B] = 0
    return matrix


def graph_partitioner(Adj_matrix, node):
    N = len(Adj_matrix)
    ALL_NODES = np.arange(N)
    # find the 1-hop and 2-hop neighbors of "NODE" &
    # sort them in the ascending order of their indices
    oneHops = np.sort(np.where(Adj_matrix[node, ALL_NODES] == 1)[0])
    twoHops = np.unique(
        np.sort(np.where(Adj_matrix[oneHops][:, ALL_NODES] == 1)[1], axis=0)
    )
    twoHops = np.setdiff1d(twoHops, np.hstack((node, oneHops)))
    theRest = np.setdiff1d(ALL_NODES, np.hstack((node, oneHops, twoHops)))
    # populate the reordered adjacency matrix
    # Adj_reordered = Adj_matrix(
    #     np.hstack(node, oneHops, twoHops, theRest),
    #     np.hstack((node, oneHops, twoHops, theRest)),
    # )
    # populate the adjacency matrix of the 1-hop proximity graph :
    # it does not include the edges between the 1-hop neighbors
    Adj_1hop = Adj_matrix.copy()
    Adj_1hop[oneHops[:, None], oneHops] = 0
    Adj_1hop = Adj_1hop[np.hstack((node, oneHops))][:, np.hstack((node, oneHops))]
    # populate the adjacency matrix of the 2-hop proximity graph :
    # it does not include the edges between the 2-hop neighbors
    Adj_2hop = Adj_matrix.copy()
    Adj_2hop[twoHops[:, None], twoHops] = 0
    Adj_2hop = Adj_2hop[np.hstack((node, oneHops, twoHops))][
        :, np.hstack((node, oneHops, twoHops))
    ]

    return oneHops, twoHops, theRest, Adj_1hop, Adj_2hop


def get_PE_algebraic_connectivity(comm_data, T, T_end, Ts, coop_ID=[], weighted=True):
    from scipy import linalg

    # comm_data = network_manager.data | see dataclass NetworkData() in types.py
    Adj = comm_data.Adj.copy()
    T_sw = [*comm_data.t_mode.copy(), T_end]
    # Tspan = np.linspace(0, T_end, np.int64(T_end / Ts) + 1)

    # interpolate Adj matrices over 0:Ts:T_end
    Adj_list = []
    for k in range(len(T_sw) - 1):
        i, j = np.int16(T_sw[k] / Ts), np.int16(T_sw[k + 1] / Ts)
        Adj_list[i:j] = [Adj[k]] * (j - i + 1)

    coop_ID = np.arange(len(Adj[0])) if coop_ID == [] else coop_ID
    iAdj_list = [adj[coop_ID][:, coop_ID] for adj in Adj_list]

    # Initialize the integrated matrices list
    integ_Adj = []
    integ_iAdj = []
    lambda2, bar_lambda2 = [], []
    mu, mu_bar = [], []

    # Perform the integration | summation
    N = len(Adj_list)  # number of Adj. matrices
    for i in range(N):
        lambda2 += [np.sort(abs(linalg.eigvals(Adj_to_Lap(Adj_list[i]))))[1]]
        bar_lambda2 += [np.sort(abs(linalg.eigvals(Adj_to_Lap(iAdj_list[i]))))[1]]

        if i < T:
            T_window = Adj_list[: i + 1]
            T_windo_ = iAdj_list[: i + 1]
        else:
            T_window = Adj_list[i - (T - 1) : i + 1]
            T_windo_ = iAdj_list[i - (T - 1) : i + 1]

        # the binary Adj or weighted Adj
        _avg_adj = (
            np.sum(T_window, axis=0) / T
            if weighted
            else ((np.sum(T_window, axis=0) / T) != 0).astype(int)
        )
        _avg_iadj = (
            np.sum(T_windo_, axis=0) / T
            if weighted
            else ((np.sum(T_windo_, axis=0) / T) != 0).astype(int)
        )

        integ_Adj += [_avg_adj]
        integ_iAdj += [_avg_iadj]

        Lap = Adj_to_Lap(integ_Adj[-1])
        iLap = Adj_to_Lap(integ_iAdj[-1])

        # algebraic connectivity in the PE sense
        mu += [np.sort(abs(linalg.eigvals(Lap)))[1]]
        mu_bar += [np.sort(abs(linalg.eigvals(iLap)))[1]]

    return mu, integ_Adj, mu_bar, integ_iAdj, lambda2, bar_lambda2


def get_DoS(p, num_dos, t_s, t_end, seed=None):
    # independent Bernoulli
    # n=1, p are num of binary trials for DoS, success probability of each trial
    seed = 703093 if seed is None else seed
    np.random.seed(seed)

    dos_dur = t_end / num_dos  # 1/dos_freq must be 0.1, 0.01, 0.001

    # independent Bernoulli
    dos_ = np.random.binomial(n=1, p=p, size=num_dos + 1)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 2.5))
    # t = np.arange(0, t_end + dos_dur, dos_dur)
    t = np.linspace(0, t_end, num_dos + 1)

    # if dos_dur == t_s:
    num_itr = np.int64(t_end / t_s)
    if num_dos == num_itr:
        DoS = dos_
    else:
        # t_span = np.arange(0, t_end, t_s)
        t_span = np.linspace(0, t_end, num_itr + 1)
        f_dos = interp1d(t, dos_, kind="previous")
        DoS = f_dos(t_span)

        plt.fill_between(t_span, 2 * DoS, where=(DoS > 0), color="C1", alpha=0.3)

    plt.fill_between(t, dos_, step="post")
    plt.xlabel("Time [sec]")

    return DoS


def run_consensus(cfg, adj=None, T_s=None, t_end=None, x0=None):
    import matplotlib.pyplot as plt

    Adj = np.array(cfg.Adj, np.float32) if adj is None else adj
    Lap = np.diag(np.sum(Adj, 1)) - Adj
    N = len(Adj)
    Ts = cfg.Ts if T_s is None else T_s
    alpha = cfg.alpha
    gamma = cfg.gamma
    px0 = np.arange(1, N + 1) if x0 is None else x0
    # np.array(cfg.IC.px)
    vx0 = 0 * px0
    # np.array(cfg.IC.vx)
    T_end = cfg.t_end if t_end is None else t_end

    # ==== initilization =======
    X = np.hstack((px0, vx0))
    X_data = []

    A = np.eye(2 * N) + Ts * np.block(
        [
            [np.zeros((N, N)), np.eye(N)],
            [-alpha * Lap, -gamma * np.eye(N)],
        ]
    )
    k = 0
    while (k * Ts) < t_end:
        t = k * Ts
        k += 1  # time increment
        X = A.dot(X)
        X_data += [X]

    t_span = np.arange(0, T_end, Ts)
    plt.plot(t_span, X_data)
    plt.xlabel("Time [sec]")
    plt.ylabel("pos. and vel.")
    return t_span, X_data


def DPMSR_update_neighbors(x, F, Adj, malicious_agents=[], threshold=0):
    """
    Update the adjacency matrix based on DP-MSR algorithem.
    Refs. [Dibaji, S. M., & Ishii, H., Elsevier 2015, 2017].

    Parameters:
    - x: agents' position states (N,)
    - F: maximum number of tolerable non-cooperative (malicious) agents
    - adj: adjacency matrix
    - threshold: the upper bound on the maximum distance between agents
    - malicious_agents: list of malicious agents indices

    Returns:
    - adj: updated adjacency matrix based on DPMSR
    """
    adj = Adj.copy()
    # Remove F values "strictly" larger and smaller than x_i[k]
    for i in range(x.shape[0]):
        if i in malicious_agents:
            continue

        neighbors_indices = np.where(adj[i, :])[0]
        msgs_w_ids = np.column_stack((x[neighbors_indices], neighbors_indices))
        order_ids = np.argsort(msgs_w_ids[:, 0])[::-1]
        sorted_msgs_w_ids = msgs_w_ids[order_ids, :]

        # Find the ID of x_j[k]'s "strictly" larger and smaller than x_i[k]
        mask_up = sorted_msgs_w_ids[:, 0] - x[i] > threshold
        mask_down = sorted_msgs_w_ids[:, 0] - x[i] < -threshold

        outliers_up = sorted_msgs_w_ids[mask_up, :]
        outliers_down = sorted_msgs_w_ids[mask_down, :]
        # Drop the first F x_j's larger than that x_i
        if outliers_up.shape[0] >= F:
            # print(adj[i, outliers_up[:1, 1]])
            adj[i, outliers_up[:F, 1].astype(int)] = 0
        else:
            adj[i, outliers_up[:, 1].astype(int)] = 0

        # Drop the last F x_j's smaller than that x_i
        if outliers_down.shape[0] >= F:
            adj[i, outliers_down[-F:, 1].astype(int)] = 0
        else:
            adj[i, outliers_down[:, 1].astype(int)] = 0

    return adj
