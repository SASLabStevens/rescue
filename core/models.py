""" Copyright 2023, Rayan Bahrami, 
    Safe Autonomous Systems Lab (SAS Lab)
    Stevens Institute of Technology
    See LICENSE file for the license information.
"""

import numpy as np
import math
from scipy.io import loadmat, savemat
from scipy import linalg
from copy import deepcopy

from omegaconf import OmegaConf

from core.utils import (
    adjacency_distance_based,
    adjacency_nullify_connections,
    graph_partitioner,
    Adj_to_Lap,
    Lap_to_Adj,
    run_consensus,
    get_DoS,
)


from core.types import NetworkData, PhiData, AgentData

# import matplotlib as mpl
# import matplotlib.pyplot as plt


class OveralyNetwokManager:
    """creates a netwrok (adjacency matrix) and update it based on three criteria
    p-level = 0: connection availablity (Denial-of-service)
    p-level = 1: Update adjacent neighbors based on their physical proximity
    p-level = 2: Update connections by disconnecting from the detected malicious neighbors
    """

    def __init__(self, Adj, comm_radios=np.inf, dwell_times=[]):
        # alg_connectivity = np.sort(abs(linalg.eigvals(Lap)))[1]
        # Case: single-mode comm. topology
        if isinstance(Adj, np.ndarray):
            self.overlay_Adj = [Adj]
        # Case: multi-mode comm. topology
        elif isinstance(Adj, list) and all(isinstance(adj, np.ndarray) for adj in Adj):
            self.overlay_Adj = Adj
            if len(dwell_times) != 0:
                self.dwell_times = dwell_times
                if min(dwell_times) <= 0.2:
                    msg = """
                    Warning!: minimum dwell time for each mode must
                    be more than sample_time Ts by orders of magnitude
                    to avoid too many switches and meet Assumption 1 in the paper.
                    """
                    print(msg)
            else:
                self.dwell_times = [0.1] * len(Adj)
                print(
                    "Warning!: No dwell time was given for each mode! T_mode=0.5 is used for all."
                )
            self.PERIOD = sum(self.dwell_times)

        self.N = len(self.overlay_Adj[0])
        self.R = comm_radios
        self.data = NetworkData()

    def update(
        self,
        time: float,
        agents_states=None,
        DDoS=0,
        agents_detections={},
    ):
        """a method that check if the network config has changed. If so, it returns NetworkData(), and otherwise None.

        Args:
            time (float): time!
            agents_states (nd.array, optional): (2*N,d) array of the pos. and vel. of N agents with d as their state dim.. Defaults to None.
            DDoS (int, optional): 0 or 1 . Defaults to 0.
            malicious_agents_ID (set, optional): indices of detected malicious agents. Defaults to empty set.

        Returns:
            status as "Network_Down | No_Change | New_Mode" AND [Phi & updated NetworkData()]
        """

        # the set of IDs of all of the detected malicious agents
        malicious_agents_ID = set()
        for agent_ID, detection_info in agents_detections.items():
            idx = detection_info.get("malicious_neighbor_ID", [])
            malicious_agents_ID |= idx

        # p-level = 0: connection availablity
        if DDoS == 1:
            if time == 0.0:
                # mode increment | has no functionality
                self.data.mode += [1]
                self.data.t_mode += [time]
                self.data.comm_level += [0]
                self.data.Adj += [np.zeros((self.N, self.N))]
                self.data.Lap += [np.zeros((self.N, self.N))]
                # self.data.Lap += [Adj_to_Lap(self.data.Adj[-1])] # all zero
                # update phi info
                status = "Network_Down"
                self.phi_info = self.dispense_local_data(
                    agents_states, "New_Mode", malicious_agents_ID
                )  # different status because of initilization @ time == 0.0
                return status, self.phi_info
            else:  # time != 0.0
                if self.data.comm_level[-1] != 0:
                    # mode increment | has no functionality
                    self.data.mode += [self.data.mode[-1] + 1]
                    self.data.t_mode += [time]
                    self.data.comm_level += [0]
                    self.data.Adj += [np.zeros((self.N, self.N))]
                    self.data.Lap += [np.zeros((self.N, self.N))]
                    # self.data.Lap += [Adj_to_Lap(self.data.Adj[-1])] # all zero

                # update phi info
                # self.phi_info = [None] * self.N
                status = "Network_Down"
                self.phi_info = self.dispense_local_data(
                    agents_states, status, malicious_agents_ID
                )
                return status, self.phi_info

        # p-level = 1: Update Neighbors, distance_based
        if self.R < np.inf:
            Adj = adjacency_distance_based(
                P=agents_states[: self.N], comm_radios=self.R
            )
        else:
            if len(self.overlay_Adj) == 1:
                Adj = self.overlay_Adj[0].copy()
            else:
                # Determine the active mode based on periodic repetition
                elapsed_time = time % self.PERIOD
                accumulated_time = 0.0
                active_mode = None
                # Hint: slow if there are many modes
                for _adj, dwell_time in zip(self.overlay_Adj, self.dwell_times):
                    accumulated_time += dwell_time

                    if elapsed_time <= accumulated_time:
                        Adj = _adj.copy()
                        break

        # p-level = 2: Update network by disconnecting from detected malicious_agents
        if len(agents_detections) != 0:
            for agent_ID, detection_info in agents_detections.items():
                neighbors_ID = list(detection_info["malicious_neighbor_ID"])
                Adj[agent_ID, neighbors_ID] = 0
                Adj[neighbors_ID, agent_ID] = 0
        # ============= legacy :) ==================================================
        # if len(malicious_agents_ID) != 0:
        #     Adj = adjacency_nullify_connections(Adj, index_set=malicious_agents_ID)
        # ===========================================================================

        # check if there is any change w.r.t the previous mode
        if time == 0.0:
            # mode increment | has no functionality
            self.data.mode += [1]
            self.data.t_mode += [time]
            self.data.comm_level += [1] if len(malicious_agents_ID) == 0 else [2]
            self.data.Adj += [Adj]
            self.data.Lap += [Adj_to_Lap(self.data.Adj[-1])]
            # update phi info
            status = "New_Mode"
            self.phi_info = self.dispense_local_data(
                agents_states, status, malicious_agents_ID
            )
            return status, self.phi_info
        else:  # time != 0.0
            if np.any(Adj != self.data.Adj[-1]):
                # mode increment | has no functionality
                self.data.mode += [self.data.mode[-1] + 1]
                self.data.t_mode += [time]
                self.data.comm_level += [1] if len(malicious_agents_ID) == 0 else [2]
                self.data.Adj += [Adj]
                self.data.Lap += [Adj_to_Lap(self.data.Adj[-1])]
                # update phi info
                status = "New_Mode"
                self.phi_info = self.dispense_local_data(
                    agents_states, status, malicious_agents_ID
                )
                return status, self.phi_info
            else:
                status = "No_Change"
                self.phi_info = self.dispense_local_data(
                    agents_states, status, malicious_agents_ID
                )
                return status, self.phi_info

    def dispense_local_data(self, agents_states, status, malicious_agents_ID):
        phi_info = []

        if status == "Network_Down":
            for id in range(self.N):
                phi = PhiData()
                phi.agent_ID = id
                phi.xp_set_I = agents_states[
                    np.hstack((id, phi.oneHops_ID, phi.twoHops_ID)).astype(int)
                ]  # ==  agents_states[id]
                phi.xv_ = agents_states[self.N + id]
                phi_info += [phi]
        else:
            for id in range(self.N):
                phi = PhiData()
                phi.agent_ID = id
                if id in malicious_agents_ID:
                    phi_info += [phi]
                    continue

                (
                    phi.oneHops_ID,
                    phi.twoHops_ID,
                    _,
                    phi.Adj_oneHop,
                    phi.Adj_twoHop,
                ) = graph_partitioner(self.data.Adj[-1], node=id)

                phi.xp_set_I = agents_states[
                    np.hstack((id, phi.oneHops_ID, phi.twoHops_ID)).astype(int)
                ]
                phi.xv_ = agents_states[self.N + id]
                phi_info += [phi]
        return phi_info


# def threshold_fn(k, L, eps, t):
#     return k*np.exp(-L*t) + eps


class LocalObs:
    def __init__(self, id, cfg, threshold_fn, xp0=0, xv0=0):
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.Ts = cfg.Ts
        self.threshold = threshold_fn
        self.hv = cfg.hv  # Obs. gains
        self.Hp = self.alpha * self.hv
        # =========
        self.id = id
        self.set_I = np.array([self.id])  # the set of included two-hop neighbors
        self.n = len(self.set_I)

        self.A_HC = np.eye(2 * self.n) + self.Ts * np.block(
            [
                [np.zeros((self.n, self.n)), np.eye(self.n)],
                [
                    -self.alpha * 0 - self.Hp,
                    -self.gamma * np.eye(self.n)
                    - self.hv
                    * np.float16(
                        np.outer(np.arange(self.n) == 0, np.arange(self.n) == 0)
                    ),
                ],
            ]
        )
        self.x_hat = np.hstack((xp0, xv0))
        self.data = []  # to log obs data
        self.detections = []  # to log time-stamped detections of malicious neighbors

    def update(self, time, phi, status):
        set_I_curr = np.hstack((phi.agent_ID, phi.oneHops_ID, phi.twoHops_ID)).astype(int)  # fmt: skip

        if len(set_I_curr) == self.n:
            New_Neighbor = np.any(set_I_curr != self.set_I)
        else:
            New_Neighbor = True

        if time == 0.0 or New_Neighbor == True:
            # reset and reconfigure the Obs.
            self.set_I = set_I_curr
            self.n = len(self.set_I)
            self.x_hat = np.hstack((phi.xp_set_I, phi.xv_, np.zeros(self.n - 1)))

            if len(phi.oneHops_ID) == 0:  # no neighbor ~ DoS or far away
                Lap = np.zeros((self.n, self.n))
                self.Hp = self.alpha * self.hv  # Obs. gain
            else:
                Lap = Adj_to_Lap(phi.Adj_twoHop)
                # this is to increase (by maximum self.hv units) the diagonal elements of the local laplacian matrix
                self.Hp = (
                    (max(np.diag(Lap)) + self.hv) * np.eye(self.n)
                    - np.diag(np.diag(Lap))
                ) * self.alpha

            a_hc = np.block(
                [
                    [np.zeros((self.n, self.n)), np.eye(self.n)],
                    [
                        -self.alpha * Lap - self.Hp,
                        -self.gamma * np.eye(self.n)
                        - self.hv
                        * np.float16(
                            np.outer(np.arange(self.n) == 0, np.arange(self.n) == 0)
                        ),
                    ],
                ]
            )
            self.A_HC = np.eye(2 * self.n) + self.Ts * a_hc

        # update the estimation - a posteriori
        # xhat_next = (A-HC) * xhat + H*y
        Hy = np.hstack(
            (
                np.zeros(self.n),
                np.dot(self.Hp, phi.xp_set_I)
                + (self.hv * np.float16(np.arange(self.n) == 0)) * phi.xv_,
            )
        )

        self.x_hat = self.A_HC.dot(self.x_hat) + self.Ts * Hy

        res = np.hstack((phi.xp_set_I, phi.xv_)) - self.x_hat[: self.n + 1]

        return self.x_hat[: self.n], self.x_hat[self.n :], res, New_Neighbor

    def update_detect_and_logdata(self, time, phi, status):
        xp_hat, xv_hat, res, New_Neighbor = self.update(time, phi, status)
        if time == 0.0 or New_Neighbor == True:
            data = AgentData()
            data.xp_hat += [xp_hat]
            data.xv_hat += [xv_hat]
            data.res += [res]
            data.t_span += [time]
            data.agent_ID = phi.agent_ID
            data.oneHops_ID = phi.oneHops_ID
            data.twoHops_ID = phi.twoHops_ID
            data.Adj_oneHop = phi.Adj_oneHop
            data.Adj_twoHop = phi.Adj_twoHop
            self.data += [data]  # store the new track
        else:
            self.data[-1].xp_hat += [xp_hat]
            self.data[-1].xv_hat += [xv_hat]
            self.data[-1].res += [res]
            self.data[-1].t_span += [time]
        # ==================================================
        # attack detection | hypothesis testing on residuals
        n_ = len(phi.oneHops_ID)
        if n_ != 0:
            # check one-hops => |r_[i,j]| >= eps, for j in oneHops_ID
            indices = np.where(abs(res[1 : n_ + 1]) > self.threshold(time))[0]
            if len(indices) != 0:
                detect = {"t_d": time, "malicious_neighbor_ID": phi.oneHops_ID[indices]}
                self.detections += [detect]
                # print for debuging | monitoring
                print(f"coop. agent {phi.agent_ID} at t={time}, detected neigh. {phi.oneHops_ID[indices]}")  # fmt: skip
                # print(f"detected_malicious_neighbors: {phi.oneHops_ID[indices]}")
                # print(f"residuals:{res}")
                # print(f"indices: {indices + 1}")
                # print(f"oneHops_ID: {phi.oneHops_ID}")


def update_detections(agents_detections, observers, coop_agents):
    """collects the list of N agents' detected adversarial neighbors, if any

    Args:{"t_d": time, "malicious_neighbor_ID": phi.oneHops_ID[indices]}
        agents_detections (dict): {id: "malicious_neighbor_ID"}
        observers (list): the list of observers (instances of LocalObs class)
        coop_agents (array): list of id's of cooperative agents

    Returns:
        dict: updated agents_detections
    """
    for id in coop_agents:
        set_det = set().union(
            *[set(det["malicious_neighbor_ID"]) for det in observers[id].detections]
        )
        if id in agents_detections:
            # Update existing entry
            agents_detections[id]["malicious_neighbor_ID"] |= set_det
        else:
            # Add a new entry
            agents_detections[id] = {"malicious_neighbor_ID": set_det}
    return agents_detections
