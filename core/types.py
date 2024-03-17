""" Copyright 2023, Rayan Bahrami, 
    Safe Autonomous Systems Lab (SAS Lab)
    Stevens Institute of Technology
    See LICENSE file for the license information.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class NetworkData:
    mode: list[int] = field(default_factory=list)
    t_mode: list[float] = field(default_factory=list)
    comm_level: list[int] = field(default_factory=list)
    Adj: list[np.ndarray] = field(default_factory=list)
    Lap: list[np.ndarray] = field(default_factory=list)


@dataclass
class PhiData:
    """Local message for each agent"""

    agent_ID: int = field(default_factory=int)
    oneHops_ID: list[int] = field(default_factory=list)
    twoHops_ID: list[int] = field(default_factory=list)
    Adj_oneHop: np.ndarray = field(default=np.array([]))
    Adj_twoHop: np.ndarray = field(default=np.array([]))
    xp_set_I: np.ndarray = field(default=np.array([]))  # pos. states - 2hop - included
    xv_: np.ndarray = field(default=np.array([]))  # vel. state - i-th agent


@dataclass
class AgentData:
    agent_ID: int = field(default_factory=int)
    oneHops_ID: list[int] = field(default_factory=list)
    twoHops_ID: list[int] = field(default_factory=list)
    Adj_oneHop: np.ndarray = field(default=np.array([]))
    Adj_twoHop: np.ndarray = field(default=np.array([]))
    xp_hat: list[np.ndarray] = field(default_factory=list)
    xv_hat: list[np.ndarray] = field(default_factory=list)
    res: list[np.ndarray] = field(default_factory=list)
    t_span: list[float] = field(default_factory=list)
