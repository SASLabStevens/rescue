import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


FONT_SIZE = 14
mpl.rcParams["axes.labelsize"] = FONT_SIZE
mpl.rcParams["xtick.labelsize"] = FONT_SIZE - 1
mpl.rcParams["ytick.labelsize"] = FONT_SIZE - 1
mpl.rcParams["legend.fontsize"] = 10

mpl.rcParams["font.family"] = "serif"


def plot_pos(
    P,
    PO=None,
    P_msr=None,
    Z1=None,
    coop_agents_ID=[],
    adv_agents_ID=[],
    T=None,
    DoS=None,
    fig_size=(6, 3),
    ylim1=None,
    titles=[],
    yticks1=[0, 4, 8],
    titles_=False,
    lg_fontsize=11,
    bbox_to_anchor=(0, 0.95, 1, 0.2),
):
    fig, axes = plt.subplots(
        1, 2, sharex=True, sharey=True, figsize=fig_size, layout="constrained"
    )
    # layout='constrained' # or # tight_layout=True
    # plot DoS
    if DoS is not None:
        gain = max(P[0, :]) * 1.5
        axes[0].fill_between(T, gain * DoS, color="#dbd7d7")
        axes[1].fill_between(T, gain * DoS, color="#dbd7d7")
        axes[0].fill_between(T, -gain * DoS, color="#dbd7d7")
        axes[1].fill_between(T, -gain * DoS, color="#dbd7d7")

    if PO is not None:
        axes[0].plot(T, PO, alpha=0.95, color="tab:gray")
    axes[0].plot(T, P[:, coop_agents_ID], "k")
    axes[0].plot(T, P[:, adv_agents_ID], "r")
    axes[0].axhline(
        y=np.average(P[0, :]), xmin=0.05, xmax=0.955, linestyle="--", color="g"
    )
    if Z1 is not None:
        axes[1].plot(T, Z1, alpha=0.95, color="tab:gray")
    if P_msr is not None:
        axes[1].plot(T, P_msr[:, coop_agents_ID], "k")
        axes[1].plot(T, P_msr[:, adv_agents_ID], "r")
        axes[1].axhline(
            y=np.average(P_msr[0, :]), xmin=0.05, xmax=0.955, linestyle="--", color="g"
        )
        axes[1].axhline(y=max(P_msr[0, :]), xmin=0.05, xmax=0.955, color="g")
        axes[1].axhline(y=min(P_msr[0, :]), xmin=0.05, xmax=0.955, color="g")

    if ylim1 is not None:
        axes[0].set_ylim(ylim1)
        # axes[1].set_ylim(ylim1)

    if yticks1 is not None:
        axes[0].set_yticks(yticks1)
        # axes[1].set_yticks(yticks1)

    axes[0].set_xlabel(r"Time [sec]")
    axes[1].set_xlabel(r"Time [sec]")
    axes[0].set_ylabel(r"Position $\mathbf{p}$($\mathit{t}$)")

    for i, ax in enumerate(axes.flatten()):
        if titles_:
            ax.set_title(rf"{titles[i]}")
        else:
            ax.text(
                0.95,
                0.2,
                rf"{titles[i]}",
                ha="right",
                va="bottom",
                fontsize=12,
                transform=ax.transAxes,
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round"),
            )

    legend_lines = [
        Line2D([0], [0], color="k", lw=2, label=r"Coop. Agents"),
        Line2D([0], [0], color="r", lw=2, label=r"Malicious Agents"),
        Line2D([0], [0], color="tab:gray", alpha=0.95, lw=2, label=r"w/o Malicious Agents"),
        Line2D([0], [0], color="green", linestyle="--", lw=2, label=r"Avg. of I.C."),
        Line2D([0], [0], color="green", lw=2, label=r"Safety Interval")]  # fmt: skip

    legend_patches = [mpatches.Patch(color="#dbd7d7", label=r"DoS")]

    fig.legend(
        handles=legend_lines + legend_patches,
        loc="upper center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=6,
        mode="expand",
        fontsize=lg_fontsize,
    )


def plot_states(
    P,
    V,
    PO=None,
    VO=None,
    P_msr=None,
    V_msr=None,
    Z1=None,
    Z2=None,
    coop_agents_ID=[],
    adv_agents_ID=[],
    T=None,
    DoS=None,
    fig_size=(6, 3),
    ylim1=None,
    ylim2=None,
    titles=[],
    yticks1=[0, 4, 8],
    yticks2=None,
    lg_fontsize=11,
    bbox_to_anchor=(0, 0.95, 1, 0.2),
):
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=fig_size, layout="constrained")
    # layout='constrained' # or # tight_layout=True
    # plot DoS
    if DoS is not None:
        gain = max(P[0, :]) * 1.5
        axes[0, 0].fill_between(T, gain * DoS, color="#dbd7d7")
        axes[0, 1].fill_between(T, gain * DoS, color="#dbd7d7")
        axes[1, 0].fill_between(T, gain * DoS, color="#dbd7d7")
        axes[1, 1].fill_between(T, gain * DoS, color="#dbd7d7")
        axes[0, 0].fill_between(T, -gain * DoS, color="#dbd7d7")
        axes[0, 1].fill_between(T, -gain * DoS, color="#dbd7d7")
        axes[1, 0].fill_between(T, -gain * DoS, color="#dbd7d7")
        axes[1, 1].fill_between(T, -gain * DoS, color="#dbd7d7")

    if len(titles) != 0:
        axes[0, 0].set_title(rf"{titles[0]}")
        axes[0, 1].set_title(rf"{titles[1]}")

    if PO is not None:
        axes[0, 0].plot(T, PO, alpha=0.95, color="tab:gray")
    axes[0, 0].plot(T, P[:, coop_agents_ID], "k")
    axes[0, 0].plot(T, P[:, adv_agents_ID], "r")
    axes[0, 0].axhline(
        y=np.average(P[0, :]), xmin=0.05, xmax=0.955, linestyle="--", color="g"
    )
    if Z1 is not None:
        axes[0, 1].plot(T, Z1, alpha=0.95, color="tab:gray")
    if P_msr is not None:
        axes[0, 1].plot(T, P_msr[:, coop_agents_ID], "k")
        axes[0, 1].plot(T, P_msr[:, adv_agents_ID], "r")
        axes[0, 1].axhline(
            y=np.average(P_msr[0, :]), xmin=0.05, xmax=0.955, linestyle="--", color="g"
        )
        axes[0, 1].axhline(y=max(P_msr[0, :]), xmin=0.05, xmax=0.955, color="g")
        axes[0, 1].axhline(y=min(P_msr[0, :]), xmin=0.05, xmax=0.955, color="g")

    if ylim1 is not None:
        axes[0, 0].set_ylim(ylim1)
        axes[0, 1].set_ylim(ylim1)

    if yticks1 is not None:
        axes[0, 0].set_yticks(yticks1)
        axes[0, 1].set_yticks(yticks1)

    axes[0, 0].set_ylabel(r"Position $\mathbf{p}$($\mathit{t}$)")
    axes[1, 0].set_ylabel(r"Velocity $\mathbf{v}$($\mathit{t}$)")

    if VO is not None:
        axes[1, 0].plot(T, VO, alpha=0.95, color="tab:gray")
    axes[1, 0].plot(T, V[:, coop_agents_ID], "--k")
    axes[1, 0].plot(T, V[:, adv_agents_ID], "--r")
    if Z2 is not None:
        axes[1, 1].plot(T, Z2, alpha=0.95, color="tab:gray")
    if P_msr is not None:
        axes[1, 1].plot(T, V_msr[:, coop_agents_ID], "--k")
        axes[1, 1].plot(T, V_msr[:, adv_agents_ID], "--r")

    if ylim2 is not None:
        axes[1, 0].set_ylim(ylim2)
        axes[1, 1].set_ylim(ylim2)
    if yticks2 is not None:
        axes[1, 0].set_yticks(yticks2)
        axes[1, 1].set_yticks(yticks2)

    axes[-1, -1].set_xlabel(r"Time [sec]")
    axes[-1, -2].set_xlabel(r"Time [sec]")

    legend_lines = [
        Line2D([0], [0], color="k", lw=2, label=r"Coop. Agents"),
        Line2D([0], [0], color="r", lw=2, label=r"Malicious Agents"),
        Line2D([0], [0], color="tab:gray", alpha=0.95, lw=2, label=r"w/o Malicious Agents"),
        Line2D([0], [0], color="green", linestyle="--", lw=2, label=r"Avg. of I.C."),
        Line2D([0], [0], color="green", lw=2, label=r"Safety Interval")]  # fmt: skip

    legend_patches = [mpatches.Patch(color="#dbd7d7", label=r"DoS")]

    fig.legend(
        handles=legend_lines + legend_patches,
        loc="upper center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=6,
        mode="expand",
        fontsize=lg_fontsize,
    )


def plot_selective_residuals(
    observers_data,
    coop_agent_ID,
    adv_agents_ID=[],
    T=None,
    DoS=None,
    num_modes=None,
    fig_size=(6, 3),
    thre=None,
    ylim=None,
    yticks=None,
    eq_label=False,
    bbox1_to_anchor=(0, 0.4, 1, 0.2),
    bbox2_to_anchor=(0, 0.65, 1, 0.2)
):
    from copy import deepcopy

    N_a = len(adv_agents_ID)
    obs = deepcopy(observers_data[coop_agent_ID])

    fig, ax = plt.subplots(figsize=fig_size, layout="constrained")
    # layout='constrained' # or # tight_layout=True
    # fig.suptitle(r"residuals", y=1.04)

    # plot DoS
    if DoS is not None:
        gain = 1 if ylim is None else ylim[1]
        ax.fill_between(T, gain * DoS, where=(DoS > 0), color="#dbd7d7")

    H1s = set()  # collective 1-hop neighbors
    for m in range(len(obs.data)):
        H1s |= set(obs.data[m].oneHops_ID)
    H1s = sorted(H1s)
    print(f"collective 1-hops:{H1s}")

    # SS = sorted(set(H1s) | set(adv_agents_ID))
    # print(f"combined_agents:{SS}")

    ccmap = mpl.colors.ListedColormap(
        [
            color
            for i, color in enumerate(plt.cm.tab10.colors)
            if i not in {6, 7, 14, 15}  # Remove reds and grays
        ]
    )

    color_mapping = {
        elem: (
            plt.cm.gist_heat(elem / (max(adv_agents_ID) + 3))
            # "tab:red"
            if elem in adv_agents_ID
            else ccmap(i / len(H1s))
        )
        for i, elem in enumerate(H1s)
    }

    # plot:  res_1hops = res[:, 1 : len(oneHops_ID) + 1]
    for m in range(len(obs.data)):
        H1 = obs.data[m].oneHops_ID
        t = np.array(obs.data[m].t_span)
        res = np.array(obs.data[m].res)
        for j in range(len(H1)):  # one hops
            if H1[j] in adv_agents_ID and (H1[j] % 2) == 0:
                ls = "--"
            else:
                ls = "-"
            ax.plot(t, abs(res[:, j + 1]), linestyle=ls, color=color_mapping[H1[j]])
        # ax.axvline(x=t[0], linewidth=1, color="k")
        # ax.set_ylabel(rf"Res. {coop_agent_ID+1}")

        if eq_label:
            ax.set_ylabel(
                rf"$|\mathbf{{r}}^{{{coop_agent_ID},j}}_{{\sigma}}|, \,\; j \in \mathcal{{N}}^{{{coop_agent_ID}(1)}}_{{\sigma}}$"
            )
        else:
            ax.set_ylabel(rf"1-hop res. of agent {coop_agent_ID+1}")
        if callable(thre):
            ax.plot(t, thre(t), linewidth=1.4, color="k")

    if thre is not None and not callable(thre):
        ax.axhline(y=thre, linewidth=1.4, color="k")

    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.set_xlabel(r"Time [sec]")

    legend_lines = [
        Line2D(
            [0],
            [0],
            color=color_mapping[j],
            lw=2,
            linestyle="--" if j in adv_agents_ID and (j % 2) == 0 else "-",
            label=(f"Mali.: {j+1}" if j in adv_agents_ID else f"Coop.: {j+1}"),
        )
        for j in H1s
    ]

    legend_line2 = [Line2D([0], [0], color="k", lw=2, label=r"threshold $\epsilon$")]

    # legend_patches_2 = [
    #     mpatches.Patch(color=colors[i], label=f"agent {i+1}") for i in range(N)
    # ]

    legend_patches = [mpatches.Patch(color="#dbd7d7", label=f"DoS")]

    # fig.legend(
    #     handles=legend_lines + legend_line2 + legend_patches,
    #     loc="upper right",
    #     bbox_to_anchor=(0, 1.01, 1.01, 0.2),
    #     ncol=(len(H1s) + 2) // 2,
    #     mode="expand",
    # )

    fig.legend(
        handles=legend_lines,
        loc="right",
        bbox_to_anchor=bbox1_to_anchor,
        ncol=(len(H1s) + 2) // 3,
        # mode="expand",
    )
    fig.legend(
        handles=legend_line2 + legend_patches,
        loc="right",
        bbox_to_anchor=bbox2_to_anchor,
        ncol=1,
        # mode="expand",
    )
