import matplotlib.pyplot as plt

def format_axis(ax,xlabel='',ylabel='',title='',xticks=[],yticks=[],xlims=(),ylims=()):

    ax.spines[['right','top']].set_visible(False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    if len(xlims) > 0:
        ax.set_xlim(xlims)
    if len(ylims) > 0:
        ax.set_ylim(ylims)