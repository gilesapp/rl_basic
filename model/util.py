import matplotlib.pyplot as plt
import numpy as np


def plot_train(plot_data, plot_name):
    if not isinstance(plot_data, np.ndarray):
        plot_data = np.array(plot_data)
    
    plt.figure(figsize=(8, 6))

    
    y_low = np.min(plot_data)
    y_high = np.max(plot_data)

    plt.plot(plot_data, c='royalblue')
    
    
    plt.title(f'RL Training reward - {plot_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.legend()
    plt.ylim([int(y_low*1.03), int(y_high*1.03)])
    plt.xlim([0, int(plot_data.shape[0]*1.03)])

    # x_tick = np.arange(0, plot_data.shape[0]+100, 100)
    # plt.xticks(x_tick, rotation=0)
    
    plt.grid(axis='y', linestyle='-', alpha=0.7)

    # plt.show()
    plt.savefig(f'fig/{plot_name}.png', bbox_inches='tight')
    plt.close()