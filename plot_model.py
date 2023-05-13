def copy_model_and_plot_learning_curve():
    import pickle
    import matplotlib.pyplot as plt
    from collections import deque
    import os
    import datetime
    import shutil
    
    datetime_stamp = datetime.datetime.now().strftime('%y%m%d_%H%M')
    plot_path = f'results/{datetime_stamp}'
    
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    else:
        print(f'directory {plot_path} already exists')
        return
    
    shutil.copyfile(f'{brain_name}_scores.pickle', f'{plot_path}/scores.pickle')
    shutil.copyfile(f'{brain_name}_model_checkpoint.pickle', f'{plot_path}/model.pickle')

    smoothed = []
    queue = deque([], maxlen=10)
    for r in total_rewards:
        queue.append(r)
        smoothed.append(sum(queue)/len(queue))
    fig,ax = plt.subplots()
    ax.plot(smoothed)
    ax.set_xlabel('total episodes (across all agents)')
    plt.savefig(f'{plot_path}/learning_curve.png')
    plt.show()
