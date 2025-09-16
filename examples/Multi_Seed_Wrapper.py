#New File to run the examples multiple times without having to reset parameters manually
import subprocess
import numpy as np

file_name="unconstrained_branin_currin"
test_function_tag="BC"
multitask=False
partial_info=False
var_thresh=False

num_tests=1

if multitask is True:
    if partial_info is True:
        if var_thresh is True:
            type_tag="var_thresh"
        else:
            type_tag="rand"
    else:
        type_tag="joint"
else:
    type_tag="Model_list"


train_y_all=[]
train_x_all=[]
hv_all=[]
times_all=[]
pf_all=[]
if partial_info is True:
    train_y_filled_all=[]
    iteration_tracker_all=[]

for manual_seed in range(num_tests):
    print(f"##################################### RUNNING SEED {manual_seed} #####################################")
    subprocess.run(["python", "-m", f"examples.{file_name}", str(manual_seed)])
    train_y=np.load(f"../{test_function_tag}_{type_tag}_train_y.npy")
    train_x=np.load(f"../{test_function_tag}_{type_tag}_train_x.npy")
    hv=np.load(f"../{test_function_tag}_{type_tag}_hv.npy")
    times=np.load(f"../{test_function_tag}_{type_tag}_times.npy")
    pf=np.load(f"../{test_function_tag}_{type_tag}_pareto_front.npy")
    train_y_all.append(train_y)
    train_x_all.append(train_x)
    hv_all.append(hv)
    times_all.append(times)
    pf_all.append(pf)
    if partial_info is True:
        train_y_filled=np.load(f"../{test_function_tag}_{type_tag}_train_y_filled.npy")
        iteration_tracker=np.load(f"../{test_function_tag}_{type_tag}_iteration_tracker.npy")
        train_y_filled_all.append(train_y_filled)
        iteration_tracker_all.append(iteration_tracker)

np.save(f"../{test_function_tag}_{type_tag}_train_y_all.npy", np.array(train_y_all, dtype=object))
np.save(f"../{test_function_tag}_{type_tag}_train_x_all.npy", np.array(train_x_all, dtype=object))
np.save(f"../{test_function_tag}_{type_tag}_hv_all.npy", np.array(hv_all, dtype=object))
np.save(f"../{test_function_tag}_{type_tag}_times_all.npy", np.array(times_all, dtype=object))
np.save(f"../{test_function_tag}_{type_tag}_pareto_front_all.npy", np.array(pf_all, dtype=object))
if partial_info is True:
    np.save(f"../{test_function_tag}_{type_tag}_train_y_filled_all.npy", np.array(train_y_filled_all, dtype=object))
    np.save(f"../{test_function_tag}_{type_tag}_iteration_tracker_all.npy", np.array(iteration_tracker_all, dtype=object))

