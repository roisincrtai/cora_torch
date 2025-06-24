  python solve_optimal_rank.py --model_savepath=torchvision/resnet18 --num_bits=4 --init_norm_optimal_rank=0.9 --loss_norm_target_budget=0.05 --loss_gamma=1 --init_order=4 --learning_rate=0.01 --k=5.0 --solution_file=solutions/resnet18_more.pickle --epochs=5 --dataset_split=0.8

  python solve_optimal_rank.py --model_savepath=torchvision/resnet18 --num_bits=4 --init_norm_optimal_rank=0.1 --loss_norm_target_budget=0.05 --loss_gamma=1 --init_order=4 --learning_rate=0.01 --k=5.0 --solution_file=solutions/resnet18_init_0.1.pickle --epochs=5 



