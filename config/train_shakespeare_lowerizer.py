# uses the shakespeare_lower dataset, otherwise identical to train_shakespeare

wandb_log = True
wandb_project = 'shakespeare'
wandb_run_name='GPT'

dataset = 'shakespeare_lower'

eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 50

gradient_accumulation_steps = 1
batch_size = 64
block_size = 128 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
