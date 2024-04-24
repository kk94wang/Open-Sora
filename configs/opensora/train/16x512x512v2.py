num_frames = 16
image_size = (512, 512)

# Define dataset
root = "path/to/video/clips/root"
data_path = "path/to/general/data/csv/file"
data_path2 = "path/to/finetune/data/csv"
use_image_transform = False
num_workers = 4

# Define acceleration
dtype = "bf16"
grad_checkpoint = False
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained="path/to/pretrained/512model.pth",
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=128,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs/test"
wandb = False

epochs = 10
log_every = 10
ckpt_every = 100
load = None

batch_size = 2
lr = 2e-5
grad_clip = 1.0
