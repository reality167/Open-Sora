num_frames = 8
image_size = (384, 1024)
transform_name = "resize_crop"
fps = 10

dtype = "bf16"
batch_size = 1
seed = 42
save_dir = "samples/vae_video"
cal_stats = True
log_stats_every = 100

dataset = dict(
    type="PhiVideoDataset",
    data_path=None,
    num_frames=num_frames,
    frame_interval=1,
    image_size=image_size,
    transform_name=transform_name,
    undistorted = False
)
num_samples = 100
num_workers = 4

# Define model
model = dict(
    type="OpenSoraVAE_V1_2",
    #from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    from_pretrained="outputs/vae_stage2/model.safetensors",
    micro_frame_size=None,
    micro_batch_size=4,
    cal_loss=True,
)

# loss weights
perceptual_loss_weight = 0.1  # use vgg is not None and more than 0
kl_loss_weight = 1e-6
