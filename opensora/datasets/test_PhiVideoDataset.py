import os
from glob import glob
import glob
import numpy as np
import torch
import cv2
import pandas as pd

from opensora.registry import DATASETS
from utils import get_transforms_image, get_transforms_video, read_file, temporal_random_crop


class PhiVideoDataset(torch.utils.data.Dataset):
    """load video according to the json file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path=None,
        num_frames=50,
        frame_interval=1,
        image_size=(None, None),
        transform_name="center",
        undistorted=True,
        video_fps=20,
    ):
        if not isinstance(data_path,list):
            if os.path.isdir(data_path):
                data_path = glob.glob(data_path+'/*.json')
                data_path.sort()

        data = read_file(data_path)
        self.data = pd.DataFrame.from_dict(data)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }
        self.undistorted = undistorted
        self.video_fps = video_fps
        self.transform_name = transform_name

    def get_image_path(self, sample_dict, current_index):
        img_path = sample_dict["frames"][current_index]
        return img_path.replace('/data-preprocess','') if '/data-preprocess/' in img_path else img_path

    def getitem(self, index):
        sample = self.data.iloc[index]
        vframes_list = []

        for i in range(len(sample['frames'])):
            img_path = self.get_image_path(sample, i)
            if not os.path.exists(img_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            image = torch.from_numpy(image).permute(2, 0, 1)
            vframes_list.append(image)

        if len(vframes_list) == 0:
            raise ValueError(f"No valid frames found for index {index}")
        
        vframes = torch.stack(vframes_list)
        video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)
        transform = self.transforms["video"]
        video = transform(video)  # T C H W

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        ret = {
            "video": video,
            "fps": self.video_fps,
        }
        return ret

    def __getitem__(self, index):
        for _ in range(5):
            try:
                return self.getitem(index)
            except Exception as e:
                print(f"Error fetching data at index {index}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many errors while fetching data.")

    def __len__(self):
        return len(self.data)
    
def test_phi_video_dataset():
    # 设置测试参数
    data_path = "/mnt/cfs2/algorithm/zhuang.ma/cyz/Open-Sora-main/data"  # 替换为你的数据路径
    num_frames = 20
    frame_interval = 1
    image_size = (384, 1024)
    transform_name = "resize_crop"
    undistorted = True
    video_fps = 10
    
    # 创建数据集实例
    dataset = PhiVideoDataset(
        data_path=data_path,
        num_frames=num_frames,
        frame_interval=frame_interval,
        image_size=image_size,
        transform_name=transform_name,
        undistorted=undistorted,
        video_fps=video_fps
    )
    
    # 打印数据集大小
    print(f"Dataset size: {len(dataset)}")

    # 获取和打印前3个样本
    for i in range(50):
        sample = dataset[i]
        video_tensor = sample['video']  # 形状为 (C, T, H, W)
        fps = sample['fps']

        # 转换为 (T, H, W, C) 格式
        video_tensor = video_tensor.permute(1, 2, 3, 0)  # 使用 PyTorch 的 permute 方法
        
        # 获取视频的高度、宽度和帧数
        T, H, W, C = video_tensor.shape
        print("帧数", T)
        frame_count = T
        
        # 定义视频编码器和输出文件
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
        output_file = f'output_video_{i}.mp4'
        out = cv2.VideoWriter(output_file, fourcc, fps, (W, H))

        # 写入每一帧
        for t in range(frame_count):
            frame = video_tensor[t].numpy()  # 转换为 NumPy 数组
            frame = frame * 0.5 + 0.5
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)  # 确保像素值在 [0, 255] 范围内，并转换为 uint8 类型
            # frame = np.clip(frame, 0, 255).astype(np.uint8)
            out.write(frame)
        
        # 释放 VideoWriter 对象
        out.release()
        print(f"Video {i} saved as {output_file}")

if __name__ == "__main__":
    test_phi_video_dataset()