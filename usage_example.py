# 파일 처리
import os, shutil 
from pathlib import Path

# 시각화 
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 깨짐 방지 설정
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지 설정

# 반복 처리
from tqdm.auto import tqdm  # 진행율 표시

# 이미지 기반 인공신경망 사용을 위한 라이브러리
import numpy as np 
from PIL import Image
import torch 
import torch.optim as optim 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

# 이미지 변환기 생성, pretrained model 인스턴스 생성을 위한 라이브러리
import torchvision 
from torchvision.transforms import transforms 
from torchvision.models import resnet50, ResNet50_Weights

# 장치 설정
# GPU 사용 가능하면 cuda, 아니면 CPU로 선택됨

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# 사용할 사전학습 모델이 학습시 사용한 transform의 조건 확인
mean_imgnet = ResNet50_Weights.IMAGENET1K_V1.transforms().mean
std_imgnet  = ResNet50_Weights.IMAGENET1K_V1.transforms().std
input_size = ResNet50_Weights.IMAGENET1K_V1.transforms().crop_size
resize_size = ResNet50_Weights.IMAGENET1K_V1.transforms().resize_size
interpolation = ResNet50_Weights.IMAGENET1K_V1.transforms().interpolation
antialias = ResNet50_Weights.IMAGENET1K_V1.transforms().antialias  # 이미지를 resize할 때 생기는 계단 현상을 줄여주는 역할. 경계를 부드럽게 하면서 이미지를 축소하기 위함

print(f"mean_imgnet: {mean_imgnet}")
print(f"std_imgnet : {std_imgnet}")
print(f"resize_size : {resize_size}")
print(f"input_size : {input_size}")
print(f"interpolation: {interpolation}")
print(f"antialias: {antialias}")  


