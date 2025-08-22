import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import glob
import random
from math import ceil
import pickle
import torch.nn.functional as F

# 파일 경로 - 외장하드 경로로 변경
WACCM_FILES = glob.glob(r'F:\ionodata\*.nc')
IONO_FILES = glob.glob(r'F:\ionodata\*.nc')
IONO_FILE = None  # 사용하지 않음

# 모델 저장 경로
MODEL_SAVE_PATH = r'C:\Users\sunma\.vscode\models\hybrid_model.pth'
MODEL_200KM_SAVE_PATH = r'C:\Users\sunma\.vscode\models\hybrid_model_200km.pth'

# 고도별 필터링 설정
ALTITUDE_THRESHOLD = 200  # 200km 이상 고도 필터링 기준
HIGH_ALTITUDE_VARS = [
    'T', 'U', 'V', 'H', 'O', 'O2', 'NO', 'QNO', 'PS', 'Z3', 'QRL', 'QRS', 'TTGW', 'UTGW_TOTAL',
    'PHIM2D', 'OMEGA', 'DTCOND', 'UI', 'VI', 'WI'
]  # 고도별 변수 (대기 변수와 동일하지만 필요시 조정 가능)

# models 디렉토리 생성
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# WGAN 하이퍼파라미터
LATENT_DIM = 100
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# 슬라이딩 윈도우 파라미터
WINDOW_SIZE = 4  # patch 및 time 슬라이딩 윈도우 크기 (기존 체크포인트와 맞추기 위해 4로 변경)
STRIDE = 2  # 슬라이딩 윈도우 간격

# 대기/이온 변수 리스트 정의
ATMOS_VARS = [
    'T', 'U', 'V', 'H', 'O', 'O2', 'NO', 'QNO', 'PS', 'Z3', 'QRL', 'QRS', 'TTGW', 'UTGW_TOTAL',
    'PHIM2D', 'OMEGA', 'DTCOND', 'UI', 'VI', 'WI'
]
IONO_VARS = [
    'EDens', 'ElecColDens', 'TElec', 'TIon', 'e', 'ED1', 'ED2', 'EDYN_ZIGM11_PED', 'EDYN_ZIGM2_HAL',
    'kp', 'ap', 'f107', 'f107a', 'f107p', 'UI', 'VI', 'WI'
]

class CombinedDataset(Dataset):
    """
    메모리 효율적으로 patch 인덱스만 저장하고, 실제 patch 데이터는 __getitem__에서 파일에서 직접 읽어옴
    파일 핸들 잠금 문제를 피하기 위해 파일 객체를 캐싱함.
    """
    def __init__(self, waccm_files, iono_files=None, use_atmos=True, use_iono=True, use_dims=None, var_filter=None, stride=None, altitude_filter=None):
        self._file_cache = {}  # 파일 핸들 캐시
        self.patch_index = []  # (파일, 변수, time, lev, lat, lon, is_atmos) 튜플 리스트
        self.use_dims = use_dims
        self.var_filter = var_filter
        self.stride = stride if stride is not None else STRIDE
        self.altitude_filter = altitude_filter  # 고도 필터링 설정
        self.variable_names = []
        # patch 인덱스만 생성
        if use_atmos and waccm_files:
            for file in waccm_files:
                try:
                    with xr.open_dataset(file, chunks={'time': 1}) as ds:
                        all_vars = [v for v in ds.variables.keys() if v in ATMOS_VARS]
                        for var in all_vars:
                            arr = ds[var]
                            if 'time' in arr.dims:
                                arr = arr.transpose(*arr.dims)
                                t_len = arr.sizes['time']
                                for t in range(t_len):
                                    if 'lev' in arr.dims:
                                        lev_len = arr.sizes['lev']
                                        for lev in range(lev_len):
                                            # 고도 필터링 적용
                                            if self.altitude_filter is not None:
                                                # lev 인덱스를 고도로 변환 (대략적인 추정)
                                                # WACCM에서 lev는 보통 하위 레벨이 높은 고도
                                                estimated_altitude = self._estimate_altitude_from_lev(lev, lev_len)
                                                if estimated_altitude < self.altitude_filter:
                                                    continue  # 200km 미만이면 스킵
                                            
                                            chunk_shape = arr.isel(time=t, lev=lev).shape
                                            lat_dim = chunk_shape[-2]
                                            lon_dim = chunk_shape[-1]
                                            # 안전한 슬라이스 범위 계산
                                            for lat_start in range(0, max(1, lat_dim - WINDOW_SIZE + 1), self.stride):
                                                for lon_start in range(0, max(1, lon_dim - WINDOW_SIZE + 1), self.stride):
                                                    self.patch_index.append((file, var, t, lev, lat_start, lon_start, True))
                                                    self.variable_names.append(f"[ATMOS]{var}")
                                    elif 'lat' in arr.dims and 'lon' in arr.dims:
                                        chunk_shape = arr.isel(time=t).shape
                                        lat_dim = chunk_shape[-2]
                                        lon_dim = chunk_shape[-1]
                                        # 안전한 슬라이스 범위 계산
                                        for lat_start in range(0, max(1, lat_dim - WINDOW_SIZE + 1), self.stride):
                                            for lon_start in range(0, max(1, lon_dim - WINDOW_SIZE + 1), self.stride):
                                                self.patch_index.append((file, var, t, None, lat_start, lon_start, True))
                                                self.variable_names.append(f"[ATMOS]{var}")
                                    elif 'mlat' in arr.dims and 'mlon' in arr.dims:
                                        chunk_shape = arr.isel(time=t).shape
                                        lat_dim = chunk_shape[-2]
                                        lon_dim = chunk_shape[-1]
                                        # 안전한 슬라이스 범위 계산
                                        for lat_start in range(0, max(1, lat_dim - WINDOW_SIZE + 1), self.stride):
                                            for lon_start in range(0, max(1, lon_dim - WINDOW_SIZE + 1), self.stride):
                                                self.patch_index.append((file, var, t, None, lat_start, lon_start, True))
                                                self.variable_names.append(f"[ATMOS]{var}")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue
        if use_iono and iono_files:
            for file in iono_files:
                try:
                    with xr.open_dataset(file, chunks={'time': 1}) as ds:
                        all_vars = [v for v in ds.variables.keys() if v in IONO_VARS]
                        for var in all_vars:
                            arr = ds[var]
                            if 'time' in arr.dims:
                                arr = arr.transpose(*arr.dims)
                                t_len = arr.sizes['time']
                                for t in range(t_len):
                                    if 'lev' in arr.dims:
                                        lev_len = arr.sizes['lev']
                                        for lev in range(lev_len):
                                            # 고도 필터링 적용
                                            if self.altitude_filter is not None:
                                                estimated_altitude = self._estimate_altitude_from_lev(lev, lev_len)
                                                if estimated_altitude < self.altitude_filter:
                                                    continue  # 200km 미만이면 스킵
                                            
                                            chunk_shape = arr.isel(time=t, lev=lev).shape
                                            lat_dim = chunk_shape[-2]
                                            lon_dim = chunk_shape[-1]
                                            # 안전한 슬라이스 범위 계산
                                            for lat_start in range(0, max(1, lat_dim - WINDOW_SIZE + 1), self.stride):
                                                for lon_start in range(0, max(1, lon_dim - WINDOW_SIZE + 1), self.stride):
                                                    self.patch_index.append((file, var, t, lev, lat_start, lon_start, False))
                                                    self.variable_names.append(f"[IONO]{var}")
                                    elif 'lat' in arr.dims and 'lon' in arr.dims:
                                        chunk_shape = arr.isel(time=t).shape
                                        lat_dim = chunk_shape[-2]
                                        lon_dim = chunk_shape[-1]
                                        # 안전한 슬라이스 범위 계산
                                        for lat_start in range(0, max(1, lat_dim - WINDOW_SIZE + 1), self.stride):
                                            for lon_start in range(0, max(1, lon_dim - WINDOW_SIZE + 1), self.stride):
                                                self.patch_index.append((file, var, t, None, lat_start, lon_start, False))
                                                self.variable_names.append(f"[IONO]{var}")
                                    elif 'mlat' in arr.dims and 'mlon' in arr.dims:
                                        chunk_shape = arr.isel(time=t).shape
                                        lat_dim = chunk_shape[-2]
                                        lon_dim = chunk_shape[-1]
                                        # 안전한 슬라이스 범위 계산
                                        for lat_start in range(0, max(1, lat_dim - WINDOW_SIZE + 1), self.stride):
                                            for lon_start in range(0, max(1, lon_dim - WINDOW_SIZE + 1), self.stride):
                                                self.patch_index.append((file, var, t, None, lat_start, lon_start, False))
                                                self.variable_names.append(f"[IONO]{var}")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                    continue
    def __len__(self):
        return len(self.patch_index)
    def __getitem__(self, idx):
        file, var, t, lev, lat_start, lon_start, is_atmos = self.patch_index[idx]
        
        if file not in self._file_cache:
            try:
                # 외장하드 접근을 위한 추가 옵션
                self._file_cache[file] = xr.open_dataset(file, engine='netcdf4', chunks={'time': 1})
            except PermissionError as e:
                print(f"권한 오류로 파일을 열 수 없습니다: {file}")
                print("외장하드가 연결되어 있는지 확인해주세요.")
                return torch.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=torch.float32)
            except Exception as e:
                print(f"Failed to open file {file}: {e}")
                # 파일 접근 실패 시 기본값 반환
                return torch.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=torch.float32)
        ds = self._file_cache[file]
        
        try:
            arr = ds[var]
            # 차원명 동적 확인
            if 'lat' in arr.dims and 'lon' in arr.dims:
                lat_dim, lon_dim = 'lat', 'lon'
            elif 'mlat' in arr.dims and 'mlon' in arr.dims:
                lat_dim, lon_dim = 'mlat', 'mlon'
            else:
                raise ValueError(f"Expected lat/lon or mlat/mlon dimensions, got: {arr.dims}")
            
            # 안전한 인덱싱을 위한 차원 크기 확인
            if 'time' in arr.dims and t >= arr.sizes['time']:
                print(f"Warning: time index {t} out of range for {var}, using 0")
                t = 0
            
            if lev is not None and 'lev' in arr.dims:
                if lev >= arr.sizes['lev']:
                    print(f"Warning: lev index {lev} out of range for {var}, using 0")
                    lev = 0
                
                # lat/lon 차원 크기 확인
                lat_size = arr.sizes[lat_dim]
                lon_size = arr.sizes[lon_dim]
                
                # 슬라이스 범위 조정
                lat_end = min(lat_start + WINDOW_SIZE, lat_size)
                lon_end = min(lon_start + WINDOW_SIZE, lon_size)
                
                if lat_start >= lat_size or lon_start >= lon_size:
                    print(f"Warning: Invalid slice indices for {var}, using zeros")
                    # 기본값으로 0부터 시작하는 패치 생성
                    patch = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
                else:
                    patch = arr.isel(time=t, lev=lev, **{lat_dim: slice(lat_start, lat_end), lon_dim: slice(lon_start, lon_end)}).values
                    
                    # 패치 크기가 WINDOW_SIZE보다 작으면 패딩
                    if patch.shape != (WINDOW_SIZE, WINDOW_SIZE):
                        padded_patch = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
                        padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                        patch = padded_patch
            else:
                # lat/lon 차원 크기 확인
                lat_size = arr.sizes[lat_dim]
                lon_size = arr.sizes[lon_dim]
                
                # 슬라이스 범위 조정
                lat_end = min(lat_start + WINDOW_SIZE, lat_size)
                lon_end = min(lon_start + WINDOW_SIZE, lon_size)
                
                if lat_start >= lat_size or lon_start >= lon_size:
                    print(f"Warning: Invalid slice indices for {var}, using zeros")
                    # 기본값으로 0부터 시작하는 패치 생성
                    patch = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
                else:
                    patch = arr.isel(time=t, **{lat_dim: slice(lat_start, lat_end), lon_dim: slice(lon_start, lon_end)}).values
                    
                    # 패치 크기가 WINDOW_SIZE보다 작으면 패딩
                    if patch.shape != (WINDOW_SIZE, WINDOW_SIZE):
                        padded_patch = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
                        padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                        patch = padded_patch
                        
        except Exception as e:
            print(f"Error reading patch from {file}, variable {var}: {e}")
            print(f"Creating zero patch as fallback")
            patch = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
            
        return torch.tensor(patch, dtype=torch.float32)

    def close_files(self):
        """캐시된 모든 파일 핸들을 닫습니다."""
        for file, ds in self._file_cache.items():
            try:
                ds.close()
            except Exception as e:
                print(f"Error closing file {file}: {e}")
                # 파일 핸들을 강제로 None으로 설정
                try:
                    del ds
                except:
                    pass
        self._file_cache = {}

    def _estimate_altitude_from_lev(self, lev_index, total_levs):
        """
        레벨 인덱스를 고도로 변환 (대략적인 추정)
        WACCM에서 lev는 보통 하위 레벨이 높은 고도
        """
        # WACCM의 일반적인 고도 분포 (대략적)
        # 하위 레벨: 0-20km, 중간 레벨: 20-100km, 상위 레벨: 100-200km+
        if total_levs <= 30:  # 일반적인 WACCM 레벨 수
            # 하위 1/3: 0-50km, 중간 1/3: 50-150km, 상위 1/3: 150-300km
            if lev_index < total_levs // 3:
                return 25 + (lev_index / (total_levs // 3)) * 25  # 25-50km
            elif lev_index < 2 * total_levs // 3:
                return 50 + ((lev_index - total_levs // 3) / (total_levs // 3)) * 100  # 50-150km
            else:
                return 150 + ((lev_index - 2 * total_levs // 3) / (total_levs // 3)) * 150  # 150-300km
        else:
            # 더 많은 레벨이 있는 경우
            altitude_ratio = lev_index / total_levs
            return altitude_ratio * 300  # 0-300km 범위로 추정
    
    def __del__(self):
        """객체 소멸 시 파일 핸들을 닫습니다."""
        try:
            self.close_files()
        except:
            pass

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
        
        # 모든 가중치를 float32로 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data = module.weight.data.float()
            if module.bias is not None:
                module.bias.data = module.bias.data.float()
        
    def forward(self, z):
        return self.model(z)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
        
        # 모든 가중치를 float32로 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data = module.weight.data.float()
            if module.bias is not None:
                module.bias.data = module.bias.data.float()
        
    def forward(self, x):
        return self.model(x)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, sequence_length=10):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
        # 모든 가중치를 float32로 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data = module.weight.data.float()
            if module.bias is not None:
                module.bias.data = module.bias.data.float()
        
    def forward(self, x):
        # 입력 데이터를 시퀀스 형태로 변환
        if len(x.shape) == 2:
            # (batch_size, features) -> (batch_size, sequence_length, features)
            x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_checkpoint(model_path, device):
    """체크포인트 로드"""
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        print(f"체크포인트 로드됨: {model_path}")
        return checkpoint
    else:
        print(f"체크포인트 파일이 없습니다: {model_path}")
        return None

def save_checkpoint(model_dict, save_path, epoch, losses):
    """체크포인트 저장"""
    try:
        # 저장 디렉토리 확인 및 생성
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'input_size': model_dict.get('input_size'),
            'latent_dim': model_dict.get('latent_dim'),
            'all_losses': losses
        }
        # 필요한 키만 추가
        if 'generator_state_dict' in model_dict:
            checkpoint['generator_state_dict'] = model_dict['generator_state_dict']
        if 'critic_state_dict' in model_dict:
            checkpoint['critic_state_dict'] = model_dict['critic_state_dict']
        if 'lstm_state_dict' in model_dict:
            checkpoint['lstm_state_dict'] = model_dict['lstm_state_dict']
        if 'g_losses' in model_dict:
            checkpoint['g_losses'] = model_dict['g_losses']
        if 'c_losses' in model_dict:
            checkpoint['c_losses'] = model_dict['c_losses']
        if 'lstm_losses' in model_dict:
            checkpoint['lstm_losses'] = model_dict['lstm_losses']

        torch.save(checkpoint, save_path)
        print(f"체크포인트 저장됨: {save_path}")
    except Exception as e:
        print(f"체크포인트 저장 실패: {e}")
        print(f"저장 경로: {save_path}")
        # 대체 경로로 저장 시도
        try:
            alt_path = f"./backup_checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, alt_path)
            print(f"대체 경로로 저장됨: {alt_path}")
        except Exception as e2:
            print(f"대체 경로 저장도 실패: {e2}")

def train_wgan_with_checkpoint(generator, critic, dataloader, num_epochs, latent_dim, device, 
                              checkpoint_path=None, start_epoch=0, flatten_input=False):
    """체크포인트 기반 WGAN 학습"""
    
    # 옵티마이저 설정 - learning rate 더 낮춤
    g_optimizer = optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999))
    c_optimizer = optim.Adam(critic.parameters(), lr=0.00005, betas=(0.5, 0.999))
    
    # 체크포인트에서 옵티마이저 상태 복원
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'g_optimizer_state_dict' in checkpoint:
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        if 'c_optimizer_state_dict' in checkpoint:
            c_optimizer.load_state_dict(checkpoint['c_optimizer_state_dict'])
    
    g_losses = []
    c_losses = []
    
    print(f"\n=== WGAN 학습 시작 (에포크 {start_epoch+1}부터 {start_epoch+num_epochs}까지) ===")
    print(f"총 에포크: {num_epochs}")
    print(f"시작 에포크: {start_epoch}")
    print(f"배치 크기: {dataloader.batch_size}")
    print(f"총 배치 수: {len(dataloader)}")
    print(f"학습 장치: {device}")
    print("=" * 50)
    
    for epoch in range(start_epoch, start_epoch+num_epochs):
        g_epoch_loss = 0.0
        c_epoch_loss = 0.0
        
        for i, real_data in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device).float()  # float32로 통일
            # 항상 Flatten
            real_data = real_data.view(batch_size, -1)
            
            # 데이터 정규화 강화 (loss 안정화를 위해)
            real_data = torch.clamp(real_data, -5, 5)
            
            # Critic 학습
            for _ in range(CRITIC_ITERATIONS):
                c_optimizer.zero_grad()
                
                # 노이즈 생성
                noise = torch.randn(batch_size, latent_dim, dtype=torch.float32).to(device)
                fake_data = generator(noise)
                
                # 실제 데이터와 가짜 데이터에 대한 critic 점수
                real_validity = critic(real_data)
                fake_validity = critic(fake_data.detach())
                
                # Gradient penalty 계산
                gradient_penalty = compute_gradient_penalty(critic, real_data, fake_data.detach(), device)
                
                # Critic loss
                c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty
                
                # Loss 값 체크 및 clamping
                if abs(c_loss.item()) > 1e4:
                    print(f"[경고] Critic Loss가 너무 큽니다: {c_loss.item():.2e}")
                    c_loss = torch.clamp(c_loss, -1e4, 1e4)
                
                c_loss.backward()
                
                # Gradient clipping 강화
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                
                c_optimizer.step()
                
                c_epoch_loss += c_loss.item()
            
            # Generator 학습
            g_optimizer.zero_grad()
            
            # 가짜 데이터 생성
            noise = torch.randn(batch_size, latent_dim, dtype=torch.float32).to(device)
            fake_data = generator(noise)
            fake_validity = critic(fake_data)
            
            # Generator loss
            g_loss = -torch.mean(fake_validity)
            
            # Loss 값 체크 및 clamping
            if abs(g_loss.item()) > 1e4:
                print(f"[경고] Generator Loss가 너무 큽니다: {g_loss.item():.2e}")
                g_loss = torch.clamp(g_loss, -1e4, 1e4)
            
            g_loss.backward()
            
            # Gradient clipping 강화
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
            
            g_optimizer.step()
            
            g_epoch_loss += g_loss.item()
            
            if i % 10 == 0:
                print(f'[WGAN] Epoch [{epoch+1}/{start_epoch+num_epochs}], Step [{i+1}/{len(dataloader)}]')
                print(f'G Loss: {g_loss.item():.4f}, C Loss: {c_loss.item():.4f}')
                print(f'진행률: {(i+1)/len(dataloader)*100:.1f}%')
                print("-" * 30)
        
        avg_g_loss = g_epoch_loss / len(dataloader)
        avg_c_loss = c_epoch_loss / len(dataloader)
        g_losses.append(avg_g_loss)
        c_losses.append(avg_c_loss)
        
        print(f'\n[WGAN] Epoch {epoch+1}/{start_epoch+num_epochs} 완료')
        print(f'평균 G Loss: {avg_g_loss:.4f}, 평균 C Loss: {avg_c_loss:.4f}')
        print("=" * 50)
        
        # 매 10 에포크마다 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            try:
                checkpoint_path = MODEL_SAVE_PATH.replace('.pth', f'_checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint({
                    'generator_state_dict': generator.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'g_losses': g_losses,
                    'c_losses': c_losses,
                    'input_size': generator.model[0].in_features, # 입력 차원
                    'latent_dim': latent_dim
                }, checkpoint_path, epoch, {'g_losses': g_losses, 'c_losses': c_losses})
            except Exception as e:
                print(f"체크포인트 저장 실패: {e}")
    
    return g_losses, c_losses

def train_lstm_with_checkpoint(lstm, wgan_generator, dataloader, num_epochs, latent_dim, device,
                              checkpoint_path=None, start_epoch=0, flatten_input=False):
    """체크포인트 기반 LSTM 학습"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=0.001)  # learning rate 유지
    
    # 체크포인트에서 옵티마이저 상태 복원
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'lstm_optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['lstm_optimizer_state_dict'])
    
    # WGAN 생성자 고정
    for param in wgan_generator.parameters():
        param.requires_grad = False
    
    losses = []
    
    print(f"\n=== LSTM 학습 시작 (에포크 {start_epoch+1}부터 {start_epoch+num_epochs}까지) ===")
    print(f"총 에포크: {num_epochs}")
    print(f"시작 에포크: {start_epoch}")
    print(f"배치 크기: {dataloader.batch_size}")
    print(f"총 배치 수: {len(dataloader)}")
    print(f"학습 장치: {device}")
    print("=" * 50)
    
    for epoch in range(start_epoch, start_epoch+num_epochs):
        epoch_loss = 0.0
        
        for i, real_data in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device).float()  # float32로 통일
            # 항상 Flatten
            real_data = real_data.view(batch_size, -1)
            
            # LSTM 예측
            lstm_output = lstm(real_data)
            
            # WGAN 생성자를 사용하여 예측값 보정
            with torch.no_grad():
                noise = torch.randn(batch_size, latent_dim, dtype=torch.float32).to(device)
                wgan_output = wgan_generator(noise)
            
            # LSTM 출력과 WGAN 출력의 차이를 최소화하도록 학습
            loss = criterion(lstm_output, wgan_output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 10 == 0:
                print(f'[LSTM] Epoch [{epoch+1}/{start_epoch+num_epochs}], Step [{i+1}/{len(dataloader)}]')
                print(f'Loss: {loss.item():.4f}')
                print(f'진행률: {(i+1)/len(dataloader)*100:.1f}%')
                print("-" * 30)
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        print(f'\n[LSTM] Epoch {epoch+1}/{start_epoch+num_epochs} 완료')
        print(f'평균 Loss: {avg_loss:.4f}')
        print("=" * 50)
        
        # 매 10 에포크마다 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            checkpoint_path = MODEL_SAVE_PATH.replace('.pth', f'_lstm_checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint({
                'lstm_state_dict': lstm.state_dict(),
                'lstm_losses': losses,
                'input_size': lstm.fc.in_features, # LSTM의 입력 차원
                'latent_dim': latent_dim
            }, checkpoint_path, epoch, {'lstm_losses': losses})
    
    # 로스 그래프 그리기
    if len(losses) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='LSTM Loss', color='red')
        plt.title('LSTM Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('lstm_loss_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"LSTM 로스 그래프가 'lstm_loss_plot.png'로 저장되었습니다.")
    
    return losses

def plot_lstm_losses_from_checkpoint(checkpoint_path):
    """체크포인트에서 LSTM 로스를 불러와서 그래프로 표시"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'lstm_losses' in checkpoint:
            losses = checkpoint['lstm_losses']
            plt.figure(figsize=(10, 6))
            plt.plot(losses, label='LSTM Loss', color='red')
            plt.title('LSTM Training Loss from Checkpoint')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('lstm_loss_from_checkpoint.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"체크포인트에서 LSTM 로스 그래프가 'lstm_loss_from_checkpoint.png'로 저장되었습니다.")
            print(f"총 {len(losses)} 에포크의 로스 데이터가 있습니다.")
            if len(losses) > 0:
                print(f"최종 로스: {losses[-1]:.6f}")
                print(f"최소 로스: {min(losses):.6f}")
        else:
            print("체크포인트에 LSTM 로스 데이터가 없습니다.")
    else:
        print(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def integrate_outputs(outputs, target_shape):
    """
    여러 shape별 예측 결과(outputs)를 target_shape로 업샘플링 후 평균 통합
    outputs: list of torch.Tensor
    target_shape: tuple (예: (C, H, W) 또는 (N, C, H, W))
    """
    resized = []
    for o in outputs:
        t = o
        # 차원 맞추기 (예: (C, H, W) → (1, C, H, W))
        while len(t.shape) < len(target_shape):
            t = t.unsqueeze(0)
        # 마지막 2차원(H, W)만 맞추는 예시 (필요시 3D 등으로 확장)
        t_resized = F.interpolate(t, size=target_shape[-2:], mode='bilinear', align_corners=False)
        # 차원 복원
        while len(t_resized.shape) > len(target_shape):
            t_resized = t_resized.squeeze(0)
        resized.append(t_resized)
    # 평균 통합
    return sum(resized) / len(resized)

def check_external_drive():
    """외장하드 연결 상태 확인"""
    import os
    drive_path = 'F:\\'  # 일반 문자열로 변경
    
    if not os.path.exists(drive_path):
        print(f"외장하드가 연결되어 있지 않습니다: {drive_path}")
        print("외장하드를 연결한 후 다시 시도해주세요.")
        return False
    
    try:
        # 드라이브 접근 테스트
        test_file = os.path.join(drive_path, 'test_access.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"외장하드 접근 가능: {drive_path}")
        return True
    except PermissionError:
        print(f"외장하드에 쓰기 권한이 없습니다: {drive_path}")
        print("관리자 권한으로 실행하거나 권한을 확인해주세요.")
        return False
    except Exception as e:
        print(f"외장하드 접근 오류: {e}")
        return False

# create_dummy_data 함수 제거됨 - 테스트용 더미 데이터 생성 기능 삭제

def main():
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
    # 재시작 옵션 선택
    print("\n=== 학습 재시작 옵션 ===")
    print("1. 전체 고도 모델 재시작")
    print("2. 200km 이상 고도 모델 학습")
    print("3. 새로운 학습 시작")
    print("4. LSTM 로스 그래프 보기")
    
    choice = input("선택하세요 (1-4): ").strip()
    
    if choice == "1":
        restart_both_models(device)
    elif choice == "2":
        train_high_altitude_model(device)
    elif choice == "3":
        start_new_training(device)
    elif choice == "4":
        checkpoint_path = input("체크포인트 파일 경로를 입력하세요: ").strip()
        plot_lstm_losses_from_checkpoint(checkpoint_path)
    else:
        print("잘못된 선택입니다.")

def restart_both_models(device):
    """전체 고도 모델만 재시작 (testdata 폴더 내 모든 파일 사용, 그룹별로 처리)"""
    print("\n=== 전체 고도 모델 재시작 ===")
    
    # 외장하드 연결 상태 확인
    if not check_external_drive():
        print("외장하드 접근이 불가능합니다. 프로그램을 종료합니다.")
        return
    
    # 파일 존재 여부 확인
    if not WACCM_FILES and not IONO_FILES:
        print("경고: 데이터 파일을 찾을 수 없습니다.")
        print("다음 경로에서 .nc 파일을 확인해주세요:")
        print(f"대기 데이터: F:\\ionodata\\")
        print(f"이온 데이터: F:\\ionodata\\")
        
        print("데이터 파일이 없어 프로그램을 종료합니다.")
        return
    
    # 체크포인트 로드 시도
    full_checkpoint = load_checkpoint(MODEL_SAVE_PATH, device)
    
    group_size = 1
    # === 대기(ATMOS) 그룹별 학습 ===
    files = WACCM_FILES.copy()
    if not files:
        print("대기 데이터 파일이 없습니다. 이온 데이터만 처리합니다.")
        files = []
    random.shuffle(files)
    n = len(files)
    train_files = files[:int(0.8*n)] if n > 0 else []
    test_files = files[int(0.8*n):] if n > 0 else []
    print(f"[대기] 훈련 파일 개수: {len(train_files)}, 테스트 파일 개수: {len(test_files)}")
    
    if not train_files:
        print("대기 훈련 파일이 없습니다. 이온 데이터만 처리합니다.")
        input_dim = 16  # 기본값
    else:
        num_groups = ceil(len(train_files) / group_size)
        first_group_files = train_files[:group_size]
        temp_dataset = CombinedDataset(first_group_files, None, use_atmos=True, use_iono=False, var_filter=ATMOS_VARS, stride=STRIDE, altitude_filter=None)
        sample_data = temp_dataset[0]
        input_dim = sample_data.numel()
        print(f"[대기] 실제 입력 차원: {input_dim}")
    
    # 새로운 모델 생성 (체크포인트 무시)
    print("새로운 모델을 생성합니다.")
    generator = Generator(LATENT_DIM, input_dim).to(device)
    critic = Critic(input_dim).to(device)
    lstm = LSTM(input_size=input_dim, sequence_length=10).to(device)
    all_g_losses = []
    all_c_losses = []
    all_lstm_losses = []
    print("새로운 모델로 초기화 완료")
    
    additional_epochs = 1000
    
    # 대기 데이터가 있는 경우에만 처리
    if train_files:
        for i in range(num_groups):
            group_files = train_files[i*group_size:(i+1)*group_size]
            print(f"\n[대기] 그룹 {i+1}/{num_groups} 파일: {group_files}")
            dataset = CombinedDataset(group_files, None, use_atmos=True, use_iono=False, var_filter=ATMOS_VARS, stride=STRIDE, altitude_filter=None)
            total_len = len(dataset)
            if total_len < 2:
                print("  [경고] 데이터 샘플이 너무 적어 스킵합니다.")
                continue
            train_size = int(0.8 * total_len)
            val_size = total_len - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
            print(f"  [대기] 그룹 {i+1} 학습 시작 ({additional_epochs} 에포크, 시작 에포크: 0)")
            g_losses, c_losses = train_wgan_with_checkpoint(generator, critic, train_loader, additional_epochs, LATENT_DIM, device, checkpoint_path=None, start_epoch=0)
            lstm_losses = train_lstm_with_checkpoint(lstm, generator, train_loader, additional_epochs, LATENT_DIM, device, checkpoint_path=None, start_epoch=0)
            all_g_losses.extend(g_losses)
            all_c_losses.extend(c_losses)
            all_lstm_losses.extend(lstm_losses)
            dataset.close_files()
    
    # === 이온(IONO) 그룹별 학습 ===
    files_iono = IONO_FILES.copy()
    if not files_iono:
        print("이온 데이터 파일이 없습니다.")
        return
    
    random.shuffle(files_iono)
    n_iono = len(files_iono)
    train_files_iono = files_iono[:int(0.8*n_iono)]
    test_files_iono = files_iono[int(0.8*n_iono):]
    print(f"[이온] 훈련 파일 개수: {len(train_files_iono)}, 테스트 파일 개수: {len(test_files_iono)}")
    
    if not train_files_iono:
        print("이온 훈련 파일이 없습니다.")
        return
    
    num_groups_iono = ceil(len(train_files_iono) / group_size)
    first_group_files_iono = train_files_iono[:group_size]
    temp_dataset_iono = CombinedDataset([], first_group_files_iono, use_atmos=False, use_iono=True, var_filter=IONO_VARS, stride=STRIDE, altitude_filter=None)
    sample_data_iono = temp_dataset_iono[0]
    input_dim_iono = sample_data_iono.numel()
    print(f"[이온] 실제 입력 차원: {input_dim_iono}")
    
    # 새로운 이온 모델 생성
    print("새로운 이온 모델을 생성합니다.")
    generator_iono = Generator(LATENT_DIM, input_dim_iono).to(device)
    critic_iono = Critic(input_dim_iono).to(device)
    lstm_iono = LSTM(input_size=input_dim_iono, sequence_length=10).to(device)
    all_g_losses_iono = []
    all_c_losses_iono = []
    all_lstm_losses_iono = []
    print("새로운 이온 모델로 초기화 완료")
    
    for i in range(num_groups_iono):
        group_files = train_files_iono[i*group_size:(i+1)*group_size]
        print(f"\n[이온] 그룹 {i+1}/{num_groups_iono} 파일: {group_files}")
        dataset = CombinedDataset([], group_files, use_atmos=False, use_iono=True, var_filter=IONO_VARS, stride=STRIDE, altitude_filter=None)
        total_len = len(dataset)
        if total_len < 2:
            print("  [경고] 데이터 샘플이 너무 적어 스킵합니다.")
            continue
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        print(f"  [이온] 그룹 {i+1} 학습 시작 ({additional_epochs} 에포크, 시작 에포크: 0)")
        g_losses, c_losses = train_wgan_with_checkpoint(generator_iono, critic_iono, train_loader, additional_epochs, LATENT_DIM, device, checkpoint_path=None, start_epoch=0)
        lstm_losses = train_lstm_with_checkpoint(lstm_iono, generator_iono, train_loader, additional_epochs, LATENT_DIM, device, checkpoint_path=None, start_epoch=0)
        all_g_losses_iono.extend(g_losses)
        all_c_losses_iono.extend(c_losses)
        all_lstm_losses_iono.extend(lstm_losses)
        dataset.close_files()
    
    # === 모델 저장 ===
    if train_files:  # 대기 데이터가 있는 경우에만 저장
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'lstm_state_dict': lstm.state_dict(),
            'g_losses': all_g_losses,
            'c_losses': all_c_losses,
            'lstm_losses': all_lstm_losses,
            'input_size': input_dim,
            'latent_dim': LATENT_DIM
        }, 'atmosphere_model.pth')
        print("대기 모델 저장 완료: atmosphere_model.pth")
    
    if train_files_iono:  # 이온 데이터가 있는 경우에만 저장
        torch.save({
            'generator_state_dict': generator_iono.state_dict(),
            'critic_state_dict': critic_iono.state_dict(),
            'lstm_state_dict': lstm_iono.state_dict(),
            'g_losses': all_g_losses_iono,
            'c_losses': all_c_losses_iono,
            'lstm_losses': all_lstm_losses_iono,
            'input_size': input_dim_iono,
            'latent_dim': LATENT_DIM
        }, 'ionosphere_model.pth')
        print("이온 모델 저장 완료: ionosphere_model.pth")
    
    # 통합 모델 저장
    save_dict = {}
    if train_files:
        save_dict['atmosphere'] = {
            'generator_state_dict': generator.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'lstm_state_dict': lstm.state_dict(),
            'g_losses': all_g_losses,
            'c_losses': all_c_losses,
            'lstm_losses': all_lstm_losses,
            'input_size': input_dim,
            'latent_dim': LATENT_DIM
        }
    if train_files_iono:
        save_dict['ionosphere'] = {
            'generator_state_dict': generator_iono.state_dict(),
            'critic_state_dict': critic_iono.state_dict(),
            'lstm_state_dict': lstm_iono.state_dict(),
            'g_losses': all_g_losses_iono,
            'c_losses': all_c_losses_iono,
            'lstm_losses': all_lstm_losses_iono,
            'input_size': input_dim_iono,
            'latent_dim': LATENT_DIM
        }
    
    if save_dict:
        torch.save(save_dict, 'hybrid_model_all.pth')
        print("대기+이온 통합 모델 저장 완료: hybrid_model_all.pth")
    
    # 모델 정보 파일 저장 (대기시뮬레이션.py에서 사용)
    model_info = {
        'input_size': input_dim if train_files else input_dim_iono,
        'latent_dim': LATENT_DIM,
        'model_type': 'hybrid_lstm_generator',
        'atmosphere_vars': ATMOS_VARS if train_files else [],
        'ionosphere_vars': IONO_VARS if train_files_iono else [],
        'window_size': WINDOW_SIZE,
        'stride': STRIDE
    }
    torch.save(model_info, MODEL_SAVE_PATH.replace('.pth', '_info.pth'))
    print("모델 정보 파일 저장 완료")
    
    print("\n=== 모든 모델 재학습 완료 ===")
    print("전체 고도 모델 재학습 완료")
    print("모델이 성공적으로 저장되었습니다.")

def train_high_altitude_model(device):
    """200km 이상 고도 전용 모델 학습"""
    print("\n=== 200km 이상 고도 모델 학습 ===")
    
    # 외장하드 연결 상태 확인
    if not check_external_drive():
        print("외장하드 접근이 불가능합니다. 프로그램을 종료합니다.")
        return
    
    # 파일 존재 여부 확인
    if not WACCM_FILES and not IONO_FILES:
        print("경고: 데이터 파일을 찾을 수 없습니다.")
        print("다음 경로에서 .nc 파일을 확인해주세요:")
        print(f"대기 데이터: F:\\ionodata\\")
        print(f"이온 데이터: F:\\ionodata\\")
        
        print("데이터 파일이 없어 프로그램을 종료합니다.")
        return
    
    # 체크포인트 로드 시도
    high_altitude_checkpoint = load_checkpoint(MODEL_200KM_SAVE_PATH, device)
    
    group_size = 1
    additional_epochs = 1000
    
    # === 대기(ATMOS) 200km 이상 고도 데이터 학습 ===
    files = WACCM_FILES.copy()
    if not files:
        print("대기 데이터 파일이 없습니다. 이온 데이터만 처리합니다.")
        files = []
    random.shuffle(files)
    n = len(files)
    train_files = files[:int(0.8*n)] if n > 0 else []
    test_files = files[int(0.8*n):] if n > 0 else []
    print(f"[대기-고도] 훈련 파일 개수: {len(train_files)}, 테스트 파일 개수: {len(test_files)}")
    
    if not train_files:
        print("대기 훈련 파일이 없습니다. 이온 데이터만 처리합니다.")
        input_dim = 16  # 기본값
    else:
        num_groups = ceil(len(train_files) / group_size)
        first_group_files = train_files[:group_size]
        # 200km 이상 고도 필터링 적용
        temp_dataset = CombinedDataset(
            first_group_files, None, 
            use_atmos=True, use_iono=False, 
            var_filter=HIGH_ALTITUDE_VARS, 
            stride=STRIDE,
            altitude_filter=ALTITUDE_THRESHOLD  # 200km 이상만
        )
        sample_data = temp_dataset[0]
        input_dim = sample_data.numel()
        print(f"[대기-고도] 실제 입력 차원: {input_dim}")
        print(f"[대기-고도] 200km 이상 고도 데이터만 사용")
    
    # 새로운 고도 모델 생성
    print("새로운 200km 이상 고도 모델을 생성합니다.")
    generator_high = Generator(LATENT_DIM, input_dim).to(device)
    critic_high = Critic(input_dim).to(device)
    lstm_high = LSTM(input_size=input_dim, sequence_length=10).to(device)
    all_g_losses_high = []
    all_c_losses_high = []
    all_lstm_losses_high = []
    print("새로운 고도 모델로 초기화 완료")
    
    # 대기 데이터가 있는 경우에만 처리
    if train_files:
        for i in range(num_groups):
            group_files = train_files[i*group_size:(i+1)*group_size]
            print(f"\n[대기-고도] 그룹 {i+1}/{num_groups} 파일: {group_files}")
            dataset = CombinedDataset(
                group_files, None, 
                use_atmos=True, use_iono=False, 
                var_filter=HIGH_ALTITUDE_VARS, 
                stride=STRIDE,
                altitude_filter=ALTITUDE_THRESHOLD  # 200km 이상만
            )
            total_len = len(dataset)
            if total_len < 2:
                print("  [경고] 200km 이상 고도 데이터 샘플이 너무 적어 스킵합니다.")
                continue
            train_size = int(0.8 * total_len)
            val_size = total_len - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
            print(f"  [대기-고도] 그룹 {i+1} 학습 시작 ({additional_epochs} 에포크, 시작 에포크: 0)")
            print(f"  [대기-고도] 200km 이상 고도 데이터: {total_len}개 샘플")
            g_losses, c_losses = train_wgan_with_checkpoint(generator_high, critic_high, train_loader, additional_epochs, LATENT_DIM, device, checkpoint_path=None, start_epoch=0)
            lstm_losses = train_lstm_with_checkpoint(lstm_high, generator_high, train_loader, additional_epochs, LATENT_DIM, device, checkpoint_path=None, start_epoch=0)
            all_g_losses_high.extend(g_losses)
            all_c_losses_high.extend(c_losses)
            all_lstm_losses_high.extend(lstm_losses)
            dataset.close_files()
    
    # === 이온(IONO) 200km 이상 고도 데이터 학습 ===
    files_iono = IONO_FILES.copy()
    if not files_iono:
        print("이온 데이터 파일이 없습니다.")
        return
    
    random.shuffle(files_iono)
    n_iono = len(files_iono)
    train_files_iono = files_iono[:int(0.8*n_iono)]
    test_files_iono = files_iono[int(0.8*n_iono):]
    print(f"[이온-고도] 훈련 파일 개수: {len(train_files_iono)}, 테스트 파일 개수: {len(test_files_iono)}")
    
    if not train_files_iono:
        print("이온 훈련 파일이 없습니다.")
        return
    
    num_groups_iono = ceil(len(train_files_iono) / group_size)
    first_group_files_iono = train_files_iono[:group_size]
    temp_dataset_iono = CombinedDataset(
        [], first_group_files_iono, 
        use_atmos=False, use_iono=True, 
        var_filter=IONO_VARS, 
        stride=STRIDE,
        altitude_filter=ALTITUDE_THRESHOLD  # 200km 이상만
    )
    sample_data_iono = temp_dataset_iono[0]
    input_dim_iono = sample_data_iono.numel()
    print(f"[이온-고도] 실제 입력 차원: {input_dim_iono}")
    print(f"[이온-고도] 200km 이상 고도 데이터만 사용")
    
    # 새로운 이온 고도 모델 생성
    print("새로운 이온 200km 이상 고도 모델을 생성합니다.")
    generator_iono_high = Generator(LATENT_DIM, input_dim_iono).to(device)
    critic_iono_high = Critic(input_dim_iono).to(device)
    lstm_iono_high = LSTM(input_size=input_dim_iono, sequence_length=10).to(device)
    all_g_losses_iono_high = []
    all_c_losses_iono_high = []
    all_lstm_losses_iono_high = []
    print("새로운 이온 고도 모델로 초기화 완료")
    
    for i in range(num_groups_iono):
        group_files = train_files_iono[i*group_size:(i+1)*group_size]
        print(f"\n[이온-고도] 그룹 {i+1}/{num_groups_iono} 파일: {group_files}")
        dataset = CombinedDataset(
            [], group_files, 
            use_atmos=False, use_iono=True, 
            var_filter=IONO_VARS, 
            stride=STRIDE,
            altitude_filter=ALTITUDE_THRESHOLD  # 200km 이상만
        )
        total_len = len(dataset)
        if total_len < 2:
            print("  [경고] 200km 이상 고도 데이터 샘플이 너무 적어 스킵합니다.")
            continue
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        print(f"  [이온-고도] 그룹 {i+1} 학습 시작 ({additional_epochs} 에포크, 시작 에포크: 0)")
        print(f"  [이온-고도] 200km 이상 고도 데이터: {total_len}개 샘플")
        g_losses, c_losses = train_wgan_with_checkpoint(generator_iono_high, critic_iono_high, train_loader, additional_epochs, LATENT_DIM, device, checkpoint_path=None, start_epoch=0)
        lstm_losses = train_lstm_with_checkpoint(lstm_iono_high, generator_iono_high, train_loader, additional_epochs, LATENT_DIM, device, checkpoint_path=None, start_epoch=0)
        all_g_losses_iono_high.extend(g_losses)
        all_c_losses_iono_high.extend(c_losses)
        all_lstm_losses_iono_high.extend(lstm_losses)
        dataset.close_files()
    
    # === 200km 이상 고도 모델 저장 ===
    if train_files:  # 대기 데이터가 있는 경우에만 저장
        torch.save({
            'generator_state_dict': generator_high.state_dict(),
            'critic_state_dict': critic_high.state_dict(),
            'lstm_state_dict': lstm_high.state_dict(),
            'g_losses': all_g_losses_high,
            'c_losses': all_c_losses_high,
            'lstm_losses': all_lstm_losses_high,
            'input_size': input_dim,
            'latent_dim': LATENT_DIM,
            'altitude_threshold': ALTITUDE_THRESHOLD,
            'model_type': 'high_altitude_hybrid'
        }, 'atmosphere_high_altitude_model.pth')
        print("대기 200km 이상 고도 모델 저장 완료: atmosphere_high_altitude_model.pth")
    
    if train_files_iono:  # 이온 데이터가 있는 경우에만 저장
        torch.save({
            'generator_state_dict': generator_iono_high.state_dict(),
            'critic_state_dict': critic_iono_high.state_dict(),
            'lstm_state_dict': lstm_iono_high.state_dict(),
            'g_losses': all_g_losses_iono_high,
            'c_losses': all_c_losses_iono_high,
            'lstm_losses': all_lstm_losses_iono_high,
            'input_size': input_dim_iono,
            'latent_dim': LATENT_DIM,
            'altitude_threshold': ALTITUDE_THRESHOLD,
            'model_type': 'high_altitude_hybrid'
        }, 'ionosphere_high_altitude_model.pth')
        print("이온 200km 이상 고도 모델 저장 완료: ionosphere_high_altitude_model.pth")
    
    # 통합 고도 모델 저장
    save_dict = {}
    if train_files:
        save_dict['atmosphere_high'] = {
            'generator_state_dict': generator_high.state_dict(),
            'critic_state_dict': critic_high.state_dict(),
            'lstm_state_dict': lstm_high.state_dict(),
            'g_losses': all_g_losses_high,
            'c_losses': all_c_losses_high,
            'lstm_losses': all_lstm_losses_high,
            'input_size': input_dim,
            'latent_dim': LATENT_DIM,
            'altitude_threshold': ALTITUDE_THRESHOLD,
            'model_type': 'high_altitude_hybrid'
        }
    if train_files_iono:
        save_dict['ionosphere_high'] = {
            'generator_state_dict': generator_iono_high.state_dict(),
            'critic_state_dict': critic_iono_high.state_dict(),
            'lstm_state_dict': lstm_iono_high.state_dict(),
            'g_losses': all_g_losses_iono_high,
            'c_losses': all_c_losses_iono_high,
            'lstm_losses': all_lstm_losses_iono_high,
            'input_size': input_dim_iono,
            'latent_dim': LATENT_DIM,
            'altitude_threshold': ALTITUDE_THRESHOLD,
            'model_type': 'high_altitude_hybrid'
        }
    
    if save_dict:
        torch.save(save_dict, MODEL_200KM_SAVE_PATH)
        print(f"200km 이상 고도 통합 모델 저장 완료: {MODEL_200KM_SAVE_PATH}")
    
    # 모델 정보 파일 저장
    model_info = {
        'input_size': input_dim if train_files else input_dim_iono,
        'latent_dim': LATENT_DIM,
        'model_type': 'high_altitude_hybrid_lstm_generator',
        'atmosphere_vars': HIGH_ALTITUDE_VARS if train_files else [],
        'ionosphere_vars': IONO_VARS if train_files_iono else [],
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
        'altitude_threshold': ALTITUDE_THRESHOLD
    }
    torch.save(model_info, MODEL_200KM_SAVE_PATH.replace('.pth', '_info.pth'))
    print("200km 이상 고도 모델 정보 파일 저장 완료")
    
    print("\n=== 200km 이상 고도 모델 학습 완료 ===")
    print("200km 이상 고도 전용 모델 학습 완료")
    print("모델이 성공적으로 저장되었습니다.")

def start_new_training(device):
    # 캐시 파일 삭제(항상 새로 데이터셋 생성)
    for cache_file in ["atmosphere_data_cache.pkl", "iono_data_cache.pkl", "variable_names_cache.pkl"]:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"[INFO] 캐시 파일 삭제: {cache_file}")
            except Exception as e:
                print(f"[WARN] 캐시 파일 삭제 실패: {cache_file}, {e}")
    print("\n=== 새로운 학습 시작 ===")
    group_size = 1
    # 1. 대기 파일 분리
    files = WACCM_FILES.copy()
    random.shuffle(files)
    n = len(files)
    train_files = files[:int(0.8*n)]
    test_files = files[int(0.8*n):]
    print(f"[대기] 훈련 파일 개수: {len(train_files)}, 테스트 파일 개수: {len(test_files)}")
    num_groups = ceil(len(train_files) / group_size)
    first_group_files = train_files[:group_size]
    temp_dataset = CombinedDataset(first_group_files, None, use_atmos=True, use_iono=False, var_filter=ATMOS_VARS, stride=STRIDE, altitude_filter=None)
    # 실제 데이터 샘플로 input_dim 계산
    sample_data = temp_dataset[0]
    input_dim = sample_data.numel()  # 실제 데이터 차원
    print(f"[대기] 실제 입력 차원: {input_dim}")
    generator = Generator(LATENT_DIM, input_dim).to(device)
    critic = Critic(input_dim).to(device)
    lstm = LSTM(input_size=input_dim, sequence_length=10).to(device)
    for i in range(num_groups):
        group_files = train_files[i*group_size:(i+1)*group_size]
        print(f"\n[대기] 그룹 {i+1}/{num_groups} 파일: {group_files}")
        dataset = CombinedDataset(group_files, None, use_atmos=True, use_iono=False, var_filter=ATMOS_VARS, stride=STRIDE, altitude_filter=None)
        total_len = len(dataset)
        if total_len < 2:
            print("  [경고] 데이터 샘플이 너무 적어 스킵합니다.")
            continue
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        print(f"  [대기] 그룹 {i+1} 학습 시작 (1000 에포크)")
        g_losses, c_losses = train_wgan_with_checkpoint(generator, critic, train_loader, 1000, LATENT_DIM, device, flatten_input=True)
        lstm_losses = train_lstm_with_checkpoint(lstm, generator, train_loader, 1000, LATENT_DIM, device, flatten_input=True)
        dataset.close_files()
    # 모든 그룹 학습 후 테스트셋 평가
    print("\n[대기] 테스트셋 평가")
    test_dataset = CombinedDataset(test_files, None, use_atmos=True, use_iono=False, var_filter=ATMOS_VARS, altitude_filter=None)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    # (test_loader로 평가 코드 필요시 추가)
    test_dataset.close_files()
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'lstm_state_dict': lstm.state_dict(),
        'input_size': input_dim,
        'latent_dim': LATENT_DIM
    }, 'atmosphere_model.pth')
    print("[대기] 모델 저장 완료: atmosphere_model.pth")
    print("\n[대기 변수 리스트 및 shape]")
    for v in ATMOS_VARS:
        print(v)
    print(f"총 변수 개수: {len(ATMOS_VARS)}")

    # 2. 이온 파일 분리
    files = IONO_FILES.copy()
    random.shuffle(files)
    n = len(files)
    train_files = files[:int(0.8*n)]
    test_files = files[int(0.8*n):]
    print(f"[이온] 훈련 파일 개수: {len(train_files)}, 테스트 파일 개수: {len(test_files)}")
    num_groups = ceil(len(train_files) / group_size)
    first_group_files = train_files[:group_size]
    temp_dataset = CombinedDataset([], first_group_files, use_atmos=False, use_iono=True, var_filter=IONO_VARS, stride=STRIDE, altitude_filter=None)
    # 실제 데이터 샘플로 input_dim 계산
    sample_data = temp_dataset[0]
    input_dim = sample_data.numel()  # 실제 데이터 차원
    print(f"[이온] 실제 입력 차원: {input_dim}")
    generator_iono = Generator(LATENT_DIM, input_dim).to(device)
    critic_iono = Critic(input_dim).to(device)
    lstm_iono = LSTM(input_size=input_dim, sequence_length=10).to(device)
    for i in range(num_groups):
        group_files = train_files[i*group_size:(i+1)*group_size]
        print(f"\n[이온] 그룹 {i+1}/{num_groups} 파일: {group_files}")
        dataset = CombinedDataset([], group_files, use_atmos=False, use_iono=True, var_filter=IONO_VARS, stride=STRIDE, altitude_filter=None)
        total_len = len(dataset)
        if total_len < 2:
            print("  [경고] 데이터 샘플이 너무 적어 스킵합니다.")
            continue
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        print(f"  [이온] 그룹 {i+1} 학습 시작 (1000 에포크)")
        g_losses_iono, c_losses_iono = train_wgan_with_checkpoint(generator_iono, critic_iono, train_loader, 1000, LATENT_DIM, device, flatten_input=True)
        lstm_losses_iono = train_lstm_with_checkpoint(lstm_iono, generator_iono, train_loader, 1000, LATENT_DIM, device, flatten_input=True)
        dataset.close_files()
    print("\n[이온] 테스트셋 평가")
    test_dataset = CombinedDataset([], test_files, use_atmos=False, use_iono=True, var_filter=IONO_VARS, altitude_filter=None)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    # (test_loader로 평가 코드 필요시 추가)
    test_dataset.close_files()
    torch.save({
        'generator_state_dict': generator_iono.state_dict(),
        'critic_state_dict': critic_iono.state_dict(),
        'lstm_state_dict': lstm_iono.state_dict(),
        'input_size': input_dim,
        'latent_dim': LATENT_DIM
    }, 'ionosphere_model.pth')
    print("[이온] 모델 저장 완료: ionosphere_model.pth")
    print("\n[이온 변수 리스트 및 shape]")
    for v in IONO_VARS:
        print(v)
    print(f"총 변수 개수: {len(IONO_VARS)}")

if __name__ == "__main__":
    # 진단: 첫 번째 WACCM 파일의 변수명, 차원명, shape 모두 출력
    if WACCM_FILES:
        import xarray as xr
        print("[진단] 첫 번째 WACCM 파일 변수/차원/shape 정보:")
        with xr.open_dataset(WACCM_FILES[0]) as ds:
            print(ds)
            for var in ds.variables:
                print(f"{var}: dims={ds[var].dims}, shape={ds[var].shape}")
    main() 