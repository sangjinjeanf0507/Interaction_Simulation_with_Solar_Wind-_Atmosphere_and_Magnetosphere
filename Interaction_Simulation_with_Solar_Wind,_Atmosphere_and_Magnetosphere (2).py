import taichi as ti
import numpy as np
import math
from datetime import datetime
import ppigrf  # pyigrf를 ppigrf로 수정
import pandas as pd
import os
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
import time

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU 메모리 정리 함수
def clear_gpu_memory():
    """GPU 메모리를 정리하여 메모리 부족 문제를 방지합니다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
def print_gpu_memory():
    """현재 GPU 메모리 사용량을 출력합니다."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
        print(f"GPU 메모리: 할당됨 {allocated:.2f}GB, 예약됨 {reserved:.2f}GB")

# 예측 시간 범위 설정 (6시간부터 18시간까지)
PREDICTION_START_HOURS = 6   # 예측 시작 시간
PREDICTION_END_HOURS = 30    # 예측 종료 시간
PREDICTION_HOURS = PREDICTION_END_HOURS - PREDICTION_START_HOURS  # 총 12시간 예측

# 200km에 해당하는 시뮬레이션 단위 상수 정의
ALTITUDE_200KM_SIM_UNIT = 4.0

# 시뮬레이션 실행 시간 범위 설정 (06:00 ~ 30:00)
SIM_START_HOUR = 6
SIM_END_HOUR = 30
SIM_START_SECONDS = SIM_START_HOUR * 3600.0
SIM_END_SECONDS = SIM_END_HOUR * 3600.0

# LSTM + Generator 하이브리드 모델 클래스 정의
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

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, sequence_length=None):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # sequence_length가 None이면 PREDICTION_HOURS 사용 (6-18시간 범위)
        self.sequence_length = sequence_length if sequence_length is not None else PREDICTION_HOURS
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 출력을 6-18시간 범위로 변경
        self.fc = nn.Linear(hidden_size, input_size * self.sequence_length)
        
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
            x = x.unsqueeze(1).repeat(1, 18, 1)  # 18시간 입력 시퀀스
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 마지막 은닉 상태에서 6-18시간 범위 예측 (12시간)
        out = self.fc(out[:, -1, :])
        # (batch_size, input_size * sequence_length) -> (batch_size, sequence_length, input_size)
        out = out.view(out.size(0), self.sequence_length, -1)
        return out

# 모델 저장 경로 설정
MODEL_SAVE_PATH = r'C:\Users\sunma\.vscode\models\hybrid_model.pth'
MODEL_200KM_SAVE_PATH = r'C:\Users\sunma\.vscode\models\hybrid_model_200km.pth'
LATENT_DIM = 100

# 하이브리드 모델 가용성 확인
HYBRID_MODELS_AVAILABLE = False
try:
    if os.path.exists(MODEL_200KM_SAVE_PATH):
        print("200km 이상 고도 하이브리드 모델 파일 발견")
        HYBRID_MODELS_AVAILABLE = True
    elif os.path.exists(MODEL_SAVE_PATH):
        print("전체 고도 하이브리드 모델 파일 발견")
        HYBRID_MODELS_AVAILABLE = True
    else:
        print("하이브리드 모델 파일을 찾을 수 없습니다.")
        print("atmosphpere_AI_learnig.py를 먼저 실행하여 모델을 훈련시켜주세요.")
except Exception as e:
    print(f"하이브리드 모델 확인 실패: {e}")

# 레거시 LSTM import 시도
try:
    from lstm_generator_hybrid import (
        load_lstm_only,  # LSTM만 로드하는 함수 사용
        predict_with_saved_model, 
        HybridTimeSeriesDataset
    )
    LSTM_AVAILABLE = True
    print("레거시 LSTM 모델 로드 함수 import 성공")
except Exception as e:
    print(f"레거시 LSTM 모델 로드 실패: {e}")
    LSTM_AVAILABLE = False

# 하이브리드 모델 로드 및 보간 함수들
def load_hybrid_model(model_path, model_type="atmosphere"):
    """하이브리드 모델(LSTM + GAN) 로드"""
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return None, None, None, None
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"하이브리드 모델 로드됨: {model_path}")
        
        # 200km 이상 고도 모델인지 확인
        if 'atmosphere_high' in checkpoint:
            model_data = checkpoint['atmosphere_high']
            print("200km 이상 고도 대기 모델 감지")
        elif 'ionosphere_high' in checkpoint:
            model_data = checkpoint['ionosphere_high']
            print("200km 이상 고도 이온 모델 감지")
        elif 'atmosphere' in checkpoint:
            model_data = checkpoint['atmosphere']
            print("전체 고도 대기 모델 감지")
        elif 'ionosphere' in checkpoint:
            model_data = checkpoint['ionosphere']
            print("전체 고도 이온 모델 감지")
        else:
            # 직접적인 모델 형태
            model_data = checkpoint
            print("직접 모델 데이터 감지")
        
        input_dim = model_data.get('input_size', 16)
        latent_dim = model_data.get('latent_dim', LATENT_DIM)
        
        # 모델 생성 (6-18시간 예측 범위)
        generator = Generator(latent_dim, input_dim).to(device)
        lstm = LSTM(input_size=input_dim, sequence_length=PREDICTION_HOURS).to(device)
        
        # 가중치 로드 (크기 호환성 체크)
        if 'generator_state_dict' in model_data:
            try:
                generator.load_state_dict(model_data['generator_state_dict'])
                print("Generator 가중치 로드 성공")
            except Exception as e:
                print(f"Generator 가중치 로드 실패: {e}")
        
        if 'lstm_state_dict' in model_data:
            try:
                # LSTM 가중치 크기 호환성 체크
                lstm_state = model_data['lstm_state_dict']
                current_fc_weight_shape = lstm.fc.weight.shape
                saved_fc_weight_shape = lstm_state['fc.weight'].shape
                
                if current_fc_weight_shape == saved_fc_weight_shape:
                    lstm.load_state_dict(lstm_state)
                    print(f"LSTM 가중치 로드 성공 (크기: {current_fc_weight_shape})")
                else:
                    print(f"LSTM 모델 크기 불일치:")
                    print(f"  기존 모델: {saved_fc_weight_shape}")
                    print(f"  새 모델: {current_fc_weight_shape}")
                    print("  새로운 모델로 초기화됩니다.")
            except Exception as e:
                print(f"LSTM 가중치 로드 실패: {e}")
                print("LSTM 가중치는 로드하지 않고 새로 초기화된 상태로 진행합니다.")
        
        print(f"하이브리드 모델 로드 완료 (입력 차원: {input_dim}, 잠재 차원: {latent_dim})")
        return generator, lstm, input_dim, latent_dim
        
    except Exception as e:
        print(f"하이브리드 모델 로드 실패: {e}")
        return None, None, None, None

def interpolate_with_hybrid_model(generator, lstm, input_data, num_interpolations=10):
    """하이브리드 모델을 사용한 보간"""
    if generator is None or lstm is None:
        print("하이브리드 모델이 로드되지 않았습니다.")
        return None
    
    print(f"하이브리드 모델 보간 시작: 입력 크기 {input_data.shape}")
    
    interpolated_results = []
    
    with torch.no_grad():
        generator.eval()
        lstm.eval()
        
        for i in range(num_interpolations):
            # LSTM 6-18시간 범위 예측
            lstm_output = lstm(input_data)  # (1, 12, input_size)
            
            # Generator로 보간 데이터 생성
            noise = torch.randn(1, LATENT_DIM, device=device)
            generated_data = generator(noise)  # (1, input_size)
            
            # LSTM 출력의 시간별 평균을 Generator 출력과 결합
            if len(lstm_output.shape) == 3:  # (1, 12, input_size)
                lstm_mean = torch.mean(lstm_output, dim=1)  # (1, input_size)
            else:
                lstm_mean = lstm_output
            
            # LSTM 출력과 Generator 출력의 가중 평균으로 보간
            interpolation_weight = 0.7  # LSTM 가중치
            interpolated = (interpolation_weight * lstm_mean + 
                          (1 - interpolation_weight) * generated_data)
            
            interpolated_results.append(interpolated.cpu().numpy())
            
            if (i + 1) % 5 == 0:
                print(f"6-18시간 보간 진행률: {(i+1)/num_interpolations*100:.1f}%")
    
    print("하이브리드 모델 보간 완료!")
    return np.array(interpolated_results)

def get_altitude_from_position(sim_y):
    """시뮬레이션 y 좌표를 고도로 변환 (km)"""
    return sim_y / KM_TO_SIM

def should_use_high_altitude_model(sim_y):
    """200km 이상 고도인지 확인"""
    altitude_km = get_altitude_from_position(sim_y)
    return altitude_km >= 200.0

# 전역 하이브리드 모델 변수들
GLOBAL_HYBRID_GENERATOR = None
GLOBAL_HYBRID_LSTM = None
GLOBAL_HIGH_ALT_GENERATOR = None
GLOBAL_HIGH_ALT_LSTM = None
HYBRID_INPUT_DIM = 16
HIGH_ALT_INPUT_DIM = 16

def initialize_hybrid_models():
    """하이브리드 모델들 초기화"""
    global GLOBAL_HYBRID_GENERATOR, GLOBAL_HYBRID_LSTM, HYBRID_INPUT_DIM
    global GLOBAL_HIGH_ALT_GENERATOR, GLOBAL_HIGH_ALT_LSTM, HIGH_ALT_INPUT_DIM
    
    print("\n=== 하이브리드 모델 초기화 ===")
    
    # GPU 메모리 정리
    clear_gpu_memory()
    print_gpu_memory()
    
    models_loaded = 0
    
    # 200km 이상 고도 모델 로드 시도
    if os.path.exists(MODEL_200KM_SAVE_PATH):
        print("200km 이상 고도 모델 로드 시도...")
        try:
            gen, lstm, input_dim, latent_dim = load_hybrid_model(MODEL_200KM_SAVE_PATH)
            if gen is not None and lstm is not None:
                GLOBAL_HIGH_ALT_GENERATOR = gen
                GLOBAL_HIGH_ALT_LSTM = lstm
                HIGH_ALT_INPUT_DIM = input_dim
                print("200km 이상 고도 모델 로드 성공")
                models_loaded += 1
            else:
                print("200km 이상 고도 모델 로드 실패")
        except Exception as e:
            print(f"200km 이상 고도 모델 로드 중 오류: {e}")
    
    # 전체 고도 모델 로드 시도
    if os.path.exists(MODEL_SAVE_PATH):
        print("전체 고도 모델 로드 시도...")
        try:
            gen, lstm, input_dim, latent_dim = load_hybrid_model(MODEL_SAVE_PATH)
            if gen is not None and lstm is not None:
                GLOBAL_HYBRID_GENERATOR = gen
                GLOBAL_HYBRID_LSTM = lstm
                HYBRID_INPUT_DIM = input_dim
                print("전체 고도 모델 로드 성공")
                models_loaded += 1
            else:
                print("전체 고도 모델 로드 실패")
        except Exception as e:
            print(f"전체 고도 모델 로드 중 오류: {e}")
    
    success = models_loaded > 0
    print(f"하이브리드 모델 초기화 {'성공' if success else '실패'} (로드된 모델: {models_loaded}개)")
    
    if not success:
        print("하이브리드 모델을 사용할 수 없습니다. 기본 모드로 진행합니다.")
    
    return success

# 안정성을 위한 CPU 모드 강제 사용
print("안정성을 위해 CPU 모드로 초기화합니다...")
print("(6-18시간 하이브리드 예측은 GPU에서 정상 작동)")
ti.init(arch=ti.cpu, cpu_max_num_threads=8)  # 8코어 최적화

# --- 1. 전역 상수 및 필드 선언 ---
mu_0 = 4 * np.pi * 1e-7 # 투자율 (상수, SI 단위: H/m 또는 N/A^2)
sigma_kernel = 10.0 / (7.0 * np.pi) # 2D Cubic Spline Kernel 정규화 상수 (무차원)
gamma = 5.0 / 3.0 # 비열비 (이상 기체 가정, 무차원)

# IGRF 모델 설정
IGRF_DATE = datetime.now().year + (datetime.now().month - 1) / 12.0  # 현재 날짜
EARTH_RADIUS = 6371.0  # 지구 반지름 (km)
BASE_ALTITUDE = 0.0   # 시뮬레이션 시작 고도 (지표면, km)
SIMULATION_HEIGHT = 1000.0  # 시뮬레이션 영역 높이 (km)
SIMULATION_WIDTH = 1000.0  # 시뮬레이션 영역 너비 (km)

# 시뮬레이션 스케일 변환 상수
KM_TO_SIM = 0.02  # 1km를 시뮬레이션 단위로 변환 (50km = 1 시뮬레이션 단위)
TESLA_TO_SIM = 1e9  # Tesla를 nT로 변환

dimension = 2
num_particles_max = 18550 # 최대 입자 수 - 5만개로 조정 (개)
dt = 300.0 # 시간 간격 (5분 단위, 초)

# 시뮬레이션 영역 (시뮬레이션 길이 단위)
domain_min = -10.0  # x축: -1000km에 해당
domain_max = 10.0   # x축: +1000km에 해당
y_domain_min = 0.0  # y축: 0km 고도에 해당 (지표면)
y_domain_max = 20.0 # y축: 1000km 고도에 해당
domain_center = ti.Vector([0.0, 0.0]) # 시뮬레이션 영역의 중앙 (세종기지 위치)
magnet_center = ti.Vector([0.0, 0.0]) # 세종기지 위치로 설정

# 입자 배치 간격
particle_spacing = (domain_max - domain_min) / 50  # GUI 넓이에 맞게 조정
particle_rows = 20  # 초기 입자 행 수
particle_height = domain_max - domain_min  # GUI 높이에 맞게 조정

hex_spacing = 0.02

# 자기 확산 계수 (자기장 확산 제어, 시뮬레이션 스케일에 맞춰 조정된 값)
magnetic_diffusivity = 5e-4 

# 중력 관련 상수 (중력 없음: 0으로 설정, 비활성화)
gravitational_constant = 5e-3 # 중력 상수 (비활성화)
center_mass = 0 # 중심 질량 (비활성화)

# 영구 자석 (자기 쌍극자) 관련 상수 (배경 자기장)
initial_magnet_moment_strength = 0.0  # 배경 자기장 비활성화
magnet_moment_strength = initial_magnet_moment_strength
magnet_moment_direction = ti.Vector([-1.0, 0.0])

# 시각화 관련 상수
magnetic_field_scale = 0.5 # GUI에서 보이는 자기장 벡터 시각화 스케일 (무차원)
num_grid_points = 50 # 전체 자기장 흐름 시각화를 위한 격자 해상도 (개)

# 각 입자의 자기장 크기의 기준이 됩니다. (테슬라, T)
base_B_magnitude = 0.05 # 시뮬레이션 스케일에 맞춰 조정된 값

# --- 배경 자기장 임계값 --- (테슬라, T)
B_magnitude_threshold = 5e-8 # 시뮬레이션 스케일에 맞춰 조정된 값 (예: 50 uT)

# --- 압력 상한선 --- (시뮬레이션 압력 단위, Pa로 추정)
max_pressure_cap = 100000000000000000000000.0

# --- 밀도 상한선 --- (실제 물리 단위: m^-3)
max_density_cap = 5.0 * 1e10 # 5e6 m^-3 (이는 5 cm^-3를 m^-3으로 변환한 값)
density_cap_activation_distance = 0.2 # (시뮬레이션 길이 단위)

# --- 속도 상한선 --- (미터/초, m/s)
max_velocity_cap = 1000000000000000000.0

# --- 새로운 입자 그룹 관련 상수 (특수 입자) ---
# 각 유형별 입자 수 (개)
num_ions = 2000
num_electron_core = 2000
num_electron_halo = 2000
num_special_particles = num_ions + num_electron_core + num_electron_halo # 총 특수 입자 수

# 특수 입자 초기화 및 재배치 스폰 영역 (왼쪽 끝, 위에서 아래까지)
special_particle_spawn_x_min = domain_min + 0.1 - 2.0 # 왼쪽 끝에서 약간 오른쪽으로
special_particle_spawn_x_max = domain_min + 0.5 - 2.0 # 왼쪽 끝에서 약간 오른쪽으로, 범위 조정
special_particle_spawn_y_min = domain_min
special_particle_spawn_y_max = domain_max

# 이온, 전자 코어, 전자 헤일로의 초기 속도 (미터/초, m/s)
ion_initial_velocity = ti.Vector([4.0, 0.0]) # 400 km/s -> 400,000 m/s
electron_core_initial_velocity = ti.Vector([4.0, 0.0]) # 400 km/s -> 400,000 m/s
electron_halo_initial_velocity_low = ti.Vector([2.0, 0.0]) # 200 km/s -> 200,000 m/s
electron_halo_initial_velocity_high = ti.Vector([10.0, 0.0]) # 1000 km/s -> 1,000,000 m/s

# 이온의 초기 자기장 (나노테슬라 -> 테슬라, nT -> T)
ion_initial_B_magnitude = 9000000000000000000000.0 # 5 nT -> 5e-9 T

# 입자 유형 식별을 위한 상수 (0: 일반, 1: 이온, 2: 전자 코어, 3: 전자 헤일로, 4: SPMHD 입자, 무차원)
PARTICLE_TYPE_NORMAL = 0
PARTICLE_TYPE_ION = 1
PARTICLE_TYPE_ELECTRON_CORE = 2
PARTICLE_TYPE_ELECTRON_HALO = 3
PARTICLE_TYPE_SPMHD = 4

is_special_particle_type = ti.field(dtype=ti.i32, shape=num_particles_max) # 특수 입자인지 여부 및 유형 (0: 일반, 1: 이온, 2: 전자 코어, 3: 전자 헤일로, 4: SPMHD 입자)

# 자기장 없는 구역 및 입자 재배치 관련 상수 (시뮬레이션 길이 단위)
magnetic_free_zone_radius = 0.0  # 자기장 없는 구역 비활성화

# --- 사용자 조절 변수: 헥사곤 격자의 '최소 배치 반경' --- (시뮬레이션 길이 단위)
initial_placement_min_radius = 1.1 # 초기값 설정 (자기장 없는 구역 반경보다 약간 크게)

reposition_radius_min = magnetic_free_zone_radius + 0.1 # 재배치 시 최소 반경
reposition_radius_max = magnetic_free_zone_radius + 0.5 # 재배치 시 최대 반경

# --- 일반 입자 재배치 (Repopulation) 관련 상수 ---
repopulation_check_radius = 2.0  # 이 반경 내에서 입자 수를 확인 (시뮬레이션 길이 단위)
num_desired_particles_in_center = 100 # 중앙 영역에 유지하고 싶은 최소 일반 입자 수
repopulation_search_attempts = 100 # 재배치 시도 횟수

# --- 특수 입자 재배치 (Repopulation) 관련 상수 --- (새로 추가)
special_particle_repop_check_center = ti.Vector([special_particle_spawn_x_min + (special_particle_spawn_x_max - special_particle_spawn_x_min) / 2.0, domain_center.y])
special_particle_repop_check_radius = 2.0 # 이 반경 내에서 특수 입자 수를 확인 (시뮬레이션 길이 단위)
num_desired_total_special_particles = num_ions + num_electron_core + num_electron_halo # 총 특수 입자 수 유지
special_repopulation_search_attempts = 100 # 재배치 시도 횟수

# 입자 간 최소 거리 (충돌 방지 및 고른 배치, 시뮬레이션 길이 단위)
min_particle_distance = 0.03
min_particle_distance_sq = min_particle_distance * min_particle_distance

# 격자점에서의 자기장 벡터를 저장할 필드
grid_pos = ti.Vector.field(dimension, dtype=ti.f32, shape=num_grid_points * num_grid_points) # (시뮬레이션 길이 단위)
grid_B_interpolated = ti.Vector.field(dimension, dtype=ti.f32, shape=num_grid_points * num_grid_points) # (테슬라, T)

# 순수한 영구 자석 자기장을 위한 격자 필드 (배경 자기장 시각화)
grid_B_dipole_only = ti.Vector.field(dimension, dtype=ti.f32, shape=num_grid_points * num_grid_points) # (테슬라, T)

# --- 2. Taichi 필드 정의 (모든 입자 데이터는 여기에 저장) ---
pos = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max) # 위치 (시뮬레이션 길이 단위)
vel = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max) # 속도 (m/s)
mass = ti.field(dtype=ti.f32, shape=num_particles_max) # 질량 (정규화된 값, 무차원)
u = ti.field(dtype=ti.f32, shape=num_particles_max) # 내부 에너지 (질량당 에너지, 시뮬레이션 에너지 단위)
rho = ti.field(dtype=ti.f32, shape=num_particles_max) # 밀도 (m^-3)
P_pressure = ti.field(dtype=ti.f32, shape=num_particles_max) # 압력 (시뮬레이션 압력 단위, Pa로 추정)
B = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max) # 자기장 벡터 (테슬라, T)
acc = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max) # 가속도 (시뮬레이션 가속도 단위, m/s^2로 추정)

# 실제 입자 수를 저장할 필드들
num_particles = ti.field(dtype=ti.i32, shape=())
num_actual_particles = ti.field(dtype=ti.i32, shape=())
num_normal_particles = ti.field(dtype=ti.i32, shape=())


etha_a_dt_field = ti.field(dtype=ti.f32, shape=num_particles_max) # 입자별 내부 에너지 변화율
ae_k_field = ti.field(dtype=ti.f32, shape=num_particles_max) # 입자별 운동 에너지
etha_a_field = ti.field(dtype=ti.f32, shape=num_particles_max) # 입자별 총 에너지 (혹은 누적 에너지)
B_unit_field = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max) # 자기장 단위 벡터 (무차원)

# 스트레스 텐서 필드
S_a_field = ti.Matrix.field(dimension, dimension, dtype=ti.f32, shape=num_particles_max)
S_b_field = ti.Matrix.field(dimension, dimension, dtype=ti.f32, shape=num_particles_max)

# 인공 점성 등을 위한 smoothing length (시뮬레이션 길이 단위)
h_smooth = ti.field(dtype=ti.f32, shape=num_particles_max)

# 자기장 업데이트를 위한 임시 필드 (dB/dt 저장)
dB_dt = ti.Vector.field(dimension, dtype=ti.f32, shape=num_particles_max)

# --- 입자별 인공 점성 계수 필드 ---
alpha_visc_p = ti.field(dtype=ti.f32, shape=num_particles_max) # 입자별 alpha_visc (무차원)
beta_visc_p = ti.field(dtype=ti.f32, shape=num_particles_max) # 입자별 beta_visc (무차원)

# 전자의 전하량 상수 정의 (쿨롱, C)
electron_charge = -1.602176634e-19 # 전자의 기본 전하량 (C)
# SPH 입자당 유효 전하를 시뮬레이션 스케일에 맞게 조정
# 이 값은 물리적 정확성보다는 시뮬레이션 내 자기장 강도를 조절하는 용도로 사용될 수 있습니다.
# 예를 들어, 한 입자가 '특정 부피'의 전하를 대표한다고 가정.
# 여기에 시뮬레이션 스케일에 맞는 임의의 승수를 곱함.
effective_electron_charge_per_particle = electron_charge * 1e25 # 시뮬레이션 스케일에 맞는 유효 전하 (조정 가능, C)

# 비오-사바르 법칙에 의한 자기장 강도를 조절하는 스케일 팩터.
# 이 값을 조정하여 전자 움직임에 의한 자기장의 영향력을 조절할 수 있습니다.
biot_savart_scale_factor = 9000000000000000.0 # (시뮬레이션 스케일에 맞춰 조정된 값, 단위는 복합적)

# --- GAN으로 생성된 입자를 위한 필드 추가 ---
gan_pos = ti.Vector.field(dimension, dtype=ti.f32, shape=30900)  # 3만900개의 보간 입자
gan_vel = ti.Vector.field(dimension, dtype=ti.f32, shape=30900)
gan_B = ti.Vector.field(dimension, dtype=ti.f32, shape=30900)
gan_rho = ti.field(dtype=ti.f32, shape=30900)
gan_P_pressure = ti.field(dtype=ti.f32, shape=30900)
gan_u = ti.field(dtype=ti.f32, shape=30900)
gan_acc = ti.Vector.field(dimension, dtype=ti.f32, shape=30900)
gan_is_200km_model = ti.field(dtype=ti.i32, shape=30900)  # 200km 학습 모델 입자 플래그

# GAN 입자 초기화 상태 추적
gan_particles_initialized = ti.field(dtype=ti.i32, shape=())
gan_particle_count = ti.field(dtype=ti.i32, shape=())
gan_reposition_count = ti.field(dtype=ti.i32, shape=())  # 재배치된 입자 수 추적

# IGRF 자기장 계산을 위한 전역 변수
base_lat = -74.6270  # 장보고 과학기지 위도 (남위 74.6270도)
base_lon = 164.2288  # 장보고 과학기지 경도 (동경 164.2288도)
current_date = datetime(2024, 1, 1)

# 배경 자기장 계산 함수
@ti.func
def calculate_magnetic_field(pos):
    # 시뮬레이션 좌표를 실제 지리 좌표로 변환
    x_km = pos[0] / KM_TO_SIM
    y_km = pos[1] / KM_TO_SIM
    
    # x, y 좌표를 위도, 경도 변화로 변환
    delta_lat = y_km / 111.0  # 1도는 약 111km
    delta_lon = x_km / (111.0 * ti.cos(ti.math.radians(base_lat)))
    
    lat = base_lat + delta_lat
    lon = base_lon + delta_lon
    alt = BASE_ALTITUDE + y_km
    
    # IGRF 모델 계산 (정적 값 사용 - 실시간 계산은 Taichi에서 불가능)
    # 한반도 주변 평균적인 IGRF 값 사용
    Be = 4500.0   # 동쪽 방향 자기장 (nT)
    Bn = 30000.0  # 북쪽 방향 자기장 (nT)
    
    # 고도에 따른 자기장 감쇠 계산 (거리의 세제곱에 반비례)
    r = (EARTH_RADIUS + alt) / EARTH_RADIUS
    field_factor = 1.0 / (r * r * r)
    
    # 위도에 따른 자기장 변화 (단순화된 쌍극자 모델)
    lat_rad = ti.math.radians(lat)
    field_factor *= (1.0 + 3.0 * ti.math.sin(lat_rad) * ti.math.sin(lat_rad)) ** 0.5
    
    # nT를 시뮬레이션 단위로 변환
    Bx = Bn * field_factor
    By = Be * field_factor
    
    return ti.Vector([Bx, By]) / TESLA_TO_SIM

# --- 3. SPH 커널 함수 (@ti.kernel 및 @ti.func) ---
@ti.func
def W(r, h):
    """SPH Cubic Spline 커널 함수."""
    q = r / h
    alpha = sigma_kernel / (h**dimension)
    result = 0.0
    if 0 <= q < 1:
        result = alpha * (1.0 - 1.5 * q**2 + 0.75 * q**3)
    elif 1 <= q < 2:
        result = alpha * (0.25 * (2.0 - q)**3)
    return result

@ti.func
def grad_W(r_vec, r, h):
    """SPH Cubic Spline 커널 함수의 기울기."""
    q = r / h
    alpha = sigma_kernel / (h**dimension)
    gradient_result = ti.Vector([0.0, 0.0])
    if r < 1e-9: # r=0에서의 0으로 나누기 방지
        pass
    else:
        dw_dq = 0.0
        if 0 <= q < 1:
            dw_dq = alpha * (-3.0 * q + 2.25 * q**2)
        elif 1 <= q < 2:
            dw_dq = alpha * (-0.75 * (2.0 - q)**2)
        gradient_result = dw_dq * r_vec / (r * h)
    return gradient_result

@ti.func
def get_dipole_B_field(p_pos, center, moment_dir, moment_strength_val):
    """
    2D 자기 쌍극자 자기장(B-field)을 계산합니다.
    """
    r_vec = p_pos - center
    r_norm = r_vec.norm()
    result_B_field = ti.Vector([0.0, 0.0])
    if moment_strength_val >= 1e-9 and r_norm >= 1e-5: # 특이점 주변 피하기
        dx = p_pos.x - center.x
        dy = p_pos.y - center.y
        r_sq = r_norm * r_norm
        r_pow_5 = r_sq * r_sq * r_norm
        base_Bx = (3.0 * dx * dy) / r_pow_5
        base_By = (3.0 * dy * dy - r_sq) / r_pow_5
        angle = ti.atan2(moment_dir.y, moment_dir.x)
        cos_a = ti.cos(angle)
        sin_a = ti.sin(angle)
        rotated_B_x = base_Bx * cos_a - base_By * sin_a
        rotated_B_y = base_Bx * sin_a + base_By * cos_a
        result_B_field = moment_strength_val * ti.Vector([rotated_B_x, rotated_B_y])
    return result_B_field

@ti.func
def is_position_valid(candidate_pos, current_idx, current_num_particles, min_dist_sq_val, free_zone_radius_val, min_placement_radius_val):
    """
    주어진 위치가 다음 조건을 만족하는지 확인합니다:
    1. 자기장 없는 구역 (free_zone_radius_val) 밖에 있을 것.
    2. 최소 배치 반경 (min_placement_radius_val)보다 멀리 있을 것.
    3. 다른 기존 입자들과 최소 거리 이상 떨어져 있을 것.
    """
    is_valid_flag = True # 플래그 변수

    dist_from_magnet_center = (candidate_pos - magnet_center).norm()

    # 1. 자기장 없는 구역 확인
    if dist_from_magnet_center < free_zone_radius_val:
        is_valid_flag = False

    # 2. 최소 배치 반경 확인 (free_zone_radius_val보다 큰 경우에만 의미 있음)
    if is_valid_flag and dist_from_magnet_center < min_placement_radius_val:
        is_valid_flag = False

    # 3. 기존 입자들과의 거리 확인 (is_valid_flag가 아직 True일 때만 검사)
    for j in range(current_num_particles):
        if is_valid_flag:
            if j == current_idx:
                pass
            else:
                dist_sq = (candidate_pos - pos[j]).dot(candidate_pos - pos[j])
                if dist_sq < min_dist_sq_val:
                    is_valid_flag = False

    return is_valid_flag

@ti.kernel
def compute_electron_initial_B_field(num_el_core: ti.i32, num_el_halo: ti.i32, initial_idx_offset: ti.i32, charge_per_particle: ti.f32, bs_scale_factor: ti.f32):
    """
    전자 코어 및 전자 헤일로 입자들의 초기 자기장을 비오-사바르 법칙을 이용하여 계산합니다.
    (각 전자 입자의 위치에서 다른 모든 전자 입자들의 전류 요소에 의한 자기장 합산)
    """
    for i in range(num_el_core + num_el_halo): # 모든 전자 입자에 대해 반복
        current_electron_idx = initial_idx_offset + i

        target_B = ti.Vector([0.0, 0.0]) # 이 전자 입자가 받을 자기장

        # 다른 모든 전자 입자(j)가 현재 전자 입자(i) 위치에 생성하는 자기장을 합산
        for j in range(num_el_core + num_el_halo):
            if i == j:
                continue # 자기 자신은 제외

            other_electron_idx = initial_idx_offset + j

            r_vec = pos[current_electron_idx] - pos[other_electron_idx] # j에서 i로 향하는 벡터
            r = r_vec.norm()

            # 자기장 기여 계산 (2D 비오-사바르 간략화)
            if r > 1e-9: # 0으로 나누기 방지
                effective_current_x = charge_per_particle * vel[other_electron_idx].x
                effective_current_y = charge_per_particle * vel[other_electron_idx].y

                # r_vec is from j to i, so pos[i] - pos[j]
                dx = r_vec.x
                dy = r_vec.y
                r_sq = r * r

                if r_sq > 1e-9: # Avoid division by zero
                    # Bx contribution from current along X
                    dBx_from_cx = - effective_current_x * dy / r_sq
                    # Bx contribution from current along Y
                    dBx_from_cy = effective_current_y * dx / r_sq

                    # By contribution from current along X
                    dBy_from_cx = effective_current_x * dx / r_sq
                    # By contribution from current along Y
                    dBy_from_cy = - effective_current_y * dy / r_sq

                    # Sum up for the target B
                    target_B.x += (dBx_from_cx + dBx_from_cy) * bs_scale_factor
                    target_B.y += (dBy_from_cx + dBy_from_cy) * bs_scale_factor

        # 각 전자 입자의 B 필드에 계산된 비오-사바르 자기장 기여를 할당
        # 이 자기장은 초기 조건으로 한 번만 계산됩니다.
        if is_special_particle_type[current_electron_idx] == PARTICLE_TYPE_ELECTRON_CORE or \
           is_special_particle_type[current_electron_idx] == PARTICLE_TYPE_ELECTRON_HALO:
            B[current_electron_idx] = target_B
            # 자기장 크기 상한선 설정 (너무 커지는 것을 방지)
            if B[current_electron_idx].norm() > 1e-7: # 예시 상한선
                B[current_electron_idx] = B[current_electron_idx].normalized() * 1e-7


@ti.kernel
def init_particles_kernel(initial_mag_strength: ti.f32, start_row: ti.i32, end_row: ti.i32, initial_min_radius: ti.f32,
                          num_ions_param: ti.i32, num_electron_core_param: ti.i32, num_electron_halo_param: ti.i32,
                          ion_vel: ti.template(), electron_core_vel: ti.template(),
                          electron_halo_vel_low: ti.template(), electron_halo_vel_high: ti.template(),
                          ion_b_mag: ti.f32,
                          spawn_x_min: ti.f32, spawn_x_max: ti.f32, spawn_y_min: ti.f32, spawn_y_max: ti.f32):
    
    total_normal_particles = num_normal_particles[None]
    current_particle_idx = 0
    
    # 일반 입자를 랜덤하게 배치
    for i in range(total_normal_particles):
        # x 좌표: domain_min ~ domain_max 사이 랜덤
        x_pos = domain_min + ti.random(ti.f32) * (domain_max - domain_min)
        # y 좌표: y_domain_min ~ y_domain_max 사이 랜덤
        y_pos = y_domain_min + ti.random(ti.f32) * (y_domain_max - y_domain_min)
        
        pos[current_particle_idx] = ti.Vector([x_pos, y_pos])
        vel[current_particle_idx] = ti.Vector([0.0, 0.0])
        mass[current_particle_idx] = 1.0
        u[current_particle_idx] = 1.0
        rho[current_particle_idx] = 1.0
        P_pressure[current_particle_idx] = 0.0
        B[current_particle_idx] = ti.Vector([0.0, 0.0])
        h_smooth[current_particle_idx] = 0.2
        alpha_visc_p[current_particle_idx] = 1.0
        beta_visc_p[current_particle_idx] = 2.0
        is_special_particle_type[current_particle_idx] = PARTICLE_TYPE_NORMAL
        current_particle_idx += 1

    num_actual_particles[None] = current_particle_idx
    num_particles[None] = current_particle_idx

@ti.kernel
def add_special_particle_B_contributions():
    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL:
            total_B = ti.Vector([0.0, 0.0])
            for j in range(num_actual_particles[None]):
                if is_special_particle_type[j] != PARTICLE_TYPE_NORMAL:
                    r_vec = pos[i] - pos[j]
                    r = r_vec.norm()
                    if r > 1e-6:
                        current_j = effective_electron_charge_per_particle * vel[j]
                        # 2D 비오-사바르 법칙 간이 계산
                        dB = biot_savart_scale_factor * ti.Vector([-current_j.y, current_j.x]) / (r * r)
                        total_B += dB
            B[i] += total_B

@ti.kernel
def induce_B_from_special_particles():
    """특수 입자의 전류 운동이 일반 입자에게 자기장을 유도하도록 합니다."""
    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL:
            total_B = ti.Vector([0.0, 0.0])
            for j in range(num_actual_particles[None]):
                if is_special_particle_type[j] != PARTICLE_TYPE_NORMAL:
                    r_vec = pos[i] - pos[j]
                    r = r_vec.norm()
                    if r > 1e-6:
                        current_j = effective_electron_charge_per_particle * vel[j]
                        dB = biot_savart_scale_factor * ti.Vector([-current_j.y, current_j.x]) / (r * r)
                        total_B += dB
            B[i] += total_B


@ti.kernel
def compute_sph_properties():
    """최적화된 SPH 계산: 2회 루프로 밀도와 힘을 계산합니다."""
    # 모든 입자 초기화
    for i in range(num_actual_particles[None]):
        rho[i] = 0.0
        acc[i] = ti.Vector([0.0, 0.0])
        etha_a_dt_field[i] = 0.0
        dB_dt[i] = ti.Vector([0.0, 0.0])

    # GAN 입자 초기화
    for i in range(30900):
        gan_acc[i] = ti.Vector([0.0, 0.0])

    # === 루프 1: 밀도 계산 (일반 + GAN 입자 모두 포함) ===
    # 일반 입자들 간의 밀도 계산
    for i in range(num_actual_particles[None]):
        for j in range(num_actual_particles[None]):
            if i != j:
                r_vec = pos[i] - pos[j]
                r = r_vec.norm()
                h_ij = (h_smooth[i] + h_smooth[j]) / 2.0
                if r < 2.0 * h_ij:
                    density_contribution = mass[j] * W(r, h_ij)
                    rho[i] += density_contribution

    # 일반 입자와 GAN 입자 간의 밀도 계산
    for i in range(num_actual_particles[None]):
        for j in range(30900):
            r_vec = pos[i] - gan_pos[j]
            r = r_vec.norm()
            h_ij = (h_smooth[i] + 0.2) / 2.0  # GAN 입자의 smoothing length를 0.2로 가정
            if r < 2.0 * h_ij:
                density_contribution = 0.1 * W(r, h_ij)  # GAN 입자의 질량을 0.1로 가정
                rho[i] += density_contribution

    # GAN 입자들 간의 밀도 계산
    for i in range(30900):
        for j in range(30900):
            if i != j:
                r_vec = gan_pos[i] - gan_pos[j]
                r = r_vec.norm()
                h_ij = 0.2  # GAN 입자의 smoothing length
                if r < 2.0 * h_ij:
                    density_contribution = 0.1 * W(r, h_ij)
                    # GAN 입자의 밀도는 별도로 계산하지 않음 (필요시 추가)

    # === 루프 2: 힘 계산 (대칭성 활용하여 중복 제거) ===
    # 일반 입자들 간의 힘 계산 (대칭성 활용)
    for i in range(num_actual_particles[None]):
        for j in range(i + 1, num_actual_particles[None]):  # i < j만 계산
            r_vec = pos[i] - pos[j]
            r = r_vec.norm()
            h_ij = (h_smooth[i] + h_smooth[j]) / 2.0
            if r < 2.0 * h_ij:
                # SPH 힘 계산 (압력, 점성, 자기장)
                force = compute_sph_force(i, j, r_vec, r, h_ij)
                acc[i] += force
                acc[j] -= force  # 대칭성 활용

    # 일반 입자와 GAN 입자 간의 힘 계산 (한 번만 계산)
    for i in range(num_actual_particles[None]):
        for j in range(30900):
            r_vec = pos[i] - gan_pos[j]
            r = r_vec.norm()
            h_ij = (h_smooth[i] + 0.2) / 2.0
            if r < 2.0 * h_ij:
                # 일반 입자 → GAN 입자 힘
                force_normal_to_gan = compute_sph_force_normal_to_gan(i, j, r_vec, r, h_ij)
                acc[i] += force_normal_to_gan
                gan_acc[j] -= force_normal_to_gan  # 대칭성 활용

    # GAN 입자들 간의 힘 계산 (대칭성 활용)
    for i in range(30900):
        for j in range(i + 1, 30900):  # i < j만 계산
            r_vec = gan_pos[i] - gan_pos[j]
            r = r_vec.norm()
            h_ij = 0.2
            if r < 2.0 * h_ij:
                # GAN 입자 간 SPH 힘 계산
                force = compute_sph_force_gan_to_gan(i, j, r_vec, r, h_ij)
                gan_acc[i] += force
                gan_acc[j] -= force  # 대칭성 활용

@ti.func
def compute_sph_force(i: ti.i32, j: ti.i32, r_vec: ti.types.vector(2, ti.f32), r: ti.f32, h_ij: ti.f32) -> ti.types.vector(2, ti.f32):
    """일반 입자 간 SPH 힘 계산"""
    # 결과 벡터 초기화
    result = ti.Vector([0.0, 0.0])
    
    # 거리가 너무 작으면 0 반환
    if r >= 1e-9:
        # 압력 힘
        P_i = (gamma - 1.0) * rho[i] * u[i]
        P_j = (gamma - 1.0) * rho[j] * u[j]
        pressure_force = -mass[j] * (P_i + P_j) / (2.0 * rho[j]) * grad_W(r_vec, r, h_ij)
        
        # 점성 힘
        v_ij = vel[i] - vel[j]
        viscosity_force = alpha_visc_p[i] * h_ij * mass[j] * v_ij.dot(grad_W(r_vec, r, h_ij)) / rho[j] * grad_W(r_vec, r, h_ij)
        
        # 자기장 힘
        B_i_norm_sq = B[i].dot(B[i])
        B_j_norm_sq = B[j].dot(B[j])
        magnetic_force = -mass[j] * ((B[i].outer_product(B[i]) - 0.5 * B_i_norm_sq * ti.Matrix.identity(ti.f32, dimension)) / rho[i] + 
                                     (B[j].outer_product(B[j]) - 0.5 * B_j_norm_sq * ti.Matrix.identity(ti.f32, dimension)) / rho[j]) / mu_0 @ grad_W(r_vec, r, h_ij)
        
        result = pressure_force + viscosity_force + magnetic_force
    
    return result

@ti.func
def compute_sph_force_normal_to_gan(i: ti.i32, j: ti.i32, r_vec: ti.types.vector(2, ti.f32), r: ti.f32, h_ij: ti.f32) -> ti.types.vector(2, ti.f32):
    """일반 입자와 GAN 입자 간 SPH 힘 계산"""
    # 결과 벡터 초기화
    result = ti.Vector([0.0, 0.0])
    
    # 거리가 너무 작으면 0 반환
    if r >= 1e-9:
        # 압력 힘
        P_i = (gamma - 1.0) * rho[i] * u[i]
        P_j = (gamma - 1.0) * gan_rho[j] * gan_u[j]
        pressure_force = -0.1 * (P_i + P_j) / (2.0 * gan_rho[j]) * grad_W(r_vec, r, h_ij)
        
        # 점성 힘
        v_ij = vel[i] - gan_vel[j]
        viscosity_force = alpha_visc_p[i] * h_ij * 0.1 * v_ij.dot(grad_W(r_vec, r, h_ij)) / gan_rho[j] * grad_W(r_vec, r, h_ij)
        
        # 자기장 힘
        B_i_norm_sq = B[i].dot(B[i])
        B_j_norm_sq = gan_B[j].dot(gan_B[j])
        magnetic_force = -0.1 * ((B[i].outer_product(B[i]) - 0.5 * B_i_norm_sq * ti.Matrix.identity(ti.f32, dimension)) / rho[i] + 
                                 (gan_B[j].outer_product(gan_B[j]) - 0.5 * B_j_norm_sq * ti.Matrix.identity(ti.f32, dimension)) / gan_rho[j]) / mu_0 @ grad_W(r_vec, r, h_ij)
        
        result = pressure_force + viscosity_force + magnetic_force
    
    return result

@ti.func
def compute_sph_force_gan_to_gan(i: ti.i32, j: ti.i32, r_vec: ti.types.vector(2, ti.f32), r: ti.f32, h_ij: ti.f32) -> ti.types.vector(2, ti.f32):
    """GAN 입자 간 SPH 힘 계산"""
    # 결과 벡터 초기화
    result = ti.Vector([0.0, 0.0])
    
    # 거리가 너무 작으면 0 반환
    if r >= 1e-9:
        # 압력 힘
        P_i = (gamma - 1.0) * gan_rho[i] * gan_u[i]
        P_j = (gamma - 1.0) * gan_rho[j] * gan_u[j]
        pressure_force = -0.1 * (P_i + P_j) / (2.0 * gan_rho[j]) * grad_W(r_vec, r, h_ij)
        
        # 점성 힘
        v_ij = gan_vel[i] - gan_vel[j]
        viscosity_force = 1.0 * h_ij * 0.1 * v_ij.dot(grad_W(r_vec, r, h_ij)) / gan_rho[j] * grad_W(r_vec, r, h_ij)
        
        # 자기장 힘
        B_i_norm_sq = gan_B[i].dot(gan_B[i])
        B_j_norm_sq = gan_B[j].dot(gan_B[j])
        magnetic_force = -0.1 * ((gan_B[i].outer_product(gan_B[i]) - 0.5 * B_i_norm_sq * ti.Matrix.identity(ti.f32, dimension)) / gan_rho[i] + 
                                 (gan_B[j].outer_product(gan_B[j]) - 0.5 * B_j_norm_sq * ti.Matrix.identity(ti.f32, dimension)) / gan_rho[j]) / mu_0 @ grad_W(r_vec, r, h_ij)
        
        result = pressure_force + viscosity_force + magnetic_force
    
    return result

@ti.kernel
def update_particles(dt: ti.f32, current_initial_placement_min_radius: ti.f32):
    """
    입자 속성 (내부 에너지, 자기장, 속도, 위치)를 업데이트하고 경계 조건을 처리합니다.
    current_initial_placement_min_radius는 현재 적용될 최소 배치 반경입니다.
    """
    for i in range(num_actual_particles[None]):
        if rho[i] > 1e-9:
            u[i] += (etha_a_dt_field[i] / rho[i]) * dt

        # 일반 입자 또는 이온에 대해서만 자기장 업데이트
        # 전자 입자의 B 필드는 초기 조건으로만 설정되며, 이후는 직접적으로 업데이트되지 않습니다.
        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL or \
           is_special_particle_type[i] == PARTICLE_TYPE_ION:
            B[i] += dB_dt[i] * dt

            B_norm_current = B[i].norm()
            if B_norm_current < B_magnitude_threshold:
               B[i] = ti.Vector([0.0, 0.0])

        ae_k_field[i] = 0.5 * mass[i] * vel[i].dot(vel[i])

        B_norm_i = B[i].norm()
        if B_norm_i > 1e-9:
            B_unit_field[i] = B[i] / B_norm_i
        else:
            B_unit_field[i] = ti.Vector([0.0, 0.0])

        vel[i] += acc[i] * dt

        # 속도 상한선 적용
        velocity_magnitude = vel[i].norm()
        if velocity_magnitude > max_velocity_cap:
            vel[i] = vel[i].normalized() * max_velocity_cap

        pos[i] += vel[i] * dt

        # --- 1. 중앙 자기장 0 구역 침범 시 입자 반사 ---
        dist_from_center = (pos[i] - magnet_center).norm()
        effective_inner_boundary_for_reflection = ti.max(magnetic_free_zone_radius, current_initial_placement_min_radius)

        if dist_from_center < effective_inner_boundary_for_reflection:
            new_pos_found = False
            attempts = 0
            max_reflection_attempts = 50

            original_pos = pos[i]
            original_vel = vel[i]

            while attempts < max_reflection_attempts and not new_pos_found:
                direction_from_center = original_pos - magnet_center
                if direction_from_center.norm() < 1e-9:
                    direction_from_center = ti.Vector([ti.random(ti.f32)*2-1, ti.random(ti.f32)*2-1]).normalized()

                candidate_pos = magnet_center + direction_from_center.normalized() * \
                                ti.random(ti.f32) * (reposition_radius_max - effective_inner_boundary_for_reflection) + effective_inner_boundary_for_reflection

                if is_position_valid(candidate_pos, i, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                    pos[i] = candidate_pos
                    new_pos_found = True

                attempts += 1

            if not new_pos_found:
                pos[i] = magnet_center + (original_pos - magnet_center).normalized() * effective_inner_boundary_for_reflection * 1.05
                vel[i] = -vel[i] * 0.8
                u[i] *= 0.9
            else:
                reflect_direction = (pos[i] - magnet_center).normalized()
                vel_dot_reflect = original_vel.dot(reflect_direction)
                vel[i] = original_vel - 2 * vel_dot_reflect * reflect_direction
                vel[i] *= 0.8
                u[i] *= 0.9


        # --- 2. 화면 밖으로 나간 입자 재배치 (일반 입자) ---
        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL and \
           not (domain_min <= pos[i].x <= domain_max and domain_min <= pos[i].y <= domain_max):

            new_pos_found = False
            attempts = 0
            max_reposition_attempts = 100

            while attempts < max_reposition_attempts and not new_pos_found:
                angle = ti.random(ti.f32) * 2 * math.pi
                effective_reposition_min_radius = ti.max(reposition_radius_min, current_initial_placement_min_radius)

                radius = ti.random(ti.f32) * (reposition_radius_max - effective_reposition_min_radius) + effective_reposition_min_radius

                candidate_pos = magnet_center + ti.Vector([radius * ti.cos(angle), radius * ti.sin(angle)])

                if is_position_valid(candidate_pos, i, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                    pos[i] = candidate_pos
                    vel[i] = ti.Vector([ti.random(ti.f32) * 2 - 1, ti.random(ti.f32) * 2 - 1]) * 5.0
                    u[i] = 1.0
                    rho[i] = 1.0 # 일반 입자의 밀도 초기화
                    B[i] = get_dipole_B_field(pos[i], magnet_center, magnet_moment_direction, magnet_moment_strength)
                    new_pos_found = True

                attempts += 1

            if not new_pos_found:
                pos[i].x = ti.max(domain_min, ti.min(domain_max, pos[i].x))
                pos[i].y = ti.max(domain_min, ti.min(domain_max, pos[i].y))
                vel[i] *= -0.5

@ti.kernel
def count_normal_particles_in_radius(check_radius: ti.f32) -> ti.i32:
    """
    중앙 영역(magnet_center) 내의 일반 입자 수를 세어 반환합니다.
    """
    count = 0
    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL:
            dist_from_center = (pos[i] - magnet_center).norm()
            if dist_from_center < check_radius:
                count += 1
    return count

@ti.kernel
def repopulate_particles_kernel(initial_mag_strength: ti.f32, current_initial_placement_min_radius: ti.f32,
                                desired_count: ti.i32, max_attempts_per_particle: ti.i32,
                                max_placement_rad_rep: ti.f32):
    """
    중앙 영역에 일반 입자가 부족할 경우, 새로 입자를 생성하여 배치합니다.
    """
    current_normal_particle_count = 0
    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL and \
           (pos[i] - magnet_center).norm() < repopulation_check_radius:
            current_normal_particle_count += 1

    num_to_add = desired_count - current_normal_particle_count

    if num_to_add > 0:
        test_pos_for_max_B = magnet_center + ti.Vector([magnetic_free_zone_radius + 0.1, 0.0])
        max_B_norm_estimate = get_dipole_B_field(test_pos_for_max_B, magnet_center, magnet_moment_direction, initial_mag_strength).norm()
        if max_B_norm_estimate < 1e-9:
            max_B_norm_estimate = 1.0

        for _ in range(num_to_add):
            if num_actual_particles[None] < num_particles_max:
                found_position = False
                attempts = 0
                while attempts < max_attempts_per_particle and not found_position:
                    # 새로운 입자가 생성될 수 있는 영역을 repopulation_check_radius 근처로 제한
                    angle = ti.random(ti.f32) * 2 * math.pi
                    radius = ti.random(ti.f32) * (max_placement_rad_rep - current_initial_placement_min_radius) + current_initial_placement_min_radius
                    candidate_pos = magnet_center + ti.Vector([radius * ti.cos(angle), radius * ti.sin(angle)])

                    if not (domain_min <= candidate_pos.x <= domain_max and \
                                         domain_min <= candidate_pos.y <= domain_max):
                        attempts += 1
                        continue

                    if is_position_valid(candidate_pos, -1, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                        idx = ti.atomic_add(num_actual_particles[None], 1)

                        if idx < num_particles_max: # 여기서 다시 한번 최종 확인
                            B_initial_direction = ti.Vector([0.0, 0.0])
                            current_B_at_pos = get_dipole_B_field(candidate_pos, magnet_center, magnet_moment_direction, initial_mag_strength)
                            current_B_norm = current_B_at_pos.norm()

                            if current_B_norm >= B_magnitude_threshold:
                                B_initial_direction = current_B_at_pos / current_B_norm

                            is_special_particle_type[idx] = PARTICLE_TYPE_NORMAL
                            alpha_visc_p[idx], beta_visc_p[idx] = 0.5, 0.5
                            vel[idx] = B_initial_direction * 1000.0
                            pos[idx] = candidate_pos
                            mass[idx] = 1.0 / num_particles_max
                            u[idx] = 1.0
                            rho[idx] = 1.0
                            acc[idx] = ti.Vector([0.0, 0.0])
                            P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                            h_smooth[idx] = 0.04
                            etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                            B_unit_field[idx] = ti.Vector([0.0, 0.0])
                            S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                            dB_dt[idx] = ti.Vector([0.0, 0.0])
                            B[idx] = get_dipole_B_field(candidate_pos, magnet_center, magnet_moment_direction, initial_mag_strength)

                            found_position = True
                        else:
                            num_actual_particles[None] -= 1 # 이미 증가된 카운트를 되돌립니다.
                    attempts += 1

@ti.kernel
def count_current_special_particles_in_zone() -> ti.i32:
    """
    특수 입자 재배치 구역 내의 특수 입자 수를 세어 반환합니다.
    """
    count = 0
    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] != PARTICLE_TYPE_NORMAL: # 특수 입자만 카운트
            dist = (pos[i] - special_particle_repop_check_center).norm()
            if dist < special_particle_repop_check_radius:
                count += 1
    return count

@ti.kernel
def repopulate_special_particles_kernel(
    initial_mag_strength: ti.f32,
    current_initial_placement_min_radius: ti.f32,
    desired_ions: ti.i32,
    desired_electron_core: ti.i32,
    desired_electron_halo: ti.i32,
    max_attempts_per_particle: ti.i32,
    spawn_x_min: ti.f32, spawn_x_max: ti.f32, spawn_y_min: ti.f32, spawn_y_max: ti.f32, # 스폰 영역
    ion_vel: ti.template(),
    electron_core_vel: ti.template(),
    electron_halo_vel_low: ti.template(),
    electron_halo_vel_high: ti.template(),
    ion_b_mag: ti.f32
):
    """
    특수 입자 스폰 영역 내의 특수 입자 수가 부족할 경우, 새로 입자를 생성하여 배치합니다.
    """
    current_ion_count = 0
    current_electron_core_count = 0
    current_electron_halo_count = 0

    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] == PARTICLE_TYPE_ION:
            current_ion_count += 1
        elif is_special_particle_type[i] == PARTICLE_TYPE_ELECTRON_CORE:
            current_electron_core_count += 1
        elif is_special_particle_type[i] == PARTICLE_TYPE_ELECTRON_HALO:
            current_electron_halo_count += 1

    # 이온 재배치
    num_ions_to_add = desired_ions - current_ion_count
    for _ in range(num_ions_to_add):
        if num_actual_particles[None] < num_particles_max:
            found_position = False
            attempts = 0
            while attempts < max_attempts_per_particle and not found_position:
                candidate_pos = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                           spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])

                if is_position_valid(candidate_pos, -1, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                    idx = ti.atomic_add(num_actual_particles[None], 1)
                    if idx < num_particles_max:
                        is_special_particle_type[idx] = PARTICLE_TYPE_ION
                        alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1
                        vel[idx] = ion_vel
                        pos[idx] = candidate_pos
                        mass[idx] = 1.0 / num_particles_max
                        u[idx] = 1.0
                        rho[idx] = 5.0 * 1e6
                        acc[idx] = ti.Vector([0.0, 0.0])
                        P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                        h_smooth[idx] = 0.04
                        etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                        B_unit_field[idx] = ti.Vector([0.0, 0.0])
                        S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                        dB_dt[idx] = ti.Vector([0.0, 0.0])
                        current_B_at_pos_dir = get_dipole_B_field(pos[idx], magnet_center, magnet_moment_direction, initial_mag_strength).normalized()
                        if current_B_at_pos_dir.norm() < 1e-9:
                            current_B_at_pos_dir = ti.Vector([1.0, 0.0])
                        B[idx] = current_B_at_pos_dir * ion_b_mag
                        found_position = True
                    else:
                        num_actual_particles[None] -= 1
                attempts += 1

    # 전자 코어 재배치
    num_electron_core_to_add = desired_electron_core - current_electron_core_count
    for _ in range(num_electron_core_to_add):
        if num_actual_particles[None] < num_particles_max:
            found_position = False
            attempts = 0
            while attempts < max_attempts_per_particle and not found_position:
                candidate_pos = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                           spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])
                if is_position_valid(candidate_pos, -1, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                    idx = ti.atomic_add(num_actual_particles[None], 1)
                    if idx < num_particles_max:
                        is_special_particle_type[idx] = PARTICLE_TYPE_ELECTRON_CORE
                        alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1
                        vel[idx] = electron_core_vel
                        pos[idx] = candidate_pos
                        mass[idx] = 1.0 / num_particles_max
                        u[idx] = 1.0
                        rho[idx] = 5.0 * 1e6
                        acc[idx] = ti.Vector([0.0, 0.0])
                        P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                        h_smooth[idx] = 0.04
                        etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                        B_unit_field[idx] = ti.Vector([0.0, 0.0])
                        S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                        dB_dt[idx] = ti.Vector([0.0, 0.0])
                        B[idx] = ti.Vector([0.0, 0.0]) # 초기에는 0으로 설정
                        found_position = True
                    else:
                        num_actual_particles[None] -= 1
                attempts += 1

    # 전자 헤일로 재배치
    num_electron_halo_to_add = desired_electron_halo - current_electron_halo_count
    for _ in range(num_electron_halo_to_add):
        if num_actual_particles[None] < num_particles_max:
            found_position = False
            attempts = 0
            while attempts < max_attempts_per_particle and not found_position:
                candidate_pos = ti.Vector([spawn_x_min + ti.random(ti.f32) * (spawn_x_max - spawn_x_min),
                                           spawn_y_min + ti.random(ti.f32) * (spawn_y_max - spawn_y_min)])
                if is_position_valid(candidate_pos, -1, num_actual_particles[None], min_particle_distance_sq, magnetic_free_zone_radius, current_initial_placement_min_radius):
                    idx = ti.atomic_add(num_actual_particles[None], 1)
                    if idx < num_particles_max:
                        is_special_particle_type[idx] = PARTICLE_TYPE_ELECTRON_HALO
                        alpha_visc_p[idx], beta_visc_p[idx] = 0.1, 0.1
                        if ti.random(ti.f32) < 0.5:
                            vel[idx] = electron_halo_vel_low
                        else:
                            vel[idx] = electron_halo_vel_high
                        pos[idx] = candidate_pos
                        mass[idx] = 1.0 / num_particles_max
                        u[idx] = 1.0
                        rho[idx] = 5.0 * 1e6
                        acc[idx] = ti.Vector([0.0, 0.0])
                        P_pressure[idx] = (gamma - 1.0) * rho[idx] * u[idx]
                        h_smooth[idx] = 0.04
                        etha_a_dt_field[idx], ae_k_field[idx], etha_a_field[idx] = 0.0, 0.0, u[idx] * mass[idx]
                        B_unit_field[idx] = ti.Vector([0.0, 0.0])
                        S_a_field[idx], S_b_field[idx] = ti.Matrix.zero(ti.f32, dimension, dimension), ti.Matrix.zero(ti.f32, dimension, dimension)
                        dB_dt[idx] = ti.Vector([0.0, 0.0])
                        B[idx] = ti.Vector([0.0, 0.0]) # 초기에는 0으로 설정
                        found_position = True
                    else:
                        num_actual_particles[None] -= 1
                attempts += 1


@ti.kernel
def visualize_magnetic_field_grid_kernel(current_mag_strength: ti.f32):
    """전반적인 자기장 흐름 시각화를 위해 자기장을 균일한 격자로 보간합니다."""
    x_grid_spacing = (domain_max - domain_min) / num_grid_points
    y_grid_spacing = (y_domain_max - y_domain_min) / num_grid_points
    
    for i_grid, j_grid in ti.ndrange(num_grid_points, num_grid_points):
        idx = i_grid * num_grid_points + j_grid
        
        # x 좌표는 -10 ~ +10 범위
        x_pos = i_grid * x_grid_spacing + x_grid_spacing / 2.0 + domain_min
        # y 좌표는 0 ~ 20 범위
        y_pos = j_grid * y_grid_spacing + y_grid_spacing / 2.0 + y_domain_min
        
        p_grid_pos = ti.Vector([x_pos, y_pos])
        grid_pos[idx] = p_grid_pos
        
        # IGRF 자기장 계산 - 강도를 증가시켜 더 잘 보이게 함
        B_field = get_igrf_B_field(p_grid_pos)
        grid_B_interpolated[idx] = B_field * 2.0  # 자기장 강도를 2배로 증가

@ti.func
def get_igrf_B_field(pos_sim):
    # 시뮬레이션 좌표를 실제 물리적 거리로 변환 (km)
    x_km = pos_sim[0] / KM_TO_SIM  # x 방향 거리 (km)
    y_km = pos_sim[1] / KM_TO_SIM  # y 방향 거리 (km)
    
    # 지구 곡률을 고려한 위도, 경도, 고도 계산
    surface_distance = ti.sqrt(x_km * x_km + y_km * y_km)  # km
    bearing = ti.atan2(x_km, y_km)  # 방위각 (라디안)
    
    # 지구 곡률을 고려한 중심각 계산
    central_angle = surface_distance / EARTH_RADIUS
    
    # 위도 계산 (세종기지 위도 기준)
    base_lat_rad = ti.math.radians(base_lat)  # base_lat = -62.2
    lat_rad = ti.asin(
        ti.math.sin(base_lat_rad) * ti.math.cos(central_angle) +
        ti.math.cos(base_lat_rad) * ti.math.sin(central_angle) * ti.math.cos(bearing)
    )
    lat = ti.math.degrees(lat_rad)
    
    # 경도 계산 (세종기지 경도 기준)
    delta_lon_rad = ti.atan2(
        ti.math.sin(bearing) * ti.math.sin(central_angle) * ti.math.cos(base_lat_rad),
        ti.math.cos(central_angle) - ti.math.sin(base_lat_rad) * ti.math.sin(lat_rad)
    )
    lon = base_lon + ti.math.degrees(delta_lon_rad)  # base_lon = -58.8
    
    # 고도 계산
    curvature_height = EARTH_RADIUS * (1.0 - ti.math.cos(central_angle))
    alt = curvature_height + y_km  # 지표면 위 고도
    
    # 남극 지역 IGRF 평균값 (2024년 기준 근사값)
    # 세종기지 위치에서의 실제 IGRF 값에 가깝게 조정
    Be = -18000.0  # 동쪽 방향 자기장 (nT)
    Bn = 5000.0    # 북쪽 방향 자기장 (nT)
    Bu = -52000.0  # 수직 방향 자기장 (nT, 남극에서는 위쪽이 음수)
    
    # 고도에 따른 자기장 감쇠 계산 (거리의 세제곱에 반비례)
    r = (EARTH_RADIUS + alt) / EARTH_RADIUS
    field_factor = 1.0 / (r * r * r)
    
    # 위도에 따른 자기장 변화 (쌍극자 모델)
    dipole_factor = ti.sqrt(3.0 * ti.math.sin(lat_rad) * ti.math.sin(lat_rad) + 1.0)
    
    # 자기장 성분 계산
    # 수직 성분 (위도에 따라 크게 변화)
    Bz = Bu * field_factor * ti.math.sin(lat_rad) * dipole_factor
    
    # 수평 성분 (위도에 따라 감소)
    Bh = ti.sqrt(Bn*Bn + Be*Be) * field_factor * ti.math.cos(lat_rad)
    
    # 2D 평면에서의 자기장 벡터 계산
    # x 방향: 수평 성분과 수직 성분의 합성
    # y 방향: 동쪽 방향 성분
    Bx = Bh * ti.math.cos(lat_rad) + Bz * ti.math.sin(lat_rad)
    By = Be * field_factor
    
    result_B = ti.Vector([Bx, By]) / TESLA_TO_SIM
    return result_B

@ti.kernel
def update_background_B_field():
    for i in range(num_particles[None]):
        # 각 입자 위치에서 IGRF 자기장 계산
        B[i] = get_igrf_B_field(pos[i])

# 지구 곡률 관련 계산 함수 추가
@ti.func
def calculate_curvature_height(distance_km):
    """지구 곡률로 인한 고도 변화 계산 (km)"""
    R = EARTH_RADIUS  # 지구 반지름 (km)
    h = R - ti.sqrt(R * R - distance_km * distance_km)
    return h

@ti.kernel
def calculate_grid_coordinates(grid_positions: ti.template(),
                             grid_lats: ti.template(),
                             grid_lons: ti.template(),
                             grid_alts: ti.template(),
                             grid_curvature_heights: ti.template()):
    """격자점의 위도, 경도, 고도를 계산"""
    n = num_grid_points
    fixed_lon = base_lon  # 세종기지 경도로 고정
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            
            # y축: 고도 (80km에서 1000km까지)
            altitude_ratio = j / (n - 1)
            altitude_km = BASE_ALTITUDE + altitude_ratio * SIMULATION_HEIGHT
            
            # x축: 위도 변화 (남에서 북으로)
            lat_range = 20.0  # 위도 범위 ±10도
            lat_ratio = i / (n - 1)
            latitude = base_lat + (lat_ratio - 0.5) * lat_range
            
            # 지구 곡률 효과 계산
            distance_from_base = ti.abs(latitude - base_lat) * 111.0  # km
            curvature_height = calculate_curvature_height(distance_from_base)
            
            # 결과 저장
            grid_positions[idx] = ti.Vector([i / (n - 1) * domain_max * 2 - domain_max,
                                          j / (n - 1) * domain_max])
            grid_lats[idx] = latitude
            grid_lons[idx] = fixed_lon
            grid_alts[idx] = altitude_km
            grid_curvature_heights[idx] = curvature_height

def save_grid_coordinates_to_csv():
    """격자점 좌표를 CSV 파일로 저장"""
    n = num_grid_points
    
    # Taichi 필드 생성
    grid_lats = ti.field(dtype=ti.f32, shape=n * n)
    grid_lons = ti.field(dtype=ti.f32, shape=n * n)
    grid_alts = ti.field(dtype=ti.f32, shape=n * n)
    grid_curvature_heights = ti.field(dtype=ti.f32, shape=n * n)
    
    # 좌표 계산
    calculate_grid_coordinates(
        grid_pos,
        grid_lats,
        grid_lons,
        grid_alts,
        grid_curvature_heights
    )
    
    # NumPy 배열로 변환
    grid_pos_np = grid_pos.to_numpy()
    grid_lats_np = grid_lats.to_numpy()
    grid_lons_np = grid_lons.to_numpy()
    grid_alts_np = grid_alts.to_numpy()
    grid_curvature_heights_np = grid_curvature_heights.to_numpy()
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'sim_x': grid_pos_np[:, 0],
        'sim_y': grid_pos_np[:, 1],
        'latitude': grid_lats_np,
        'longitude': grid_lons_np,
        'altitude_km': grid_alts_np,
        'curvature_height_km': grid_curvature_heights_np
    })
    
    # CSV 파일로 저장
    output_file = 'C:\\Users\\sunma\\.vscode\\simuldatasave\\grid.csv'
    df.to_csv(output_file, index=False)
    print(f"격자점 좌표가 {output_file}에 저장되었습니다.")
    
    # 데이터 요약 출력
    print("\n데이터 요약:")
    print(f"총 격자점 수: {len(df)}")
    print("\n위도 범위:")
    print(f"최소: {df['latitude'].min():.2f}°")
    print(f"최대: {df['latitude'].max():.2f}°")
    print("\n고도 범위:")
    print(f"최소: {df['altitude_km'].min():.2f} km")
    print(f"최대: {df['altitude_km'].max():.2f} km")
    print("\n지구 곡률에 의한 고도 변화:")
    print(f"최대: {df['curvature_height_km'].max():.2f} km")

# --- 4. 메인 시뮬레이션 루프 (Python) ---
def main_simulation_loop(resume_from_checkpoint=False):
    global magnet_moment_strength
    global initial_placement_min_radius
    global sim_time, frame_count
    
    print(f"[DEBUG] main_simulation_loop 시작: resume_from_checkpoint={resume_from_checkpoint}")
    
    initial_placement_min_radius = 0.0
    num_normal_particles[None] = 100000  # 입자 수를 100000개로 설정
    
    print("[DEBUG] 1단계: GAN 입자 로드 시작")
    # GAN으로 생성된 입자 데이터 로드
    try:
        load_gan_particles()
        print("GAN 입자 로드 성공")
    except Exception as e:
        print(f"GAN 입자 로드 실패: {e}")
        print("GAN 입자 없이 시뮬레이션을 계속합니다.")
    
    print("[DEBUG] 2단계: 시뮬레이션 시간 설정")
    # 체크포인트에서 복원된 경우가 아니라면 시뮬레이션 시간 초기화
    if not resume_from_checkpoint:
        sim_time = SIM_START_SECONDS
        frame_count = 0
        print(f"새로운 시뮬레이션 시작: {sim_time/3600:.2f}시간")
    else:
        print(f"체크포인트에서 복원된 시뮬레이션 계속: {sim_time/3600:.2f}시간, 프레임 {frame_count}")
        print(f"[DEBUG] main_simulation_loop: sim_time={sim_time}, frame_count={frame_count}")
    
    print("[DEBUG] 3단계: WACCM 데이터 로드 시작")
    # WACCM 데이터 로드
    waccm_data = load_waccm_data()
    
    # 관측 파일 초기화
    initialize_observation_files(waccm_data, resume_from_checkpoint)
    
    if waccm_data is None:
        print("WACCM 데이터를 불러올 수 없습니다. 기본 초기화를 사용합니다.")
        # WACCM 데이터가 없을 때는 WACCM에서 초기화된 입자 수를 0으로 설정
        n_particles = 0
        init_particles_kernel(
            initial_magnet_moment_strength,
            -10, 10,
            initial_placement_min_radius,
            num_ions, num_electron_core, num_electron_halo,
            ion_initial_velocity, electron_core_initial_velocity,
            electron_halo_initial_velocity_low, electron_halo_initial_velocity_high,
            ion_initial_B_magnitude,
            special_particle_spawn_x_min, special_particle_spawn_x_max,
            y_domain_min, y_domain_max
        )
    else:
        # WACCM 데이터로 입자 초기화
        print("WACCM 데이터로 입자 초기화 중...")
        
        # 필요한 입자 수만큼 샘플링 (CPU 모드 최적화: 최대 10000개)
        max_waccm_particles = 10000  # CPU 모드에서 충분한 입자 수
        n_particles = min(max_waccm_particles, num_normal_particles[None], len(waccm_data))
        if len(waccm_data) > n_particles:
            selected_indices = np.random.choice(len(waccm_data), size=n_particles, replace=False)
            selected_data = waccm_data.iloc[selected_indices]
        else:
            selected_data = waccm_data
            n_particles = len(selected_data)
        
        # 사용 가능한 컬럼 확인 및 기본값 설정
        columns = selected_data.columns
        
        # 위치 데이터 (sim_x, sim_y가 있으면 사용, 없으면 랜덤)
        if 'sim_x' in columns and 'sim_y' in columns:
            x_pos = selected_data['sim_x'].to_numpy()
            y_pos = selected_data['sim_y'].to_numpy()
        else:
            print("시뮬레이션 좌표를 찾을 수 없습니다. 랜덤 위치를 사용합니다.")
            x_pos = np.random.uniform(domain_min, domain_max, n_particles)
            y_pos = np.random.uniform(y_domain_min, y_domain_max, n_particles)
        
        # 속도/온도 데이터
        u_data = np.zeros(n_particles)
        v_data = np.zeros(n_particles)
        
        if 'U' in columns:
            u_data = selected_data['U'].to_numpy()
        elif 'T' in columns:
            u_data = selected_data['T'].to_numpy() * 0.01  # 온도를 속도로 변환
        else:
            u_data = np.random.uniform(-1, 1, n_particles)
        
        if 'V' in columns:
            v_data = selected_data['V'].to_numpy()
        elif 'PS' in columns:
            v_data = selected_data['PS'].to_numpy() * 1e-5  # 압력을 속도로 변환
        else:
            v_data = np.random.uniform(-1, 1, n_particles)
        
        # 밀도 데이터
        if 'Q' in columns:
            density_data = selected_data['Q'].to_numpy()
        elif 'RELHUM' in columns:
            density_data = selected_data['RELHUM'].to_numpy()
        elif 'T' in columns:
            density_data = selected_data['T'].to_numpy()
        else:
            density_data = np.ones(n_particles)
        
        # 압력 데이터
        if 'PS' in columns:
            pressure_data = selected_data['PS'].to_numpy()
        elif 'T' in columns:
            pressure_data = selected_data['T'].to_numpy()
        else:
            pressure_data = np.ones(n_particles) * 1e5
        
        # 날짜/시간 데이터
        date_data = selected_data['date'].to_numpy() if 'date' in columns else np.full(n_particles, 20141222)
        datesec_data = selected_data['datesec'].to_numpy() if 'datesec' in columns else np.zeros(n_particles)
        
        print(f"WACCM 데이터로 {n_particles}개 입자 초기화")
        print(f"사용된 컬럼: {list(columns)}")
        
        init_particles_from_waccm_kernel(
            x_pos, y_pos, u_data, v_data,
            density_data, pressure_data,
            date_data, datesec_data,
            n_particles
        )
    
    # SPMHD 입자 초기화
    print("SPMHD 입자 초기화 중...")
    print(f"SPMHD 격자: {num_spmhd_particles_x}x{num_spmhd_particles_y} = {num_spmhd_particles}개")
    
    import time
    start_time = time.time()
    init_spmhd_particles()
    end_time = time.time()
    
    print(f"SPMHD 입자 초기화 완료! (소요시간: {end_time-start_time:.2f}초)")
    
    # 총 입자 수 확인
    total_particles = num_actual_particles[None] + num_spmhd_particles
    print(f"총 활성 입자 수: {total_particles}개")
    print(f"  - WACCM 입자: {n_particles}개")
    print(f"  - 하이브리드 입자: 135,000개")  
    print(f"  - SPMHD 입자: {num_spmhd_particles}개")
    
    print("GUI 윈도우 생성 중...")
    window = ti.ui.Window('Earth Magnetosphere Simulation', (800, 800))
    canvas = window.get_canvas()
    print("GUI 초기화 완료! 시뮬레이션 루프 시작...")
    
    # 관측 주기(5분 = 300초)
    OBS_INTERVAL = 300.0
    
    # 체크포인트에서 복원된 경우가 아니라면 관측 시간 초기화
    if not resume_from_checkpoint:
        next_obs_time = SIM_START_SECONDS
        frame_count = 0
        sim_time = SIM_START_SECONDS  # 시뮬레이션 경과 시간 (초), 06:00 시작
        print(f"새로운 시뮬레이션: 관측 시작 시간 {next_obs_time/3600:.2f}시간")
    else:
        # 체크포인트에서 복원된 경우: 다음 관측 시간을 현재 시간부터 OBS_INTERVAL 후로 설정
        next_obs_time = sim_time + OBS_INTERVAL
        print(f"체크포인트 복원: 관측 시작 시간 {next_obs_time/3600:.2f}시간 (현재 시간 {sim_time/3600:.2f}시간 + {OBS_INTERVAL/60:.0f}분)")
    
    time_display_interval = 10  # 10프레임마다 시간 출력

    while window.running and sim_time < SIM_END_SECONDS:
        frame_count += 1
        # 다음 시간으로 진행하되 18:00을 넘지 않도록 클램프
        sim_time = min(sim_time + dt, SIM_END_SECONDS)  # 매 프레임마다 시간 증가 (5분 = 300초)
        
        # 시뮬레이션 시간 표시
        if frame_count <= 3 or frame_count % time_display_interval == 0:
            # 절대 시뮬레이션 시각(HH:MM:SS)과 경과 시간(시작 대비)을 함께 출력
            hours = int(sim_time // 3600)
            minutes = int((sim_time % 3600) // 60)
            seconds = int(sim_time % 60)

            elapsed = max(0.0, sim_time - SIM_START_SECONDS)
            e_h = int(elapsed // 3600)
            e_m = int((elapsed % 3600) // 60)
            e_s = int(elapsed % 60)

            if frame_count <= 3:
                print(f"시뮬레이션 프레임 {frame_count} | 시각: {hours:02d}:{minutes:02d}:{seconds:02d} | 경과: {e_h:02d}:{e_m:02d}:{e_s:02d}")
            else:
                print(f"시뮬레이션 시각: {hours:02d}:{minutes:02d}:{seconds:02d} | 경과: {e_h:02d}:{e_m:02d}:{e_s:02d} (프레임 {frame_count})")
        
        canvas.set_background_color((0, 0, 0))
        
        # 전기력 계산
        compute_electric_forces()
        
        # SPH 상호작용 계산 (최적화된 2회 루프)
        compute_sph_properties()
        
        # 입자 위치 업데이트
        update_particle_positions(dt)
        
        # 재배치된 보간 입자 수 출력
        repositioned_count = gan_reposition_count[None]
        if repositioned_count > 0:
            print(f"200km 학습 모델 입자 {repositioned_count}개가 200km 미만 고도로 내려가 재배치 완료")
        
        # 완전한 체크포인트 저장 (매 5프레임마다)
        if frame_count % 5 == 0:
            try:
                save_complete_checkpoint(sim_time, frame_count)
                
                # 메모리 사용량 모니터링 (매 20프레임마다)
                if frame_count % 20 == 0:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    print(f"메모리 사용량: {memory_mb:.1f} MB")
                    
            except Exception as e:
                print(f"체크포인트 저장 실패: {e}")
                import traceback
                traceback.print_exc()

        # 자기장 격자점 업데이트 및 시각화
        visualize_magnetic_field_grid_kernel(initial_magnet_moment_strength)
        
        # 자기장 시각화
        grid_pos_np = grid_pos.to_numpy()
        grid_B_np = grid_B_interpolated.to_numpy()
        
        # 격자점 위치를 GUI 좌표계로 변환
        display_grid_pos = np.zeros_like(grid_pos_np)
        display_grid_pos[:, 0] = (grid_pos_np[:, 0] - domain_min) / (domain_max - domain_min)
        display_grid_pos[:, 1] = (grid_pos_np[:, 1] - y_domain_min) / (y_domain_max - y_domain_min)
        
        # 자기장 벡터 정규화 및 스케일 조정
        max_B = np.max(np.linalg.norm(grid_B_np, axis=1))
        if max_B > 0:
            normalized_B = grid_B_np / max_B
            
            # 자기력선을 그리기 위한 배열 준비
            lines_start = []
            lines_end = []
            
            # 자기력선 데이터 수집
            for i in range(num_grid_points * num_grid_points):
                start_pos = display_grid_pos[i]
                B_direction_x = normalized_B[i][0] * magnetic_field_scale * 0.02
                B_direction_y = normalized_B[i][1] * magnetic_field_scale * 0.02
                B_direction = np.array([B_direction_x, B_direction_y])
                end_pos = start_pos + B_direction
                
                # 화면 내부의 벡터만 포함
                if (0 <= start_pos[0] <= 1 and 0 <= start_pos[1] <= 1 and
                    0 <= end_pos[0] <= 1 and 0 <= end_pos[1] <= 1):
                    lines_start.append(start_pos)
                    lines_end.append(end_pos)
            
            # 자기력선 그리기
            if lines_start:
                vertices = []
                for start, end in zip(lines_start, lines_end):
                    vertices.extend([start, end])
                
                vertices = np.array(vertices, dtype=np.float32)
                vertices_field = ti.Vector.field(2, dtype=ti.f32, shape=len(vertices))
                vertices_field.from_numpy(vertices)
                
                # 자기력선을 초록색으로 그리기
                canvas.lines(vertices_field, 
                           width=0.001, 
                           indices=None,
                           color=(0.0, 0.8, 0.0))
        
        # 입자 위치 데이터 준비
        pos_np = pos.to_numpy()[:num_actual_particles[None]]
        particle_types = is_special_particle_type.to_numpy()[:num_actual_particles[None]]
        
        # 좌표 변환
        display_pos_np = np.zeros_like(pos_np)
        display_pos_np[:, 0] = (pos_np[:, 0] - domain_min) / (domain_max - domain_min)
        display_pos_np[:, 1] = pos_np[:, 1] / y_domain_max
        
        # SPMHD 입자와 일반 입자를 분리
        spmhd_mask = particle_types == PARTICLE_TYPE_SPMHD
        normal_mask = ~spmhd_mask
        
        # 일반 입자 표시 (흰색)
        if np.any(normal_mask):
            normal_pos = display_pos_np[normal_mask]
            normal_circles = ti.Vector.field(2, dtype=ti.f32, shape=len(normal_pos))
            normal_circles.from_numpy(normal_pos)
            canvas.circles(normal_circles, radius=0.002, color=(1.0, 1.0, 1.0))
        
        # SPMHD 입자 표시 (파란색)
        if np.any(spmhd_mask):
            spmhd_pos = display_pos_np[spmhd_mask]
            spmhd_circles = ti.Vector.field(2, dtype=ti.f32, shape=len(spmhd_pos))
            spmhd_circles.from_numpy(spmhd_pos)
            canvas.circles(spmhd_circles, radius=0.002, color=(0.0, 0.0, 1.0))
        
        # GAN으로 생성된 입자 표시 (고도별 색상 구분)
        gan_pos_np = gan_pos.to_numpy()  # 모든 보간 입자 사용
        display_gan_pos = np.zeros_like(gan_pos_np)
        display_gan_pos[:, 0] = (gan_pos_np[:, 0] - domain_min) / (domain_max - domain_min)
        display_gan_pos[:, 1] = gan_pos_np[:, 1] / y_domain_max
        
        # 하이브리드 입자를 학습 모델별로 구분
        # 처음 절반(15450개): 200km 학습 모델(GLOBAL_HIGH_ALT) → 노란색 (전체 고도 범위)
        # 나머지 절반(15450개): 전체 고도 학습 모델(GLOBAL_HYBRID) → 초록색 (전체 고도 범위)
        particles_per_model = len(gan_pos_np) // 2
        
        # 200km 학습 모델로 생성된 입자 (노란색)
        if particles_per_model > 0:
            high_alt_model_pos = display_gan_pos[:particles_per_model]
            high_alt_model_circles = ti.Vector.field(2, dtype=ti.f32, shape=len(high_alt_model_pos))
            high_alt_model_circles.from_numpy(high_alt_model_pos)
            canvas.circles(high_alt_model_circles, radius=0.002, color=(1.0, 1.0, 0.0))  # 노란색
        
        # 전체 고도 학습 모델로 생성된 입자 (초록색)
        if len(gan_pos_np) > particles_per_model:
            general_model_pos = display_gan_pos[particles_per_model:]
            general_circles = ti.Vector.field(2, dtype=ti.f32, shape=len(general_model_pos))
            general_circles.from_numpy(general_model_pos)
            canvas.circles(general_circles, radius=0.002, color=(0.0, 1.0, 0.0))  # 초록색
        
        # 원점 표시 (빨간 점)
        origin_pos = ti.Vector.field(2, dtype=ti.f32, shape=1)
        origin_pos[0] = ti.Vector([(0 - domain_min) / (domain_max - domain_min), 0])  # 원점 위치 변환
        canvas.circles(origin_pos, radius=0.005, color=(1.0, 0.0, 0.0))  # 빨간색, 크기는 일반 입자보다 크게
        
        # 관측점 표시 (빨간 점)
        for obs_point in observation_points:
            obs_pos = ti.Vector.field(2, dtype=ti.f32, shape=1)
            obs_pos[0] = ti.Vector([
                (obs_point[0] - domain_min) / (domain_max - domain_min),
                obs_point[1] / y_domain_max
            ])
            canvas.circles(obs_pos, radius=0.005, color=(1.0, 0.0, 0.0))
        
        # 관측점 근처 입자 확인 및 데이터 수집 (10분 단위로만 실행)
        if sim_time >= next_obs_time:
            check_particles_near_observation_points(sim_time)
            save_observation_data(sim_time)
            next_obs_time += OBS_INTERVAL

        # 하이브리드 모델 기반 동적 입자 생성 (매 시간마다)
        if (HYBRID_MODELS_AVAILABLE or LSTM_AVAILABLE) and should_update_lstm(sim_time):
            perform_hybrid_update(sim_time)
        
        # Taichi UI는 동적 윈도우 제목 변경을 지원하지 않음
        # 대신 콘솔에 시간 정보 출력 (30프레임마다)
        if frame_count % 30 == 0:
            hours = int(sim_time // 3600)
            minutes = int((sim_time % 3600) // 60)
            elapsed = max(0.0, sim_time - SIM_START_SECONDS)
            print(f"시뮬레이션 시각: {hours:02d}:{minutes:02d} | 경과: {elapsed/3600.0:.2f}h")
        
        window.show()

    # 루프 종료 시점에 종료 로그 출력
    end_hours = int(SIM_END_SECONDS // 3600)
    end_minutes = int((SIM_END_SECONDS % 3600) // 60)
    end_seconds = int(SIM_END_SECONDS % 60)
    print(f"시뮬레이션 종료: {end_hours:02d}:{end_minutes:02d}:{end_seconds:02d}")

@ti.kernel
def update_particle_positions(dt: ti.f32):
    """입자들의 위치를 시간에 따라 업데이트"""
    # 재배치 카운터 초기화
    reposition_count = 0
    
    # 일반 입자 업데이트
    for i in range(num_actual_particles[None]):
        # 위치 업데이트
        pos[i] += vel[i] * dt
        
        # GUI 영역을 벗어난 입자 재배치
        if pos[i].x < domain_min or pos[i].x > domain_max or pos[i].y > y_domain_max:
            # x 좌표: domain_min ~ domain_max 사이 랜덤
            pos[i].x = domain_min + ti.random(ti.f32) * (domain_max - domain_min)
            # y 좌표: y_domain_min ~ y_domain_max 사이 랜덤
            pos[i].y = y_domain_min + ti.random(ti.f32) * (y_domain_max - y_domain_min)
        # 지표면 충돌 처리
        elif pos[i].y < y_domain_min:
            pos[i].y = y_domain_min  # 지표면 위치로 보정
            vel[i].y = -vel[i].y * 0.1  # y축 속도를 반대 방향으로 바꾸고 90% 감쇠

    # 보간 입자 업데이트
    for i in range(30900):  # 보간 입자 수 - 입자 수 3만900개로 조정
        # 위치 업데이트
        gan_pos[i] += gan_vel[i] * dt
        
        # 200km 학습 모델로 생성된 입자만 200km 미만 고도 제한 적용
        if gan_is_200km_model[i] == 1 and gan_pos[i].y < ALTITUDE_200KM_SIM_UNIT:
            # 200km 이상 고도로 재배치 (4.0 ~ 20.0 시뮬레이션 단위)
            gan_pos[i].x = domain_min + ti.random(ti.f32) * (domain_max - domain_min)
            gan_pos[i].y = ALTITUDE_200KM_SIM_UNIT + ti.random(ti.f32) * (y_domain_max - ALTITUDE_200KM_SIM_UNIT)
            # 속도 재설정 (아래쪽으로 이동하는 속도)
            gan_vel[i] = ti.Vector([0.0, -2.0])  # 200 km/s, 아래쪽 방향
            reposition_count += 1
        
        # GUI 영역을 벗어난 입자 재배치
        elif gan_pos[i].x < domain_min or gan_pos[i].x > domain_max or gan_pos[i].y > y_domain_max:
            # x 좌표: domain_min ~ domain_max 사이 랜덤
            gan_pos[i].x = domain_min + ti.random(ti.f32) * (domain_max - domain_min)
            # y 좌표: y_domain_min ~ y_domain_max 사이 랜덤
            gan_pos[i].y = y_domain_min + ti.random(ti.f32) * (y_domain_max - y_domain_min)
        # 지표면 충돌 처리
        elif gan_pos[i].y < y_domain_min:
            gan_pos[i].y = y_domain_min  # 지표면 위치로 보정
            gan_vel[i].y = -gan_vel[i].y * 0.1  # y축 속도를 반대 방향으로 바꾸고 90% 감쇠
    
    # 재배치된 입자 수를 전역 변수에 저장
    gan_reposition_count[None] = reposition_count

def load_waccm_data():
    try:
        # WACCM-X 데이터 파일 로드 (NetCDF 형식) - 메모리 효율적 처리
        waccm_file = 'F:\\ionodata\\f.e20.FXSD.f19_f19.001.cam.h1.2014-12-22-00000.nc'
        
        print("WACCM-X 데이터 파일 정보 확인 중...")
        
        # 먼저 데이터셋 정보만 확인 (지연 로딩)
        with xr.open_dataset(waccm_file, chunks={'time': 1}) as ds:
            print(f"데이터셋 차원: {dict(ds.dims)}")
            print(f"사용 가능한 변수: {list(ds.data_vars.keys())[:10]}...")  # 처음 10개만 표시
            
            # 필요한 차원과 변수만 선택
            # 시간 차원에서 06:00 시간 스텝 선택
            if 'time' in ds.dims:
                try:
                    target_time = np.datetime64('2014-12-22T06:00:00')
                    ds_subset = ds.sel(time=target_time, method='nearest')
                    print("06:00 시간 스텝 선택")
                except Exception:
                    # hour == 6인 인덱스를 탐색하거나, 없으면 근사 인덱스 사용
                    try:
                        hours = ds['time'].dt.hour.values
                        idx_candidates = np.where(hours == 6)[0]
                        time_index = int(idx_candidates[0]) if len(idx_candidates) > 0 else (6 if ds.sizes.get('time', 0) > 6 else 0)
                        ds_subset = ds.isel(time=time_index)
                        print(f"{int(hours[time_index]) if len(hours) > time_index else '인덱스'}시 시간 스텝 선택")
                    except Exception:
                        ds_subset = ds.isel(time=6 if ds.sizes.get('time', 0) > 6 else 0)
                        print("시간 좌표 정보 부족으로 인덱스 기반 06:00 근사 선택")
            else:
                ds_subset = ds
            
            # 필요한 변수들만 선택 (존재하는 것만)
            available_vars = list(ds_subset.data_vars.keys())
            required_vars = []
            
            # 위치 정보
            if 'lat' in available_vars:
                required_vars.append('lat')
            if 'lon' in available_vars:
                required_vars.append('lon')
            if 'lev' in available_vars or 'ilev' in available_vars:
                required_vars.append('lev' if 'lev' in available_vars else 'ilev')
            
            # 물리량 (존재하는 것만 선택)
            possible_vars = ['T', 'U', 'V', 'PS', 'PHIS', 'OMEGA', 'Q', 'RELHUM']
            for var in possible_vars:
                if var in available_vars:
                    required_vars.append(var)
                    if len(required_vars) >= 8:  # 최대 8개 변수만 선택
                        break
            
            print(f"선택된 변수: {required_vars}")
            
            if not required_vars:
                print("사용 가능한 변수를 찾을 수 없습니다.")
                return None
            
            # 선택된 변수들만 로드 (메모리 절약)
            ds_selected = ds_subset[required_vars]
            
            # 위도/경도 정보를 명시적으로 추가
            if 'lat' in ds_selected.dims and 'lon' in ds_selected.dims:
                # 위도/경도 좌표를 데이터 변수로 추가
                ds_selected['latitude'] = ds_selected.lat
                ds_selected['longitude'] = ds_selected.lon
                print("위도/경도 정보를 데이터 변수로 추가")
            
            # 지역 범위 제한 (남극 지역만 선택)
            if 'lat' in ds_selected.dims and 'lon' in ds_selected.dims:
                # 남극 지역 (-90도 ~ -50도, 전체 경도)
                lat_slice = slice(-90, -50)
                ds_selected = ds_selected.sel(lat=lat_slice)
                print("남극 지역으로 제한")
            
            # 고도 제한 (하위 30개 레벨만 선택)
            if 'lev' in ds_selected.dims:
                if len(ds_selected.lev) > 30:
                    ds_selected = ds_selected.isel(lev=slice(0, 30))
                    print("하위 30개 고도 레벨만 선택")
            elif 'ilev' in ds_selected.dims:
                if len(ds_selected.ilev) > 30:
                    ds_selected = ds_selected.isel(ilev=slice(0, 30))
                    print("하위 30개 고도 레벨만 선택")
            
            # 청크 단위로 로드하여 메모리 사용량 제한
            print("데이터를 청크 단위로 로드 중...")
            
            # DataFrame으로 변환하기 전에 크기 확인
            total_size_gb = ds_selected.nbytes / (1024**3)
            print(f"예상 메모리 사용량: {total_size_gb:.2f} GB")
            
            if total_size_gb > 10:  # 10GB 이상이면 추가 샘플링
                print("데이터가 너무 큽니다. 추가 샘플링을 수행합니다.")
                # 위도/경도를 2배씩 건너뛰어 샘플링
                if 'lat' in ds_selected.dims:
                    ds_selected = ds_selected.isel(lat=slice(None, None, 2))
                if 'lon' in ds_selected.dims:
                    ds_selected = ds_selected.isel(lon=slice(None, None, 2))
                print("공간 해상도를 1/4로 줄였습니다.")
            
            # float32로 변환하여 메모리 절약
            print("데이터 타입을 float32로 변환 중...")
            for var in ds_selected.data_vars:
                if ds_selected[var].dtype == 'float64':
                    ds_selected[var] = ds_selected[var].astype('float32')
            
            # 실제 로딩 및 DataFrame 변환
            print("DataFrame으로 변환 중...")
            df = ds_selected.to_dataframe()
            
            # NaN 값 제거
            df = df.dropna()
            
            # 시뮬레이션 좌표로 변환
            if 'latitude' in df.columns and 'longitude' in df.columns:
                print(f"위도/경도 컬럼 발견: latitude 범위 [{df['latitude'].min():.2f}, {df['latitude'].max():.2f}], longitude 범위 [{df['longitude'].min():.2f}, {df['longitude'].max():.2f}]")
                
                # 세종기지 기준 상대 좌표로 변환
                df['relative_lat'] = df['latitude'] - base_lat
                df['relative_lon'] = df['longitude'] - base_lon
                
                # 시뮬레이션 좌표로 변환 (km를 시뮬레이션 단위로)
                df['sim_x'] = df['relative_lon'] * 111.0 * np.cos(np.radians(df['latitude'])) * KM_TO_SIM
                df['sim_y'] = df['relative_lat'] * 111.0 * KM_TO_SIM
                
                print(f"시뮬레이션 좌표 변환 완료: sim_x 범위 [{df['sim_x'].min():.4f}, {df['sim_x'].max():.4f}], sim_y 범위 [{df['sim_y'].min():.4f}, {df['sim_y'].max():.4f}]")
                
                # 시뮬레이션 영역 내의 데이터만 선택
                mask = ((df['sim_x'] >= domain_min) & (df['sim_x'] <= domain_max) & 
                       (df['sim_y'] >= y_domain_min) & (df['sim_y'] <= y_domain_max))
                df = df[mask]
                
                print(f"시뮬레이션 영역 필터링 후 데이터 크기: {len(df)} 행")
            else:
                print("위도/경도 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼:", list(df.columns))
                print("가상의 시뮬레이션 좌표를 생성합니다.")
                
                # 가상의 시뮬레이션 좌표 생성 (테스트용)
                n_points = len(df)
                df['sim_x'] = np.random.uniform(domain_min, domain_max, n_points)
                df['sim_y'] = np.random.uniform(y_domain_min, y_domain_max, n_points)
                print(f"가상 좌표 생성 완료: {n_points}개 포인트")
            
            # 날짜/시간 정보 추가 (06:00으로 설정)
            df['date'] = 20141222
            df['datesec'] = 6 * 3600
            
            print(f"최종 데이터 크기: {len(df)} 행, {len(df.columns)} 열")
            print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            
            if len(df) == 0:
                print("시뮬레이션 영역 내에 데이터가 없습니다.")
                return None
            
            return df
            
    except MemoryError as e:
        print(f"메모리 부족 오류: {e}")
        print("더 작은 데이터 청크를 사용하거나 메모리를 늘려주세요.")
        return None
    except Exception as e:
        print(f"WACCM-X 데이터 파일 로드 오류: {e}")
        print("generated_particles.csv 파일을 사용합니다.")
        return None

@ti.kernel
def init_particles_from_waccm_kernel(
    x_pos: ti.types.ndarray(),
    y_pos: ti.types.ndarray(),
    u_vel: ti.types.ndarray(),  # 속도 성분 또는 온도
    v_vel: ti.types.ndarray(),  # 속도 성분 또는 압력
    density_data: ti.types.ndarray(),  # 밀도 관련 데이터
    pressure_data: ti.types.ndarray(), # 압력 관련 데이터
    date: ti.types.ndarray(),
    datesec: ti.types.ndarray(),
    num_particles_to_init: ti.i32
):
    """WACCM 데이터를 사용하여 입자를 초기화합니다."""
    for i in range(num_particles_to_init):
        current_particle_idx = i
        
        pos[current_particle_idx] = ti.Vector([x_pos[i], y_pos[i]])
        
        # 속도 설정 (물리적 단위를 시뮬레이션 단위로 변환)
        vel_scale = 0.001  # 속도 스케일링 팩터
        vel[current_particle_idx] = ti.Vector([u_vel[i] * vel_scale, v_vel[i] * vel_scale])
        
        # 기본 속성 설정
        mass[current_particle_idx] = 1.0
        u[current_particle_idx] = ti.abs(pressure_data[i]) * 1e-5  # 내부 에너지로 압력 사용 (스케일링)
        rho[current_particle_idx] = ti.abs(density_data[i]) * 1e-6  # 밀도 스케일링
        P_pressure[current_particle_idx] = ti.abs(pressure_data[i]) * 1e-3  # 압력 스케일링
        B[current_particle_idx] = ti.Vector([0.0, 0.0])
        h_smooth[current_particle_idx] = 0.2
        alpha_visc_p[current_particle_idx] = 1.0
        beta_visc_p[current_particle_idx] = 2.0
        is_special_particle_type[current_particle_idx] = PARTICLE_TYPE_NORMAL

    num_actual_particles[None] = num_particles_to_init
    num_particles[None] = num_particles_to_init

# SPMHD 입자 관련 상수
num_spmhd_particles_x = 23  # x 방향 격자 수 - 5만개로 조정
num_spmhd_particles_y = 24  # y 방향 격자 수 - 5만개로 조정
num_spmhd_particles = num_spmhd_particles_x * num_spmhd_particles_y
spmhd_charge = electron_charge * 1e20  # SPMHD 입자의 전하량 (C)

# 전기력 상수 (쿨롱 법칙)
k_coulomb = 8.9875517923e9  # 쿨롱 상수 (N⋅m²/C²)
electric_force_scale = 1e-20  # 전기력 스케일 팩터

@ti.kernel
def init_spmhd_particles():
    """SPMHD 입자들을 격자 패턴으로 초기화하고 자기장 정보를 설정"""
    dx = (domain_max - domain_min) / (num_spmhd_particles_x - 1)
    dy = (y_domain_max - y_domain_min) / (num_spmhd_particles_y - 1)
    
    for i, j in ti.ndrange(num_spmhd_particles_x, num_spmhd_particles_y):
        idx = num_actual_particles[None] + i * num_spmhd_particles_y + j
        
        # 격자 위치 설정
        pos[idx] = ti.Vector([
            domain_min + i * dx,
            y_domain_min + j * dy
        ])
        
        # 해당 위치의 배경 자기장 계산
        B_field = calculate_magnetic_field(pos[idx])
        B[idx] = B_field
        
        # 자기장 방향으로의 초기 속도 설정
        B_magnitude = ti.sqrt(B_field[0]**2 + B_field[1]**2)
        if B_magnitude > 1e-10:  # 0으로 나누기 방지
            B_direction = B_field / B_magnitude
            vel[idx] = B_direction * B_magnitude * 1e5  # 자기장 세기에 비례하는 속도
        else:
            vel[idx] = ti.Vector([0.0, 0.0])
        
        # 기본 속성 설정
        mass[idx] = 1.0
        u[idx] = 0.0
        rho[idx] = 1.0
        P_pressure[idx] = 0.0
        h_smooth[idx] = 0.2
        alpha_visc_p[idx] = 1.0
        beta_visc_p[idx] = 2.0
        is_special_particle_type[idx] = PARTICLE_TYPE_SPMHD
    
    # 입자 수 업데이트
    num_actual_particles[None] += num_spmhd_particles

@ti.kernel
def compute_electric_forces():
    """최적화된 전기력 계산: 대칭성 활용하여 중복 제거"""
    # 일반 입자와 SPMHD 입자 간의 전기력 계산 (대칭성 활용)
    for i in range(num_actual_particles[None]):
        if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL:  # 일반 입자
            for j in range(i + 1, num_actual_particles[None]):  # i < j만 계산
                if is_special_particle_type[j] == PARTICLE_TYPE_SPMHD:  # SPMHD 입자
                    # 두 입자 사이의 거리 벡터
                    r_vec = pos[j] - pos[i]
                    r_magnitude = ti.sqrt(r_vec[0]**2 + r_vec[1]**2)
                    
                    if r_magnitude > 1e-10:  # 0으로 나누기 방지
                        # 쿨롱 법칙에 따른 전기력 계산
                        force_magnitude = k_coulomb * electron_charge * spmhd_charge / (r_magnitude * r_magnitude)
                        force_magnitude *= electric_force_scale  # 시뮬레이션 스케일에 맞게 조정
                        
                        # 힘의 방향 설정 (척력)
                        force_direction = r_vec / r_magnitude
                        force = force_direction * force_magnitude
                        
                        # 가속도 업데이트 (대칭성 활용)
                        acc[i] += force / mass[i]
                        acc[j] -= force / mass[j]  # 작용-반작용

# 관측점 설정
observation_heights = [87, 97, 250]  # 관측 고도 (km)
observation_radius = 0.5  # 관측 반경을 0.5 시뮬레이션 단위로 설정 (25km)
observation_points = []
for height in observation_heights:
    sim_height = height * KM_TO_SIM
    observation_points.append(ti.Vector([0.0, sim_height]))  # 장보고 기지 상공

# 관측 데이터를 저장할 CSV 파일들과 시간 정보를 저장할 변수
observation_files = []
waccm_date = None
waccm_time = None

# 관측 데이터를 임시 저장할 필드들
max_observed_particles = 500  # 메모리 최적화
observed_particles = ti.field(dtype=ti.i32, shape=(len(observation_heights), max_observed_particles))
num_observed_particles = ti.field(dtype=ti.i32, shape=len(observation_heights))

def initialize_observation_files(waccm_data, resume_from_checkpoint=False):
    global observation_files, waccm_date, waccm_time
    
    # WACCM 데이터에서 날짜와 시간 정보 추출
    if waccm_data is not None:
        waccm_date = str(waccm_data['date'].iloc[0])
        seconds = waccm_data['datesec'].iloc[0]
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        waccm_time = f"{hours:02d}{minutes:02d}"
    else:
        waccm_date = "00000000"
        waccm_time = "0000"
    
    # 파일 이름 생성 (간단한 형식)
    observation_files = [
        f"C:\\Users\\sunma\\.vscode\\simuldatasave\\obs_{int(height)}km.csv" 
        for height in observation_heights
    ]
    
    if resume_from_checkpoint:
        # 체크포인트에서 이어서 할 때는 기존 파일 유지
        print("\n=== 체크포인트에서 이어서 시작: 기존 관측 파일 유지 ===")
        for file in observation_files:
            if os.path.exists(file):
                print(f"- 기존 파일 유지: {file}")
            else:
                # 파일이 없으면 헤더만 작성
                observation_headers = ["time", "particle_id", "pos_x", "pos_y", "vel_x", "vel_y", 
                              "density", "pressure", "temperature", "internal_energy",
                              "wind_dir_rad", "wind_dir_deg",
                              "B_x", "B_y", "acc_x", "acc_y"]
                with open(file, 'w') as f:
                    f.write(','.join(observation_headers) + '\n')
                print(f"- 새 파일 생성: {file}")
        return
    
    # 새로운 시뮬레이션 시작할 때만 헤더 작성
    observation_headers = ["time", "particle_id", "pos_x", "pos_y", "vel_x", "vel_y", 
                      "density", "pressure", "temperature", "internal_energy",
                      "wind_dir_rad", "wind_dir_deg",
                      "B_x", "B_y", "acc_x", "acc_y"]
    
    print("\n=== 새로운 시뮬레이션 시작: 관측 데이터 파일 생성 ===")
    print(f"WACCM 데이터 날짜: {waccm_date}")
    print(f"WACCM 데이터 시간: {waccm_time[:2]}:{waccm_time[2:]}") 
    print("\n생성된 파일:")
    
    for file in observation_files:
        with open(file, 'w') as f:
            f.write(','.join(observation_headers) + '\n')
        print(f"- {file}")

@ti.kernel
def check_particles_near_observation_points(sim_time: ti.f32):
    """관측점 근처의 모든 입자들을 확인하고 데이터를 수집"""
    # 관측된 입자 수 초기화
    for i in ti.static(range(3)):  # observation_heights의 길이가 3으로 고정
        num_observed_particles[i] = 0
    
    # 일반 입자 확인
    for i in range(num_actual_particles[None]):
        if i < 10:  # 처음 10개 입자의 위치만 출력
            print(f"입자 {i} 위치: {pos[i]}")
        
        for obs_idx in ti.static(range(3)):
            if num_observed_particles[obs_idx] < max_observed_particles:
                # 모든 관측점에 동일한 반경 사용
                effective_radius = observation_radius
                
                dist = (pos[i] - observation_points[obs_idx]).norm()
                if i < 10:  # 처음 10개 입자에 대해서만 거리 출력
                    print(f"입자 {i}와 관측점 {obs_idx} 사이 거리: {dist}, 반경: {effective_radius}")
                if dist <= effective_radius:
                    curr_idx = ti.atomic_add(num_observed_particles[obs_idx], 1)
                    if curr_idx < max_observed_particles:
                        observed_particles[obs_idx, curr_idx] = i
                        if obs_idx == 2:  # 250km 관측점
                            print(f"250km 관측점에 입자 {i} 감지됨 (거리: {dist})")
    
    # GAN 입자 확인 (초기화된 경우에만)
    if gan_particles_initialized[None] == 1:
        actual_gan_count = gan_particle_count[None]
        for i in range(actual_gan_count):
            if i < gan_pos.shape[0]:  # 배열 범위 체크
                if i < 10:  # 처음 10개 GAN 입자의 위치만 출력
                    print(f"GAN 입자 {i} 위치: {gan_pos[i]}")
                    
                for obs_idx in ti.static(range(3)):
                    if num_observed_particles[obs_idx] < max_observed_particles:
                        # 모든 관측점에 동일한 반경 사용
                        effective_radius = observation_radius
                        
                        dist = (gan_pos[i] - observation_points[obs_idx]).norm()
                        if i < 10: # 처음 10개 GAN 입자에 대해서만 거리 출력
                            print(f"GAN 입자 {i}와 관측점 {obs_idx} 사이 거리: {dist}, 반경: {effective_radius}")
                        if dist <= effective_radius:
                            curr_idx = ti.atomic_add(num_observed_particles[obs_idx], 1)
                            if curr_idx < max_observed_particles:
                                observed_particles[obs_idx, curr_idx] = i + 100000  # GAN 입자 ID 구분을 위해 100000 더함
                                if obs_idx == 2: # 250km 관측점
                                    print(f"250km 관측점에 GAN 입자 {i} 감지됨 (거리: {dist})")

def save_observation_data(sim_time):
    """관측 데이터를 CSV 파일에 저장"""
    # 시뮬레이션 시간을 시간 단위로 변환 (초 -> 시간)
    time_in_hours = sim_time / 3600.0
    
    print(f"\n=== 관측 데이터 저장 시도 ===")
    print(f"시뮬레이션 시간: {time_in_hours:.2f} 시간")
    
    for obs_idx in range(len(observation_heights)):
        print(f"\n관측점 {obs_idx} (고도: {observation_heights[obs_idx]}km)")
        print(f"관측된 입자 수: {num_observed_particles[obs_idx]}")
        print(f"관측점 위치: {observation_points[obs_idx]}")
        
        # 250km 관측점 특별 디버그 출력
        if obs_idx == 2:  # 250km
            print(f"=== 250km 관측점 상세 정보 ===")
            print(f"감지된 입자 수: {num_observed_particles[obs_idx]}")
            print(f"관측 반경: 0.5 (25km) - 모든 관측점과 동일")
            if num_observed_particles[obs_idx] > 0:
                print("감지된 입자들:")
                for i in range(min(num_observed_particles[obs_idx], 5)):  # 처음 5개만 출력
                    particle_id = observed_particles[obs_idx, i]
                    is_gan = particle_id >= 100000
                    actual_id = particle_id - 100000 if is_gan else particle_id
                    particle_type = "GAN" if is_gan else "일반"
                    print(f"  - {particle_type} 입자 {actual_id} (ID: {particle_id})")
            else:
                print("⚠️ 250km 관측점에 감지된 입자가 없습니다!")
                print("가능한 원인: 1) 입자가 250km 근처에 없음, 2) 관측 반경이 부족함")
        
        # 간단한 파일명
        filename = f"C:\\Users\\sunma\\.vscode\\simuldatasave\\obs_{int(observation_heights[obs_idx])}km.csv"
        with open(filename, 'a') as f:
            num_particles = num_observed_particles[obs_idx]
            for i in range(num_particles):
                particle_id = observed_particles[obs_idx, i]
                
                # GAN 입자와 일반 입자 구분
                is_gan = particle_id >= 100000
                actual_id = particle_id - 100000 if is_gan else particle_id
                
                if is_gan:
                    # GAN 입자 데이터 (안전한 접근)
                    if gan_particles_initialized[None] == 1 and actual_id < gan_pos.shape[0]:
                        particle_pos = gan_pos[actual_id].to_numpy()
                        particle_vel = gan_vel[actual_id].to_numpy()
                        particle_B = gan_B[actual_id].to_numpy()
                        temperature = gan_u[actual_id]
                        density = gan_rho[actual_id]
                        pressure = gan_P_pressure[actual_id]
                    else:
                        # GAN 입자가 초기화되지 않았거나 범위를 벗어난 경우 기본값 사용
                        particle_pos = np.array([0.0, 0.0])
                        particle_vel = np.array([0.0, 0.0])
                        particle_B = np.array([0.0, 0.0])
                        temperature = 0.0
                        density = 0.0
                        pressure = 0.0
                else:
                    # 일반 입자 데이터
                    particle_pos = pos[actual_id].to_numpy()
                    particle_vel = vel[actual_id].to_numpy()
                    particle_B = B[actual_id].to_numpy()
                    temperature = u[actual_id]
                    density = rho[actual_id]
                    pressure = P_pressure[actual_id]
                
                data = [
                    f"{time_in_hours:.6f}",  # 시간
                    f"{actual_id}",          # 원래 입자 ID
                    f"{particle_pos[0]:.6f}",
                    f"{particle_pos[1]:.6f}",
                    f"{particle_vel[0]:.6f}",
                    f"{particle_vel[1]:.6f}",
                    f"{density:.6f}",
                    f"{pressure:.6f}",
                    f"{temperature:.6f}",
                    f"{u[actual_id]:.6f}",  # internal_energy
                    f"{0.0:.6f}",          # wind_dir_rad (현재 데이터 없음, 0으로 설정)
                    f"{0.0:.6f}",          # wind_dir_deg (현재 데이터 없음, 0으로 설정)
                    f"{particle_B[0]:.6f}",
                    f"{particle_B[1]:.6f}",
                    f"{acc[actual_id][0]:.6f}",
                    f"{acc[actual_id][1]:.6f}"
                ]
                f.write(','.join(data) + '\n')
                print(f"입자 {actual_id} 데이터 저장 완료")



@ti.kernel
def init_gan_particles(
    pos_x: ti.types.ndarray(),
    pos_y: ti.types.ndarray(),
    vel_x: ti.types.ndarray(),
    vel_y: ti.types.ndarray(),
    density: ti.types.ndarray(),
    pressure: ti.types.ndarray(),
    temperature: ti.types.ndarray(),
    B_x: ti.types.ndarray(),
    B_y: ti.types.ndarray(),
    acc_x: ti.types.ndarray(),
    acc_y: ti.types.ndarray(),
    is_200km_model: ti.types.ndarray()
):
    """하이브리드 모델로 생성된 입자 데이터를 Taichi 필드에 초기화 (모델별 색상 구분)"""
    # 초기화 상태 설정
    gan_particles_initialized[None] = 1
    gan_particle_count[None] = 30900  # 입자 수 3만900개로 조정
    
    for i in range(30900):  # 3만900개의 보간 입자 - 입자 수 조정
        # DataFrame에서 받은 정확한 위치 데이터 사용
        # 처음 15450개: 200km 이상 고도 모델 (pos_y: 4-20)
        # 나머지 15450개: 전체 고도 모델 (pos_y: 0-4)
        gan_pos[i] = ti.Vector([ti.f32(pos_x[i]), ti.f32(pos_y[i])])
        gan_vel[i] = ti.Vector([ti.f32(vel_x[i]), ti.f32(vel_y[i])])
        gan_B[i] = ti.Vector([ti.f32(B_x[i]), ti.f32(B_y[i])])
        gan_rho[i] = ti.f32(density[i])
        gan_P_pressure[i] = ti.f32(pressure[i])
        gan_u[i] = ti.f32(temperature[i])
        gan_acc[i] = ti.Vector([ti.f32(acc_x[i]), ti.f32(acc_y[i])])
        
        # 200km 학습 모델 플래그 설정
        gan_is_200km_model[i] = ti.i32(is_200km_model[i])

def generate_particles_with_hybrid_model(sim_time, force_regenerate=False):
    """하이브리드 모델을 사용하여 입자 생성 및 업데이트"""
    global GLOBAL_HYBRID_GENERATOR, GLOBAL_HYBRID_LSTM
    global GLOBAL_HIGH_ALT_GENERATOR, GLOBAL_HIGH_ALT_LSTM
    
    if not HYBRID_MODELS_AVAILABLE:
        print("하이브리드 모델이 사용 불가능합니다.")
        return False
    
    # 시뮬레이션 시간(초)을 시/분 문자열로 표기
    sim_hours = int(sim_time // 3600)
    sim_minutes = int((sim_time % 3600) // 60)
    print(f"하이브리드 모델을 사용한 입자 생성 시작 (시간: {sim_hours:02d}:{sim_minutes:02d})")
    
    n_particles = 30900  # 입자 수 3만900개로 조정 - 중형 시뮬레이션
    generated_data = {
        'pos_x': np.zeros(n_particles),
        'pos_y': np.zeros(n_particles),
        'vel_x': np.zeros(n_particles),
        'vel_y': np.zeros(n_particles),
        'density': np.zeros(n_particles),
        'pressure': np.zeros(n_particles),
        'temperature': np.zeros(n_particles),
        'B_x': np.zeros(n_particles),
        'B_y': np.zeros(n_particles),
        'acc_x': np.zeros(n_particles),
        'acc_y': np.zeros(n_particles)
    }
    
    # 각 모델이 전체 고도 범위에 입자 생성 (겹쳐서 작동)
    particles_per_model = n_particles // 2
    
    # 200km 학습 모델로 입자 생성 (전체 고도 범위)
    if GLOBAL_HIGH_ALT_GENERATOR is not None and GLOBAL_HIGH_ALT_LSTM is not None:
        print("200km 학습 모델로 입자 생성 중...")
        try:
            # 더미 입력 데이터 생성 (실제로는 현재 시뮬레이션 상태를 사용해야 함)
            input_data = torch.randn(1, HIGH_ALT_INPUT_DIM, device=device)
            
            # 보간 수행
            interpolated = interpolate_with_hybrid_model(
                GLOBAL_HIGH_ALT_GENERATOR, 
                GLOBAL_HIGH_ALT_LSTM, 
                input_data, 
                num_interpolations=particles_per_model
            )
            
            if interpolated is not None:
                # 생성된 데이터를 파티클 속성으로 변환
                for i in range(particles_per_model):
                    data_idx = i % len(interpolated)
                    particle_data = interpolated[data_idx].flatten()
                    
                    # 전체 고도 범위에 배치 (0-1000km)
                    generated_data['pos_x'][i] = np.random.uniform(-10, 10)
                    generated_data['pos_y'][i] = np.random.uniform(0, 20)   # 0-1000km 고도
                    generated_data['vel_x'][i] = particle_data[0] if len(particle_data) > 0 else np.random.uniform(-5, 5)
                    generated_data['vel_y'][i] = particle_data[1] if len(particle_data) > 1 else np.random.uniform(-5, 5)
                    generated_data['density'][i] = abs(particle_data[2]) * 1e6 if len(particle_data) > 2 else np.random.uniform(1e5, 5e6)
                    generated_data['pressure'][i] = abs(particle_data[3]) * 1e5 if len(particle_data) > 3 else np.random.uniform(1e4, 1e6)
                    generated_data['temperature'][i] = abs(particle_data[4]) * 1000 if len(particle_data) > 4 else np.random.uniform(100, 1000)
                    generated_data['B_x'][i] = particle_data[5] if len(particle_data) > 5 else 0
                    generated_data['B_y'][i] = particle_data[6] if len(particle_data) > 6 else 0
                    generated_data['acc_x'][i] = particle_data[7] if len(particle_data) > 7 else 0
                    generated_data['acc_y'][i] = particle_data[8] if len(particle_data) > 8 else 0
                    
                print(f"200km 학습 모델로 {particles_per_model}개 입자 생성 완료 (전체 고도 범위)")
        except Exception as e:
            print(f"200km 학습 모델 입자 생성 실패: {e}")
    
    # 전체 고도 학습 모델로 입자 생성 (전체 고도 범위)
    if GLOBAL_HYBRID_GENERATOR is not None and GLOBAL_HYBRID_LSTM is not None:
        print("전체 고도 학습 모델로 입자 생성 중...")
        try:
            # 더미 입력 데이터 생성
            input_data = torch.randn(1, HYBRID_INPUT_DIM, device=device)
            
            # 보간 수행
            interpolated = interpolate_with_hybrid_model(
                GLOBAL_HYBRID_GENERATOR, 
                GLOBAL_HYBRID_LSTM, 
                input_data, 
                num_interpolations=particles_per_model
            )
            
            if interpolated is not None:
                # 생성된 데이터를 파티클 속성으로 변환
                for i in range(particles_per_model):
                    data_idx = i % len(interpolated)
                    particle_data = interpolated[data_idx].flatten()
                    idx = particles_per_model + i
                    
                    # 전체 고도 범위에 배치 (0-1000km)
                    generated_data['pos_x'][idx] = np.random.uniform(-10, 10)
                    generated_data['pos_y'][idx] = np.random.uniform(0, 20)   # 0-1000km 고도
                    generated_data['vel_x'][idx] = particle_data[0] if len(particle_data) > 0 else np.random.uniform(-5, 5)
                    generated_data['vel_y'][idx] = particle_data[1] if len(particle_data) > 1 else np.random.uniform(-5, 5)
                    generated_data['density'][idx] = abs(particle_data[2]) * 1e6 if len(particle_data) > 2 else np.random.uniform(1e5, 5e6)
                    generated_data['pressure'][idx] = particle_data[3] * 1e5 if len(particle_data) > 3 else np.random.uniform(1e4, 1e6)
                    generated_data['temperature'][idx] = abs(particle_data[4]) * 1000 if len(particle_data) > 4 else np.random.uniform(100, 1000)
                    generated_data['B_x'][idx] = particle_data[5] if len(particle_data) > 5 else 0
                    generated_data['B_y'][idx] = particle_data[6] if len(particle_data) > 6 else 0
                    generated_data['acc_x'][idx] = particle_data[7] if len(particle_data) > 7 else 0
                    generated_data['acc_y'][idx] = particle_data[8] if len(particle_data) > 8 else 0
                
                print(f"전체 고도 학습 모델로 {particles_per_model}개 입자 생성 완료 (전체 고도 범위)")
        except Exception as e:
            print(f"전체 고도 학습 모델 입자 생성 실패: {e}")
    
    # 200km 학습 모델 플래그 추가
    generated_data['is_200km_model'] = np.zeros(n_particles, dtype=int)
    # 처음 particles_per_model개는 200km 학습 모델 (플래그 = 1)
    generated_data['is_200km_model'][:particles_per_model] = 1
    # 나머지는 전체 고도 학습 모델 (플래그 = 0)
    generated_data['is_200km_model'][particles_per_model:] = 0
    
    print(f"200km 학습 모델 플래그 설정 완료: {particles_per_model}개 (플래그=1), {particles_per_model}개 (플래그=0)")
    
    # DataFrame 생성
    df = pd.DataFrame(generated_data)
    
    return df

def load_gan_particles():
    """하이브리드 모델을 우선 사용하여 입자 생성, 실패시 기존 방식으로 fallback"""
    
    print("[DEBUG] load_gan_particles 함수 시작")
    
    # 하이브리드 모델 초기화
    print("[DEBUG] 하이브리드 모델 초기화 시도")
    hybrid_initialized = initialize_hybrid_models()
    
    df = None
    
    if hybrid_initialized:
        print("하이브리드 모델을 사용한 입자 생성 시도...")
        try:
            df = generate_particles_with_hybrid_model(0.0)  # 초기 시간 0
            if df is not None:
                print("하이브리드 모델로 입자 생성 성공")
        except Exception as e:
            print(f"하이브리드 모델 입자 생성 실패: {e}")
    
    # 하이브리드 모델 실패시 기존 방식 시도
    if df is None:
        print("기존 방식으로 입자 생성 시도...")
        
        # 레거시 LSTM 초기화 시도
        lstm_initialized = initialize_lstm_model() if LSTM_AVAILABLE else False
        
        if lstm_initialized:
            print("레거시 LSTM 시계열 예측 모드로 동작합니다.")
        
        # 기존 파일 로드 시도
        try:
            # 하이브리드 파일 시도
            hybrid_file_path = r'C:\Users\sunma\.vscode\simuldatasave\hybrid.csv'
            if os.path.exists(hybrid_file_path):
                print(f"하이브리드 입자 파일 로드: {hybrid_file_path}")
                df = pd.read_csv(hybrid_file_path)
                print("하이브리드 생성 입자 데이터 로드 완료")
        except Exception as e:
            print(f"하이브리드 데이터 로드 실패: {e}")
        
        # 기존 GAN 파일 시도
        if df is None:
            try:
                gan_file_path = r'C:\Users\sunma\.vscode\simuldatasave\gan.csv'
                if os.path.exists(gan_file_path):
                    print(f"기존 GAN 입자 파일 로드: {gan_file_path}")
                    df = pd.read_csv(gan_file_path)
                    print("기존 GAN 생성 입자 데이터 로드 완료")
            except Exception as e:
                print(f"GAN 생성 입자 데이터 로드 실패: {e}")
        
        # 모든 시도 실패시 랜덤 생성
        if df is None:
            print("모든 입자 로드 실패, 랜덤 보간 입자 생성 중...")
            n_particles = 100000
            data = {
                'pos_x': np.random.uniform(-10, 10, n_particles),
                'pos_y': np.random.uniform(0, 20, n_particles),
                'vel_x': np.random.uniform(-5, 5, n_particles),
                'vel_y': np.random.uniform(-5, 5, n_particles),
                'density': np.random.uniform(1e6, 5e6, n_particles),
                'pressure': np.random.uniform(1e5, 1e6, n_particles),
                'temperature': np.random.uniform(100, 1000, n_particles),
                'B_x': np.random.uniform(-1e-8, 1e-8, n_particles),
                'B_y': np.random.uniform(-1e-8, 1e-8, n_particles),
                'acc_x': np.random.uniform(-1, 1, n_particles),
                'acc_y': np.random.uniform(-1, 1, n_particles)
            }
            df = pd.DataFrame(data)
            print("랜덤 보간 입자 생성 완료")
    
    # 데이터 크기 조정
    if len(df) >= 30900:
        df = df.iloc[:30900]
    elif len(df) < 30900:
        df = pd.concat([df] * (30900 // len(df) + 1), ignore_index=True)[:30900]

    # Taichi 필드에 데이터 초기화
    init_gan_particles(
        np.ascontiguousarray(df['pos_x'].to_numpy()),
        np.ascontiguousarray(df['pos_y'].to_numpy()),
        np.ascontiguousarray(df['vel_x'].to_numpy()),
        np.ascontiguousarray(df['vel_y'].to_numpy()),
        np.ascontiguousarray(df['density'].to_numpy()),
        np.ascontiguousarray(df['pressure'].to_numpy()),
        np.ascontiguousarray(df['temperature'].to_numpy()),
        np.ascontiguousarray(df['B_x'].to_numpy()),
        np.ascontiguousarray(df['B_y'].to_numpy()),
        np.ascontiguousarray(df['acc_x'].to_numpy()),
        np.ascontiguousarray(df['acc_y'].to_numpy()),
        np.ascontiguousarray(df['is_200km_model'].to_numpy())
    )
    
    print(f"보간 입자 초기화 완료: {gan_particle_count[None]}개 입자")
    print("[DEBUG] load_gan_particles 함수 완료")

def save_normal_particles_data(sim_time):
    """일반 입자들의 정보를 CSV 파일에 저장합니다."""
    # 시뮬레이션 시간을 시간 단위로 변환 (초 -> 시간)
    time_in_hours = sim_time / 3600.0
    
    # 파일 이름에 시간 정보 포함
    output_file = 'C:\\Users\\sunma\\.vscode\\simuldatasave\\normal.csv'
    
    # 파일이 없으면 헤더와 함께 새로 생성
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            headers = ['time', 'particle_id', 'pos_x', 'pos_y', 'vel_x', 'vel_y',
                      'density', 'pressure', 'temperature', 'B_x', 'B_y',
                      'acc_x', 'acc_y', 'smoothing_length']
            f.write(','.join(headers) + '\n')
    
    # 데이터를 파일에 추가
    with open(output_file, 'a') as f:
        # 모든 입자에 대해 반복
        for i in range(num_actual_particles[None]):
            # 일반 입자만 저장
            if is_special_particle_type[i] == PARTICLE_TYPE_NORMAL:
                particle_pos = pos[i].to_numpy()
                particle_vel = vel[i].to_numpy()
                particle_B = B[i].to_numpy()
                particle_acc = acc[i].to_numpy()
                
                data = [
                    f"{time_in_hours:.6f}",     # 시간
                    f"{i}",                      # 입자 ID
                    f"{particle_pos[0]:.6f}",    # x 위치
                    f"{particle_pos[1]:.6f}",    # y 위치
                    f"{particle_vel[0]:.6f}",    # x 속도
                    f"{particle_vel[1]:.6f}",    # y 속도
                    f"{rho[i]:.6f}",            # 밀도
                    f"{P_pressure[i]:.6f}",     # 압력
                    f"{u[i]:.6f}",              # 온도/내부 에너지
                    f"{particle_B[0]:.6f}",     # x 자기장
                    f"{particle_B[1]:.6f}",     # y 자기장
                    f"{particle_acc[0]:.6f}",   # x 가속도
                    f"{particle_acc[1]:.6f}",   # y 가속도
                    f"{h_smooth[i]:.6f}"        # smoothing length
                ]
                f.write(','.join(data) + '\n')

# LSTM + Generator를 위한 전역 변수들
LSTM_MODEL = None
LSTM_MODEL_INFO = None
time_series_buffer = []  # 시계열 데이터 버퍼 (18시간 분량)
HOURS_TO_LEARN = 18      # 학습할 시간 (시간)
HOURS_TO_PREDICT = PREDICTION_HOURS  # 예측할 시간 (6-18시간 범위, 12시간)
LSTM_UPDATE_INTERVAL = 300.0  # LSTM 업데이트 간격 (초, 5분)
last_lstm_update_time = 0.0    # 마지막 LSTM 업데이트 시간

def dt_to_hours(dt_seconds):
    """시뮬레이션 dt를 시간 단위로 변환"""
    return dt_seconds / 3600.0

def collect_simulation_state():
    """
    LSTM 학습에 사용한 feature와 동일한 순서/개수로 시뮬레이션 상태를 추출.
    실제 시뮬레이션에서 해당 feature의 평균값/대표값을 추출해야 함.
    없는 feature는 0 또는 적절한 값으로 대체(사용자가 직접 수정 가능).
    """
    state = {
        'pos_x': np.mean(pos.to_numpy()[:, 0]) if 'pos' in globals() else 0.0,
        'pos_y': np.mean(pos.to_numpy()[:, 1]) if 'pos' in globals() else 0.0,
        'vel_x': np.mean(vel.to_numpy()[:, 0]) if 'vel' in globals() else 0.0,
        'vel_y': np.mean(vel.to_numpy()[:, 1]) if 'vel' in globals() else 0.0,
        'vel_z': 0.0,  # 2D 시뮬레이션이면 0, 3D면 np.mean(vel[:,2])
        'density': np.mean(rho.to_numpy()) if 'rho' in globals() else 0.0,
        'pressure': np.mean(P_pressure.to_numpy()) if 'P_pressure' in globals() else 0.0,
        'temperature': np.mean(u.to_numpy()) if 'u' in globals() else 0.0,  # 예시: 내부에너지 또는 온도
        'internal_energy': np.mean(u.to_numpy()) if 'u' in globals() else 0.0,  # 실제 내부에너지 변수로 교체
        'wind_dir_rad': 0.0,  # 풍향(라디안) 추출 코드 필요
        'wind_dir_deg': 0.0,  # 풍향(도) 추출 코드 필요
        'B_x': np.mean(B.to_numpy()[:, 0]) if 'B' in globals() else 0.0,
        'B_y': np.mean(B.to_numpy()[:, 1]) if 'B' in globals() else 0.0,
        'acc_x': np.mean(acc.to_numpy()[:, 0]) if 'acc' in globals() else 0.0,
        'acc_y': np.mean(acc.to_numpy()[:, 1]) if 'acc' in globals() else 0.0,
        'U010': 0.0,  # 실제 U010 값 추출 코드 필요
        'U030': 0.0,  # 실제 U030 값 추출 코드 필요
        'U050': 0.0,  # 실제 U050 값 추출 코드 필요
        'U100': 0.0,  # 실제 U100 값 추출 코드 필요
        'U200': 0.0,  # 실제 U200 값 추출 코드 필요
        'V200': 0.0,  # 실제 V200 값 추출 코드 필요
        'V850': 0.0,  # 실제 V850 값 추출 코드 필요
        'VBOT': 0.0,  # 실제 VBOT 값 추출 코드 필요
        'Z010': 0.0,  # 실제 Z010 값 추출 코드 필요
        'Z030': 0.0,  # 실제 Z030 값 추출 코드 필요
        'Z050': 0.0,  # 실제 Z050 값 추출 코드 필요
        'Z200': 0.0,  # 실제 Z200 값 추출 코드 필요
        'Z500': 0.0,  # 실제 Z500 값 추출 코드 필요
        'Z850': 0.0,  # 실제 Z850 값 추출 코드 필요
    }
    feature_names = [
        'pos_x', 'pos_y', 'vel_x', 'vel_y', 'vel_z', 'density', 'pressure', 'temperature', 'internal_energy',
        'wind_dir_rad', 'wind_dir_deg', 'B_x', 'B_y', 'acc_x', 'acc_y',
        'U010', 'U030', 'U050', 'U100', 'U200', 'V200', 'V850', 'VBOT',
        'Z010', 'Z030', 'Z050', 'Z200', 'Z500', 'Z850'
    ]
    return [state[k] for k in feature_names]

def initialize_lstm_model():
    """LSTM 모델 초기화 (LSTM만 로드)"""
    global LSTM_MODEL, LSTM_MODEL_INFO
    
    if not LSTM_AVAILABLE:
        return False
    
    try:
        # LSTM만 로드하는 경로 사용
        lstm_path = r'C:\Users\sunma\.vscode\trained_lstm_only.pth'
        info_path = r'C:\Users\sunma\.vscode\model_info.pth'
        
        print(f"LSTM 모델 파일 확인 중...")
        print(f"LSTM 파일: {lstm_path}")
        print(f"정보 파일: {info_path}")
        
        if os.path.exists(lstm_path) and os.path.exists(info_path):
            LSTM_MODEL, LSTM_MODEL_INFO = load_lstm_only(lstm_path, info_path)
            print("저장된 LSTM 모델 로드 성공")
            return True
        else:
            print("저장된 LSTM 모델 파일을 찾을 수 없습니다.")
            print(f"LSTM 파일 존재: {os.path.exists(lstm_path)}")
            print(f"정보 파일 존재: {os.path.exists(info_path)}")
            print("lstm_generator_hybrid.py를 먼저 실행하여 모델을 훈련시켜주세요.")
            return False
    except Exception as e:
        print(f"LSTM 모델 초기화 실패: {e}")
        return False

def update_time_series_buffer(sim_time):
    """시계열 버퍼를 현재 시뮬레이션 상태로 업데이트"""
    global time_series_buffer
    
    current_state = collect_simulation_state()
    if current_state is None:
        return
    
    current_hour = sim_time / 3600.0
    timestamped_state = [current_hour] + current_state
    time_series_buffer.append(timestamped_state)
    
    max_buffer_size = HOURS_TO_LEARN + HOURS_TO_PREDICT
    if len(time_series_buffer) > max_buffer_size:
        time_series_buffer.pop(0)

@ti.kernel 
def update_gan_particles_from_lstm_kernel(particle_data: ti.types.ndarray()):
    """LSTM + Generator 출력으로 GAN 입자 필드 업데이트"""
    for i in range(30900):
        feature_idx = i % 12
        
        if feature_idx == 0:  # pos_x
            gan_pos[i][0] = particle_data[0] * (domain_max - domain_min) + domain_min
        elif feature_idx == 1:  # pos_y  
            gan_pos[i][1] = particle_data[1] * y_domain_max
        elif feature_idx == 2:  # vel_x
            gan_vel[i][0] = particle_data[2] * 10.0
        elif feature_idx == 3:  # vel_y
            gan_vel[i][1] = particle_data[3] * 10.0
        elif feature_idx == 4:  # density
            gan_rho[i] = particle_data[4] * 1e7
        elif feature_idx == 5:  # pressure
            gan_P_pressure[i] = particle_data[5] * 1e7

def generate_particles_with_lstm(sim_time):
    """LSTM만 사용하여 시계열 예측하고 간단한 입자 생성"""
    global LSTM_MODEL, time_series_buffer
    if LSTM_MODEL is None:
        print("[DEBUG] LSTM_MODEL is None, trying to re-initialize...")
        success = initialize_lstm_model()
        print(f"[DEBUG] initialize_lstm_model() called, success: {success}, LSTM_MODEL is None? {LSTM_MODEL is None}")
    # 각 조건을 개별적으로 체크하여 정확한 원인 파악
    if LSTM_MODEL is None:
        print(f"LSTM 모델이 로드되지 않았습니다. 모델 파일을 확인하세요.")
        return False
    
    # 입력 데이터 준비 (부족하면 패딩 보간)
    if len(time_series_buffer) < HOURS_TO_LEARN:
        print(f"LSTM 예측을 위한 데이터 부족: {len(time_series_buffer)}/{HOURS_TO_LEARN} (패딩 보간)")
        recent_data = np.array(time_series_buffer)
        if len(recent_data) > 0:
            pad = np.tile(recent_data[-1], (HOURS_TO_LEARN - len(recent_data), 1))
            input_data = np.concatenate([recent_data, pad], axis=0)[:, 1:]
        else:
            input_data = np.zeros((HOURS_TO_LEARN, 6))  # feature 수에 맞게
    else:
        recent_data = np.array(time_series_buffer[-HOURS_TO_LEARN:])
        input_data = recent_data[:, 1:]
    try:
        print(f"LSTM 입력 데이터 형태: {input_data.shape}")
        print(f"시뮬레이션 시간: {sim_time/3600.0:.2f}시간")
        # LSTM만 사용하여 6-18시간 범위 예측
        LSTM_MODEL.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
            lstm_prediction = LSTM_MODEL(input_tensor)  # (1, 12, input_size)
            lstm_pred_np = lstm_prediction.cpu().numpy().squeeze()
        
        # LSTM 예측 결과를 기반으로 간단한 입자 데이터 생성
        # 6-18시간 예측 결과 (12시간 분량)
        if len(lstm_pred_np.shape) == 2:  # (12, input_size)
            # 시간별 예측값의 평균 사용
            pred_mean = np.mean(lstm_pred_np, axis=0)
            # 시간별 변화량 계산
            pred_variance = np.std(lstm_pred_np, axis=0)
        else:  # 1차원인 경우
            pred_mean = lstm_pred_np
            pred_variance = np.zeros_like(pred_mean)
        
        # 입자 특성 생성 (12개 특성)
        particle_features = np.zeros(12)
        # 예측 결과를 입자 특성으로 매핑 (6-18시간 범위 고려)
        for i in range(min(len(pred_mean), 6)):
            if i < len(pred_mean):
                # 시간 변화를 고려한 특성 생성
                time_factor = 1.0 + pred_variance[i] * 0.1  # 변화량을 시간 팩터로 활용
                particle_features[i] = pred_mean[i] * 0.1 * time_factor  # 위치/속도 특성
                particle_features[i + 6] = pred_mean[i] * 1e6 * time_factor  # 물리 특성
        # === LSTM 6-18시간 예측값을 실제 입자 속도에 반영 ===
        try:
            if 'vel' in globals() and vel.shape[0] > 0:
                # 6-18시간 예측 범위의 평균값 사용
                vel[:, 0] = pred_mean[2] if len(pred_mean) > 2 else 0  # mean_vel_x
                vel[:, 1] = pred_mean[3] if len(pred_mean) > 3 else 0  # mean_vel_y
                print(f"6-18시간 예측값을 입자 속도에 반영: vel_x={pred_mean[2]:.4f}, vel_y={pred_mean[3]:.4f}")
        except Exception as e:
            print(f"입자 속도에 LSTM 6-18시간 예측값 반영 실패: {e}")
        
        # GAN 입자 필드 업데이트
        update_gan_particles_from_lstm_kernel(particle_features.astype(np.float32))
        print(f"LSTM 기반 입자 생성 완료 (6-18시간 예측 범위, {PREDICTION_HOURS}시간 분량)")
        return True
    except Exception as e:
        print(f"LSTM 입자 생성 중 오류 발생: {e}")
        print(f"오류 타입: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def should_update_lstm(sim_time):
    """LSTM 업데이트가 필요한지 확인"""
    global last_lstm_update_time
    return (sim_time - last_lstm_update_time) >= LSTM_UPDATE_INTERVAL

def perform_hybrid_update(sim_time):
    """하이브리드 모델 기반 입자 업데이트"""
    global last_lstm_update_time
    
    # 시간을 시, 분, 초로 변환
    hours = int(sim_time // 3600)
    minutes = int((sim_time % 3600) // 60)
    seconds = int(sim_time % 60)
    
    print(f"\n=== 하이브리드 모델 업데이트 수행 ===")
    print(f"시뮬레이션 시간: {hours:02d}:{minutes:02d}:{seconds:02d} ({sim_time/3600.0:.2f}h)")
    print(f"예측 범위: {PREDICTION_START_HOURS}-{PREDICTION_END_HOURS}시간 ({PREDICTION_HOURS}시간 분량)")
    
    update_time_series_buffer(sim_time)
    
    # 하이브리드 모델 사용 시도
    if HYBRID_MODELS_AVAILABLE:
        try:
            df = generate_particles_with_hybrid_model(sim_time, force_regenerate=True)
            if df is not None:
                print(f"하이브리드 모델 기반 입자 업데이트 성공 (6-18시간 예측 적용)")
                
                # Taichi 필드 업데이트
                init_gan_particles(
                    np.ascontiguousarray(df['pos_x'].to_numpy()),
                    np.ascontiguousarray(df['pos_y'].to_numpy()),
                    np.ascontiguousarray(df['vel_x'].to_numpy()),
                    np.ascontiguousarray(df['vel_y'].to_numpy()),
                    np.ascontiguousarray(df['density'].to_numpy()),
                    np.ascontiguousarray(df['pressure'].to_numpy()),
                    np.ascontiguousarray(df['temperature'].to_numpy()),
                    np.ascontiguousarray(df['B_x'].to_numpy()),
                    np.ascontiguousarray(df['B_y'].to_numpy()),
                    np.ascontiguousarray(df['acc_x'].to_numpy()),
                    np.ascontiguousarray(df['acc_y'].to_numpy()),
                    np.ascontiguousarray(df['is_200km_model'].to_numpy())
                )
                
                last_lstm_update_time = sim_time
                return True
        except Exception as e:
            print(f"하이브리드 모델 업데이트 실패: {e}")
    
    # 하이브리드 모델 실패시 레거시 LSTM 시도
    if LSTM_AVAILABLE and hasattr(sys.modules[__name__], 'LSTM_MODEL'):
        try:
            if generate_particles_with_lstm(sim_time):
                print("레거시 LSTM 기반 입자 생성 성공")
                last_lstm_update_time = sim_time
                return True
            else:
                print("레거시 LSTM 기반 입자 생성 실패")
        except Exception as e:
            print(f"레거시 LSTM 업데이트 실패: {e}")
    
    print("모든 AI 모델 업데이트 실패")
    last_lstm_update_time = sim_time
    return False

def perform_lstm_update(sim_time):
    """레거시 함수: 하이브리드 업데이트로 리디렉션"""
    return perform_hybrid_update(sim_time)

# 사용되지 않는 함수 - 주석 처리
# @ti.kernel
# def visualize_particles():
#     """입자 시각화 (일반 입자: 흰색, 전자: 초록색)"""
#     for i in range(num_normal_particles[None]):
#         pos = particles.position[i]
#         if 0 <= pos[0] < window_width and 0 <= pos[1] < window_height:
#             if particles.is_ionospheric[i] == 1:  # 전자
#                 gui.circle(pos, radius=2, color=0x00FF00)  # 초록색
#             else:  # 일반 입자
#                 gui.circle(pos, radius=2, color=0xFFFFFF)  # 흰색

def save_complete_checkpoint(sim_time, frame_count):
    """완전한 체크포인트 저장 - 모든 입자 상태 포함"""
    try:
        checkpoint_data = {
            'sim_time': sim_time,
            'frame_count': frame_count,
            'timestamp': time.time(),
            
            # SPH 입자 상태 저장 (NumPy 배열을 리스트로 변환)
            'sph_positions': pos.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_velocities': vel.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_masses': mass.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_internal_energies': u.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_densities': rho.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_pressures': P_pressure.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_magnetic_fields': B.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_accelerations': acc.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_special_types': is_special_particle_type.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_smoothing_lengths': h_smooth.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_alpha_visc': alpha_visc_p.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_beta_visc': beta_visc_p.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_etha_a_dt': etha_a_dt_field.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_ae_k': ae_k_field.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_etha_a': etha_a_field.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_B_unit': B_unit_field.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_S_a': S_a_field.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_S_b': S_b_field.to_numpy()[:num_actual_particles[None]].tolist(),
            'sph_dB_dt': dB_dt.to_numpy()[:num_actual_particles[None]].tolist(),
            
            # GAN 입자 상태 저장 (초기화된 경우에만)
            'gan_initialized': gan_particles_initialized[None],
            'gan_count': gan_particle_count[None] if gan_particles_initialized[None] == 1 else 0,
        }
        
        # GAN 입자가 초기화된 경우에만 저장
        if gan_particles_initialized[None] == 1:
            checkpoint_data.update({
                'gan_positions': gan_pos.to_numpy()[:gan_particle_count[None]].tolist(),
                'gan_velocities': gan_vel.to_numpy()[:gan_particle_count[None]].tolist(),
                'gan_magnetic_fields': gan_B.to_numpy()[:gan_particle_count[None]].tolist(),
                'gan_densities': gan_rho.to_numpy()[:gan_particle_count[None]].tolist(),
                'gan_pressures': gan_P_pressure.to_numpy()[:gan_particle_count[None]].tolist(),
                'gan_internal_energies': gan_u.to_numpy()[:gan_particle_count[None]].tolist(),
                'gan_accelerations': gan_acc.to_numpy()[:gan_particle_count[None]].tolist(),
                'gan_is_200km_model': gan_is_200km_model.to_numpy()[:gan_particle_count[None]].tolist(),
            })
        
        # 체크포인트 파일에 저장
        checkpoint_file = 'simulation_checkpoint_complete.json'
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"완전한 체크포인트 저장 완료: 프레임 {frame_count}, 시각 {sim_time/3600:.2f}시간")
        print(f"저장된 데이터: SPH {num_actual_particles[None]}개, GAN {gan_particle_count[None] if gan_particles_initialized[None] == 1 else 0}개")
        
    except Exception as e:
        print(f"체크포인트 저장 실패: {e}")
        import traceback
        traceback.print_exc()

def load_complete_checkpoint():
    """완전한 체크포인트에서 시뮬레이션 상태 복원"""
    try:
        checkpoint_file = 'simulation_checkpoint_complete.json'
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            print(f"\n=== 완전한 체크포인트 발견 ===")
            print(f"프레임: {checkpoint_data['frame_count']}")
            print(f"시각: {checkpoint_data['sim_time']/3600:.2f}시간")
            print(f"저장 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint_data['timestamp']))}")
            print(f"SPH 입자: {len(checkpoint_data['sph_positions'])}개")
            print(f"GAN 입자: {checkpoint_data['gan_count']}개")
            
            print("\n" + "="*50)
            print("시뮬레이션 시작 방법을 선택하세요:")
            print("="*50)
            print("1. 체크포인트에서 이어서 (y)")
            print("2. 새로 시작 (n)")
            print("="*50)
            print("선택: ", end="")
            
            while True:
                user_input = input().strip().lower()
                if user_input in ['y', 'yes', '1']:
                    print("\n체크포인트에서 시뮬레이션을 이어서 시작합니다...")
                    return checkpoint_data
                elif user_input in ['n', 'no', '2']:
                    print("\n새로운 시뮬레이션을 시작합니다...")
                    return None
                else:
                    print("잘못된 입력입니다. 'y' 또는 'n'을 입력하세요: ", end="")
        else:
            print("완전한 체크포인트 파일이 없습니다. 새로운 시뮬레이션을 시작합니다.")
            return None
    except Exception as e:
        print(f"체크포인트 로드 실패: {e}")
        print("새로운 시뮬레이션을 시작합니다.")
        return None

def restore_simulation_state(checkpoint_data):
    """체크포인트 데이터로 시뮬레이션 상태 완전 복원"""
    global sim_time, frame_count
    
    print("시뮬레이션 상태 복원 중...")
    
    # 기본 정보 복원
    sim_time = float(checkpoint_data['sim_time'])
    frame_count = int(checkpoint_data['frame_count'])
    
    print(f"[DEBUG] restore_simulation_state: sim_time={sim_time}, frame_count={frame_count}")
    
    # SPH 입자 상태 복원
    sph_count = len(checkpoint_data['sph_positions'])
    num_actual_particles[None] = sph_count
    
    print(f"SPH 입자 {sph_count}개 상태 복원 중...")
    
    # JSON에서 로드된 리스트를 NumPy 배열로 변환 후 Taichi 필드로 복원
    print("[DEBUG] SPH 입자 데이터 Taichi 필드 복원 시작")
    
    try:
        # 데이터 크기 및 타입 확인
        sph_positions = np.array(checkpoint_data['sph_positions'], dtype=np.float32)
        print(f"[DEBUG] sph_positions shape: {sph_positions.shape}, dtype: {sph_positions.dtype}")
        
        # pos 필드 크기 확인
        print(f"[DEBUG] pos 필드 크기: {pos.shape}")
        
        # Vector.field는 다차원 데이터를 다르게 처리해야 함
        if len(pos.shape) == 1 and len(sph_positions.shape) == 2:
            print(f"[DEBUG] pos는 Vector.field입니다. 다차원 데이터로 복원합니다.")
            # Vector.field에 맞게 데이터 재구성
            if sph_positions.shape[0] > pos.shape[0]:
                print(f"[WARNING] 복원할 데이터({sph_positions.shape[0]})가 pos 필드({pos.shape[0]})보다 큽니다. 잘라서 복원합니다.")
                sph_positions = sph_positions[:pos.shape[0]]
            
            # Vector.field에 복원 - from_numpy 대신 개별 설정
            print(f"[DEBUG] {sph_positions.shape[0]}개 입자 위치를 개별적으로 설정합니다.")
            try:
                for i in range(sph_positions.shape[0]):
                    pos[i] = ti.Vector([sph_positions[i, 0], sph_positions[i, 1]])
                print("[DEBUG] pos 복원 완료 (개별 설정 방식)")
            except Exception as e:
                print(f"[ERROR] pos 복원 중 오류: {e}")
                # 오류 발생 시에도 복원을 강제로 시도
                print("[WARNING] pos 복원 실패, 강제로 복원을 시도합니다.")
                try:
                    # 다른 방법으로 복원 시도
                    pos.from_numpy(sph_positions.astype(np.float32))
                    print("[DEBUG] pos 강제 복원 성공")
                except Exception as e2:
                    print(f"[ERROR] 강제 복원도 실패: {e2}")
                    # 마지막 수단으로 개별 설정
                    print("[DEBUG] 마지막 수단으로 개별 설정 시도")
                    for i in range(sph_positions.shape[0]):
                        pos[i] = ti.Vector([float(sph_positions[i, 0]), float(sph_positions[i, 1])])
                    print("[DEBUG] pos 개별 설정 완료")
        else:
            print(f"[DEBUG] pos는 일반 필드입니다. 1차원 데이터로 복원합니다.")
            # 1차원 데이터로 변환하여 복원
            if sph_positions.shape[0] > pos.shape[0]:
                sph_positions = sph_positions[:pos.shape[0]]
            pos.from_numpy(sph_positions.flatten())
        
        print("[DEBUG] pos 복원 완료")
        
        print("[DEBUG] vel 복원 시작")
        # vel도 개별 설정 방식으로 복원
        try:
            sph_velocities = np.array(checkpoint_data['sph_velocities'], dtype=np.float32)
            print(f"[DEBUG] vel 데이터 shape: {sph_velocities.shape}")
            for i in range(min(sph_velocities.shape[0], pos.shape[0])):
                vel[i] = ti.Vector([sph_velocities[i, 0], sph_velocities[i, 1]])
            print("[DEBUG] vel 복원 완료 (개별 설정 방식)")
        except Exception as e:
            print(f"[ERROR] vel 복원 중 오류: {e}")
            print("[DEBUG] vel 강제 복원 시도")
            try:
                vel.from_numpy(sph_velocities[:pos.shape[0]])
                print("[DEBUG] vel 강제 복원 성공")
            except Exception as e2:
                print(f"[ERROR] vel 강제 복원도 실패: {e2}")
                for i in range(min(sph_velocities.shape[0], pos.shape[0])):
                    vel[i] = ti.Vector([0.0, 0.0])
                print("[DEBUG] vel 기본값 설정 완료")
        
        print("[DEBUG] mass 복원 시작")
        # mass도 개별 설정 방식으로 복원
        try:
            sph_masses = np.array(checkpoint_data['sph_masses'], dtype=np.float32)
            for i in range(min(len(sph_masses), pos.shape[0])):
                mass[i] = float(sph_masses[i])
            print("[DEBUG] mass 복원 완료 (개별 설정 방식)")
        except Exception as e:
            print(f"[ERROR] mass 복원 중 오류: {e}")
            for i in range(min(len(sph_masses), pos.shape[0])):
                mass[i] = 1.0
            print("[DEBUG] mass 기본값 설정 완료")
        
        print("[DEBUG] u 복원 시작")
        # u도 개별 설정 방식으로 복원
        try:
            sph_internal_energies = np.array(checkpoint_data['sph_internal_energies'], dtype=np.float32)
            for i in range(min(len(sph_internal_energies), pos.shape[0])):
                u[i] = float(sph_internal_energies[i])
            print("[DEBUG] u 복원 완료 (개별 설정 방식)")
        except Exception as e:
            print(f"[ERROR] u 복원 중 오류: {e}")
            for i in range(min(len(sph_internal_energies), pos.shape[0])):
                u[i] = 0.0
            print("[DEBUG] u 기본값 설정 완료")
        
        print("[DEBUG] rho 복원 시작")
        # rho도 개별 설정 방식으로 복원
        try:
            sph_densities = np.array(checkpoint_data['sph_densities'], dtype=np.float32)
            for i in range(min(len(sph_densities), pos.shape[0])):
                rho[i] = float(sph_densities[i])
            print("[DEBUG] rho 복원 완료 (개별 설정 방식)")
        except Exception as e:
            print(f"[ERROR] rho 복원 중 오류: {e}")
            for i in range(min(len(sph_densities), pos.shape[0])):
                rho[i] = 1.0
            print("[DEBUG] rho 기본값 설정 완료")
        
        print("[DEBUG] P_pressure 복원 시작")
        # P_pressure도 개별 설정 방식으로 복원
        try:
            sph_pressures = np.array(checkpoint_data['sph_pressures'], dtype=np.float32)
            for i in range(min(len(sph_pressures), pos.shape[0])):
                P_pressure[i] = float(sph_pressures[i])
            print("[DEBUG] P_pressure 복원 완료 (개별 설정 방식)")
        except Exception as e:
            print(f"[ERROR] P_pressure 복원 중 오류: {e}")
            for i in range(min(len(sph_pressures), pos.shape[0])):
                P_pressure[i] = 0.0
            print("[DEBUG] P_pressure 기본값 설정 완료")
        
        print("[DEBUG] B 복원 시작")
        # B도 개별 설정 방식으로 복원
        try:
            sph_magnetic_fields = np.array(checkpoint_data['sph_magnetic_fields'], dtype=np.float32)
            for i in range(min(sph_magnetic_fields.shape[0], pos.shape[0])):
                B[i] = ti.Vector([sph_magnetic_fields[i, 0], sph_magnetic_fields[i, 1]])
            print("[DEBUG] B 복원 완료 (개별 설정 방식)")
        except Exception as e:
            print(f"[ERROR] B 복원 중 오류: {e}")
            for i in range(min(sph_magnetic_fields.shape[0], pos.shape[0])):
                B[i] = ti.Vector([0.0, 0.0])
            print("[DEBUG] B 기본값 설정 완료")
        
        print("[DEBUG] acc 복원 시작")
        # acc도 개별 설정 방식으로 복원
        try:
            sph_accelerations = np.array(checkpoint_data['sph_accelerations'], dtype=np.float32)
            for i in range(min(sph_accelerations.shape[0], pos.shape[0])):
                acc[i] = ti.Vector([sph_accelerations[i, 0], sph_accelerations[i, 1]])
            print("[DEBUG] acc 복원 완료 (개별 설정 방식)")
        except Exception as e:
            print(f"[ERROR] acc 복원 중 오류: {e}")
            for i in range(min(sph_accelerations.shape[0], pos.shape[0])):
                acc[i] = ti.Vector([0.0, 0.0])
            print("[DEBUG] acc 기본값 설정 완료")
        
        print("[DEBUG] 나머지 필드들 복원 시작")
        # 나머지 필드들도 개별 설정 방식으로 복원
        try:
            sph_special_types = np.array(checkpoint_data['sph_special_types'], dtype=np.int32)
            sph_smoothing_lengths = np.array(checkpoint_data['sph_smoothing_lengths'], dtype=np.float32)
            sph_alpha_visc = np.array(checkpoint_data['sph_alpha_visc'], dtype=np.float32)
            sph_beta_visc = np.array(checkpoint_data['sph_beta_visc'], dtype=np.float32)
            sph_etha_a_dt = np.array(checkpoint_data['sph_etha_a_dt'], dtype=np.float32)
            sph_ae_k = np.array(checkpoint_data['sph_ae_k'], dtype=np.float32)
            sph_etha_a = np.array(checkpoint_data['sph_etha_a'], dtype=np.float32)
            sph_B_unit = np.array(checkpoint_data['sph_B_unit'], dtype=np.float32)
            sph_S_a = np.array(checkpoint_data['sph_S_a'], dtype=np.float32)
            sph_S_b = np.array(checkpoint_data['sph_S_b'], dtype=np.float32)
            sph_dB_dt = np.array(checkpoint_data['sph_dB_dt'], dtype=np.float32)
            
            for i in range(min(len(sph_special_types), pos.shape[0])):
                is_special_particle_type[i] = int(sph_special_types[i])
                h_smooth[i] = float(sph_smoothing_lengths[i])
                alpha_visc_p[i] = float(sph_alpha_visc[i])
                beta_visc_p[i] = float(sph_beta_visc[i])
                etha_a_dt_field[i] = float(sph_etha_a_dt[i])
                ae_k_field[i] = float(sph_ae_k[i])
                etha_a_field[i] = float(sph_etha_a[i])
                B_unit_field[i] = ti.Vector([sph_B_unit[i, 0], sph_B_unit[i, 1]])
                S_a_field[i] = ti.Vector([sph_S_a[i, 0], sph_S_a[i, 1]])
                S_b_field[i] = ti.Vector([sph_S_b[i, 0], sph_S_b[i, 1]])
                dB_dt[i] = ti.Vector([sph_dB_dt[i, 0], sph_dB_dt[i, 1]])
            
            print("[DEBUG] 나머지 필드들 복원 완료 (개별 설정 방식)")
        except Exception as e:
            print(f"[ERROR] 나머지 필드들 복원 중 오류: {e}")
            print("[DEBUG] 나머지 필드들 기본값 설정")
            for i in range(pos.shape[0]):
                is_special_particle_type[i] = 0
                h_smooth[i] = 0.2
                alpha_visc_p[i] = 1.0
                beta_visc_p[i] = 2.0
                etha_a_dt_field[i] = 0.0
                ae_k_field[i] = 0.0
                etha_a_field[i] = 0.0
                B_unit_field[i] = ti.Vector([0.0, 0.0])
                S_a_field[i] = ti.Vector([0.0, 0.0])
                S_b_field[i] = ti.Vector([0.0, 0.0])
                dB_dt[i] = ti.Vector([0.0, 0.0])
            print("[DEBUG] 나머지 필드들 기본값 설정 완료")
        
        print("[DEBUG] 모든 SPH 입자 데이터 Taichi 필드 복원 완료")
        
    except Exception as e:
        print(f"[ERROR] SPH 입자 데이터 복원 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("[WARNING] 체크포인트 복원에 실패했습니다. 하지만 이어서 진행하기 위해 복원을 강제합니다.")
        # 복원 실패 시에도 시뮬레이션을 계속할 수 있도록 강제로 진행
        print("[DEBUG] 오류가 발생했지만 시뮬레이션을 계속 진행합니다.")
    
    # GAN 입자 상태 복원 (초기화된 경우에만)
    if checkpoint_data['gan_initialized'] == 1:
        gan_count = checkpoint_data['gan_count']
        gan_particles_initialized[None] = 1
        gan_particle_count[None] = gan_count
        
        print(f"GAN 입자 {gan_count}개 상태 복원 중...")
        
        gan_pos.from_numpy(np.array(checkpoint_data['gan_positions']))
        gan_vel.from_numpy(np.array(checkpoint_data['gan_velocities']))
        gan_B.from_numpy(np.array(checkpoint_data['gan_magnetic_fields']))
        gan_rho.from_numpy(np.array(checkpoint_data['gan_densities']))
        gan_P_pressure.from_numpy(np.array(checkpoint_data['gan_pressures']))
        gan_u.from_numpy(np.array(checkpoint_data['gan_internal_energies']))
        gan_acc.from_numpy(np.array(checkpoint_data['gan_accelerations']))
        gan_is_200km_model.from_numpy(np.array(checkpoint_data['gan_is_200km_model']))
    
    print(f"시뮬레이션 상태 복원 완료!")
    print(f"복원된 상태: 프레임 {frame_count}, 시각 {sim_time/3600:.2f}시간")
    print(f"SPH 입자: {num_actual_particles[None]}개")
    print(f"GAN 입자: {gan_particle_count[None] if gan_particles_initialized[None] == 1 else 0}개")
    print("[DEBUG] restore_simulation_state 함수 완료")
    print("[DEBUG] restore_simulation_state 함수에서 나가기 직전")

def resume_simulation_from_checkpoint(checkpoint_data):
    """체크포인트에서 시뮬레이션 재개"""
    global sim_time, frame_count
    
    print("[DEBUG] resume_simulation_from_checkpoint 함수 시작")
    
    # 시뮬레이션 상태 완전 복원
    print("[DEBUG] restore_simulation_state 호출 시작")
    restore_simulation_state(checkpoint_data)
    print("[DEBUG] restore_simulation_state 호출 완료")
    
    print(f"체크포인트에서 시뮬레이션 재개: 프레임 {frame_count}, 시각 {sim_time/3600:.2f}시간")
    
    # 체크포인트에서 복원된 시뮬레이션 시간과 프레임 수를 유지하면서 메인 루프 실행
    resume_from_checkpoint = True
    
    # 디버깅을 위한 출력
    print(f"[DEBUG] resume_simulation_from_checkpoint: sim_time={sim_time}, frame_count={frame_count}")
    
    # 메인 루프 실행
    print("[DEBUG] main_simulation_loop 호출 시작")
    main_simulation_loop(resume_from_checkpoint=resume_from_checkpoint)
    print("[DEBUG] main_simulation_loop 호출 완료")
    print("[DEBUG] resume_simulation_from_checkpoint 함수 완료")

if __name__ == "__main__":
    # 완전한 체크포인트 확인 및 재시작 옵션 제공
    checkpoint_data = load_complete_checkpoint()
    
    if checkpoint_data:
        resume_simulation_from_checkpoint(checkpoint_data)
    else:
        main_simulation_loop(resume_from_checkpoint=False)