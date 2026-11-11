# devsim-sic-mosfet-guide

오픈소스 TCAD **DEVSIM**으로 4H-SiC 평판형 MOSFET을 시뮬레이션하는 과정을 처음부터 정리한 레포.
velog 시리즈 **DEVSIM SiC MOSFET** 의 코드, 로그, 그림이 전부 여기 있음.

- 시뮬레이터: DEVSIM 2.11.0 (Python 모듈) + MKL
- 소자: 4H-SiC 평판형 MOSFET 2D half-cell (셀 피치 7 µm, 채널 0.5 µm, 게이트 산화막 50 nm), 300 K / 423 K
- 물리 모델과 숫자는 논문 레포 [sic-mosfet-devsim](https://github.com/mseokq23/sic-mosfet-devsim)과 같음 (기준 소자 I–V가 비트 단위로 같게 나옴)

| 결과 | 300 K | 423 K |
|---|---|---|
| Vth (I<sub>D</sub> = 1e-4 A/cm, V<sub>DS</sub> = 0.1 V) | 3.765 V | 3.530 V |
| SS | 112.5 mV/dec | 155.7 mV/dec |
| R<sub>on,sp</sub> (V<sub>GS</sub> = 18 V) | 2.000 mΩ·cm² | 2.871 mΩ·cm² |
| 메시 검증 (medium vs fine) | Vth 0.03 mV, Ron 0.9 % | |
| 역추정 (논문 데이터, GP) | 300 K 특징 0.112 → 300 + 423 K 특징 0.072 | |
| adjoint 민감도 vs 다시 풀기 | 상대 차이 4e-6 이하 | |

## 시작

**GitHub Codespaces**: Code → Codespaces → Create codespace on main (2-core면 충분함).
처음 만들 때 `requirements.txt` 설치와 `steps/02_check_env.py`가 자동으로 돌아감.

**로컬 Linux (x86-64)**

```bash
git clone https://github.com/mseokq23/devsim-sic-mosfet-guide.git
cd devsim-sic-mosfet-guide
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python steps/02_check_env.py
```

`devsim`만 설치하면 import에서 `Error loading math libraries`가 나옴. `mkl`을 같이 설치해야 함 ([logs/02_no_mkl.log](logs/02_no_mkl.log)).

## 폴더

```
sicdev/          공통 코드
  params.py        4H-SiC 물성, 온도 의존
  physics.py       DEVSIM 모델식 (Poisson, 전자/홀 연속, SG 전류, 이동도, 불완전 이온화, 계면전하, 접촉)
  devices.py       1D PN 다이오드, 1D MOS 커패시터, 2D half-cell MOSFET (메시, 도핑), 평형 해
  solver.py        DC 풀기, 바이어스 램프 (실패하면 스텝 절반)
  mosfet.py        MOSFET 한 번 계산 = 정해진 바이어스 순서 + 특징 추출
  extract.py       Vth, SS, gm, Ion, Ron,sp, log ID 표본
  worker.py        run 하나를 새 프로세스에서
  runner.py        설계표 전체를 병렬로, 끝난 run은 건너뜀
  adjoint.py       adjoint 민감도
  plot.py          그림 공통
configs/         기준 설정(baseline.yaml), 설계표 csv
steps/           포스트 순서대로 실행하는 스크립트 (파일 번호 = 포스트 번호)
results/         계산 결과 json, csv
logs/            실행 로그 (파일 번호 = 포스트 번호)
docs/img/        포스트에 쓴 그림 (폴더 번호 = 포스트 번호)
```

## 포스트와 스크립트

| 편 | 내용 | 스크립트 |
|---|---|---|
| 1 | 시작하며 | - |
| 2 | 실습 환경 | `02_check_env.py` |
| 3 | 이론 1 - 반도체 소자 방정식 | - |
| 4 | 이론 2 - 유한체적법, Scharfetter-Gummel, Newton | `04_sg_vs_central.py` |
| 5 | DEVSIM 구조 - 1D 저항 | `05_resistor_1d.py` |
| 6 | 4H-SiC 물성 1 - 밴드갭, 상태밀도, n<sub>i</sub> | `06_sic_bands.py` |
| 7 | 4H-SiC 물성 2 - 불완전 이온화 | `07_ionization.py` |
| 8 | 4H-SiC 물성 3 - 이동도 | `08_mobility.py` |
| 9 | 1D PN 다이오드 1 - 메시, 도핑, 평형 | `09_pn_equilibrium.py` |
| 10 | 1D PN 다이오드 2 - 해석해 비교, 정밀도 | `10_pn_check.py` |
| 11 | 1D MOS 커패시터 1 - 게이트, 산화막, 계면 | `11_moscap_bands.py` |
| 12 | 1D MOS 커패시터 2 - Vfb, Vth, C-V | `12_moscap_cv.py` |
| 13 | 평판형 SiC MOSFET 구조와 half-cell | `13_structure.py` |
| 14 | 2D 메시 | `14_mesh.py` |
| 15 | 도핑, 계면전하, 표면 이동도 | `15_doping.py` |
| 16 | 평형 해와 전자만 푸는 이유 | `16_equilibrium.py` |
| 17 | 전달 특성 | `17_transfer.py` |
| 18 | 출력 특성과 R<sub>on,sp</sub> | `18_output.py` |
| 19 | 특징 추출 | `19_features.py` |
| 20 | 메시, 정밀도, 반복성 검증 | `20_mesh_study.py` |
| 21 | 300 K vs 423 K | `21_temperature.py` |
| 22 | 공정 파라미터 하나씩 바꾸기 | `22_one_at_a_time.py` |
| 23 | 여러 run 자동화 | `23_doe.py` |
| 24 | 특징으로 공정 파라미터 역추정 | `24_inverse_gp.py` |
| 25 | adjoint 민감도 1 - 원리, 1D | `25_adjoint_resistor.py` |
| 26 | adjoint 민감도 2 - SiC MOSFET | `26_adjoint_mosfet.py` |
| 27 | 정리 | - |

## 한 번에 다시 돌리기

```bash
git clone https://github.com/mseokq23/sic-mosfet-devsim.git ../sic-mosfet-devsim   # 24편에서 씀
bash run_all.sh        # 2-core에서 약 45분, logs/ 를 다시 씀
```

22, 23편은 `results/oat`, `results/doe`에 이미 있는 run을 건너뜀. 처음부터 다시 계산하려면 두 폴더를 지우고 돌리면 됨.

## 라이선스

MIT. DEVSIM은 Apache 2.0.
