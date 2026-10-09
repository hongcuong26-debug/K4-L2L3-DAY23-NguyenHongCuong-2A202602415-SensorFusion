# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

Part E–H đã hoàn thành và kiểm thử. Báo cáo sử dụng số liệu từ lần chạy
Waymo `compare --seed 0`, frame 0–198, sau lần sửa code cuối cùng.

## Thông tin học viên

- Họ tên: Nguyễn Hồng Cường
- MSSV: 2A202602415
- Email: 26ai.cuongnh2@vinuni.edu.vn
- Link repo (fork): https://github.com/hongcuong26-debug/K4-L2L3-DAY23-NguyenHongCuong-2A202602415-SensorFusion
- Commit code + artifacts CP5: `c17bfca2252581ba20dbfa4b1ed58edae2f0f3dd`.
- Commit hash nộp (`git rev-parse HEAD`): dùng hash commit cuối chứa báo cáo này, lấy sau khi commit và push; hash đó được nộp cùng URL fork trên VLearn.

## Tóm tắt kết quả

- `fusion_mode`: `compare`; `frames`: `[0,198]` (199 frames/mode); `seed`: `0`.
- `segment`: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`.
- Detection dùng chung hai mode: precision `0.9700934579439252`, recall `0.7004048582995951`, TP `519`, FP `16`, FN `222`.

Số gốc dưới đây lấy từ [metrics.json](artifacts/metrics.json) và đối chiếu với
[grade_run.log](artifacts/grade_run.log) của chính lần full compare sau code cuối:

| Metric | LiDAR | Fused |
|---|---:|---:|
| rmse (m, 3D) | 0.15032268781360134 | 0.1358667883353908 |
| matches | 502 | 502 |
| sum_sq_err (m²) | 11.343649056695735 | 9.26681165463209 |
| ghost_track_frames | 0 | 0 |
| missed_gt_frames | 239 | 239 |
| mean_confirmed_tracks | 2.522613065326633 | 2.522613065326633 |
| precision_track | 1.0 | 1.0 |
| coverage | 0.9672447013487476 | 0.9672447013487476 |

`precision_track = 502/(502+0) = 1`; `coverage = 502/519 = 0.9672447013487476`.
Delta RMSE (fused − lidar) = `-0.014455899478210549 m`.
Cả hai mode đạt các ngưỡng metric RMSE ≤0.45 m, precision_track ≥0.75,
coverage ≥0.70; delta ≤0.05 m. Đây là đối chiếu ngưỡng số liệu,
không thay cho kết quả chấm code, báo cáo hoặc vấn đáp của giảng viên.

Fused giảm RMSE khoảng 0.01446 m trong lần chạy này, giữ cùng 502 matches,
0 ghost và 239 miss, nên cải thiện tổng hợp không đến từ giảm số cặp ghép.
Tuy nhiên camera không cải thiện ở mọi frame:

- Mode lidar/fused, frame 93: cùng 2 matches, 0 ghost, 1 miss; SSE lần lượt
  `0.042219327427724024` / `0.032096904387808135 m²`.
- Mode lidar/fused, frame 188: cùng 3 matches, 0 ghost, 9 misses; SSE lần lượt
  `0.033894733286664966` / `0.0632841891715982 m²`, fused xấu hơn tại frame này.
- Frame 188 có 12 valid_gt, detection TP=2/FN=10 nhưng tracking matches=3:
  track có thể tồn tại qua frame detector bỏ sót. Coverage là tỷ lệ tổng
  matches/det_tp theo rubric, không phải tỷ lệ ghép từng detection của frame.
- Mode lidar/fused, frame 0: detection TP=2, confirmed=0, matches=0, misses=2;
  frame 4 đã có confirmed=2, matches=2, misses=0. Giai đoạn chưa confirmed
  cũng ảnh hưởng tracking misses, nên không đánh giá chỉ từ RMSE.

Runner đã tạo đủ `metrics.json`, `grade_run.log`, `metrics_lidar.json`,
`metrics_fused.json`, `grade_run_lidar.log`, `grade_run_fused.log`.
Validator `fusion_lab.evaluation.validate_metrics_records` đạt cho file gộp
và hai mode; có đúng 398 records tổng/199 mỗi mode, mỗi `(mode,frame)` duy nhất,
đủ frame 0–198, detection từng frame giống nhau giữa modes. Mọi count/SSE,
RMSE và trung bình tái tạo khớp metrics; không sửa tay artifacts.

Môi trường kiểm thử: Python 3.12.14; tests gốc `student/tests` đạt
128 passed, 0 failed/xfailed (530 warnings về np.matrix và input nonfinite).
Probes ngoài repo đạt: params/zero override, ndarray/matrix, camera depth/FOV/extrinsic,
association empty/all-inf và covariance khởi tạo xoay bất đẳng hướng.
Trên Windows chạy với `$env:PYTHONUTF8="1"` để fixtures tiếng Việt dùng UTF-8.

Lệnh đã chạy từ root repo, với weights pretrained được nạp vào model eval:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

## Giải thích kỹ thuật Parts E–H

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   LiDAR đo vị trí 3D trong hệ cảm biến, z (3,1) tính bằng m, R (3,3) bằng m²;
   camera đo pixel z (2,1), R (2,2) bằng pixel². `kalman.py::innovation` dùng
   z−h(x); `camera_fusion.py::camera_measurement_prediction` chiếu pinhole phi tuyến.
   `build_camera_measurement` lấy sigma_cam_i/j=5 từ params (R=diag(25,25)),
   còn Jacobian H do `platform/fusion_lab/tracking/sensors.py::Sensor.get_H` cung cấp.

2. Vì sao cần gating Mahalanobis trước khi gán?

   `association.py::mahalanobis_distance` tính d²=γᵀS⁻¹γ, S=HPHᵀ+R.
   Khác khoảng cách Euclidean, cùng residual có d² nhỏ hơn theo hướng uncertainty lớn.
   `chi2_gate` nhận d² hữu hạn, không âm và nhỏ hơn χ² ppf(0.995,dim_meas),
   với 3 chiều LiDAR/2 chiều camera, trước chọn global minimum greedy.
   `association_cost_matrix` kiểm FOV trước gọi MHD/H/h, tránh projection invalid.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   Track-then-fuse: `platform/fusion_lab/scripts/run_lab.py::run` predict mỗi track
   một lần/frame, association/update LiDAR rồi management, sau đó camera update
   nếu có nhóm FRONT. Waymo Frame gom đo cùng tick; Measurement.t=frame*dt.
   `associate_and_update` không predict. Trong `grade_run.log`, lidar/fused frame 93
   cùng có 2 confirmed/2 matches nhưng SSE lần lượt 0.042219327427724024 và
   0.032096904387808135 m². Log chứng minh các lượt đánh giá mỗi mode/frame;
   thứ tự nội bộ predict→LiDAR→camera được chứng minh bằng source runner,
   vì log chỉ chứa counts/SSE, không chứa residual/calibration/update events.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Extrinsic sai làm p_s=R_vs p_v+t_vs sai, nên h(x) lệch pixel có hệ thống;
   innovation z−h(x) bị bias và d² có thể vượt gate. Nếu vẫn lọt gate, EKF
   có thể kéo state lệch. Cơ chế ở projection student và H chain-rule platform.
   Chưa chạy bonus calibration drift, chưa đo residual hay RMSE của hiện tượng này.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi meas_list rỗng, không thể lấy loại sensor từ đo. Hàm vẫn gọi
   manager.manage_tracks(unmatched_tracks,unmatched_meas,sensor) đúng một lần.
   `platform/fusion_lab/tracking/manager.py::TrackManager` chỉ ghi hit/miss,
   init/delete cho LiDAR; miss chỉ trong FOV. Camera cập nhật x/P bằng EKF,
   không tăng/giảm score, xác nhận, tạo hoặc xóa track, kể cả lượt rỗng.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   `track_management.py`: init score=1/window=1/6, state initialized; hit +1/6
   cap 1, miss −1/6. Score >0.8 mới confirmed; hit chưa đủ ngưỡng thì tentative;
   đã confirmed giữ state sau miss. Xóa nếu Pxx hoặc Pyy >9, hoặc confirmed score <0.6,
   hoặc unconfirmed score ≤0. Các biên 0.8/0.6/9 dùng strictness đúng params.
   Init đổi position/covariance LiDAR sang vehicle; velocity zero với covariance
   diag(sigma_p44²,sigma_p55²,sigma_p66²). Tests gốc và probe xoay kiểm các quy tắc này.

## Bonus

Không thực hiện bonus.

## Khai báo sử dụng AI

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): ChatGPT/Codex, Superpowers và Google Drive connector.
- Dùng cho phần nào (hàm, câu hỏi, debug): ChatGPT/Superpowers giúp lập plan; Codex/Superpowers executing-plans triển khai 17 hàm Part E–H, xử lý môi trường UTF-8, viết probes ngoài repo, chạy Waymo/validator và hỗ trợ sáu câu giải thích cùng báo cáo từ metrics/log thật. Google Drive connector tìm nguồn lớp và tải weights; học viên bổ sung segment local. Một agent tuần tự, self-review toàn diff; không có reviewer độc lập.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Codex chạy tests nguyên bản trước/sau từng Part, suite cuối 128 passed; bốn probes ngoài repo đạt; đối chiếu công thức EKF, gate, FOV và lifecycle. Full Waymo compare seed 0/frame 0–198 exit 0; validator kiểm sáu artifacts và 398 records đạt. Checker sau commit báo `KẾT QUẢ: SẴN SÀNG NỘP`. Học viên chịu trách nhiệm đọc, kiểm tra và giải thích code/câu trả lời khi vấn đáp.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: giữ nguyên
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP` sau commit báo cáo
- [x] Fork Public đúng tên; code và sáu artifacts đã push lên `main`
- [ ] Học viên nộp URL fork + hash commit cuối 40 ký tự trên VLearn ([hướng dẫn nộp](../SUBMISSION.md))
