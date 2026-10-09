# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

> Trạng thái ngày 10/10/2026: Part E–H đã kiểm thử; CP5–CP6 chưa hoàn tất.
> Thiếu dữ liệu/weights. Học viên đã xác nhận MSSV và cung cấp email trong phiên.

## Thông tin học viên

- Họ tên: Nguyễn Hồng Cường
- MSSV: 2A202602415
- Email: 26ai.cuongnh2@vinuni.edu.vn
- Link repo (fork): https://github.com/hongcuong26-debug/K4-L2L3-DAY23-NguyenHongCuong-2A202602415-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): Chưa có hash nộp. Hash code CP4: `ab5c82b164f5d5f3cdf439ae6a2405e0488cd596` (chưa có artifacts; hash nộp cuối lấy sau khi CP5–CP6 đạt).

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: Chưa chạy. Cấu hình dự kiến: compare, [0,198], training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, seed 0.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: Chưa có metrics thật.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: Chưa có metrics thật.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: Chưa có metrics thật.
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: Chưa thể kết luận; chưa có log mode/frame thực nghiệm. Khi chạy xong sẽ đối chiếu RMSE, matches, ghosts/misses và frame cụ thể.

`precision_track = matches/(matches+ghost_track_frames)`;
`coverage = matches/detection.tp`; delta RMSE = fused − lidar.
Chưa có dữ liệu để tính các đại lượng này. Không thay bằng 0 hoặc số từ tests.

Blocker CP5: chưa có hai file local:

- `data/Waymo/training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`
- `data/weights/pretrained_fpn-resnet/fpn_resnet_18_epoch_300.pth`

Cần cung cấp file hoặc đường dẫn tương đương trong `student/config/paths.yaml`.
Sau khi có dữ liệu, chạy full compare sau code cuối, xác minh sáu artifacts,
398 records tổng và 199 records/mode bằng validator có sẵn trước khi điền số liệu.

Bằng chứng kiểm thử local: Python 3.12.14; tests gốc `student/tests` đạt
128 passed, 0 failed/xfailed (530 warnings về np.matrix và input nonfinite).
Probes ngoài repo đạt: params/zero override, ndarray/matrix, camera depth/FOV/extrinsic,
association empty/all-inf và covariance khởi tạo xoay bất đẳng hướng.
Trên Windows chạy với `$env:PYTHONUTF8="1"` để fixtures tiếng Việt dùng UTF-8.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

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
   `associate_and_update` không predict. Chưa có log thực nghiệm để dẫn mode/frame.
   Khi có log, counts/SSE chứng minh các lượt đánh giá, không chứng minh thứ tự nội bộ update.

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

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): ChatGPT/Codex và Superpowers.
- Dùng cho phần nào (hàm, câu hỏi, debug): ChatGPT/Superpowers giúp lập plan; Codex/Superpowers executing-plans triển khai 17 hàm Part E–H, xử lý môi trường UTF-8, viết probes ngoài repo và phần giải thích/báo cáo trạng thái. Một agent tuần tự, self-review toàn diff; không có reviewer độc lập.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Codex chạy tests nguyên bản trước/sau từng Part, suite cuối 128 passed; bốn probes ngoài repo đạt; đối chiếu công thức EKF, gate, FOV và lifecycle. Chưa chạy full Waymo/validator artifacts vì thiếu dữ liệu/weights. Học viên cần tự đọc, kiểm tra và giải thích mọi dòng trước khi nộp.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: giữ nguyên
- [ ] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [ ] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [ ] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
