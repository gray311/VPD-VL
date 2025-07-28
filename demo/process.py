import cv2
import glob
import os

# ① 按排序规则收集所有帧
frames = sorted(glob.glob("./waymo_605a964cc30b61fedf8a41bdf130f505/*.jpg"))

# ② 读取第一帧获取分辨率
first = cv2.imread(frames[0])
height, width, _ = first.shape

# ③ 创建视频写入器
fps = 12                       # 帧率
out = cv2.VideoWriter(
    "2.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),  # 也可改 'avc1'、'H264'
    fps,
    (width, height)
)

# ④ 逐帧写入
for fn in frames:
    img = cv2.imread(fn)
    out.write(img)

out.release()
print("Done! -> output.mp4")
