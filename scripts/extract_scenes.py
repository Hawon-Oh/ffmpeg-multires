from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

# 입력 영상
video_path = "examples/Golden - KPop Demon Hunters.mp4"
output_dir = "results/Golden - KPop Demon Hunters/scenes"

# 1. 비디오/장면 매니저 준비
video_manager = VideoManager([video_path])
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=30.0))  
# threshold ↓: 더 민감하게 감지 (페이드도 잡힘)

video_manager.start()

# 2. 장면 탐지 실행
scene_manager.detect_scenes(frame_source=video_manager)

# 3. 탐지 결과 얻기
scene_list = scene_manager.get_scene_list()
print(f"Detected {len(scene_list)} scenes.")

for i, (start, end) in enumerate(scene_list):
    print(f"Scene {i+1}: Start {start.get_timecode()} / End {end.get_timecode()}")

# 4. 장면별 대표 프레임 저장
save_images(
    scene_list,
    video_manager,
    num_images=1,              # 각 장면마다 1장
    image_name_template="$SCENE_NUMBER-$TIMECODE",
    output_dir=output_dir
)

video_manager.release()
