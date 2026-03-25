# Control Regression Report

- Status: `completed`
- Cases: `2`

## Overview

| Case | Status | Baseline | Pose+Depth | Routing Differs | Score Delta |
| --- | --- | --- | --- | --- | --- |
| robot-walk-city | completed | succeeded | succeeded | True | -0.044 |
| robot-hero-pose | completed | succeeded | succeeded | True | -0.0809 |

## robot-walk-city

- Case status: `completed`
- Baseline status: `succeeded`
- Pose+Depth status: `succeeded`
- Routing differs: `True`
- Score delta: `-0.044`

### Baseline

- Job id: `robot-walk-city-baseline`
- Status: `succeeded`
- Error: `None`
- Output: `storage\outputs\robot-walk-city-baseline.json`
- Preview: `storage\outputs\robot-walk-city-baseline.png`
- Video: `storage\outputs\robot-walk-city-baseline.mp4`
- Score: `0.626`
- Provider artifacts: `-`
- Pose control: `False`
- Depth control: `False`
- Used image conditioning: `False`
- Used video conditioning: `False`
- Video call summary: `{"branch": "video", "has_control_input": false, "control_image_count": 0, "control_image_sizes": [], "controlnet_conditioning_scale": null, "motion_bucket_id": null, "noise_aug_strength": null, "pose_control": false, "depth_control": false}`

### Pose+Depth

- Job id: `robot-walk-city-pose_depth`
- Status: `succeeded`
- Error: `None`
- Output: `storage\outputs\robot-walk-city-pose_depth.json`
- Preview: `storage\outputs\robot-walk-city-pose_depth.png`
- Video: `storage\outputs\robot-walk-city-pose_depth.mp4`
- Score: `0.582`
- Provider artifacts: `camera_manifest, depth_manifest, skeleton_sequence, transition_manifest`
- Pose control: `True`
- Depth control: `True`
- Used image conditioning: `False`
- Used video conditioning: `True`
- Video call summary: `{"branch": "video", "has_control_input": false, "control_image_count": 0, "control_image_sizes": [], "controlnet_conditioning_scale": null, "motion_bucket_id": 139, "noise_aug_strength": 0.056, "pose_control": true, "depth_control": true}`

### Assertions

- baseline_succeeded: `True`
- controlled_succeeded: `True`
- controlled_has_pose: `True`
- controlled_has_depth: `True`
- controlled_video_conditioning_used: `True`
- routing_differs_from_baseline: `True`
- score_delta: `-0.044`

## robot-hero-pose

- Case status: `completed`
- Baseline status: `succeeded`
- Pose+Depth status: `succeeded`
- Routing differs: `True`
- Score delta: `-0.0809`

### Baseline

- Job id: `robot-hero-pose-baseline`
- Status: `succeeded`
- Error: `None`
- Output: `storage\outputs\robot-hero-pose-baseline.json`
- Preview: `storage\outputs\robot-hero-pose-baseline.png`
- Video: `storage\outputs\robot-hero-pose-baseline.mp4`
- Score: `0.6085`
- Provider artifacts: `-`
- Pose control: `False`
- Depth control: `False`
- Used image conditioning: `False`
- Used video conditioning: `False`
- Video call summary: `{"branch": "video", "has_control_input": false, "control_image_count": 0, "control_image_sizes": [], "controlnet_conditioning_scale": null, "motion_bucket_id": null, "noise_aug_strength": null, "pose_control": false, "depth_control": false}`

### Pose+Depth

- Job id: `robot-hero-pose-pose_depth`
- Status: `succeeded`
- Error: `None`
- Output: `storage\outputs\robot-hero-pose-pose_depth.json`
- Preview: `storage\outputs\robot-hero-pose-pose_depth.png`
- Video: `storage\outputs\robot-hero-pose-pose_depth.mp4`
- Score: `0.5276`
- Provider artifacts: `camera_manifest, depth_manifest, skeleton_sequence, transition_manifest`
- Pose control: `True`
- Depth control: `True`
- Used image conditioning: `False`
- Used video conditioning: `True`
- Video call summary: `{"branch": "video", "has_control_input": false, "control_image_count": 0, "control_image_sizes": [], "controlnet_conditioning_scale": null, "motion_bucket_id": 139, "noise_aug_strength": 0.056, "pose_control": true, "depth_control": true}`

### Assertions

- baseline_succeeded: `True`
- controlled_succeeded: `True`
- controlled_has_pose: `True`
- controlled_has_depth: `True`
- controlled_video_conditioning_used: `True`
- routing_differs_from_baseline: `True`
- score_delta: `-0.0809`

