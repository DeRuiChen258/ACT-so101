# Dataset schema audit

- root: `/home/violet/Workspace/Data/lerobot_act/lerobot/svla_so101_pickplace`
- codebase_version: `v3.0`
- robot_type: `so100_follower`
- total_episodes: 50
- total_frames: 11939
- total_tasks: 1
- fps: 30
- camera_keys: observation.images.up, observation.images.side
- state_dim: 6
- action_dim: 6

| feature | dtype | shape |
| --- | --- | --- |
| `action` | float32 | [6] |
| `observation.state` | float32 | [6] |
| `observation.images.up` | video | [480, 640, 3] |
| `observation.images.side` | video | [480, 640, 3] |
| `timestamp` | float32 | [1] |
| `frame_index` | int64 | [1] |
| `episode_index` | int64 | [1] |
| `index` | int64 | [1] |
| `task_index` | int64 | [1] |

**problems**: none (schema PASS)
