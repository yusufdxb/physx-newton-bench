# Runtime policy-contract comparison

- Publication safe: **False**
- Left backend: physx
- Right backend: newton

| check | match |
|---|---:|
| robot_joint_order | False |
| robot_body_order | False |
| action_contract | False |
| observation_schema | True |

## Left joint order

`FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint, FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint, FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint`

## Right joint order

`FL_hip_joint, FL_thigh_joint, FL_calf_joint, FR_hip_joint, FR_thigh_joint, FR_calf_joint, RL_hip_joint, RL_thigh_joint, RL_calf_joint, RR_hip_joint, RR_thigh_joint, RR_calf_joint`

The cross-backend fixed-checkpoint result is invalid: action and/or joint-observation semantics differ at runtime.
