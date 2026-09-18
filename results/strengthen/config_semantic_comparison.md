# Semantic configuration comparison

Publication safe: **False**

## Recursive differences

| path | classification | PhysX | Newton | rationale |
|---|---|---|---|---|
| `env.events.add_base_mass` | uncontrolled confound | `{"func": "isaaclab.envs.mdp.events:randomize_rigid_body_mass", "interval_range_s": null, "is_global_time": false, "min_step_count_between_reset": 0, "mode": "startup", "params":...` | `'<missing>'` | PhysX randomizes base mass at startup with add range [-1, 3], while Newton has no matching event. |
| `env.events.base_com` | harmless representation | `None` | `'<missing>'` | PhysX dumps a disabled event as null while Newton omits it. |
| `env.events.physics_material` | uncontrolled confound | `{"func": "isaaclab.envs.mdp.events:randomize_rigid_body_material", "interval_range_s": null, "is_global_time": false, "min_step_count_between_reset": 0, "mode": "startup", "para...` | `'<missing>'` | PhysX randomizes robot rigid-body material at startup, while Newton has no matching event. |
| `env.log_dir` | harmless representation | `'$ISAACLAB_PATH/logs/rsl_rl/go2flat_bench10/2026-07-02_18-16-56_physx_s0'` | `'$ISAACLAB_PATH/logs/rsl_rl/go2flat_bench10/2026-07-02_18-36-10_newton_s0'` | Run timestamp and backend name identify output directories only. |
| `env.scene.contact_forces.class_type` | required backend-specific | `'isaaclab_physx.sensors.contact_sensor.contact_sensor:ContactSensor'` | `'isaaclab_newton.sensors.contact_sensor.contact_sensor:ContactSensor'` | Contact sensor implementation follows the selected backend package. |
| `env.scene.contact_forces.filter_shape_prim_expr` | harmless representation | `'<missing>'` | `[]` | Newton dumps explicit empty contact sensor shape filters where PhysX omits them. |
| `env.scene.contact_forces.sensor_shape_prim_expr` | harmless representation | `'<missing>'` | `[]` | Newton dumps explicit empty contact sensor shape filters where PhysX omits them. |
| `env.sim.physics.bounce_threshold_velocity` | required backend-specific | `0.5` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.class_type` | required backend-specific | `'isaaclab_physx.physics.physx_manager:PhysxManager'` | `'isaaclab_newton.physics.newton_manager:NewtonManager'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.debug_mode` | required backend-specific | `'<missing>'` | `False` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.enable_ccd` | required backend-specific | `False` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.enable_enhanced_determinism` | required backend-specific | `False` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.enable_external_forces_every_iteration` | required backend-specific | `False` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.enable_scene_query_support` | required backend-specific | `False` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.enable_stabilization` | required backend-specific | `False` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.friction_correlation_distance` | required backend-specific | `0.025` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.friction_offset_threshold` | required backend-specific | `0.04` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.gpu_collision_stack_size` | required backend-specific | `67108864` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.gpu_found_lost_aggregate_pairs_capacity` | required backend-specific | `33554432` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.gpu_found_lost_pairs_capacity` | required backend-specific | `2097152` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.gpu_heap_capacity` | required backend-specific | `67108864` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.gpu_max_num_partitions` | required backend-specific | `8` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.gpu_max_particle_contacts` | required backend-specific | `1048576` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.gpu_max_rigid_contact_count` | required backend-specific | `8388608` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.gpu_max_rigid_patch_count` | required backend-specific | `327680` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.gpu_max_soft_body_contacts` | required backend-specific | `1048576` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.gpu_temp_buffer_capacity` | required backend-specific | `16777216` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.gpu_total_aggregate_pairs_capacity` | required backend-specific | `2097152` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.max_position_iteration_count` | required backend-specific | `255` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.max_velocity_iteration_count` | required backend-specific | `255` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.min_position_iteration_count` | required backend-specific | `1` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.min_velocity_iteration_count` | required backend-specific | `0` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.num_substeps` | required backend-specific | `'<missing>'` | `1` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.solve_articulation_contact_last` | required backend-specific | `False` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.solver_cfg` | required backend-specific | `'<missing>'` | `{"actuator_gears": null, "ccd_iterations": 35, "cone": "pyramidal", "default_actuator_gear": null, "disable_contacts": false, "impratio": 1, "integrator": "implicitfast", "itera...` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.solver_type` | required backend-specific | `1` | `'<missing>'` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `env.sim.physics.use_cuda_graph` | required backend-specific | `'<missing>'` | `True` | Backend manager and solver/contact buffer settings are selected by the physics preset. |
| `agent.run_name` | harmless representation | `'physx_s0'` | `'newton_s0'` | Run name labels the TensorBoard/log directory only. |

## Field comparison

| field | classification | verdict |
|---|---|---|
| task | harmless representation | Same task is specified by run_strengthen.sh and probe artifacts; task id is not embedded in the committed env YAML. |
| observations | harmless representation | Policy observation terms, noise, and concatenation match. |
| actions | harmless representation | Joint position action config matches. |
| control frequency | harmless representation | Same policy rate from dt=0.005 and decimation=4. |
| timestep | harmless representation | Same simulation timestep. |
| decimation | harmless representation | Same action decimation. |
| solver/contact | required backend-specific | Physics manager and solver/contact buffer settings intentionally differ by backend. |
| actuator | harmless representation | DCMotor stiffness, damping, limits, friction, and saturation match. |
| rewards | harmless representation | Reward terms, weights, and params match. |
| terminations | harmless representation | Timeout and base-contact termination configs match. |
| reset distributions | uncontrolled confound | Startup material and mass randomization are not controlled across backends. |
| policy architecture | harmless representation | Actor/critic MLP shapes, activations, and distribution config match. |
| PPO hyperparams | harmless representation | PPO algorithm config and rollout length match. |
| normalization | harmless representation | No empirical normalization and no agent obs groups in either arm. |
| seeds | harmless representation | Manifest has seeds 0-9 and exit code 0 for both backends. |
| env count | harmless representation | Training resolved env count is 2048 for both arms. |
| training horizon | harmless representation | Both arms have 300 configured iterations and 300 CSV rows per seed. |
| eval protocol | harmless representation | Claims must be limited to training reward, throughput, and probes; there is no held-out policy evaluation protocol. |
| checkpoint selection | harmless representation | Training starts fresh; load_checkpoint is inert because resume=false. Final CSV iteration is used for reported training reward. |
| software versions | harmless representation | Both arms are from one strengthening suite under the same captured Isaac Lab SHA, package freeze, driver, and CUDA stack. |

## Required rerun

The committed training artifacts are not publication safe because the Newton arm is missing startup mass and robot material randomization events present in the PhysX arm. Existing results cannot be repaired post hoc.

Exact rerun commands after fixing the task preset so both resolved configs contain identical non-backend events:

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
export ISAACLAB_PYTHON=/path/to/IsaacLab/_isaac_sim/python.sh  # or the Isaac Lab venv python used for the original suite
./scripts/run_strengthen.sh
python3 scripts/compare_semantic_configs.py --fail-on-unsafe
python3 scripts/analyze_strengthen.py
```
