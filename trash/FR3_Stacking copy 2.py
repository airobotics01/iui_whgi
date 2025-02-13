# 2012 이걸
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import numpy as np
import time
from omni.isaac.core import World
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.manipulators import SingleManipulator
import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.tasks import PickPlace
import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim, create_prim




import time
import omni.kit.commands
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim, create_prim


class FR3RMPFlowController(mg.MotionPolicyController):
    def __init__(
        self,
        name: str,
        robot_articulation: Articulation,
        end_effector_frame_name="fr3_hand",
        physics_dt: float = 1.0 / 60.0,
    ) -> None:
        # TODO: change the follow paths
        mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
        rmp_config_dir = os.path.join(
            mg_extension_path, "motion_policy_configs", "FR3", "rmpflow"
        )
        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=os.path.join(
                rmp_config_dir, "fr3_robot_description.yaml"
            ),
            urdf_path=os.path.join(
                mg_extension_path, "motion_policy_configs", "FR3", "fr3.urdf"
            ),
            rmpflow_config_path=os.path.join(rmp_config_dir, "fr3_rmpflow_config.yaml"),
            end_effector_frame_name=end_effector_frame_name,
            maximum_substep_size=0.00334,
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, self.rmpflow, physics_dt
        )

        mg.MotionPolicyController.__init__(
            self, name=name, articulation_motion_policy=self.articulation_rmp
        )
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )


class FR3PickPlaceController(manipulators_controllers.PickPlaceController):
    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: Articulation,
        end_effector_initial_height: float = None,
        events_dt: list[float] = None,
    ) -> None:
        if events_dt is None:
            events_dt = [
                0.005,  # Phase 0: 1/0.005 steps
                0.001,  # Phase 1: 1/0.002 steps
                0.1,  # Phase 2: 10 steps - waiting phase, can have larger dt
                0.05,  # Phase 3: 20 steps - gripper closing
                0.0008,  # Phase 4: 1/0.0008 steps
                0.005,  # Phase 5: 1/0.005 steps
                0.0008,  # Phase 6: 1/0.0008 steps
                0.05,  # Phase 7: 20 steps - gripper opening
                0.0008,  # Phase 8: 1/0.0008 steps
                0.008,  # Phase 9: 1/0.008 steps
            ]
        super().__init__(
            name=name,
            cspace_controller=FR3RMPFlowController(
                name=name + "_cspace_controller",
                robot_articulation=robot_articulation,
            ),
            gripper=gripper,
            end_effector_initial_height=end_effector_initial_height,
            events_dt=events_dt,
        )


class FR3PickPlaceTask(PickPlace):
    def __init__(
        self,
        name: str = "FR3_pick_place",
        cube_initial_position: np.ndarray = None,
        cube_initial_orientation: np.ndarray = None,
        target_position: np.ndarray = None,
        offset: np.ndarray = None,
    ) -> None:
        super().__init__(
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=np.array([0.0515, 0.0515, 0.0515]),
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        robot_prim_path = "/World/FR3"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/FR3/fr3.usd"
        add_reference_to_stage(usd_path=path_to_robot_usd, prim_path=robot_prim_path)
        gripper = ParallelGripper(  # For the finger only the first value is used, second is ignored
            end_effector_prim_path="/World/FR3/fr3_hand",
            joint_prim_names=["fr3_finger_joint1", "fr3_finger_joint2"],
            joint_opened_positions=np.array([0.04, 0.04]),
            joint_closed_positions=np.array([0, 0]),
            action_deltas=np.array([0.04, 0.04]),
        )
        fr3_robot = SingleManipulator(
            prim_path=robot_prim_path,
            name="my_fr3",
            end_effector_prim_name="fr3_hand",
            gripper=gripper,
        )
        joints_default_positions = np.array(
            [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.7, +0.04, -0.04]
        )
        joints_default_positions[7] = 0.04
        joints_default_positions[8] = 0.04
        fr3_robot.set_joints_default_state(positions=joints_default_positions)
        return fr3_robot
    



class FR3Stacking(PickPlace):
    def __init__(self, name="FR3_stacking", cube_positions=None, target_positions=None):
        if cube_positions is None or target_positions is None:
            raise ValueError("cube_positions and target_positions must be provided.")
        
        self.num_cubes = len(cube_positions)
        self.cube_positions = cube_positions
        self.target_positions = target_positions
        self.current_index = 0  # 현재 처리 중인 블록 인덱스
        self.task_done = False  # 완료 여부

        # 기본적으로 첫 번째 블록을 기준으로 부모 클래스 초기화
        super().__init__(
            name=name,
            cube_initial_position=self.cube_positions[0],
            cube_initial_orientation=np.array([0, 0, 0, 1]),
            target_position=self.target_positions[0],
            cube_size=np.array([0.0515, 0.0515, 0.0515])
        )

    def set_robot(self) -> SingleManipulator:
        robot_prim_path = "/World/FR3"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/FR3/fr3.usd"
        add_reference_to_stage(usd_path=path_to_robot_usd, prim_path=robot_prim_path)
        
        gripper = ParallelGripper(
            end_effector_prim_path="/World/FR3/fr3_hand",
            joint_prim_names=["fr3_finger_joint1", "fr3_finger_joint2"],
            joint_opened_positions=np.array([0.04, 0.04]),
            joint_closed_positions=np.array([0, 0]),
            action_deltas=np.array([0.04, 0.04])
        )
        
        fr3_robot = SingleManipulator(
            prim_path=robot_prim_path,
            name="my_fr3",
            end_effector_prim_name="fr3_hand",
            gripper=gripper
        )
        
        joints_default_positions = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.7, 0.04, -0.04])
        fr3_robot.set_joints_default_state(positions=joints_default_positions)
        return fr3_robot

    def update_target(self):
        if self.current_index >= self.num_cubes - 1:
            print("✅ 모든 블록이 스택되었습니다! 종료합니다.")
            self.task_done = True
            return
        
        self.current_index += 1
        self._cube_initial_position = self.cube_positions[self.current_index]
        self._target_position = self.target_positions[self.current_index]
        print(f"🔄 다음 블록 타겟 설정: {self.current_index}")








# 시뮬레이션 환경 설정
my_world = World(stage_units_in_meters=1.0)

# 블록 위치 정의
cube_positions = [
    np.array([-0.3, 0.4, 0.0515 / 2.0]),
    np.array([-0, 0.2, 0.0515 / 2.0]),
    np.array([-0.3, 0.2, 0.0515 / 2.0]),
]

stack_positions = [
    np.array([-0.3, 0.6, 0.0515 / 2.0]),
    np.array([-0.3, 0.6, 0.0515 + 0.0515 / 2.0]),
    np.array([-0.3, 0.6, 2 * 0.0515 + 0.0515 / 2.0]),
]

# 태스크 생성
my_task = FR3Stacking(cube_positions=cube_positions, target_positions=stack_positions)
my_world.add_task(my_task)
my_world.reset()

# 로봇 초기화
fr3_robot = my_task.set_robot()
fr3_robot.initialize()
gripper = fr3_robot.gripper

# 컨트롤러 설정
my_controller = manipulators_controllers.PickPlaceController(
    name="FR3_controller",
    cspace_controller=None,  # 변경된 인자 적용
    gripper=gripper,
    end_effector_initial_height=0.3,
)

# 시뮬레이션 실행
while simulation_app.is_running():
    my_world.step(render=True)
    
    if my_world.is_playing():
        observations = my_world.get_observations()
        if my_task.task_done:
            print("✅ 모든 블록이 스택되었습니다! 시뮬레이션을 종료합니다.")
            break
        
        if my_controller.is_done():
            my_task.update_target()
        
        actions = my_controller.forward(
            picking_position=observations["cube_position"],
            placing_position=observations["target_position"],
            current_joint_positions=observations["joint_positions"],
            end_effector_offset=np.array([0, 0, 0.0925]),
        )
        
        fr3_robot.get_articulation_controller().apply_action(actions)
