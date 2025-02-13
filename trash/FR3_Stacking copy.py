# 2012
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import numpy as np
from omni.isaac.core import World
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.manipulators import SingleManipulator
import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.tasks import PickPlace
import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import create_new_stage
from omni.isaac.core.utils.prims import create_prim
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
    






class FR3Stacking(FR3PickPlaceTask):
    def __init__(self, name: str = "FR3_stacking", cube_initial_positions=None, 
                 cube_initial_orientations=None, target_positions=None, offset=None) -> None:
        if cube_initial_positions is None:
            raise ValueError("cube_initial_positions must be provided and cannot be None.")
        if cube_initial_orientations is None:
            cube_initial_orientations = [np.array([0, 0, 0, 1]) for _ in cube_initial_positions]

        # ✅ `self.world`를 먼저 생성
        self.world = World(stage_units_in_meters=1.0)

        # ✅ 부모 클래스 초기화
        super().__init__(
            name=name,
            cube_initial_position=cube_initial_positions[0],  
            cube_initial_orientation=cube_initial_orientations[0],
            target_position=target_positions[0],  
            offset=offset,
        )

        self.cube_initial_positions = cube_initial_positions
        self.cube_initial_orientations = cube_initial_orientations
        self.target_positions = target_positions
        self.current_index = 0  
        self.task_done = False  # ✅ 완료 상태 변수 추가

        # ✅ 블록을 환경에 추가 (중복 생성 방지)
        self.cube_prims = []
        for i, pos in enumerate(cube_initial_positions):
            cube_name = f"cube_{i}"
            prim_path = f"/World/{cube_name}"

            # ✅ 기존 Prim 삭제 (완전히 삭제될 때까지 확인)
            existing_prim = get_prim_at_path(prim_path)
            if existing_prim:
                print(f"⚠️ 기존 Prim 삭제 시도: {prim_path}")
                delete_prim(prim_path)
                self.world.scene.wait_for_usd()  # 🚀 USD 업데이트 대기
                time.sleep(0.2)  # 🔥 추가 대기 시간
                
                # 🔄 삭제 확인 및 재시도
                max_retries = 5
                for retry in range(max_retries):
                    if not get_prim_at_path(prim_path):
                        break
                    print(f"🔄 삭제 확인 중... ({retry+1}/{max_retries})")
                    delete_prim(prim_path)
                    time.sleep(0.2)
                else:
                    raise Exception(f"❌ Prim 삭제 실패: {prim_path}")

            # ✅ 새로운 블록 생성
            create_prim(
                prim_path,  
                "Cube",  
                position=pos,  
                orientation=cube_initial_orientations[i],
                scale=[0.0515, 0.0515, 0.0515],  
            )

            self.cube_prims.append(prim_path)

    def update_target(self):
        """ 다음 블록을 타겟으로 설정하는 함수 """
        if self.current_index >= len(self.cube_initial_positions) - 1:
            print("✅ 모든 블록이 스택되었습니다! 종료합니다.")
            self.task_done = True  # ✅ 모든 블록을 옮기면 완료 처리
            return
        
        self.current_index += 1  # 다음 블록 인덱스로 이동

        # ✅ 다음 블록 Pick & Place 위치 업데이트
        self._cube_initial_position = self.cube_initial_positions[self.current_index]
        self._cube_initial_orientation = self.cube_initial_orientations[self.current_index]
        self._target_position = self.target_positions[self.current_index]

        print(f"🔄 다음 블록을 타겟으로 설정: {self.current_index}")



















# 월드 생성
my_world = World(stage_units_in_meters=1.0)

# 여러 개의 블록을 쌓을 위치 설정
stack_positions = [
    np.array([-0.3, 0.6, 0.0515 / 2.0]),  # 첫 번째 블록 위치
    np.array([-0.3, 0.6, 0.0515 + 0.0515 / 2.0]),  # 두 번째 블록 (첫 블록 위)
    np.array([-0.3, 0.6, 2 * 0.0515 + 0.0515 / 2.0]),  # 세 번째 블록 (두 번째 블록 위)
]
# 블록 초기 위치 설정 (로봇이 집을 위치)
cube_initial_positions = [
    np.array([-0.3, 0.4, 0.0515 / 2.0]),  # 첫 번째 블록 pick 위치
    np.array([-0, 0.2, 0.0515 / 2.0]),  # 두 번째 블록 pick 위치 (같은 위치에서 pick)
    np.array([-0.3, 0.2, 0.0515 / 2.0]),  # 세 번째 블록 pick 위치
]

# 블록 초기 방향 설정 (기본적으로 모든 블록은 회전 없이 pick)
cube_initial_orientations = [np.array([0, 0, 0, 1]) for _ in cube_initial_positions]

# 블록을 놓을 위치 설정 (스택할 목표 위치)
stack_positions = [
    np.array([-0.3, 0.6, 0.0515 / 2.0]),  # 첫 번째 블록 위치
    np.array([-0.3, 0.6, 0.0515 + 0.0515 / 2.0]),  # 두 번째 블록 (첫 블록 위)
    np.array([-0.3, 0.6, 2 * 0.0515 + 0.0515 / 2.0]),  # 세 번째 블록 (두 번째 블록 위)
]

# my_task 객체 생성 (이제 cube_initial_positions과 cube_initial_orientations도 전달!)
my_task = FR3Stacking(
    cube_initial_positions=cube_initial_positions,
    cube_initial_orientations=cube_initial_orientations,
    target_positions=stack_positions,
)


# 시뮬레이션 월드에 태스크 추가
my_world.add_task(my_task)
my_world.reset()

# 로봇 세팅
fr3_robot = my_task.set_robot()
fr3_robot.initialize()
gripper = fr3_robot.gripper





my_controller = FR3PickPlaceController(
    name="FR3_controller",
    gripper=gripper,
    robot_articulation=fr3_robot,
    end_effector_initial_height=0.3,
)

task_params = my_world.get_task("FR3_stacking").get_params()
articulation_controller = fr3_robot.get_articulation_controller()
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed or my_world.current_time_step_index == 0:
            my_task.cleanup()
            my_world.reset()

            # 새로운 로봇 초기화
            fr3_robot = my_task.set_robot()
            fr3_robot.initialize()
            gripper = fr3_robot.gripper

            # 컨트롤러 재설정
            my_controller = FR3PickPlaceController(
                name="FR3_controller",
                gripper=gripper,
                robot_articulation=fr3_robot,
                end_effector_initial_height=0.3,
            )

            my_task.post_reset()
            reset_needed = False

        observations = my_world.get_observations()

        # ✅ 모든 작업이 끝나면 시뮬레이션 종료
        if my_task.task_done:
            print("✅ 모든 블록을 옮겼으므로 시뮬레이션을 종료합니다!")
            break  # 🚀 루프 종료

        # 블록이 목표 위치에 도달하면 다음 블록으로 이동
        if my_controller.is_done():
            my_task.update_target()

        # 컨트롤러 업데이트 및 동작 수행
        actions = my_controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]]["position"],
            placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, 0, 0.0925]),
        )

        articulation_controller.apply_action(actions)
