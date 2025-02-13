# 2012 이걸

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
from omni.isaac.core.tasks import Stacking
import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import get_extension_path_from_name


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



class FR3StackingController(manipulators_controllers.StackingController):
    def __init__(
        self,
        name: str,
        pick_place_controller: FR3PickPlaceController,
        picking_order_cube_names: list[str],
        robot_observation_name: str,
    ) -> None:
        super().__init__(name, pick_place_controller, picking_order_cube_names, robot_observation_name)
        self.pick_place_controller = pick_place_controller  # ✅ 명확하게 저장
        self.picking_order_cube_names = picking_order_cube_names  # ✅ 추가
        self.robot_observation_name = robot_observation_name
        self.current_index = 0  # 현재 어떤 큐브를 다루고 있는지 인덱스로 관리

    def get_current_event(self) -> int:
        """현재 진행 중인 이벤트(스택 단계)를 반환"""
        return self.current_index


    def forward(
        self, observations: dict, 
        end_effector_orientation: np.ndarray = None, 
        end_effector_offset: np.ndarray = None
    ) -> ArticulationAction:
        if self.current_index >= len(self.picking_order_cube_names):
            return ArticulationAction()  # 모든 큐브가 스택되었으면 종료
        
        # 현재 목표하는 큐브 가져오기
        cube_name = self.picking_order_cube_names[self.current_index]
        if cube_name not in observations:
            raise ValueError(f"Cube {cube_name} not found in observations")
        
        picking_position = observations[cube_name]['position']
        placing_position = observations["stack_target"]['position']
        placing_position[2] += self.current_index * 0.05  # 스택 높이 증가
        
        action = self.pick_place_controller.forward(
            picking_position=picking_position,
            placing_position=placing_position,
            current_joint_positions=observations[self.robot_observation_name]['joint_positions'],
            end_effector_offset=end_effector_offset,
            end_effector_orientation=end_effector_orientation,
        )
        
        if self.pick_place_controller.is_done():
            self.current_index += 1  # 다음 큐브로 이동
            self.pick_place_controller.reset()
        
        return action
    
    def is_done(self) -> bool:
        return self.current_index >= len(self.picking_order_cube_names)
    
def reset(self):
    print(f"🔄 Resetting Controller. Current Index: {self.current_index}")
    super().reset()
    self.current_index = 0
    self.pick_place_controller.reset()





class FR3StackTask(Stacking):
    def __init__(
        self,
        name: str = "FR3_stack_task",
        cube_initial_positions: np.ndarray = None,
        cube_initial_orientations: np.ndarray = None,
        stack_target_positions: list = None,  # ✅ 리스트로 설정
        cube_size: np.ndarray = None,
        offset: np.ndarray = None,
    ) -> None:
        num_cubes = cube_initial_positions.shape[0]  # ✅ 큐브 개수 확인

        # ✅ stack_target_positions이 None이면 자동으로 설정
        if stack_target_positions is None:
            base_position = [-0.3, 0.6, 0.0515 / 2.0]  # 첫 번째 큐브의 목표 위치
            stack_target_positions = [
                [base_position[0], base_position[1], base_position[2] + i * 0.0515]  # 위로 쌓이도록 조정
                for i in range(num_cubes)
            ]

        self.stack_target_positions = stack_target_positions  # ✅ 명확하게 저장 (리스트 형태)

        super().__init__(
            name=name,
            cube_initial_positions=cube_initial_positions,
            cube_initial_orientations=cube_initial_orientations,
            stack_target_position=None,  # ✅ 개별 저장할 것이므로 None으로 설정
            cube_size=np.array([0.0515, 0.0515, 0.0515]) if cube_size is None else cube_size,
            offset=offset,
        )

    def get_task_objects(self) -> dict:
        task_objects = super().get_task_objects()
        # ✅ 각 큐브마다 별도 목표 위치 저장
        for i, pos in enumerate(self.stack_target_positions):
            task_objects[f"stack_target_{i}"] = {"position": np.array(pos, dtype=np.float32)}  # ✅ 명확한 dtype 변환 추가
        return task_objects


    def set_robot(self) -> SingleManipulator:
        robot_prim_path = "/World/FR3"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/FR3/fr3.usd"
        add_reference_to_stage(usd_path=path_to_robot_usd, prim_path=robot_prim_path)
        
        gripper = ParallelGripper(
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
        
        joints_default_positions = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.7, 0.04, -0.04])
        fr3_robot.set_joints_default_state(positions=joints_default_positions)
        return fr3_robot






#! my_world = World(stage_units_in_meters=1.0)
#! target_position = np.array([-0.3, 0.6, 0])
#! target_position[2] = 0.0515 / 2.0
#! my_task = FR3PickPlaceTask(target_position=target_position)

my_world = World(stage_units_in_meters=1.0)

# 큐브 초기 위치 설정
cube_positions = np.array([
    [0.1, 0.3, 0.0515 / 2.0],  # 첫 번째 큐브
    [0.3, 0.3, 0.0515 / 2.0],  # 두 번째 큐브
    [0.4, 0.5, 0.0515 / 2.0],  # 세 번째 큐브
])

stack_target_positions = np.array([
    [-0.3, 0.6, 0.0515 / 2.0],   # 첫 번째 큐브 목표 위치
    [-0.3, 0.6, 0.0515 * 1.5],   # 두 번째 큐브 목표 위치
    [-0.3, 0.6, 0.0515 * 2.5],   # 세 번째 큐브 목표 위치
])
my_task = FR3StackTask(
    cube_initial_positions=cube_positions,
    stack_target_positions=stack_target_positions.tolist(),  # ✅ numpy -> list 변환
)





my_world.add_task(my_task)
my_world.reset()


fr3_robot = my_task.set_robot()
fr3_robot.initialize()
gripper = fr3_robot.gripper

# my_controller = FR3PickPlaceController(
#     name="FR3_controller",
#     gripper=gripper,
#     robot_articulation=fr3_robot,
#     end_effector_initial_height=0.3,
# )


my_controller = FR3StackingController(
    name="FR3_stacking_controller",
    pick_place_controller=FR3PickPlaceController(
        name="FR3_pick_place_controller",
        gripper=gripper,
        robot_articulation=fr3_robot,
        end_effector_initial_height=0.3,
    ),
    picking_order_cube_names=["cube_1", "cube_2", "cube"],  # 큐브 순서 지정
    robot_observation_name=my_world.get_task("FR3_stack_task").get_params()["robot_name"]["value"],
)








#! task_params = my_world.get_task("FR3_pick_place").get_params()
task_params = my_world.get_task("FR3_stack_task").get_params()

articulation_controller = fr3_robot.get_articulation_controller()
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed or my_world.current_time_step_index == 0:
            # 기존 작업 정리
            my_task.cleanup()
            my_world.reset()
            
            # 새로운 로봇 생성 및 초기화
            fr3_robot = my_task.set_robot()
            fr3_robot.initialize()
            gripper = fr3_robot.gripper

            # # 컨트롤러 재설정 (기존 컨트롤러가 이전 로봇을 참조하기 때문)
            # my_controller = FR3PickPlaceController(
            #     name="FR3_controller",
            #     gripper=gripper,
            #     robot_articulation=fr3_robot,
            #     end_effector_initial_height=0.3,
            # )

            # # task 초기화
            # my_task.post_reset()
            # reset_needed = False
            my_controller = FR3StackingController(
                name="FR3_stacking_controller",
                pick_place_controller=FR3PickPlaceController(
                    name="FR3_pick_place_controller",
                    gripper=gripper,
                    robot_articulation=fr3_robot,
                    end_effector_initial_height=0.3,
                ),
                picking_order_cube_names=["cube_2", "cube_1", "cube"],  # 큐브 순서 지정
                robot_observation_name=task_params["robot_name"]["value"],
            )

            
            

        # observations = my_world.get_observations()

        # # Observation 확인
        # if (
        #     task_params["cube_name"]["value"] in observations
        #     and "position" in observations[task_params["cube_name"]["value"]]
        #     and "target_position" in observations[task_params["cube_name"]["value"]]
        #     and task_params["robot_name"]["value"] in observations
        #     and "joint_positions" in observations[task_params["robot_name"]["value"]]
        # ):
        #     print("Observation valid")

        # # 컨트롤러 업데이트 및 동작 수행
        # actions = my_controller.forward(stack_target_1
        #     picking_position=observations[task_params["cube_name"]["value"]]["position"],
        #     placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
        #     current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
        #     end_effector_offset=np.array([0, 0, 0.0925]),
        # )
        
        print("🛠 Checking Stack Target Positions:", stack_target_positions)
        observations = my_world.get_observations()

        # ✅ stack_target_0이 없으면 기본값으로 첫 번째 목표 위치 사용
        if f"stack_target_0" not in observations:
            print("⚠ Warning: 'stack_target' not found in observations. Using first target position as default.")
            observations["stack_target"] = {"position": np.array(stack_target_positions[0], dtype=np.float32)}  

        actions = my_controller.forward(
            observations=observations,
            end_effector_offset=np.array([0, 0, 0.0925]),
        )


        observations = my_world.get_observations()
        print("Observations:", observations)  # ✅ 데이터 확인



        if my_controller.is_done():
            print("Done picking and placing")
        else:
            print(f"Phase: {my_controller.get_current_event()}")

        # 새로 생성된 articulation_controller를 사용하여 동작 적용
        articulation_controller = fr3_robot.get_articulation_controller()
        articulation_controller.apply_action(actions)