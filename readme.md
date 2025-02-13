
## 🛠 Reset이 되지 않던 원인과 해결 방법
⚠️ 1. my_world.clear() 이후 오브젝트가 정상적으로 다시 추가되지 않음
문제: clear() 후 새로운 task를 추가하지 않거나, 추가하더라도 reset() 전에 로봇이 정상적으로 배치되지 않음.
해결: clear() → add_task() → reset() 순서로 수행.
⚠️ 2. SingleManipulator가 초기화되지 않음
문제: fr3_robot.is_initialized() 함수가 존재하지 않아서 로봇이 초기화되지 않음.
해결: fr3_robot.initialize()를 직접 호출.
⚠️ 3. getTypes called on non-existent path /World/Cube 에러
문제: my_world.clear() 이후 큐브와 목표 지점(stack_target)이 정상적으로 다시 생성되지 않음.
해결: 새로운 FR3StackTask를 생성하고 add_task()를 통해 추가.
⚠️ 4. AttributeError: 'NoneType' object has no attribute 'gripper'
문제: fr3_robot이 None이거나, set_robot() 호출이 없어서 gripper 객체가 없는 상태.
해결: fr3_robot = my_task.set_robot() 후 반드시 initialize()를 실행하여 로봇을 등록.