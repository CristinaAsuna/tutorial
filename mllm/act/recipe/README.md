# ACT 真实机器人 recipe 合同

论文级 ACT 需要 ALOHA 或等价的双臂 teleoperation demonstrations、同步多相机 RGB、follower joint positions、leader target joint actions、动作统计量和控制频率。动作应保留为绝对 target joint positions，并由底层高频 PID 跟踪；policy 只预测 future action chunk。

顶层 CPU toy 不连接相机、Dynamixel、ROS 或 ALOHA。它不能用于真机控制，也不因配置存在而宣称复现论文成功率。运行 `python validate_config.py` 只检查这些数据/控制接口契约。
