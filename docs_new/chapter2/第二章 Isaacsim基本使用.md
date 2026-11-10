# Isaac Sim基本使用

### 第二章和第三章的内容大部分源自[Isaac Sim的官方文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)，教程仅做了内容的筛选、解读并做了修改，去除了一些不太常用和复杂的功能，增加了操作流程可视化，方便大家基础入门，后续一些复杂的内容例如ROS NAV2等会在其他章节用到的时候重新介绍，尽量让大家有一个从易到难的学习体验。

### 第二章主要是通过最简单的案例，来让大家跟着操作一遍，从而熟悉Isaacsim仿真的完整操作流程。

## 2.1 Issac Sim初见

跟上一章一样，打开Isaacsim的页面，ROS Bridge Extension选择isaacsim.ros2.bridge，Use Internal ROS2 Libraries选择humble即可。

```python
cd isaac-sim
./isaac-sim.selector.sh
```

我们能看到如下的页面。

<img src=".\assets\page.png"/>

### 设置Isaacsim属性

在将任何内容添加到Isaacsim之前，请验证当前的Isaacsim属性设置是否符合你的预期。

1）在菜单栏在中依次点击 Edit > Preferences 打开首选项面板。

2）在面板左侧的列中浏览 Omniverse Kit 内按类别分组的多种设置。

3）从左侧栏中选择 Stage 并查看属性：

* 决定向上的轴 (Default Up axis)：Isaac Sim 中的默认值是 Z。如果您的资产是在具有不同向上轴的程序中创建的，则会导致您的资产在导入时发生旋转。

* 单位 (Stage units)：在 2022.1 之前的 Isaac Sim 版本中，单位是厘米，但现在的默认单位是米。然而，Omniverse Kit 的默认单位仍然是厘米。如果你发现 USD 的单位似乎相差了 100 倍，可能是这个原因。

* 默认旋转顺序 (Default rotation order)：默认设置为先绕 Z 轴旋转，然后绕 Y 轴，最后绕 X 轴旋转。

<img src=".\assets\stage.png"/>

### 创建物理场景

添加物理场景以模拟真实世界的物理现象，包括重力和物理时间步长。

1）前往菜单栏，并依次点击 Create > Physics > Physics Scene。

2）验证是否已将 PhysicsScene 添加到 stage 的树中。

3）点击即可查看其属性。可以看到，重力大小设置为 Earth Gravity ，即 9.8 米/秒²。长度的默认单位是米。

4）除非你要模拟数百个刚体和机器人，否则使用 CPU 物理引擎会更高效。

* 打开物理场景的属性选项卡

* 取消勾选 Enable GPU dynamics

* 将 Broadphase 类型设置为 MBP 

<img src=".\assets\physics.png"/>

### 添加一个物理地面

地面平面会阻止任何具有物理特性的物体坠落到其下方。即使该平面在每个方向上的可见范围仅为 25 米，其碰撞属性也会无限延伸，也就是说，这个物理地面是无限大的。

在虚拟环境中添加地面：

1）在顶部的菜单栏，依次点击 Create > Physics > Ground Plane

2）点击“眼睛图标”并选择Grid打开网格 ，可以更容易看到地面。

<img src=".\assets\ground.png"/>

### 添加一个灯光

每一个新的环境都会预先加载一个defaultLight，否则你将看不到任何的内容，这个光源在Environment中，可以在stage的树中找到。

1）为了让我们添加的聚光灯更容易被看到，我们先将defaultlight中的 Main > Intensity 调成300来降低默认灯光的强度。

<img src=".\assets\defaultlight.png"/>

2）通过 Create > Light > Sphere Light 创建一个新的球型灯光。

3）调整新的球型灯光的位置，在属性页面中，选择translate位移属性设置为(0, 0, 7)，将orient方向属性设置为(0, 0, 0)

4）修改灯光颜色、亮度和范围属性，修改颜色为(0, 1, 0)，将Intensity灯光强度改为1e6，Radius半径修改为0.5，在后面的shaping部分，将cone angle修改为45度，即可看到如下的灯光。

<img src=".\assets\spherelight.png"/>

通过这些内容的学习，你应该对整个Isaacsim的页面有了一个初步的了解和认知，下面将介绍仿真环境中，机器人的基础构建过程。

## 2.2 简易移动机器人构建

### 2.2.1 构建简易机器人

### 创建一个基本的双轮机器人

制作机器人的主体：

1）点击 Create > Xform 创建一个新的坐标系，并将其重命名为body，并将它的Translate设置为（0,0,1）。注意：通过菜单 Create 创建的节点默认会挂在世界坐标系（World）根节点下，如果希望它创建在某个特定坐标系（如 wheel_right）之下，可以先在 Stage 树中选中该坐标系再点击 Create，节点会直接绑定到选中的父层级下，避免创建后再手动拖拽和二次矫正坐标参数。

<img src=".\assets\xform.png"/>

2）点击 Create > Shape > Cube 创建一个立方体，点击Cube立方体，将Transform > Scale缩放比例设置为（1,2,0.5），并将Cube拖到body坐标系上，注意此时cube的偏移是（0,0,0）。

<img src=".\assets\cube.png"/>

制作机器人的轮子：

1）点击 Create > Xform 创建一个新的坐标系，并将其重命名为wheel_left，并将它的Translate设置为（1.5,0,1），orient方向设置为（90,0,0）。

<img src=".\assets\left_xform.png"/>

2）点击 Create > Shape > Cylinder 创建圆柱体，orient方向设置为（0,90,0），将他的geometry中的radius半径改为0.5，height高度改为1.0，并将这个圆柱体重命名为wheel_left。

<img src=".\assets\wheel_left.png"/>

3）右键wheel_left的xform，选择duplicate复制一个新的xform，新的xform和圆柱体都改名为wheel_right，注意，下面一定是将xform的translate的x改为-1.5，不要改成圆柱体的了。

<img src=".\assets\wheel_right.png"/>

### 添加物理属性

目前添加的立方体和圆柱体仅仅是视觉图元，没有附加任何物理或碰撞属性。当您按下 play 播放按钮启动模拟并应用重力时，这些物体不会移动，因为它们不受物理定律的影响。

为了让机器人具有物理特性，将其转换为具有碰撞属性的刚体：

1）选中3个物体，右键选择+Add，添加Physics > Rigid Body with Colliders Preset，此时按下播放键，三个物体就会全部落到地面上。

<img src=".\assets\phy.png"/>

<img src=".\assets\fall.png"/>

### 检查碰撞网络

直观地检查物体碰撞网格的轮廓：

1）找到视口顶部的眼睛图标。

2）选择 Show By Type > Physics > Colliders > All。

3）确认所有应用了碰撞检测 API 的物体周围都显示了紫色轮廓线。例如，确认长方体、圆柱体和地面是否都显示了紫色轮廓线。

<img src=".\assets\all.png"/>

### 添加接触和摩擦参数

1）在菜单栏中依次点击 Create > Physics > Physics Material，在弹窗中选择 Rigid Body Material，则会出现一个新的 PhysicsMaterial，他的选项卡中可以修改调整 friction coefficients and restitution 摩擦系数和恢复系数等参数。

2）将指定的物理知识应用于物体，在body的选项卡中选择cube，找到Physics materials on selected models，在下拉框中选择/World/PhysicsMaterial即可。

<img src=".\assets\friction.png"/>

### 添加材料属性

新建的物体可能会反射之前添加的聚光灯的颜色，但实际上它并没有被赋予任何颜色。要更改物体的颜色，需要创建不同的材质，然后将其分配给物体，就像设置物理材质一样。例如，创建两种不同的材质，一种用于车身，一种用于车轮。

1）依次点击 Create > Materials > OmniPBR 两次，因为我们需要创建两个材料，并重命名为body和wheel。

2）将相应的刚体分配给新创建的材质，方法是转到 Property 属性选项卡中的  Materials on selected models 项，然后从下拉列表中选择匹配的材质，请注意，选择后，刚刚下面的Physics materials on selected models也会跟着变动，这个选项重新选择一下/World/PhysicsMaterial即可。

<img src=".\assets\materials.png"/>

3）body和wheel都按上面的流程添加好材质后，可以修改body和wheel的材质来修改颜色，在选项卡的 Material and Shader 中，如果出现groundplane的也变了的情况，修改 groundplane下的collisionmesh 中的 Materials on selected models 即可。

<img src=".\assets\color.png"/>

### 2.2.2 组装简易机器人

通过上面的学习，我们已经对简易机器人的基本结构构建已经很熟练了，下面我们就要开始对以上机器人进行组装，不然他还是散的几个方块圆柱。

此处建议使用Isaacsim中自带的文件，因为他Isaacsim教程构建的机器人结构和他给出的结构不一样，所以我们这里统一采用他库里的结构，跟前面的不一样的只是坐标系位置不一样而已，只是为了保证课程的一致性，大家不用担心，放心学习，后续我们自己的代码也会在每一章前放一个完整的构建好的文件，方便大家选择性跳过学习。

在Isaacsim仿真软件的content部分，依次点击 Isaacsim > Samples > Rigging > MockRobot 将mock_robot_no_joints.usd文件直接拖入环境中即可，其他的除了ground plane之外都可以去除。

<img src=".\assets\mockrobot_old.png"/>

### 添加关节

1）为了便于组织，创建一个用于存储关节的文件夹，点击 Create > Scope，并将其重命名为Joints。

2）要在两个物体之间添加关节，必须先选中它们，选择立方体物体 body ，然后按住 Ctrl ，选择圆柱体物体 wheel_left （一定要先选body，再选wheel_left，他们有一个父节点的关系）。

3）选中两个物体后，右键单击并选择 Create > Physics > Joints > Revolute Joint 。 RevoluteJoint 会出现在 wheel_left 下。将其重命名为 wheel_joint_left 并拖到Joint文件夹下统一管理，并查看一下是否body0是不是/mock_robot_no_joints/body/body，body1是不是/mock_robot_no_joints/wheel_left/wheel_left。

4）将关节​​的 X 轴的 Local Rotation 0 设置为 0.0， Local Rotation 1 设置为 -90.0 ，以考虑实体和圆柱体之间的变换。这是因为圆柱体相对于实体绕 X 轴旋转了 90 度。

5）将关节 Axis 改为 Y 轴。因为机器人没有局部旋转 0 ，所以关节与身体处于同一姿态。

6）对右侧车轮接头重复前2-5步骤。

<img src=".\assets\wheel_joint.png"/>

在添加关节之前，按下播放键后，三个刚体分别落到地面。现在添加了关节，这些刚体就像连在一起一样落下。要观察它们像旋转关节连接一样一起运动，你可以按住 Shift 键，然后在视口中点击并拖动机器人的任意部位。

### 添加联轴器

添加关节即可建立机械连接。要控制和驱动关节，必须添加关节驱动 API。选中wheel_joint_left和wheel_joint_right两个关节，在 Property 属性选项卡中选择+Add > Physics > Angular Drive，同时为两个关节添加驱动。

<img src=".\assets\angular.png"/>

* 位置控制： 对于位置控制关节，设置较高的刚度和相对较低的阻尼或零阻尼。

* 速度控制： 对于速度控制器关节，设置高阻尼和零刚度。

对于轮式关节，速度控制更为合理，因此请将两个轮子的 Damping 阻尼都设置为 1e4 ， Target Velocity 目标速度设置为 200 rad/s 。如果您使用的是活动范围有限的关节，可以在 Property 属性选项卡中的 Raw USD Properties > Lower (Upper) Limit 中进行设置。点击 “播放”按钮 ，即可看到模拟移动机器人运动。

注意，一定要两个轮子都设置。

<img src=".\assets\drive.png"/>

### 添加铰接结构

Joint 是连接两个物体的零件，而 Articulation 是一种将整个机器人作为一个整体系统来计算的高级物理求解方法。

添加方式在这个示例中很简单，了解即可，后续用的机器人都会给一个完整的usd，想要了解的可以参考[Articulations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/articulations.html)。

在这个示例中，先在 Stage 树中选中机器人的顶层 xform（即 mock_robot_no_joints），再在 Property 属性选项卡中选择+Add > Physics > Articulation Root 即可。注意 Articulation Root 应添加在机器人模型的关节节点层级上，且整个模型只能有一个根节点。

> ⚠️ **注意**：请不要在轮子或地面等其他层级上重复添加 Articulation Root，否则播放时会报错「检测到多个关节根节点」。多余的 Articulation Root 无法在界面上手动删除，只能通过修改 USD 文件移除（参见 issue #4）。

<img src=".\assets\articulation.png"/>

### 添加控制器

现在我们就可以用工具来测试机器人的运动了。

1）依次点击 Tools > Robotics > Omnigraph Controllers > Joint Velocity 添加速度控制器，在弹出来的窗口中点击Add添加按钮，选择mock_robot_no_joints，路径选择/Graphs/Velocity_Controller（这个应该是自动选择的）。

<img src=".\assets\velocity.png"/>

<img src=".\assets\add.png"/>

2）将Graphs放到mock_robot_no_joints xform下，右键选择velocity_controller，点击open graph即可打开action graph，最后点击播放，即可开始运动。

<img src=".\assets\motion_end.png"/>

至此，简易移动机器人构建移动就都完成了，恭喜你完成了第一个完整的示例，传感器部分会放到本章最后统一讲解。下面我们将开始第二个比较关键的内容，机械臂。

## 2.3 机械臂构建

对于机械臂的构建，这边省去了机械臂和机械爪的usdf转usd，以及拼装过程，因为我们大部分时候不会涉及这部分内容，且部分内容需要“魔法”工具，因此跳过，我们直接使用Isaacsim创建好的usd机械臂文件来进行教程讲解。

### 2.3.1 组装机械臂和机械爪

这边只展示如何通过固定关节和共享关节直接连接机械臂和机械爪，第二种使用isaacsim的Tools的暂时不提供，因为我们只需要了解组装过程即可，第二种需要下载完整的assets，同时后续的教程会提供组装好的，因此有组装概念就可，因此，你不想组装也可以直接使用下面的ur_gripper.usd文件，这个是组装好的机械臂，下面让我们开始。

机械爪文件：Isaac Sim/Samples/Rigging/Manipulator/import_manipulator/robotiq_2f_140/robotiq_2f_140.usd

机械臂文件：Isaac Sim/Samples/Rigging/Manipulator/import_manipulator/ur10e/ur/ur.usd

组装好的机械臂文件：Isaac Sim/Samples/Rigging/Manipulator/Configure_Manipulator/ur10e/ur/ur_gripper.usd

1）将机械臂文件和机械爪文件拖到仿真中。

2）将 robotiq_2f_140.usd 的prim重命名为ee_link，同时将ee_link的tanslate的数值设置为wrist_3_link的相同数值，从而让他们在视觉上进行装配，并将Orient换为(-90, 0, -90)。

<img src=".\assets\connect.png"/>

3）选择 ee_link/root_joint，进入属性编辑器中的 Physics Articulation Root 部分，移除 Articulation Root，为机械爪保留单一关节即可，并将joints部分的body 0 设置为 wrist_3_link，将机械爪与机械臂连接。

<img src=".\assets\body0.png"/>

4）选择ur的prim，在属性中找到 IsaacRobotAPI，在 isaac：physics：robotjoints 和 isaac：physics：robotLinks 字段中添加 /ur/ee_link，确保 UR10e 包含 2F-140 机械爪。

<img src=".\assets\schema.png"/>

### 2.3.2 生成 Lula 机器人描述文件和碰撞球体

我们需要配置文件提供关于机器人运动学、动力学及其他用于 RMPFlow、CuMotion 和 Lula 运动学求解器属性的信息，因此需要生成这些配置文件来保证求解器的正常运行，如果不想要看这部分也可以跳过，仅做了解即可，最终在机械臂运动实例中用到的配置文件都已经存放在以下地址中。

Lula配置文件：source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/rmpflow/robot_descriptor.yaml

Rmp配置文件：source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/rmpflow/ur10e_rmpflow_common.yaml

### 启用 Isaac Sim Lula 扩展

点击 Window > Extensions，在搜索框中输入 Lula，找到 Isaac Sim Lula 扩展，如果找不到，请从搜索框中移除 @feature 过滤器，通过点击标有 ENABLE 的开关按钮来启用扩展，勾选 ENABLE 右侧的 AUTOLOAD 框。

### 处理 ur_gripper.usd 资产文件

Lula 机器人描述编辑器不支持可实例化网格，因此必须移除可实例化网格。

1）打开ur_gripper.usd文件：Isaac Sim/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper_lula.usd

2）查找选择所有的visuals和collisions，取消勾选instanceable可实例化。

<img src=".\assets\visuals.png"/>

<img src=".\assets\collisions.png"/>

### 在 Lula 机器人描述编辑器中配置关节

1）按下播放键开始模拟，后续需要保证模拟一直开启，不要停止，直到保存文件。

2）依次点击 Tools > Robotics > **Lula Robot Description Editor

3）在选择面板中，选择ur关节，找到 Set Joint Properties，对于每个通用机械臂的关节，将Joint Status关节状态设置为Active Joint活动关节 ，其他设置保持默认，同时保持 Robotiq 2F-140 夹持关节为Fixed Joint固定关节，这样机器人控制器就不会试图移动机械爪以优化机器人位置。

<img src=".\assets\lula_editor.png"/>

### 生成碰撞球体并导出文件

这个首先说明，具体的情况还要具体分析，这里仅作为示例，告诉大家怎么做，具体的生成球体个数和范围，可以后续自行决定。

1）找到  Link Sphere editor，在Selection Panel/Select link中选择链接，首先是upper_arm_link（这个后面还有其他的，都是在这里选，直到铺满整个机械臂），依次选择 Link Sphere editor/Generate Spheres/Select Mesh，选择/collisions/upperarm/mesh碰撞球体所基于的网格。

2）将 Radius Offset 半径偏移设置为 0.03，这是网格半径与碰撞球半径之间的偏移。将 Number of Sphere 球体数量设置为 8，这是要生成碰撞球体数量，确认你在 upper_arm_link 上看到八个红色球体。

3）点击 Generate Spheres 生成球体，球体会变成青色，表示碰撞球体已经生成。然后对其他关节也做重复的步骤。

<img src=".\assets\spheres.png"/>

4）在 Lula Robot Description Editor 机器人描述编辑器中，到最底部，找到 Export To File 导出文件部分。

5）Export to Lula Robot Description File导出为 Lula 机器人描述文件 ，点击文件图标，并指定文件名为 ur10e.yaml，并点击Save保存。

6）你也可以通过选择 Export To File > Export to cuMotion XRDF 并指定文件名为 ur10e.xrdf 来导出 XRDF 文件。

7）导出机器人配置文件后停止模拟。

<img src=".\assets\spheres_end.png"/>

以上就是配置文件设置，后续使用建议直接采用官方提供生成的文件，因为有一些官方有些参数没有给，是我自己设置的，所以尽量使用官方参数，流程理解即可。

### 2.3.3 抓取放置示例

这里提供示例为Isaacsim官方示例供大家入门使用，后续教程中会对具体的机械臂控制规划算法基于实际的机器人进行讲解和代码编写。

### 机械爪控制示例

使用isaacsim中的Parallel Gripper类来控制抓手关节，使用Manipulator类来控制机器人关节。步骤 0 到 400：慢慢关闭抓手。步骤 400 到 800，缓慢打开夹持器，然后将抓手位置重置为 0。

```python
cd isaac-sim
./python.sh standalone_examples/api/isaacsim.robot.manipulators/ur10e/gripper_control.py
```

<img src=".\assets\gripper.png"/>

### Lula 运动学求解器跟踪目标示例

在 2.3.2中讲解了Lula的配置文件生成，此处直接使用source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/rmpflow/robot_descriptor.yaml，官方的Lula配置文件，以保证代码的稳定运行，后续更进一步的内容会在机器人的实际控制中统一讲解。

使用 Lula 运动学求解器创建跟踪目标任务，可以用立方体指定目标位置，机器人会移动其末端执行器到目标位置。

```python
cd isaac-sim
./python.sh standalone_examples/api/isaacsim.robot.manipulators/ur10e/follow_target_example.py
```

其中，ik_solver.py 脚本初始化了 KinematicsSolver 类和 LulaKinematicsSolver 类。

follow_target.py 脚本初始化 FollowTarget 类，并 manipulator 和 parallel_gripper 对象。

最终的 follow_target_example.py 脚本初始化了 FollowTarget 任务和前一步创建的 KinematicsSolver，并设定了立方体的目标位置并执行。

<img src=".\assets\lula_example.png"/>

### RMPFlow 跟踪目标示例

RMPFlow的配置文件：source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/rmpflow/ur10e_rmpflow_common.yaml

创建一个 RMPFlow 控制器，将机器人端执行器移动到目标位置。

```python
cd isaac-sim
./python.sh standalone_examples/api/isaacsim.robot.manipulators/ur10e/follow_target_example_rmpflow.py
```

rmpflow.py 使用上述 ur10e_rmpflow_common.yaml 文件、ur10e.urdf 以及 RMPFlow 配置文件，初始化 Lula 动作生成策略。

follow_target_example_rmpflow.py 脚本初始化了 FollowTarget 任务和上一步创建的 RMPFlowController，并设定了立方体后面的目标位置并执行。

<img src=".\assets\rmp_example.png"/>

### RMPFlow 的基本拣选与放置任务

用 RMPFlow 控制器拿起一个方块并将其放置在目标位置。

```python
cd isaac-sim
./python.sh standalone_examples/api/isaacsim.robot.manipulators/ur10e/pick_up_example.py
```

其中，controllers/pick_place.py 脚本会创建一个 PickPlace 控制器，它会拾取一个方块并将其放置在目标位置。

tasks/pick_place.py 脚本会创建一个选取任务 ，设置 UR10e 的操作器和夹持器，以便拾取方块并将其放置在目标位置。

pick_place_example.py 脚本负责整合所有内容并运行模拟。

<img src=".\assets\rmp_pick.png"/>
