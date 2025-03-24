## Simulation Setup

- Open Terminal

    ``` 
    roslaunch kortex_gazebo spawn_kortex_robot.launch
    ```

- In another terminal

    - For arm movement with goal end-effector cartesian pose

    ```
    roslaunch kortex_examples cartesian_poses_with_notifications_python.launch
    ```

    - For arm movement with end-effector trajectory pose to goal (Change the path to CSV containing end-effector trajectory poses)
    
    ```
    roslaunch kortex_examples cartesian_csv.launch
    ```

