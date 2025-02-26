# Project Title

## Abstract
Indoor robot navigation demands precise perception, effective decision-making, and robust adaptability to operate in complex and dynamic environments. Learning-based approaches learn directly from sensor data and environmental interactions, enabling robots to develop navigation strategies that are both flexible and efficient.

This work presents a learning-based framework for indoor navigation that combines imitation learning (IL) for pre-training and reinforcement learning (RL). Models are trained on expert demonstrations collected from a joystick-controlled robot. To optimize the navigation, multi-modal inputs are explored by training and evaluating models using RGB images, laser scans, and their combination. The most successful IL model is then used to initialize the RL policy, providing a strong foundation for further learning.

Evaluations were conducted on a mobile robot equipped with a 2D LiDAR and camera in real indoor environments measuring success rate, path efficiency, and collision rate. With the laser-based IL model, the robot showed an overall 70% success rate in various static and dynamic scenarios. The multi-modal IL model combining laser and image data showed an 86.7% success rate in static environments, including complex scenarios. However, real-world RL training proved challenging, as small actor network updates led to significant behavioral shifts, underscoring the need for further research on effectively leveraging pre-trained models for RL. The overall results demonstrate the practicality and potential of learning-based methods for indoor navigation.

## Usage Instructions

### Prerequisites
- ROS2 Humble
- Install the requirements.txt
- ROS2 bag files of expert data

### Data Preparation
1. Run `EDA/robile_data_prep.ipynb` to extract data from bag files, process it, and temporally align. The processed data will be saved in pickle files.
2. Run `Imitation_Learning/Dataset_Preparation.ipynb` to create TensorFlow datasets from the pickle files. The dataset created will be saved as tfrecord.

### Imitation Learning
Run the following files to train models that use Laser, Image, and combined modalities. Use the configurations in the `yaml_files` folder to reproduce the results.
1. `Imitation_Learning/ImitationLearning_LaserGoal.py`
2. `Imitation_Learning/ImitationLearning_ImgGoal.py`
3. `Imitation_Learning/ImitationLearning_LaserImgGoal.py`

The models will be saved at regular intervals during training.

### Critic Pre-training
1. Run `Reinforcement_Learning/Critic_Pretraining/Dataset_Preparation_Critic.ipynb` to create a dataset for critic pre-training using the pickle files created earlier.
2. Run `Reinforcement_Learning/Critic_Pretraining/critic_pretraining.py` to pre-train the critic with the created dataset.

### Reinforcement Learning
Run `Reinforcement_Learning/ddpg_laser_nav.py` to train the agent in real-time using reinforcement learning. The best IL model trained earlier is used to initialize the actor, and the pre-trained critic model is used to initiate the critic.

### Evaluation - Imitation Learning
1. Run the robot and SLAM toolbox as described in [this tutorial](https://robile-amr.readthedocs.io/en/latest/source/Tutorial/Demo%20Mapping.html).
2. Record bag files on `robot_path`, `goal_pose`, and `map` topics for future analysis.
3. Run one of the following files based on the IL model you are going to evaluate:
    1. `Imitation_Learning/real_world_implemetation/nav2goal.py`
    2. `Imitation_Learning/real_world_implemetation/nav2goal_LaserImg.py`
    3. `Imitation_Learning/real_world_implemetation/nav2goal_withImgs.py`
4. Open Rviz2 and give the goal position.
5. Repeat the experiment several times for all three IL models.
6. Run `Evaluation/potential_field_planner.py` to get results for the same experiment scenarios using the traditional potential field method for comparison.
7. After collecting all bag files, run `Evaluation/Experiment_evaluation.ipynb` to get metrics and visualize the performance of all the models.
8. Run `Evaluation/IL_with_global_planner.py` to test end-to-end navigation of the robot with the laser-based IL model integrated with A*.



