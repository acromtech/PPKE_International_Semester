# Introduction to Neuromorph Movement Control

## General Model of Motor Control

1. **Motor Task**: The intended movement or action that the body aims to perform.
2. **Sensing**: Gathering information about the environment and the state of the body through sensory inputs.
3. **Movement Planning**: The premotor cortex plans the specific details of the movement.
4. **Movement Command**: The motor cortex issues commands to initiate and control the movement.
5. **Execution**: Spinal motoneurons activate muscles and joints to carry out the movement.
6. **Feedback Loop**: The system continuously loops back to sensing to adjust and refine the movement based on new sensory information.

## Levels of Motor Control and Execution

1. **Central Nervous System (CNS)**: The CNS generates electric signals that initiate movement.
2. **Muscle**: Muscles respond to electric signals by contracting.
3. **Joint**: Joint movement results in rotation.
4. **Limb**: The coordinated movement of muscles and joints results in limb displacement.

## Relationship Between Modeling and Experimentation

1. **Mathematical Model**: Creating equations and models to describe the motor control system.
2. **Computer Simulation**: Using software like MATLAB or Python to simulate the mathematical model.
3. **Comparison**: Comparing simulation results with experimental data to check accuracy.
4. **Model Adjustment**: Adjusting the model based on the comparison to improve its accuracy.

### Experimental Protocol

1. **Planning**: Defining the objectives, participants, and methods of the experiment.
2. **Data Measurement**: Emphasizing the importance of specifying space, time, and sampling rate.
3. **Movement Analysis**: Analyzing the collected data to understand movement patterns.
4. **Comparison and Adjustment**: Comparing experimental data with model predictions and refining the model accordingly.

## Overview of Key Concepts

- **Motor Task Definition**: Understanding what a motor task is and its significance.
- **Modeling and Experimentation**: How to develop models and test them against real-world data.
- **Elementary Definitions**: Basic terms such as kinematics (study of motion) and electromyography (EMG, study of muscle electrical activity).
- **Data Acquisition Methods**: Techniques for capturing 3D joint coordinates using optical systems (e.g., Vicon) and ultrasound systems (e.g., Zebris).
- **Case Studies**: Examples of arm and leg movements to illustrate concepts.

## Execution and Redundancy

- **Movement Patterns**: Various ways to execute a motor task.
- **Redundancy Problem**: Multiple ways to achieve the same movement, creating redundancy.
- **Sensory-Motor Transformations**: How sensory information is converted into motor actions.
- **Extrinsic and Intrinsic Geometry**: Different coordinate systems used to represent movement.

## Examples of Movements

- **Upper Extremity Movements**:
  - **Pointing Movements**: Reaching out to point at an object.
  - **Tracking Movements**: Moving the index finger along a trajectory.
  - **Grasping**: Grabbing an object.
- **Lower Extremity Movements**:
  - **Walking**: The biomechanics of walking.
  - **Cycling**: The motion involved in cycling.
- **Other Movements**:
  - **Eye Movements**: Movement of the eyes, considered as a "one joint system".
  - **Head-Neck Movements**: Coordinated movements of the head and neck.

## Controlled Parameters and Measurements

- **Physical Perspective**: Dynamics (forces causing motion) and kinematics (motion itself).
- **Biomechanical Perspective**: Neural firing frequencies, muscle activity patterns (EMG), and joint rotations (torques).
- **Measurement Techniques**: Using EMG to record electrical activity of muscles with invasive and non-invasive electrodes.

## Kinematics and EMG Analysis

- **Kinematics**: Study of the geometric properties of motion over time.
- **EMG**: Recording and analyzing the electrical activity of muscles.
- **Measurement Techniques**: Using markers placed on joints and employing movement analyzers.

## Model Improvement

- **Data Processing**: Using mathematical and physical algorithms in tools like MATLAB, Python, or Excel.
- **Algorithms**: Calculating parameters like inertia, torque, gravity, and muscle force to make the model more realistic.
- **Error Correction and Normalization**: Ensuring accuracy and consistency in the data.

## Movement Analysis Software

- **Data Storage and Post-Processing**: Storing measured data in text and Excel files for further analysis.
- **Algorithm Development**: Creating and refining algorithms in MATLAB and Python to analyze the data.

## Functional Electrical Stimulation (FES)

- **FES Driven Cycling**: Application of FES in rehabilitation, specifically for cycling in paraplegic patients.

## Sensory-Motor Transformations

- **Sensation and Execution**: Converting sensory inputs (e.g., vision, proprioception) into motor outputs (e.g., muscle activation).
- **Extrinsic and Intrinsic Geometry**: Representing movements using external (e.g., Cartesian coordinates) and internal (e.g., joint angles) systems.

## Tensor Network Theory

- **Coordinate Systems**: Understanding contravariant (interdependent) and covariant (independent) coordinates.
- **Human Arm Movement**: Using intrinsic coordinate systems where each joint defines an axis, and understanding how rotations at the shoulder, elbow, and wrist move the finger.

---

This text provides a clear, concise summary of the first lecture, covering the main points and concepts in an easy-to-understand manner.