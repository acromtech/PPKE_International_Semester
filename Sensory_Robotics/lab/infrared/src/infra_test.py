#!/usr/bin/env python

from __future__ import print_function
import serial.tools.list_ports as port_list
#from serial import Serial
import time
import numpy as np
from Marci_PyPowerCube import cube_comm
from sp_c import sp_c

# This code realizes the measurement of the infrared LED+photodiode.
# It contains 4 parts:
# 1) initialization of the xyz-table,
# 2) moving the final module to starting point of the measurement,
# 3) iterative measurement  <--- please complete this part only,
# 4) reseting the position of the xyz-table along axis z.
# 
# This function should save an array and a variable to a file, which 
# can be later loaded and plotted, these variables are:
#   distance___signal_level___pairs: two-dimensional array, please store in
#                                    this array the return values of the microcontroller;
#   max_distance: the maximal measurement distance of the reflective surface
#                 from the board

print('infra distance measurement with the xyz-table has started')


#######################################################################
# PART 1: initialization of the xyz-table

s_plotter = None    # serial device name of the plotter
plotter = None      # ref. to the plotter object
s_uC = None         # serial device name of the microcontroller
uC = None           # ref. to the serial object of the uC

# check the port manufacturer name!
# select the right serial port to the plotter:
#   - 1m-long cable manufacturer contains 'ftdi'
#   - 40cm-long cable manufacturer contains 'Prolific'
available_ports = list(port_list.comports())
for p in available_ports:
    if 'Prolific' in p.manufacturer: #   <----- TODO please update this if necessary
        s_plotter = p.device
        print('FOUND: plotter (manufacturer: '+str(p.manufacturer)+')')
    elif 'FTDI' in p.manufacturer: # <- TODO please update this if necessary
        s_uC = p.device
        print('FOUND: uC (manufacturer: '+str(p.manufacturer)+')')
    print(available_ports)
    print(p.manufacturer)

if s_plotter == None or s_uC == None:
    print('the serial device belonging to the plotter or to the uC has not found, terminating')
    exit()
else:
    plotter = cube_comm(s_plotter, 57600)
    uC = sp_c(s_uC)

# home-operation along axis z:
plotter.cube_send(3,['\x01'],rw = 1)
asdf = input('If the home-operation of axis z has finished, please type a key and ENTER: ')
# home-operation of x and y:
plotter.cube_send(1,['\x01'],rw = 1)
plotter.cube_send(2,['\x01'],rw = 1)
asdf = input('If the home-operations of axis x and y have finished, please type 0 and ENTER: ')


#######################################################################
# PART 2: moving the final module to the starting point of the measurement
# the positions are measured in meter; axis x has negative sign
# a good starting point: -1.0, 0.5, 0.0
# a good max-depth of axis z: 0.27
max_distance = 0.27;
desired_position = [-1.0, 0.5, 0.0];

plotter.x_move_to(desired_position[0], 0.05, 0.05)
plotter.wait_until_pos_reached('x', desired_position[0])

plotter.y_move_to(desired_position[1], 0.05, 0.05)
plotter.wait_until_pos_reached('y', desired_position[1])

plotter.z_move_to(desired_position[2], 0.05, 0.05)
plotter.wait_until_pos_reached('z', desired_position[2])


#######################################################################
# PART 3: iterative measurement  <--- please complete this part only
# You should have ready your serial communicator function, which
# realizes the communication protocoll of one measurement:
#   - PC --> PIC: 20, 21
#   - PIC --> PC: 22, 23
#   - PIC --> PC: ADC(AN1), ADC(AN2)
#
# You have to do:
# Write a for-loop, where in every iteration:
#    - call your measurement function
#    - store its return value in the array called distance___signal_level___pairs
#    - shrink 1 millimeter with module z: for this purpose please use the following two lines:
#                                         plotter.z_move_to(<where_in_absolute coordinate>, 0.01, 0.01)
#                                         plotter.wait_until_pos_reached('z', <where_in_absolute coordinate>, epsilon=0.004)
# The condition to terminate the loop: reaching max_distance with
# module z.

distance_signal_level_pairs = []  # Initialize an empty array to store measurement results

for distance in np.arange(0, max_distance, 0.001):
    success, adc1, adc2 = uC.measure()
    
    if not success:
        print("Measurement failed at distance:", distance)
        break  # Exit the loop if the measurement fails
    
    # Store the measurement result in the array
    distance_signal_level_pairs.append([distance, adc1, adc2])
    
    # Shrink 1 millimeter with module z
    plotter.z_move_to(distance, 0.01, 0.01)
    plotter.wait_until_pos_reached('z', distance, epsilon=0.004)

# Now you have the array distance_signal_level_pairs containing the measurement results
# Save the array and the max_distance variable to a file, if needed
np.save('./numpy_array_of_infra_measurement.npy', (distance_signal_level_pairs, max_distance))


#######################################################################
# PART 4: send cube home along axis z:
plotter.cube_send(3,['\x01'],rw = 1)
asdf = input('If the home-operation of axis z has finished, please type a key and ENTER: ')

uC.close()

print('infra distance measurement with the xyz-table has finished')


