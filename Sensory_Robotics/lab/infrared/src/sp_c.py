from __future__ import print_function
import serial
import serial.tools.list_ports as port_list
from time import sleep
import struct
from types import *

class sp_c :
    def __init__(self, s_port):
        self.serial_obj = serial.Serial(port=s_port, baudrate=38400, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, timeout=5) #timeout=0
        if self.serial_obj.isOpen():
            print('successfully opened serial port to uC')
        else:
            print('opening port failed, terminating')
            exit()
    
    converter = lambda self, c: ord(struct.pack(">c", c))
    
    def measure(self):
        # PC  --> PIC: 20, 21
        # PIC --> PC : 22, 23, adc1, adc2
        data = struct.unpack(">BB", chr(20)+chr(21))
        self.serial_obj.write(data)
        sleep(0.01)
        buff = self.serial_obj.read(size=4)
        if len(buff) != 4:
            return False, None, None
        if self.converter(buff[0]) != 22 or self.converter(buff[1]) != 23:
            return False, None, None
        
        return True, self.converter(buff[2]), self.converter(buff[3])
    
    def close(self):
        self.serial_obj.close()




