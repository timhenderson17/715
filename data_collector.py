import time
import csv 

from spidev import SpiDev
 
class MCP3008:
    def __init__(self, bus = 0, device = 0):
        self.bus, self.device = bus, device
        self.spi = SpiDev()
        self.open()
        self.spi.max_speed_hz = 1000000 # 1MHz
 
    def open(self):
        self.spi.open(self.bus, self.device)
        self.spi.max_speed_hz = 1000000 # 1MHz
    
    def read(self, channel = 0):
        adc = self.spi.xfer2([1, (8 + channel) << 4, 0])
        data = ((adc[1] & 3) << 8) + adc[2]
        return data
            
    def close(self):
        self.spi.close()

adc = MCP3008()
    
# field names 
fields = ['time', 'value'] 
    
# name of csv file 
filename = "data_collection.csv"

while True:
    value = adc.read( channel = 0 ) # You can of course adapt the channel to be read out
    now = time.localtime()
    now_str = time.strftime("%b %d %Y %H:%M:%S", now)
    mydict =[{'time': now_str, 'value': str(value)}] 
    #print("Applied voltage: " + str((value / 1023.0 * 3.3)) + " at " + now_str)
    
    with open(filename, 'a') as csvfile: 
         #creating a csv dict writer object 
         writer = csv.DictWriter(csvfile, fieldnames = fields) 
            
         #writing data rows 
         writer.writerows(mydict)
    
    # wait a minute
    time.sleep(6)
