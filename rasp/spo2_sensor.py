
import max30102
import hrcalc
import time
from datetime import datetime
import json
from flask import Flask, jsonify
import matplotlib.pyplot as plt
import numpy as np

id=input("ID를 입력하세요: ")   
Measure_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))

Myspo2_data={}
My_spo2=[]

m=max30102.MAX30102()
count=0

#------------------------------sensor---------------------------

while True:
	red, ir= m.read_sequential()
	if hrcalc.calc_hr_and_spo2(ir,red)!=-999:
		x=hrcalc.calc_hr_and_spo2(ir,red)
		My_spo2.append(round(x,2))
		print("spo2: ",x)
		count+=1
	if count==10:
		break
print(id+Measure_time)
#--------------------------------graph---------------------------
RED = []
with open("./red.log", "r") as f:
	for r in f:
		RED.append(int(r))
fig=plt.figure()
ax=fig.add_subplot(111)

ax.plot(RED,c="blue",label="ppg")
plt.gca().invert_yaxis()
plt.show()

#--------------------------------server-------------------------------
Myspo2_data={
        'username':id,
        'datatime':Measure_time,
        'spo2_data':{
                'spo2':My_spo2
                },
	'wave_data':{
		'ppg':RED
		},
        'avg_spo2':round(sum(My_spo2)/len(My_spo2),2),
        'min_spo2':min(My_spo2),
        'max_spo2':max(My_spo2)
        }

app=Flask(__name__)
@app.route('/app/'+id+Measure_time)
def spo2_api():
	return jsonify(Myspo2_data)
if __name__=='__main__':
        from waitress import serve
        serve(app, host='0.0.0.0',port=8080)
