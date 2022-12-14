import os
import csv
import time
import json
print("Starting sequential.js")
if not os.path.exists("./generated"):
    os.mkdir("generated")
start_time=time.time()
return_code=os.system("node sequential.js")
end_time=time.time()
if return_code==0:
    print("Sequential.js finished")
    print("Sequential: ", round(end_time-start_time,3),"s")
    print("Image Making: ",time.strftime("%Hh %Mm %SS",time.gmtime(round(end_time-start_time))))
    print("Starting make_video.py")
    start_time_1=time.time()
    return_code_py=os.system("python3 make_video.py")
    end_time_1=time.time()
    if return_code_py==0:
        print("Make_video.py finished")
        print("Make Video: ", round(end_time_1-start_time_1,3),"s")
        print(time.strftime("%Hh %Mm %Ss",time.gmtime(round(end_time_1-start_time_1))))
        print("Starting benchmark writing")
        csv_file=open("benchmarks.csv", "r", newline="")
        json_file=open("settings.json", "r")
        json_data=json.load(json_file)
        csv_reader=csv.reader(csv_file)
        timing=end_time-start_time
        timing/=60
        timing1=end_time_1-start_time_1
        csv_object=[ row for row in csv_reader ]
        csv_file.close()
        json_file.close()
        csv_object.append([str(json_data['iterations']), str(round(timing,3)), str(round(timing1,3)),str(round(timing1+timing,3))])
        csv_header=csv_object[0]
        csv_object=csv_object[1:]
        csv_object.sort(key=lambda x: float(x[0]))
        csv_object.insert(0, csv_header)
        with open('benchmarks.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_object)
            csvfile.close()
            print("Done")
        print("Benchmark writing finished")
        print("Finished")
