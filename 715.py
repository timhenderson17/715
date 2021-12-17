import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from datetime import datetime
from datetime import timedelta
from matplotlib.gridspec import GridSpec
import calendar
import copy
import random
from sklearn import svm

pd.set_option('display.max_rows', 100)

pickling = 0    # 1 to read the original CSV's and create pickles
                # 0 to just read the pickles (faster)
hazard = 0      # 1 to compute most hazardous locations
                # 0 to do nothing
plot = 1        # 1 to show plots / dashboard
                # 1 to not show any plots
knn = 0         # 
                # 0 to do nothing
svm_run = 0     #
                # 0 to do nothing



# Pickling
   
if pickling :
    df_bl = pd.read_csv('./back left facing teacher (small).csv')
    df_br = pd.read_csv('./back right facing teacher (caution tape).csv')
    df_fl = pd.read_csv('./front left facing teacher (close to entrance).csv')
    df_fr = pd.read_csv('./front right facing teacher (by exit).csv')

    pd.to_datetime(df_bl['Time'])
    pd.to_datetime(df_br['Time'])
    pd.to_datetime(df_fl['Time'])
    pd.to_datetime(df_fr['Time'])

    # so all data starts at exact same time
    # 7:48:33 PM Nov 15
    df_fl = df_fl.iloc[61: , :]
    df_fr = df_fr.iloc[39: , :]
    df_bl = df_bl.iloc[14: , :]

    # so all data ends at exact same time
    # 12:15:53 PM
    df_fl = df_fl.iloc[:110550 , :].reset_index(drop=True)
    df_fr = df_fr.iloc[:110550 , :].reset_index(drop=True)
    df_bl = df_bl.iloc[:110550 , :].reset_index(drop=True)
    df_br = df_bl.iloc[:110550 , :].reset_index(drop=True)

    def getPPM(row):
        input_val = row['Value']
        input_val = np.clip(input_val, 0, .4)
        resistance = 100000 / input_val - 30000
        Rs_Ro = resistance / 575000
        ppm = -987.137 * math.log(Rs_Ro) + 17.347
        return max(0,ppm)

    df_fl['PPM'] = df_fl.apply(getPPM, axis=1)
    df_fr['PPM'] = df_fr.apply(getPPM, axis=1)
    df_bl['PPM'] = df_bl.apply(getPPM, axis=1)
    df_br['PPM'] = df_br.apply(getPPM, axis=1)

    df_bl.to_pickle("./df_bl.pkl")
    df_br.to_pickle("./df_br.pkl")
    df_fl.to_pickle("./df_fl.pkl")
    df_fr.to_pickle("./df_fr.pkl")
else :
    df_bl = pd.read_pickle("./df_bl.pkl")
    df_br = pd.read_pickle("./df_br.pkl")
    df_fl = pd.read_pickle("./df_fl.pkl")
    df_fr = pd.read_pickle("./df_fr.pkl")

# Most Hazardous Location
if hazard :
    df_hazard = df_bl.copy()
    df_hazard['Value'].values[:] = 0.0

    def get_peak(start, end) :
        peak_bl = df_bl.iloc[start:end][['PPM']].idxmax()
        peak_br = df_br.iloc[start:end][['PPM']].idxmax()
        peak_fl = df_fl.iloc[start:end][['PPM']].idxmax()
        peak_fr = df_fr.iloc[start:end][['PPM']].idxmax()
        return [peak_bl, peak_br, peak_fl, peak_fr]

    def cost(x, y, distances) :
        points = [[0,0,80,0],[0,0,0,60],[0,0,80,60],[80,0,0,60],[80,0,80,60],[0,60,80,60]]
        sum = 0
        for i in range(len(points)) :
            a_side = math.sqrt(math.pow(x-points[i][0],2)+math.pow(y-points[i][1],2))
            b_side = math.sqrt(math.pow(x-points[i][2],2)+math.pow(y-points[i][3],2)) - distances[i]
            dif = math.pow(a_side - b_side,2)
            sum += dif
        return sum

    def gradient_descent(x, y, distances, iterations = 4949, stopping_threshold = 1e-6):
        current_x = x
        current_y = y
        iterations = iterations

        costs = []
        best_cost = None
        best_x = 40
        best_y = 30

        for i in range(iterations):

            # Calculationg the current cost
            current_cost = cost(current_x, current_y, distances)

            # If the change in cost is less than or equal to 
            # stopping_threshold we stop the gradient descent
            if best_cost and abs(best_cost)<=stopping_threshold:
                break

            if best_cost :
                if current_cost < best_cost :
                    best_cost = current_cost
                    best_x = current_x
                    best_y = current_y
            else :
                best_cost = current_cost

            costs.append(current_cost)

            # Printing the parameters for each 1000th iteration
            # print(f"Iteration {i+1}: Cost {current_cost}, X: {current_x}, Y: {current_y}")

            # Updating x and y
            current_y = (current_y + 1) % 61 if (current_x == 80) else current_y
            current_x = (current_x + 1) % 81

        return best_x, best_y


    T = [0,0,0,0,0,0]
    for index, row in df_hazard.iterrows():
        if (index > 1000 and index % 50 == 0) :
            peaks = get_peak(index - 50, index)

            k = 0
            for i in range(1, len(peaks) - 1) :
                for j in range(i+1, len(peaks)) :
                    T[k] = peaks[i] - peaks[j]
                    T[k] = T[k] * 0.07874
                    k += 1

            x, y = gradient_descent(0, 0, T)
            print(str(index) + " " + "Most Hazardous Spot Best Estimate: " + str(x) + "," + str(y))

            df_hazard.at[index, 'x'] = x
            df_hazard.at[index, 'y'] = y

    df_hazard.to_pickle("./df_hazard_50.pkl")

# Dashboard

if plot :
    fig = plt.figure(figsize=(40, 20))
    gs = GridSpec(nrows=4, ncols=5)

    fig.suptitle('PPM at All 4 Sensors')

    ax0 = fig.add_subplot(gs[0, 0])
    ax_bl = ax0.plot(df_bl['Time'], df_bl['PPM'])
    ax0.set_xlabel('Time')
    ax0.set_ylabel('CO2 PPM')
    ax0.set_title('Far Left from Front')

    ax1 = fig.add_subplot(gs[0, 1])
    ax_br = ax1.plot(df_br['Time'], df_br['PPM'])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('CO2 PPM')
    ax1.set_title('Far Right from Front')

    ax2 = fig.add_subplot(gs[1, 0])
    ax_fl = ax2.plot(df_fl['Time'], df_fl['PPM'])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('CO2 PPM')
    ax2.set_title('Front Left from Front')

    ax3 = fig.add_subplot(gs[1, 1])
    ax_fr = ax3.plot(df_fr['Time'], df_fr['PPM'])
    ax3.set_xlabel('Time')
    ax3.set_ylabel('CO2 PPM')
    ax3.set_title('Front Right from Front')

    ax0.set_ylim(0,1000)
    ax1.set_ylim(0,1000)
    ax2.set_ylim(0,1000)
    ax3.set_ylim(0,1000)

    # hazard plotting
    df_hazard = pd.read_pickle("./df_hazard_50.pkl")
    ax4 = fig.add_subplot(gs[:2, 2:4])

    def update(i) :
        time_index = datetime(2021, 11, 15, 19, 48, 33)
        ax4.clear()
        index = i*50 + 1200
        time_index += timedelta(seconds=int(index*6))
        ax4.set_xlim(0, 80)
        ax4.set_ylim(0, 60)
        ax4.grid()
        ax4.set_title(str(time_index))
        ax4.plot(df_hazard.at[index, 'x'], df_hazard.at[index, 'y'], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")

    anim = FuncAnimation(fig, update, frames=np.arange(0, 40), interval=200)
    anim.save('hazard.gif', dpi=80, writer='imagemagick')

    # plt.show()
    #plt.savefig('foo.png')


# Beginning of ML / Verification 
attendance = []
for section in open('Month Day Year Start End Number_Of_Students.txt').read().split('\n'):
    attendance.append(section.split())
class_sizes = []
classtimes = []
abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
for lecture in attendance:
    month, day, year, start, end, students = lecture
    start = start.split(':')
    end = end.split(":")
    begin = datetime(int(year), abbr_to_num[month], int(day), int(start[0]), int(start[1]), 0)
    finish = datetime(int(year), abbr_to_num[month], int(day), int(end[0]), int(end[1]), 0)
    classtimes.append([begin, finish])
    class_sizes.append(int(students))

# 7:48:33 PM Nov 15
# datetime(year, month, day, hour, minute, second, microsecond)
zeroTime = datetime(2021, 11, 15, 19, 48, 33)
def getIndex(start, end):
    diff = end - start
    return int(diff.total_seconds()/6)

def getPeakAndAverageOfPeriod(startTime, endTime, zeroTime):
    #bl, br, fl, fr
    lower = getIndex(zeroTime, startTime)
    upper = getIndex(zeroTime, endTime)
    peak = [0.0,0.0,0.0,0.0]
    average = [0.0,0.0,0.0,0.0]
    bl = df_bl.values
    br = df_br.values
    fl = df_fl.values
    fr = df_fr.values
    for i in range(lower,upper+1):
        average[0] += bl[i][2]
        average[1] += br[i][2]
        average[2] += fl[i][2]
        average[3] += fr[i][2]
        peak[0] = max(peak[0], bl[i][2])
        peak[1] = max(peak[1], br[i][2])
        peak[2] = max(peak[2], fl[i][2])
        peak[3] = max(peak[3], fr[i][2])
    for i in range(len(average)):
        average[i] = average[i]/(upper-lower+1)
        
    return peak, average

#class_sizes, classtimes, get firstMinutePPM in first minute and average, peak across class times
#feature 4 CO2 level and number_of_students -> total of 5
# 4 CO2 levels -> those that are 1st minute of class time
# number_of_students -> assuming full attendence for 1 minute later
# good/bad -> decided by 4 CO2 levels average of the class time
# kNN nearest neighbor and SVM -> anticipate the rise of CO2 levels
firstMinutePPM = []
averages = []
peaks = []
for classtime in classtimes:
    start = classtime[0]
    end = classtime[1]
    currP, currAv = getPeakAndAverageOfPeriod(start, end, zeroTime)
    averages.append(currAv)
    peaks.append(currP)
    _, firstAv = getPeakAndAverageOfPeriod(start, end+timedelta(minutes=1), zeroTime)
    firstMinutePPM.append(firstAv)
# class size, firstMinutePPM, true/false danger (based on averages/peaks)
# positive dangerous readings are recorded
allSamples = []
dangerous = []
for i in range(0, len(class_sizes)):
    isDangerous = max(peaks[i]) >= 600  
    dangerous.append(isDangerous)
    parameters = []
    parameters.extend([class_sizes[i]])
    parameters.extend(firstMinutePPM[i])
    allSamples.append(parameters)

if knn:
    print("Of N samples, each individual sample is tested with all other N-1 samples being treated as neighbors. (Leave One Out)")
    print("For each different value of k, N tests are run and accuracy = # of correct/N")
    print("")
    def knn_distance(sample1, sample2):
        distance = 0.0
        for i in range(len(sample1)):
            distance += (sample1[i]-sample2[i])**2
        return distance**0.5
    # running kNN with different values of k   
    for numOfNeighbors in range(1,int((len(allSamples))/2), 2):
        accuracy = 0.0
        for i in range(0,len(allSamples)):
            sample = allSamples[i]
            closest = [(float("inf"), False) for _ in range(0,numOfNeighbors)]
            for j in range(0, len(allSamples)):
                neighbor = allSamples[j]
                if sample != neighbor:
                    dist = knn_distance(sample,neighbor)
                    if closest[-1][0] > dist:
                            closest[-1] = (dist, dangerous[j])
                closest.sort(key=lambda y: y[0])
            estimate = 0
            for close in closest:
                if close[1] == True:
                    estimate += 1
                else:
                    estimate -= 1
            if (dangerous[i]) == (estimate > 0):
                accuracy += 1.0
        accuracy = accuracy/(len(allSamples))
        print("k value in kNN:",numOfNeighbors,"Accuracy:",accuracy)
    print("")
        
if svm_run:
    print("Of N samples, each individual sample is tested with all other N-1 samples being used to train an SVM classifier. (Leave One Out)")
    print("For each different kernel, N tests are run and accuracy = # of correct/N")
    print("")
    accuracy = 0.0
    for kernelType in [("linear", 1), ("poly", 2), ("poly", 3), ("poly", 4), ("poly", 5), ('rbf', 0)]:
        clf = svm.SVC(kernel=kernelType[0], degree=kernelType[1])
        for i in range(0, len(allSamples)):
            trainData = copy.deepcopy(allSamples)
            trainLabel = copy.deepcopy(dangerous)
            testSample = trainData.pop(i)
            testLabel = trainLabel.pop(i)
            clf.fit(trainData, trainLabel)  
            accuracy += (testLabel == clf.predict([testSample])[0])
        accuracy = accuracy/len(allSamples)
        print("SVM Kernel Type:",kernelType[0],"Degree (if relevant):",kernelType[1],"Accuracy:",accuracy)
    print("")
