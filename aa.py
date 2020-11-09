
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from scipy.signal import savgol_filter

################path:엑셀파일 저장된 경로 , save_path: 그래프저장할경로, 그래프로 표시할 라인갯수#################################
path = 'F:/2020-10-27 세종대/rawdata/'
save_path = 'F:/2020-10-27 세종대/rawdata_con/'
line_num = 16
########################################################################################################################
filter_data = []

y = ['y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12', 'y13', 'y14', 'y15'][0:line_num]
file_list = os.listdir(path)
if not file_list:
    print('경로확인, 경로 지정안되었습니다.')
    sys.exit()
for file_name in file_list:
    if file_name.endswith(".csv"):
        df = pd.read_csv(os.path.join(path, file_name), sep='\t', header=None)  # encoding='CP949', sep= ''
        data = np.array(df)

        if line_num+2 == data.shape[1]:
            pass
        else:
            print('line 갯수를 확인해야합니다.\n 시스템을 종료합니다')
            sys.exit()

        load = data[:, len(y):len(y)+1]
        dis = data[:, len(y)+1:len(y)+2]

        num = np.shape(data)[0]
        x = np.arange(num)

        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        y[0] = data[:, 0:0 + 1]
        y[0] = savgol_filter(y[0].flatten(), 301, 1)
        data_con = y[0]
        for i in range(1, len(y)):
            y[i] = data[:, i:i + 1]
            y[i] = savgol_filter(y[i].flatten(),301,1)

            data_con =np.vstack((data_con, y[i]))

            # ax1.plot(x, y[i], label=i + 1, linewidth=0.1)
        data_con = np.vstack((data_con, load.flatten(), dis.flatten())).transpose()
        np.savetxt(save_path+file_name.split('.')[0]+'_'+str('filtering')+'.'+file_name.split('.')[1], data_con, delimiter='\t', fmt='%4e')



        # ax2.plot(x, load)
        # ax2.plot(x, dis)
        #
        # ##그래프 설정 파트 #################################################
        # ax1.set_ylabel('Sample Voltage') #y1축 이름
        # ax2.set_ylabel('Load - Displacement') #y2축 이름
        #
        # ax1.set_xlabel('Time')    # x 축 이름
        # ax1.set_ylim(6.8, 6.975)        #y축 스케일조절
        # ax2.set_ylim(-10, 0)      #y2축 스케일조절
        # ax1.set_xlim(2000,3000)  # x축 스케일 조절
        #
        # plt.savefig(save_path + '/' + file_name + '.png', dpi=300)
        # print(file_name, "완료!")
        # plt.close(fig)
print('최종 끝')
