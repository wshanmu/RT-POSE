import mat73
import numpy as np
import math
import re
import os
from tqdm import tqdm
import argparse

#for plot
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


path_folder_tail = 'radar/mat/'
path_folder_cam_tail = 'camera/left/'
path_save_dzyx_c_npy_tail =  'radar/npy_DZYX_complex/'

def parse_args():
    parser = argparse.ArgumentParser(description="4D radar tensor trans to cartesian coordinate")
    parser.add_argument("--dataset_dir", help="the file path of dataset")
    parser.add_argument(
        "--sequence",
        type=int,
        nargs='*', 
        # required=True,        
        help="select sequence to trans to cartesian coordinate, if no select will trans all sequneces",
    )
    args = parser.parse_args()

    return args


# cart to polar
def c2p(x,y,z): #array[x][y][z]

    #input  x,y,z (m)
    #output r,a,e (rad)

    r = math.sqrt(x**2+y**2+z**2)

    if (y==0.0):
        a =  math.pi/2.
    else:
        a = math.atan(x/y)

    if(a<0) :
        a+=math.pi
    

    if(x**2+y**2==0):
        e = math.pi/2.
    else:
        e = math.atan(z/math.sqrt(x**2+y**2))

    return r,a,e

    # polar to cart
def p2c(r,a,e):

    #input  r,a,e (rad)
    #output x,y,z (m)

    x = r*math.cos(a)*math.cos(e)
    y = r*math.sin(a)*math.cos(e)
    z = r*math.sin(e)

    return x,y,z

    # find index for blinear interpolation

def findIndexForBiInt( value, array): #value(rad)
    i_0 = -1
    i_1 = -1
    for i in range(len(array)-1):
        
        if(value<array[i+1]):
            i_0=i
            i_1=i+1
            break
        
    return i_0-1,i_1-1

def getBiIntValue(rea,val_rea,arr_bi_index,arrRange,arrElevation,arrAzimuth):

    r = complex(val_rea[0])
    e = complex(val_rea[1])
    a = complex(val_rea[2])

    i_r0 = arr_bi_index[0][0]
    i_r1 = arr_bi_index[0][1]
    i_e0 = arr_bi_index[1][0]
    i_e1 = arr_bi_index[1][1]
    i_a0 = arr_bi_index[2][0]
    i_a1 = arr_bi_index[2][1]

    temp_rea000 = np.reshape(rea[:,i_r0,i_e0,i_a0],(len(rea),1))
    temp_rea001 = np.reshape(rea[:,i_r0,i_e0,i_a1],(len(rea),1))
    temp_rea010 = np.reshape(rea[:,i_r0,i_e1,i_a0],(len(rea),1))
    temp_rea011 = np.reshape(rea[:,i_r0,i_e1,i_a1],(len(rea),1))
    temp_rea100 = np.reshape(rea[:,i_r1,i_e0,i_a0],(len(rea),1))
    temp_rea101 = np.reshape(rea[:,i_r1,i_e0,i_a1],(len(rea),1))
    temp_rea110 = np.reshape(rea[:,i_r1,i_e1,i_a0],(len(rea),1))
    temp_rea111 = np.reshape(rea[:,i_r1,i_e1,i_a1],(len(rea),1))        
    # print(temp_rea000.shape)
    vector = np.vectorize(np.complex_)
    rea000 =  vector(temp_rea000)
    rea001 =  vector(temp_rea001)
    rea010 =  vector(temp_rea010)
    rea011 =  vector(temp_rea011)
    rea100 =  vector(temp_rea100)
    rea101 =  vector(temp_rea101)
    rea110 =  vector(temp_rea110)
    rea111 =  vector(temp_rea111)


    v=-1
    if(   i_r0 <0 or i_r1 <0 \
    or i_e0 <0 or i_e1 <0 \
    or i_a1 <0 or i_a1 <0  ):
        v=-1
    else :
        del_r = arrRange[i_r1] - arrRange[i_r0]
        del_e = arrElevation[i_e1] - arrElevation[i_e0]
        del_a = arrAzimuth[i_a1] - arrAzimuth[i_a0]
        del_cross = 1/(del_r*del_e*del_a)
        
        v =  del_cross * ( \
            rea000*((arrRange[i_r1]-r)*(arrElevation[i_e1]-e)*(arrAzimuth[i_a1]-a))+ \
            rea001*((arrRange[i_r1]-r)*(arrElevation[i_e1]-e)*(a-arrAzimuth[i_a0]))+ \
            rea010*((arrRange[i_r1]-r)*(e-arrElevation[i_e0])*(arrAzimuth[i_a1]-a))+ \
            rea011*((arrRange[i_r1]-r)*(e-arrElevation[i_e0])*(a-arrAzimuth[i_a0]))+ \
            rea100*((r-arrRange[i_r0])*(arrElevation[i_e1]-e)*(arrAzimuth[i_a1]-a))+ \
            rea101*((r-arrRange[i_r0])*(arrElevation[i_e1]-e)*(a-arrAzimuth[i_a0]))+ \
            rea110*((r-arrRange[i_r0])*(e-arrElevation[i_e0])*(arrAzimuth[i_a1]-a))+ \
            rea111*((r-arrRange[i_r0])*(e-arrElevation[i_e0])*(a-arrAzimuth[i_a0]))  \
        )
    v = np.squeeze(v)

    return v


def viz_YX_YZ_ZX_CAM(arrZYX,path_cam,path_save):
    fig,axs = plt.subplots(2,2,figsize=(12,10))
    
    #do flip for viz 
    arrZYX_flip = np.flip(arrZYX,axis=1)

    # BEV Y-X plt
    showXY = np.transpose(np.mean(arrZYX_flip,axis=0),(1,0))
    showXY[showXY<=1] = 1
    showXY = np.log2(showXY)

    im00 = axs[0, 1].imshow(showXY[::-1, :],cmap='jet',vmin=0,vmax=20)
    fig.colorbar(im00, orientation='vertical')

    axs[0, 1].set_title('Y-X (BEV)')
    axs[0, 1].set_xlabel('axis-Y (m)')
    axs[0, 1].set_ylabel('axis-X (m)')
    axs[0,1].set_aspect(0.5)
    x_plot_loc = np.linspace(0, showXY.shape[1], 11).astype(int)
    y_plot_loc = np.linspace(0, showXY.shape[0], 11).astype(int)
    axs[0, 1].set_xticks(x_plot_loc, -( x_plot_loc/showXY.shape[1] * (y_max - y_min) + y_min).astype(int))
    axs[0, 1].set_yticks(y_plot_loc, (-y_plot_loc/showXY.shape[0] * (x_max - x_min) + x_max).astype(int))


    # Y-Z plt
    showZY = np.mean(arrZYX_flip,axis=2)
    showZY[showZY<=1] = 1
    showZY = np.log2(showZY)
    im01 = axs[1, 1].imshow(showZY[::-1, :],cmap='jet',vmin=0,vmax=20)
    fig.colorbar(im01, orientation='vertical')

    axs[1, 1].set_title('Y-Z')
    axs[1, 1].set_xlabel('axis-Y (m)')
    axs[1, 1].set_ylabel('axis-Z (m)')
    axs[1,1].set_aspect(2)
    x_plot_loc = np.linspace(0, showZY.shape[1], 11).astype(int)
    y_plot_loc = np.linspace(0, showZY.shape[0], 11).astype(int)
    axs[1, 1].set_xticks(x_plot_loc, -( x_plot_loc/showZY.shape[1] * (y_max - y_min) + y_min).astype(int))
    axs[1, 1].set_yticks(y_plot_loc, (-y_plot_loc/showZY.shape[0] * (z_max - z_min) + z_max).astype(int))


    # Z-X plt
    showZX = np.mean(arrZYX_flip,axis=1)
    showZX[showZX<=1] = 1
    showZX = np.log2(showZX)
    im10 = axs[1, 0].imshow(showZX[::-1, :],cmap='jet',vmin=0,vmax=20)
    fig.colorbar(im10, orientation='vertical')

    axs[1, 0].set_title('X-Z')
    axs[1, 0].set_xlabel('axis-X (m)')
    axs[1, 0].set_ylabel('axis-Z (m)')
    axs[1,0].set_aspect(4)
    x_plot_loc = np.linspace(0, showZX.shape[1], 11).astype(int)
    y_plot_loc = np.linspace(0, showZX.shape[0], 11).astype(int)
    axs[1, 0].set_xticks(x_plot_loc, ( x_plot_loc/showZX.shape[1] * (x_max - x_min) + x_min).astype(int))
    axs[1, 0].set_yticks(y_plot_loc, (-y_plot_loc/showZX.shape[0] * (z_max - z_min) + z_max).astype(int))
    

    # camera pic
    im11 = plt.imread(path_cam)
    imagebox = OffsetImage(im11, zoom=0.25)
    imagebox.image.axes = axs[0,0]
    ab = AnnotationBbox(imagebox, (0.42, 0.5), xycoords='axes fraction',
                        bboxprops={'lw':0})
    axs[0, 0].add_artist(ab)
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].grid("False")
    axs[0, 0].spines['top'].set_visible(False)
    axs[0, 0].spines['right'].set_visible(False)
    axs[0, 0].spines['bottom'].set_visible(False)
    axs[0, 0].spines['left'].set_visible(False)

    plt.savefig(path_save)
    fig.clear()
    plt.close()
    return True


def main():

    args = parse_args()

    DEG2RAD = math.pi / 180.
    RAD2DEG = 180. / math.pi

    x_min       = 0     #(m)
    x_per_bin   = 11.6/256     #(m)
    x_max       = 11.6  #(m) x_max = range size +x_per_bin

    y_min       = -10.05     #(m) 
    y_per_bin   = 20.1/128     #(m)
    y_max       = 10.05   #(m)  y_max(2*) = 2*128*math.sin(64*DEG2RAD) 64 is from azimuth angle(0-128) +y_per_bin

    z_min       = -5.8    #(m)
    z_per_bin   = 11.6/32  #(m)
    z_max       = 5.8   #(m)  z_max(2*) = 2*128*math.sin(30*DEG2RAD) 30 is from azimuth angle(0 to 60) +z_per_bin


    arr_x = np.arange(x_min,x_max,x_per_bin)
    arr_y = np.arange(y_min,y_max,y_per_bin)
    arr_z = np.arange(z_min,z_max,z_per_bin)

    len_x = len(arr_x)
    len_y = len(arr_y)
    len_z = len(arr_z)
    len_d = 64


    arrAzimuth      = np.arange(0,120,120/128)
    arrElevation    = np.arange(0,60,60/32)
    arrRange        = np.arange(0,11.6,11.6/256)

    arrAzimuthRad   = arrAzimuth*DEG2RAD
    arrElevationRad = arrElevation*DEG2RAD

    #set min/max r,a,e 
    r_max = 11.6
    r_min = 0.
    a_max = 120.*DEG2RAD
    a_min = 0.*DEG2RAD
    e_max = 60.*DEG2RAD
    e_min = 0.*DEG2RAD

    idx = 0

    Mean_D = []
    Max_D = []
    Min_D = []
    Mean_C_D = []
    Max_C_D = []
    Min_C_D = []

    path_dataset = args.dataset_dir
    files_no = os.listdir(path_dataset)
    
    if args.sequence == None:
        sequence_list= files_no
    else:
        sequence_list = args.sequence
        

    for file_seq in sequence_list:
        
        # The following sequences were unexpectedly damaged
        if file_seq == ('44'or'68'or'107'or'155'):
            print("The raw radar data in the {} sequence was unexpectedly damaged".format(file_seq))
            continue

        s_file_seq = str(file_seq)
        path = os.path.join(path_dataset,s_file_seq,path_folder_tail)
        path_save_dzyx_c_npy = os.path.join(path_dataset ,s_file_seq,path_save_dzyx_c_npy_tail)
        print('Save path: ',path_save_dzyx_c_npy)
        for p,subdirs,files in os.walk(path):

            print('File : ',path, 'Total Frame num is : ', len(files))
            for name in tqdm(files):
                
                
                #check npy exist
                framename = re.split('[-|.]',name)
                filename = re.split('e',framename[1])
                num_frame = int(filename[1]) #split number from filename
                filename_save_npy = filename[1].zfill(6) #filename[2] is num of frame

                if(os.path.isdir(path_save_dzyx_c_npy)== False):
                    os.makedirs(path_save_dzyx_c_npy)
                
                print(os.path.join(path_save_dzyx_c_npy,(filename_save_npy+'.npy')))
                # if done skip!
                if os.path.isfile(os.path.join(path_save_dzyx_c_npy,(filename_save_npy+'.npy'))):
                    continue

                #-----loading data-----
                
                ## name 'matr4' is setting from matlab 
                arrDRAE = mat73.loadmat(os.path.join(path,name))['matr4']

                #-----loading data-----end

                arrDREA = np.transpose(arrDRAE,(0,1,3,2))
                # Creat XYZ  numpy
                # arrZYX = np.ones((len_z,len_y,len_x), dtype='complex64')
                arrDZYX = np.ones((len_d,len_z,len_y,len_x), dtype='complex64')


                except_cnt1 = 0
                except_cnt2 = 0

                #scan all XYZ
                for i_x in range(len_x):
                    x = arr_x[i_x]
                    for i_y in range(len_y):
                        y = arr_y[i_y]
                        for i_z in range(len_z):
                            z = arr_z[i_z]

                            r,a,e = c2p(x,y,z) # r,a,e (rad)
                            e = e+30*DEG2RAD # for angle shift  from deg(0-60) to elevation deg(-30 to 30)
                            a = 150*DEG2RAD-a # for angle shift  from deg(0-120) to azimuth deg(150-30)
                            
                            #exception 1
                            if(     r<r_min or r>r_max \
                                or  e<e_min or e>e_max\
                                or  a<a_min or a>a_max ):
                                except_cnt1+=1
                                continue

                            # find index (int) of r,e,a 
                            i_r0,i_r1 = findIndexForBiInt(r,arrRange)
                            i_e0,i_e1 = findIndexForBiInt(e,arrElevationRad)
                            i_a0,i_a1 = findIndexForBiInt(a,arrAzimuthRad)

                            #exception 2
                            if(     i_r0 <0 or i_r1 <0 \
                                or  i_e0 <0 or i_e1 <0 \
                                or  i_a0 <0 or i_a1<0 ):
                                except_cnt2+=1
                                continue
                            
                            #Bilinear Interpolation
                            val_rea = [r,e,a]
                            arr_bi_index = [[i_r0,i_r1],[i_e0,i_e1],[i_a0,i_a1]]
                            val = getBiIntValue(arrDREA,val_rea,arr_bi_index,arrRange,arrElevationRad,arrAzimuthRad)

                            # with Bin Int
                            arrDZYX[:,i_z,i_y,i_x] = val
                            max_C_D = np.max(val)
                            min_C_D = np.min(val)
                            mean_C_D = np.mean(val)
                            max_D = np.max(np.abs(val))
                            min_D = np.min(np.abs(val))
                            mean_D = np.mean(np.abs(val))

                #doing reverse of XYZ axis =Y
                arrDZYX = np.flip(arrDZYX,axis=2)

                Min_D.append(min_D)
                Max_D.append(max_D)
                Mean_D.append(mean_D)
                Min_C_D.append(min_C_D)
                Max_C_D.append(max_C_D)
                Mean_C_D.append(mean_C_D)

                
                # # arrDZYX = np.abs(arrDZYX)
                # arrZYX = np.mean(np.abs(arrDZYX),axis=0)

                np.save(os.path.join(path_save_dzyx_c_npy,filename_save_npy),arrDZYX)

    print ('Max D: ' + str(max(Max_D)))
    print ('Min D: ' + str(min(Min_D)))
    print ('Max Mean D: ' + str(max(Mean_D)))
    print ('Min Mean D: ' + str(min(Mean_D)))
    print ('Complex Max D: ' + str(max(Max_C_D)))
    print ('Complex Min D: ' + str(min(Min_C_D)))
    print ('Complex Max Mean D: ' + str(max(Mean_C_D)))
    print ('Complex Min Mean D: ' + str(min(Mean_C_D)))


if __name__ == "__main__":
    main()


