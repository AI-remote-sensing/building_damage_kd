import numpy as np
import gdal
import os
import cv2
import matplotlib.pyplot as plt
import argparse
#%%
def image(path,band): 
#读为一个numpy数组  
    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(band)
    nXSize = dataset.RasterXSize #列数
    nYSize = dataset.RasterYSize #行数
    data= band.ReadAsArray(0,0,nXSize,nYSize).astype(np.int)
    return data

def writeimage(filename,dst_filename,data):
#filename用于获取坐标信息，dst_filename目标文件格式为ENVI格式，data为要写出的数据，
    dataset=gdal.Open(filename)
    projinfo=dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    format = "ENVI"
    driver = gdal.GetDriverByName( format )
    dst_ds = driver.Create( dst_filename,dataset.RasterXSize, dataset.RasterYSize,
                           1, gdal.GDT_Float32 )
    dst_ds.SetGeoTransform(geotransform )
    dst_ds.SetProjection( projinfo )
    dst_ds.GetRasterBand(1).WriteArray( data )
    dst_ds = None

def getListFiles(path):
#获取文件目录下的所有文件（包含子文件夹内的文件）
    assert os.path.isdir(path),'%s not exist,'%path
    ret=[]
    for root,dirs,files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root,filespath))
    return ret

def get_figure(file_path,save_path=None):
    band_list = [image(file_path,i) for i in range(1,4)]
    img = cv2.merge(band_list)
    plt.imshow(img)
    plt.xticks([]),plt.yticks([]) # 不显示坐标轴
    if save_path != None:
        plt.savefig(save_path)
    plt.show()

#%%
if __name__=='__main__':
    parser = argparse.ArgumentParser('将geotiff文件转成rbg图像')
    parser.add_argument("-geotiff", help="geotiff文件的path", required=True)
    parser.add_argument("-savepath", help="存储rgb文件的path", required=True)
    args = parser.parse_args()
    #file_path = '/data1/su/data/geotiffs/tier3/images/woolsey-fire_00000877_pre_disaster.tif'
    # use like following
    # python process_geotiff.py -geotiff /data1/su/data/geotiffs/tier3/images/woolsey-fire_00000877_pre_disaster.tif -savepath test.png
    get_figure(args.geotiff,args.savepath)