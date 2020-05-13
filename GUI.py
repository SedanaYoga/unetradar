
import numpy as np 
import pandas as pd 
import streamlit as st 
import time
import cv2
import matplotlib.pyplot as plt
import os
from natsort import natsorted, ns
import glob
from skimage import io
from skimage import color
from scipy import ndimage, misc
from skimage import exposure
from scipy.ndimage import gaussian_filter
import subprocess



############################################################## INISIALISASI
st.title("GUI for U-Net Radar")
list_menu = ['Home','Pre-processing', 'CNN Segmentation', 'Convertion']
sb_menu = st.sidebar.selectbox('Pilih menu : ', list_menu, index = 0)

############################################################## PERSIAPAN DATA UNTUK PREPROCESSING
data_path = './data-gui/example/1ExtractedImage/'
file_path = glob.glob(data_path + '*png')
list_data = os.listdir(data_path)
filenames = natsorted(file_path, key = lambda y : y.lower())
images = [cv2.imread(img) for img in filenames]
cropped = [cv2.imread(img)[20:276,111:367] for img in filenames]

############################################################# PERSIAPAN DATA UNTUK CNN SEGMENTASI

grafik_path = glob.glob('./grafik-training/*png')
listname_graf = natsorted(grafik_path, key = lambda y : y.lower())
grafik = [cv2.imread(graf) for graf in listname_graf]

############################################################# FUNGSI-FUNGSI
def home_sb():
    st.header('**Segmentasi Gambar Ultrasound Arteri Radialis Menggunakan Convolutional Neural Network untuk Akses Insersi Arteri**')
    st.markdown('## Abstrak')
    st.write('Kateterisasi dan kanulasi melalui arteri radialis sudah menjadi prosedur umum yang dilakukan para ahli untuk masa perioperatif. Meskipun tingkat kesuksesan yang tinggi untuk para ahli berpengalaman menggunakan teknik palpasi, ada beberapa kasus yang mana secara teknis sedikit sulit (hipotensi dan obesitas). Komplikasi yang paling sering terjadi pada saat kateterisasi melalui arteri radialis adalah oklusi arteri sementara (19.7 %) dan hematoma (14.4%), dengan infeksi pada tempat pemasukan (1.3%), haemorrhage (0.53%), dan bacteremia (0.13%) [1]. Suatu sistem ultrasound menjadi pilihan untuk membantu visualisasi para ahli terkait kelebihannya dalam aspek kenyamanan, ekonomis, dan non-ionisasi. Penelitian tentang arteri radialis masih sangat minim karena morfologi yang lebih kecil daripada arteri umum lainnya (Arteri Karotis dan Femoralis). Penelitian Smistad, et al telah membuktikan bahwa sistem ultrasound pada pembuluh darah sudah sangat berkembang—menerapkan tiga metode sekaligus, deteksi, tracking, dan rekonstruksi 3D secara real-time—dan mempunyai potensi untuk dikembangkan pada arteri lainnya, arteri radialis. [2] Sistem yang dibutuhkan saat ini adalah sistem segmentasi otomatis yang mana pada kali ini mengusulkan penggunaan metode deep learning Convolution Neural Network (CNN) untuk mendapatkan visualisasi citra pembuluh darah arteri radialis sebagai alat dukung para ahli pada saat melakukan insersi ke intra-arterial. Sistem dibagi menjadi tiga proses yaitu persiapan data, segmentasi, dan konversi. Dari hasil pengujian, didapatkan bahwa proses segmentasi citra mendapatkan nilai rata-rata dice similarity coefficient sebesar 0.935 dan rata-rata nilai error 0.124.')

def crop_func(): 
    slide_crop = st.slider('', min_value = 0, max_value = len(file_path)-1, key = 0)
    for i in range(len(file_path)):
        if slide_crop == i:
            st.image([images[i], cropped[i]], caption = ['Gambar asli '+list_data[i], 'Hasil Cropping '+list_data[i]], width = 330)

def filter_func():
    cap_list_filter = ['Gambar Cropped', 'Gaussian Filter', 'Rescale Intensity', 'Histogram Equalization', 'Median Filter']
    for i in range(len(file_path)):
        if slider_filter == i:
            imggray = cv2.cvtColor(cropped[i], cv2.COLOR_BGR2GRAY)
            img_gaussian = gaussian_filter(imggray, sigma = 2)
            rescale = exposure.rescale_intensity(img_gaussian, in_range= (40,100))
            img_histeq = cv2.normalize(src=exposure.equalize_hist(rescale),dst = None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
            img_median = ndimage.median_filter(img_histeq, size= 20)
            st.image([cropped[i], img_gaussian, rescale, img_histeq, img_median], caption = cap_list_filter, width = 125)
            st.image([cropped[i],img_median], caption = ['Gambar Hasil Cropping', 'Hasil Proses Filtering'], width = 330)

def masking(img, mask, th, height=432, width=532, color = None):
    '''.'''
    #mask_out = np.zeros((512, 470, 3), dtype = 'uint8') #phantom
    mask_out = np.zeros((height, width, 3), dtype = 'uint8') #GE
    img = np.zeros((height, width, 3), dtype = 'uint8')
    for i in range(mask.shape[0]-1):
        for j in range(mask.shape[1]-1):
            if (mask[i,j] >= th):
                mask_out[i,j,:] = mask[i,j] 
                mask_out[i,j,:] = color
                img[i,j,:] = mask_out[i,j,:]
    segmented = img
    return segmented

def masking_def():
    Merah = np.array([255, 0, 0])
    Hijau = np.array([0, 255, 0])
    Biru = np.array([0, 0, 255])
    Kuning = np.array([255, 255, 0])
    Aqua = np.array([0, 255, 255])
    Putih = np.array([255,255,255])
    input_mask_color = st.selectbox('Pilih Warna', ['Merah','Hijau','Biru','Kuning','Aqua','Putih'], index = 4)
    input_thres = st.number_input('Atur Threshold', min_value = 1, max_value=255, value = 30, step=1, key = 0)
    pred_path = './data-gui/example/test/'
    pred_list = os.listdir(pred_path)
    masked = np.zeros((len(pred_list), 432, 532, 3), dtype ='uint8')
    if input_thres:
        for i in range(len(pred_list)):
            masked[i] = masking(np.zeros((432, 532, 3), dtype = 'uint8'), io.imread(pred_path+str(i)+'_predict.png'), th=input_thres, color = vars()[input_mask_color])
        st.image([masked[0],masked[1],masked[2],masked[3],masked[4]], width = 126)
        st.image([io.imread(pred_path+'0_predict.png'), masked[0]], width = 333)

def compressing_def():
    compress_file = open('./data-gui/example/video/testFinalWeight.mp4','rb')
    compress_bytes = compress_file.read()

    st.video(compress_bytes)
    
############################################################## PROSEDUR UNTUK MENU SIDEBAR

if sb_menu == 'Home':
    home_sb()

if sb_menu == 'Pre-processing':
    st.header('**Pre-Processing**')
    st.markdown('Sebelum siap untuk menjadi masukan CNN, semua data harus melalui *pre-processing*. Pada penelitian kali ini proses tersebut meliputi *cropping* (pemotongan resolusi gambar) dan *image enhancement*.')
    st.markdown('### **Cropping**')
    st.markdown('Cropping berguna untuk memotong citra USG yang utuh agar tersisa bagian arteri yang diinginkan. Selain itu, hasil cropping harus serupa dengan resolusi masukan dari proses training CNN yaitu berukuran 256×256.')
    crop_func()
    st.markdown('### **Filter**')
    st.markdown('image enhancement dengan empat filter berbeda, yaitu Gaussian Filter, Rescale Intensity, Histogram Equalization, dan Median Filter secara berurutan. Pemilihan keempat filter tersebut karena memberikan peningkatan kualitas citra di aspek-aspek tertentu, gaussian filter dengan sigma = 2 membantu menyamarkan noise (blur), rescale intensity berguna untuk memperjelas citra pada rentang tertentu dengan cara melebarkan (stretch) atau menyusutkan (shrink) intensitas pada pada citra, histogram equalization untuk meningkatkan kontras dari citra sehingga fitur yang diinginkan lebih terlihat jelas, dan median filter pada size 20 untuk mengurangi noise berupa bintik hitam putih kecil pada citra (salt dan pepper).')
    slider_filter = st.slider('', min_value = 0, max_value = len(file_path)-1, key = 1)
    filter_func()

if sb_menu == 'CNN Segmentation':
    st.header('**Segmentation**')
    st.markdown('Convolutional Neural Network membutuhkan masukan dengan persyaratan tertentu. Gambar training pada umumnya hanya memiliki persebaran intensitas warna grayscale pada setiap pikselnya, namun membentuk array gambar 3 dimensi karena memiliki kanal warna RGB. Label yang juga berupa array biner hanya membentuk gambar 2 dimensi array gambar dengan nilai 0 dan 1 sebagai intensitas warnanya. Gambar training dan label dipasangkan berdasarkan nama yang sama. Proses training tidak hanya menggunakan satu gambar, tetapi ribuan gambar grayscale.')
    st.markdown('### **Grafik pada saat training**')
    cap_list_graf = ['Akurasi pada epoch 10', 'Akurasi pada epoch 100', 'Loss pada epoch 10', 'Loss pada epoch 100']
    st.image([grafik[0], grafik[1], grafik[2], grafik[3]], caption = cap_list_graf, width=330)
    st.markdown('### **Hasil Data Testing**')
    test_button = st.button('Show testing result', key=0)
    if test_button:
        subprocess.Popen(r'explorer /select,"D:\DATA\Institut Teknologi Sepuluh Nopember\MATERI\SEMESTER VIII\Tugas Akhir\GitRepo\data\test\0.png"')
if sb_menu == 'Convertion':
    st.header('**Konversi Sekuens Gambar ke Video**')
    st.markdown('Pada halaman ini akan menjelaskan bagaimana metode untuk mengembalikan keluaran dari sistem CNN, berupa gambar, menjadi sebuah video yang memiliki properti yang sama hanya saja ditambahkan *mask* yaitu daerah spesifik yang menunjukkan *region of interest* (ROI) dari arteri radialis. Perlu digarisbawahi bahwa hasil konversi ini akan berhasil jika data *testing* yang diuji pada model CNN harus berupa sekuens *frame* dari sebuah video. Bagian konversi ini akan dibagi menjadi dua tahap yaitu proses *masking* dan proses *compressing*.')
    st.markdown('### **Proses Masking**')
    st.markdown('Proses masking pada dasarnya hanya menempel gambar prediksi baik pada gambar training maupun gambar asli dari USG tanpa mengurangi kualitasnya (resolusi hasil masking sama dengan hasil citra USG asli). Penambahan warna pada gambar prediksi menjadi penting sebagai pembeda ROI.')
    masking_def()
    st.markdown('### **Proses Compressing**')
    st.markdown('Pada dasarnya tujuan dari *compressing* ini cukup sederhana, yaitu menggabungkan semua hasil *masking* pada satu video utuh dan kembali mempunyai format seperti video USG asli. Ada banyak cara untuk menggabungkan *frame*, namun cara yang paling sederhana dan cepat adalah menggunakan FFmpeg.')
    st.markdown('Berikut adalah kode *compressing* menggunakan aplikasi FFmpeg pada *command-line*')
    st.code("""ffmpeg -r 30 -f image2 -i %d.png -vcodec libx264 -crf 15  -pix_fmt yuv420p test.mp4
    """, language="python")
    st.markdown('Parameter yang digunakan antara lain, *framerate* (FPS) = 30, *constant rate factor* (CRF) = 15, *coder-decoder* (codec) = libx264, dan *pixel format* (pix_fmt) = yuv420p.')
    st.markdown('Hasil kompresi dalam bentuk video')
    compressing_def()
            