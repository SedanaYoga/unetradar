
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

## Inisialisasi Library Keras
import os 
import skimage.transform as trans 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras



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
    #st.markdown("<h4 style='text-align: justify; color: black;'>Kateterisasi dan kanulasi melalui arteri radialis sudah menjadi prosedur umum yang dilakukan para ahli untuk masa perioperatif. Meskipun tingkat kesuksesan yang tinggi untuk para ahli berpengalaman menggunakan teknik palpasi, ada beberapa kasus yang mana secara teknis sedikit sulit (hipotensi dan obesitas). Komplikasi yang paling sering terjadi pada saat kateterisasi melalui arteri radialis adalah oklusi arteri sementara (19.7 %) dan hematoma (14.4%), dengan infeksi pada tempat pemasukan (1.3%), haemorrhage (0.53%), dan bacteremia (0.13%) [1]. Suatu sistem ultrasound menjadi pilihan untuk membantu visualisasi para ahli terkait kelebihannya dalam aspek kenyamanan, ekonomis, dan non-ionisasi. Penelitian tentang arteri radialis masih sangat minim karena morfologi yang lebih kecil daripada arteri umum lainnya (Arteri Karotis dan Femoralis). Penelitian Smistad, et al telah membuktikan bahwa sistem ultrasound pada pembuluh darah sudah sangat berkembang—menerapkan tiga metode sekaligus, deteksi, tracking, dan rekonstruksi 3D secara real-time—dan mempunyai potensi untuk dikembangkan pada arteri lainnya, arteri radialis. [2] Sistem yang dibutuhkan saat ini adalah sistem segmentasi otomatis yang mana pada kali ini mengusulkan penggunaan metode deep learning Convolution Neural Network (CNN) untuk mendapatkan visualisasi citra pembuluh darah arteri radialis sebagai alat dukung para ahli pada saat melakukan insersi ke intra-arterial. Sistem dibagi menjadi tiga proses yaitu persiapan data, segmentasi, dan konversi. Dari hasil pengujian, didapatkan bahwa proses segmentasi citra mendapatkan nilai rata-rata dice similarity coefficient sebesar 0.935 dan rata-rata nilai error 0.124.</h1>", unsafe_allow_html=True)
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

@st.cache(allow_output_mutation=True)
def unet(pretrained_weights = None, input_size = (256,256, 1)):
    inputs = Input(input_size)
    #keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
    #                   padding='valid', data_format=None, 
    #                   dilation_rate=(1, 1), activation=None, 
    #                   use_bias=True, kernel_initializer='glorot_uniform', 
    #                   bias_initializer='zeros', kernel_regularizer=None, 
    #                   bias_regularizer=None, activity_regularizer=None, 
    #                   kernel_constraint=None, bias_constraint=None)
    
    # Rangkaian Extraction 1
    conv1 = Conv2D(64, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size = (2,2))(conv1)

    # Rangkaian Extraction 2
    conv2 = Conv2D(128, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size = (2,2))(conv2)

    # Rangkaian Extraction 3
    conv3 = Conv2D(256, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size = (2,2))(conv3)

    # Rangkaian Extraction 4
    conv4 = Conv2D(512, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size = (2,2))(drop4)

    # Rangkaian Extraction 5
    conv5 = Conv2D(1024, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3 , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Rangkaian Expansion 1
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4, up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    # Rangkaian Expansion 2
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    # Rangkaian Expansion 3
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    # Rangkaian Expansion 4
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    # Rangkaian Expansion 5
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # Membuat Model 
    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def saveResult(save_path, npyfile, flag_multi_class = False, num_class = 1):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:,:,0]
        print(img.shape)
        #img = trans.resize(img, (432,532)) # Gambar USG TA
        img = trans.resize(img, (512,470)) # Gambar USG Phantom
        io.imsave(os.path.join(save_path,"%d_predict.png"%i), img, check_contrast=False)

def testGenerator(test_path, num_image = 500, target_size = (256,256), flag_multi_class = False, as_gray = True):
    for i in range(num_image):
        #img = io.imread(os.path.join(test_path, "%d.png"%i), as_gray = as_gray)
        img = io.imread(os.path.join(test_path, "%d.png"%i), as_gray = as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,)+img.shape)
        yield img

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
        st.image([io.imread(pred_path+'1_predict.png'), masked[1]], width = 333)

def compressing_def():
    compress_file = open('./data-gui/example/video/testFinalWeight.mp4','rb')
    compress_bytes = compress_file.read()

    st.video(compress_bytes)

def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum
def dscdef():
    dsc_button = st.button('Run DSC', key = 1)
    if dsc_button:
        list_predict = os.listdir(path_predict)
        list_acuan = os.listdir(path_acuan)
        dsc = np.zeros(len(list_predict), dtype = float)
        temp = 0

        for i in range(len(list_predict)):
            img1 = cv2.cvtColor(cv2.imread(path_predict+str(i)+'.png'), cv2.COLOR_RGB2GRAY)
            img2 = cv2.cvtColor(cv2.imread(path_acuan+str(i)+'.png'), cv2.COLOR_RGB2GRAY)
            dsc[i] = dice(img1, img2)
            temp = temp + dsc[i]
            #st.write(i,dsc[i])  
        hasil = temp/len(list_predict)
        st.write('**Hasil rata-rata total: **')
        st.markdown(hasil)

def pixeld():
    pixel_button = st.button('Run Pixel Difference', key = 2)
    if pixel_button:
        list_predict = os.listdir(path_predict)
        list_acuan = os.listdir(path_acuan)
        temp = 0
        aa = 0
        bb = 0
        for i in range(len(list_acuan)):
            a = np.count_nonzero(cv2.imread(path_predict+str(i)+'.png')==255)
            b = np.count_nonzero(cv2.imread(path_acuan+str(i)+'.png')==255)
            avg = abs(a-b)/b
            #st.write(i, b, a, avg)
            temp = temp + avg
            aa = aa + a
            bb = bb + b
        temp = temp/len(list_acuan)
        aa = aa/len(list_acuan)
        bb = bb/len(list_acuan)
        st.markdown('**Hasil rata-rata error total**')
        st.markdown(temp)

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
    unet_arch = io.imread('./data-gui/unet-architecture.png')
    st.markdown('Convolutional Neural Network membutuhkan masukan dengan persyaratan tertentu. Gambar training pada umumnya hanya memiliki persebaran intensitas warna grayscale pada setiap pikselnya, namun membentuk array gambar 3 dimensi karena memiliki kanal warna RGB. Label yang juga berupa array biner hanya membentuk gambar 2 dimensi array gambar dengan nilai 0 dan 1 sebagai intensitas warnanya. Gambar training dan label dipasangkan berdasarkan nama yang sama. Proses training tidak hanya menggunakan satu gambar, tetapi ribuan gambar grayscale.')
    st.markdown('### **Grafik pada saat training**')
    cap_list_graf = ['Akurasi pada epoch 10', 'Akurasi pada epoch 100', 'Loss pada epoch 10', 'Loss pada epoch 100']
    st.image([grafik[0], grafik[1], grafik[2], grafik[3]], caption = cap_list_graf, width=330)
    st.markdown('### **Prediksi Citra dengan Model CNN**')
    st.markdown('Arsitektur yang digunakan adalah U-Net, jaringan neural network yang sudah banyak digunakan untuk segmentasi gambar biomedik. Jaringan ini diberi nama U-Net dengan alasan sederhana, karena bentuk arsitekturnya seperti huruf U dapat dilihat pada gambar dibawah. Hal ini disebabkan karena adanya dua jalur yaitu contracting path dan expansive path. Bagian Contracting sama dengan arsitektur convolutional network pada umumnya. Terdiri dari dua kali 3×3 konvolusi (padding same), setiap konvolusi diikuti rectified linear unit (ReLU) dan sebuah operasi 2×2 max pooling dengan stride 2 poin untuk downsampling. Di setiap downsampling jumlah feature channel dikali dua. Setiap langkah pada expansive path terdiri dari upsampling feature map diikuti dengan 2×2 up-convolution yang membagi dua jumlah dari feature channel, concatenation dengan feature map yang berhubungan dengan contracting path, dan dua konvolusi 3×3, setiap konvolusi diikuti dengan ReLU. Pada layer akhir, sebuah konvolusi 1×1 digunakan untuk memetakan setiap 64-komponen feature vector ke jumlah kelas yang diinginkan. Total jaringan menggunakan 23 layer konvolusi sudah termasuk up sampling convolution. ')
    st.image(unet_arch, caption='U-Net: Convolutional Networks for Biomedical Image Segmentation (Olaf Ronneberger, Philipp Fischer, Thomas Brox)', use_column_width=True)
    st.markdown('**Berikut adalah demo program prediksi model CNN: ** *(klik Run)*')

    model_checkpoint = ModelCheckpoint('unet_weights100.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model = unet(pretrained_weights='unet_weights100.hdf5')
    
    run_button = st.button('Run', key = 0)
    if run_button:
        testGene = testGenerator('data/test-phantom') #Data Phantom
        results = model.predict_generator(testGene,167,verbose=1, callbacks = [model_checkpoint])#cv2.normalize(src= model.predict_generator(testGene,237,verbose=1, callbacks = [model_checkpoint]), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #saveResult('data/test', results) #Data GE
        saveResult('data/test-phantom', results) #Data Phantom
        st.success('Proses prediksi sukses, silahkan buka folder!')
    
    test_button = st.button('Buka Folder', key=0)
    if test_button:
        subprocess.Popen(r'explorer /select,"D:\DATA\Institut Teknologi Sepuluh Nopember\MATERI\SEMESTER VIII\Tugas Akhir\GitRepo\data\test-phantom\0.png"')
    st.markdown('## **Metode Pengujian Citra**')
    st.markdown('Metode perhitungan diuji dengan data testing yang telah diberikan label biner sama seperti data *training*.')
    
    sb_dsc = st.selectbox('Pilih Threshold:', ('30','220'), index = 0)
    path_predict = './data-gui/example/result threshold '+str(sb_dsc)+'/'
    path_acuan = './data-gui/example/Ground Truth/'
    
    st.markdown('### **Dice Similarity Coefficient**')
    st.markdown('Data label ini akan digunakan untuk mencari akurasi model CNN dengan metode Dice Similarity Coefficient (DSC) dijelaskan pada dibawah ini:')
    st.markdown('$$DSC={2\cdot|A\cap B|\over |A| + |B|}$$')
    st.markdown('dimana A adalah gambar label dan B adalah gambar dari hasil prediksi model CNN.')
    dscdef()

    st.markdown('### **Pixel Difference**')
    st.markdown('Kesalahan nilai pengukuran menggunakan metode perbedaan jumlah piksel non-zero (berwarna putih) dari gambar label dan gambar prediksi. Pengukuran menggunakan persamaan perbandingan absolut selisih dari jumlah piksel non-zero gambar label dan gambar prediksi dengan jumlah piksel non-zero gambar label seperti persamaan dibawah ini:')
    st.markdown('$$Error={|RA_{label} - RA_{pred}|\over RA_{label}}$$')
    st.markdown('Dimana $${RA_{label}}$$ merupakan jumlah piksel non-zero gambar label dan $${RA_{pred}}$$ merupakan jumlah piksel non-zero gambar prediksi.')
    pixeld()
    
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
            