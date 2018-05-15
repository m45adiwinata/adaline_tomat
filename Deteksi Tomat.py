from Tkinter import *
import tkMessageBox as msgbox
import tkFileDialog
import cv2
import math
import numpy as np
import scipy as scp
from PIL import Image, ImageTk
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')

class Main:
    def __init__(self,parent,title):
        self.parent = parent
        self.parent.title(title)
        self.parent.config(background="#316068")
        global kernel_h
        global kernel_w
        kernel_h = 3
        kernel_w = 3
        self.komponen()

#==================================================================== GUI =========================================================================================

    def komponen(self):
        self.frameFoto = Frame(self.parent,bg="#316068")
        self.frameFoto.grid(row=0,column=0,sticky=NW)

        self.histFoto = Frame(self.parent,bg="#316068")
        self.histFoto.grid(row=0,column=1,sticky=NW)

        self.imgDef = Image.open("kosong.png")
        self.imgDef = ImageTk.PhotoImage(self.imgDef)
        
        label1 = Label(self.frameFoto,width=25,height=2,fg="black",text="Ambil Gambar Tomat")
        label1.grid(row=0,column=0,sticky=N,columnspan=2)
        
        label2 = Label(self.histFoto,width=25,height=2,fg="black",text="Histogram")
        label2.grid(row=0,column=0,sticky=N)

        self.fotoInput = Label(self.frameFoto,image=self.imgDef,width=256,height=256)
        self.fotoInput.grid(row=1,column=0,padx=10,pady=10,sticky=N,columnspan=2)
        self.fotoInput.image = self.imgDef

        self.fotoZoom = Label(self.frameFoto,image=self.imgDef,width=256,height=256)
        self.fotoZoom.grid(row=2,column=0,sticky=N,padx=10,pady=10, columnspan=2)
        self.fotoZoom.image = self.imgDef

        self.btnBrowse = Button(self.frameFoto, text='Load Photo',command=self.ambilImg,width=20,height=2,bg="#a1dbcd")
        self.btnBrowse.grid(row=3,column=0,sticky=N,pady=10,padx=2.5)

        self.btnXtract = Button(self.frameFoto, text='Feature Extraction',command=self.mulai,width=20,height=2,bg="#a1dbcd")
        self.btnXtract.grid(row=3,column=1,sticky=N,pady=10,padx=2.5)

        self.btnGO = Button(self.frameFoto, text='Identifikasi',command=self.identify,width=45,height=2,bg="green")
        self.btnGO.grid(row=4,column=0,sticky=N,pady=10,padx=10,columnspan=2)

        labelR = Label(self.histFoto, fg="red",text="Histogram Merah")
        labelR.grid(row=1,column=0,sticky=N, pady=10, padx=50)

        self.histMerah = Figure(figsize=(4,1.7), dpi=100)
        canvas = FigureCanvasTkAgg(self.histMerah, master=self.histFoto)
        canvas.get_tk_widget().grid(row=2,column=0,sticky=N)

        labelG = Label(self.histFoto,fg="green",text="Histogram Hijau")
        labelG.grid(row=3,column=0,sticky=N, pady=10, padx=50)

        self.histHijau = Figure(figsize=(4,1.7), dpi=100)
        canvas = FigureCanvasTkAgg(self.histHijau, master=self.histFoto)
        canvas.get_tk_widget().grid(row=4,column=0,sticky=N)

        labelB = Label(self.histFoto,fg="blue",text="Histogram Biru")
        labelB.grid(row=5,column=0,sticky=N, pady=10, padx=50)

        self.histBiru = Figure(figsize=(4,1.7), dpi=100)
        canvas = FigureCanvasTkAgg(self.histBiru, master=self.histFoto)
        canvas.get_tk_widget().grid(row=6,column=0,sticky=N)

        self.nilai = Frame(self.parent, bg="#FFFFFF")
        self.nilai.grid(row=0, column=2, sticky=NW, padx=30)
        
        labelNilaiRGB = Label(self.nilai, fg="black", text="Nilai RGB")
        labelNilaiRGB.grid(row=0, column=0, sticky=N, pady=10, columnspan=2)

        labelMaxRGB = Label(self.nilai, fg="black", text="MAX RGB")
        labelMaxRGB.grid(row=1, column=0, sticky=N, pady=10, columnspan=2)

        x = StringVar()
        self.NilaiMax = Entry(self.nilai,bd=4,width=20, textvariable=x)
        self.NilaiMax.grid(row=2,column=0,sticky=N,padx=10,pady=10, columnspan=2)
        x.set(str(self.maxRGB()))

        labelMean = Label(self.nilai, fg="black", text="MEAN")
        labelMean.grid(row=3, column=0, sticky=N, pady=5, columnspan=2)

        labelMeanR = Label(self.nilai, fg="red", text="RED")
        labelMeanR.grid(row=4, column=0, sticky=N, pady=5, columnspan=2)

        self.mR = StringVar()
        self.meanRed = Entry(self.nilai,bd=4,width=20, textvariable=self.mR)
        self.meanRed.grid(row=5,column=0,sticky=N, columnspan=2)
        self.mR.set("0")

        labelMeanG = Label(self.nilai, fg="green", text="GREEN")
        labelMeanG.grid(row=6, column=0, sticky=N, pady=5, columnspan=2)

        self.mG = StringVar()
        self.meanGreen = Entry(self.nilai,bd=4,width=20, textvariable=self.mG)
        self.meanGreen.grid(row=7,column=0,sticky=N, columnspan=2)
        self.mG.set("0")

        labelMeanB = Label(self.nilai, fg="blue", text="BLUE")
        labelMeanB.grid(row=8, column=0, sticky=N, pady=5, columnspan=2)

        self.mB = StringVar()
        self.meanBlue = Entry(self.nilai,bd=4,width=20, textvariable=self.mB)
        self.meanBlue.grid(row=9,column=0,sticky=N, columnspan=2)
        self.mB.set("")

        labelNorm = Label(self.nilai, fg="black", text="NORMALISASI")
        labelNorm.grid(row=10, column=0, sticky=N, pady=10, columnspan=2)

        self.N1 = StringVar()
        self.Norm1 = Entry(self.nilai,bd=4,width=20, textvariable=self.N1)
        self.Norm1.grid(row=11,column=0,sticky=N, columnspan=2)
        self.N1.set("")

        self.N2 = StringVar()
        self.Norm2 = Entry(self.nilai,bd=4,width=20, textvariable=self.N2)
        self.Norm2.grid(row=12,column=0,sticky=N, pady=5, columnspan=2)
        self.N2.set("")

        self.N3 = StringVar()
        self.Norm3 = Entry(self.nilai,bd=4,width=20, textvariable=self.N3)
        self.Norm3.grid(row=13,column=0,sticky=N, columnspan=2)
        self.N3.set("")

        labelHasil = Label(self.nilai, fg="black", text="HASIL PROSES ADALINE")
        labelHasil.grid(row=14, column=0, sticky=N, pady=10, columnspan=2)

        labelNet = Label(self.nilai, fg="black",text="NET")
        labelNet.grid(row=15, column=0, sticky=N, pady=10)

        self.Net = StringVar()
        self.valueNet = Entry(self.nilai,bd=3,width=20, textvariable=self.Net)
        self.valueNet.grid(row=15,column=1,sticky=NW, pady=10, padx=4)
        self.Net.set("")

        labelNet = Label(self.nilai, fg="black",text="KESIMPULAN")
        labelNet.grid(row=16, column=0, sticky=NW, pady=10)
        
        self.result = StringVar()
        self.conclusion = Entry(self.nilai,bd=3,width=20, textvariable=self.result)
        self.conclusion.grid(row=16,column=1,sticky=N, pady=10,padx=4)
        self.result.set("")

        self.Learning = Frame(self.parent,bg="#FFFFFF")
        self.Learning.grid(row=0,column=3,sticky=NW, padx=30, pady=10)

        label30 = Label(self.Learning, fg="black", text="LEARNING")
        label30.grid(row=0, column=0, sticky=N, columnspan=2)
        
        labelTarget = Label(self.Learning, fg="black", text="TARGET")
        labelTarget.grid(row=1, column=0, sticky=N, pady=10)

        self.target = StringVar()
        self.setTarget = Entry(self.Learning,fg="blue", bd=4, width=10, textvariable=self.target)
        self.setTarget.grid(row=1,column=1, sticky=N, pady=10,padx=5)
        self.target.set("unset")

        self.saveBtn = Button(self.Learning, text='Save to Dataset',command=self.simpandata,width=20,height=1,bg="#a1dbcd")
        self.saveBtn.grid(row=2,column=0,sticky=N,pady=10,padx=10, columnspan=2)

        self.learnBtn = Button(self.Learning, text='Learn', command=self.adaline, width=20,height=1, bg="green")
        self.learnBtn.grid(row=3, column=0,sticky=N, pady=10, padx=10, columnspan=2)

        self.spaceFrame = Frame(self.Learning, bg="#FFFFFF")
        self.spaceFrame.grid(row=4, column=0, sticky=N, pady=50, columnspan=2)
        
        self.resetBtn = Button(self.Learning, text='Reset Bobot', command=self.resetWeightAndBias, width=20,height=1, bg="red", fg="black")
        self.resetBtn.grid(row=5, column=0, sticky=N, pady=10, padx=10, columnspan=2)

        self.destroyBtn = Button(self.Learning, text='Hapus Dataset', command=self.clearDataset, width=20,height=1, bg="red", fg="black")
        self.destroyBtn.grid(row=6, column=0, sticky=N, pady=10, padx=10, columnspan=2)


#==================================================================== FUNGSI =========================================================================================

    def maxRGB(self):
        n = 256*256*255*3
        return n
    def tampilImg(self):
        self.fotoOutput.config(image=self.image_show,width=320,height=320)
        self.fotoOutput.image = self.image_show

    def ambilImg(self):
        self.path = tkFileDialog.askopenfilename()
        if(len(self.path) > 0):
            image = cv2.imread(self.path)
            h = image.shape[0]
            w = image.shape[1]
            for i in range(h):
                for j in range(w):
                    px_b = float(image[i,j,0])
                    px_g = float(image[i,j,1])
                    px_r = float(image[i,j,2])
                    image[i,j,0] = px_r
                    image[i,j,1] = px_g
                    image[i,j,2] = px_b
                
            resized_image = cv2.resize(image, (256,256))
            self.image_show = Image.fromarray(resized_image)
            self.image_show = ImageTk.PhotoImage(self.image_show)

            self.fotoInput.config(image=self.image_show,width=256,height=256)
            
            self.fotoZoom.config(image=self.image_show,width=256,height=256)
            
    
    def mulai(self):
        image = cv2.imread(self.path)
        resized_image = cv2.resize(image, (256,256))
        rows = resized_image.shape[0]
        cols = resized_image.shape[1]
        r=0
        g=0
        b=0
        for i in range(rows):
            for j in range(cols):
                red = int(image[i,j,2])
                r = r+red
                green = int(image[i,j,1])
                g = g+green
                blue = int(image[i,j,0])
                b = b+blue
        meanR = float(r)/self.maxRGB()
        self.mR.set(meanR)
        meanG = float(g)/self.maxRGB()
        self.mG.set(meanG)
        meanB = float(b)/self.maxRGB()
        self.mB.set(meanB)
        self.normalisasi()
    def normalisasi(self):
        d = []
        d.append(float(self.mR.get()))
        d.append(float(self.mG.get()))
        d.append(float(self.mB.get()))
        new_min = 0
        new_max = 1
        v = []
        for i in range(3):
            x = (d[i]-min(d))*(new_max - new_min)/(max(d) - min(d))
            x = x+new_min
            v.append(x)
        self.N1.set(v[0])
        self.N2.set(v[1])
        self.N3.set(v[2])
        
    def simpandata(self):
        if(self.target.get() != 'unset'):
            x = []
            file = open("dataset.txt","r")
            for item in file.read().split():
                x.append(float(item))
            file.close()
            x.append(float(self.N1.get()))
            x.append(float(self.N2.get()))
            x.append(float(self.N3.get()))
            x.append(float(self.target.get()))
            file = open("dataset.txt","w")
            endl = 0
            for item in x:
                file.write("%s " % item)
                endl += 1
                if(endl==4):
                    file.write("\n")
                    endl=0
            file.close()
            print x

    def perceptron(self):
       x = []
       x.append(float(self.N1.get()))
       x.append(float(self.N2.get()))
       x.append(float(self.N3.get()))
       lr = 0.5
       target = [1,0,-1]
       f = open("bobot.txt","r")
       w = f.readlines()
       f.close()

    def adaline(self):
       lr = 0.1
       d = 1
       f = open("dataset.txt","r")
       data = f.read().split()
       f.close()
       epoh = 1
       while d>0.05:
           print "epoh : ",epoh
           for j in range(len(data)/4):
               x = []
               i=j*4
               while i<j*4+3:
                   x.append(float(data[i]))
                   i+=1
               t = float(data[j*4+3])
               print "x : ",x
               f = open("bobot.txt","r")
               w = f.read().split()
               f.close()
               net = 0
               for k in range(3):
                   net += x[k]*float(w[k])
               f = open("bias.txt","r")
               b = float(f.read())
               f.close()
               net += b
               print "net = ",net
               y = net
               print "y = ",y
               print "t-y = ",t-y
               dw = [0,0,0]
               for k in range(3):
                   dw[k] = lr*x[k]*(t-y)
               print dw
               db = lr*(t-y)
               for k in range(3):
                   w[k] = str(float(w[k])+dw[k])
               print "w : ",w 
               f = open("bobot.txt","w")
               for item in w:
                   f.write("%s " % item)
               f.close()
               f = open("bias.txt","w")
               f.write("%s" % str(b+db))
               f.close()
           d = max(dw)
           print ""
           epoh += 1
           
    def identify(self):
        self.mulai()
        x = []
        x.append(float(self.N1.get()))
        x.append(float(self.N2.get()))
        x.append(float(self.N3.get()))
        file = open("bobot.txt","r")
        w = file.read().split()
        file.close()
        file = open("bias.txt","r")
        b = float(file.read())
        file.close()
        net = 0
        for i in range(3):
            net += x[i]*float(w[i])
        net += b
        self.Net.set(str(net))
        #fungsi aktivasi
        if net<1.5:
            r = 1
            self.result.set("MATANG")
        elif net<2.4:
            r = 2
            self.result.set("KURANG MATANG")
        else:
            r = 3
            self.result.set("MASIH MENTAH")

    def resetWeightAndBias(self):
        file = open("password.txt","r")
        p = file.read()
        file.close()
        c = raw_input("ketik password : ")
        if c == p:
            print "password benar"
            '''
            file = open("bobot.txt","w")
            for i in range(3):
                file.write("0 ")
            file.close()
            file = open("bias.txt","w")
            file.write("0")
            file.close()
            '''
        else:
            print "password salah, operasi dibatalkan"

    def clearDataset(self):
        file = open("password.txt","r")
        p = file.read()
        file.close()
        c = raw_input("ketik password : ")
        if c == p:
            print "password benar"
            '''
            file = open("bobot.txt","w")
            for i in range(3):
                file.write("0 ")
            file.close()
            file = open("bias.txt","w")
            file.write("0")
            file.close()
            '''
        else:
            print "password salah, operasi dibatalkan"

#==================================================================== MAIN =========================================================================================        
root = Tk()
Main(root,".:: APLIKASI PENDETEKSI KEMATANGAN TOMAT ::.")
root.mainloop()

