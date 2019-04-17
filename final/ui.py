import tkinter as tk
from tkinter import *
import csv
import numpy as np
import random
import tkinter.filedialog as filedialog
import requests
from tkinter import *  
from tkinter.filedialog import askopenfilename  
from PIL import Image, ImageTk

dept = []
opt = []
def choosepic():  
    path_=askopenfilename()  
    img_open = Image.open(path_) 
    img_open = img_open.resize((540,540),Image.ANTIALIAS) 
    img=ImageTk.PhotoImage(img_open)  
    l1.config(image=img)  
    l1.image=img
    
def predict():
    print('prdeict')

def construct_gui():
    global window
    window = tk.Tk()
    window.title('Cognitive Computing Final Demo')
    window.geometry('1000x800')
    global path; global l1;
    path=StringVar()  
    Button(window,text='upload image',command=choosepic).place(x=120,y=5)
    Button(window,text='predict image',command=predict).place(x=250,y=5)  
    l1=Label(window)  
    l1.place(x=10,y=60)


    chosen_var = tk.StringVar(window)
    chosen_var.set("")
    w = OptionMenu(window, chosen_var, *opt)
    w.place(x=800,y=10)

    var = tk.StringVar()
    picked_taglist = tk.Listbox(window, listvariable=var)
    picked_taglist.place(x=600,y=10)

    ## pick a word and add to list
    def add_to_list():
        tmp = chosen_var.get()
        picked_taglist.insert('end', tmp)
        chosen_var.set("")

    button = Button(window, text="Add", command=add_to_list)
    button.place(x=900,y=10)
    #####

    window.mainloop()


if __name__ == '__main__':
    opt = np.random.choice(np.arange(1000), 10)
    construct_gui()
