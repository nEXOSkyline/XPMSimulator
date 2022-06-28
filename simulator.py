
try:
    import tkinter as tk
except ModuleNotFoundError:
    # Import tkinter module for Python 2.
    import Tkinter as tk  

# ttk gives the widgets a more modern look 
try:
    from tkinter import Menu, ttk
except ModuleNotFoundError:
    # Import ttk module for Python 2.
    import ttk
    from ttk import *

import threading    
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)   
from matplotlib.figure import Figure
from matplotlib import style
style.use('ggplot')
import random
import numpy as np
from scipy import integrate
from scipy.special import erfc
from http.server import SimpleHTTPRequestHandler, HTTPServer
from lmfit.models import SkewedVoigtModel
from lmfit import Model

class MyServer(SimpleHTTPRequestHandler):
    def write_to_server(self, data):
        """ Writes to the server with data in UTF-8 encoding and text content-type """
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes(data, "utf-8"))

    def do_GET(self):
        # NOTE: The global variables--preamble and event--are found further down
        # the code

        # Write message to the server if 'curve?' query is used
        if self.path.find('curve?') >= 0:
            print('path found: curve')
            self.write_to_server(root.graph.graph.calculate_msg())

        # Write preamble to the server if 'wfmpre?' query is used
        elif self.path.find('wfmpre?') >= 0:
            print('path found: wfmpre')
            print(preamble)
            self.write_to_server(preamble)
        
        # Write event to the server if no paths are found in the URL
        else:
            print('no paths found')
            event_val = str(-event) + '\n'
            print('event:', event_val)
            self.write_to_server(event_val)

class Graph(tk.Frame):
    def __init__(self, parent, *args):
        tk.Frame.__init__(self, parent, *args)

        # Set up 2 versions of the preamble
        self.preamble_8_bit = '1;8;ASC;RP;MSB;500;"Ch1, AC coupling, 2.0E-2 V/div, 4.0E-5 s/div, 500 points, Average mode";Y;8.0E-7;0;-1.2E-4;"s";8.0E-4;0.0E0;-5.4E1;"V"\n'
        self.preamble_16_bit = '2;16;ASC;RP;MSB;500;"Ch1, AC coupling, 2.0E-2 V/div, 4.0E-5 s/div, 500 points, Average mode";Y;8.0E-7;0;-1.2E-4;"s";3.125E-6;0.0E0;-1.3824E4;"V"\n' 
        
        # Set up figure and plot
        self.figure = Figure(figsize=(3,5), dpi=100)
        self.plt = self.figure.add_subplot(111)

        # Set up electron lifetime variables
        self.lifetime = tk.StringVar(value=10000)
        self.lifetime_label = ttk.Label(parent, text="Lifetime")
        self.lifetime_entry = ttk.Entry(parent, textvariable=self.lifetime)
        self.lifetime.trace_add(
            'write', lambda name, index, mode, var=self.lifetime: self.trace_callback_lifetime(var)
        )

        # Set up cathode amplitude variables
        self.cathode = tk.StringVar(value=50)
        self.cathode_label = ttk.Label(parent, text="Cathode")
        self.cathode_entry = ttk.Entry(parent, textvariable=self.cathode)
        self.cathode.trace_add(
            'write', lambda name, index, mode, var=self.cathode: self.trace_callback_cathode(var)
        )

        # Set up 14-bit checkbutton
        self.is_14bit = tk.BooleanVar()
        self.is_14bit_chk_butn = ttk.Checkbutton(parent, text='14-bit Digitization', variable=self.is_14bit, onvalue=1, offvalue=0)
        self.is_14bit.trace_add(
            'write', lambda name, index, mode, var=self.is_14bit: self.trace_callback_14bit(var)
        )

        # Set up canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)
        
        # Set up synthetic waveform variables from synth_16bit.py
        self.preamble = self.preamble_16_bit if self.is_14bit.get() else self.preamble_8_bit
        self.ct = 0
        self.event = 0
        self.pkmodel = SkewedVoigtModel()
        self.pars = self.pkmodel.make_params()
        self.err = 0.0
        self.sqrt2 = np.sqrt(2.0)
        self.randstart = random.randint(0,1000)
        self.p_i = [65.669502,66.09416,41.967726,395.3,3.598,1.0884104,81.320328,1.80825,0.38634939]

        # Set up waveform
        lifetime = float(self.lifetime.get())
        self.deltaT = 81.9 - 10.0
        exparg = self.deltaT / lifetime
        cathode = float(self.cathode.get())
        downstroke = np.exp(-exparg)*cathode

        self.t = [ 1.0e6*(float(self.preamble.split(';')[8])*float(i)+float(self.preamble.split(';')[10])) for i in range(0,500) ]
        self.t = np.array(self.t)

        # Create wavmodel from smeared function, then create waveform
        self.wavmodel = Model(self.smeared_func, nan_policy='raise')
        self.waveform = self.wavmodel.eval(
            x=self.t,
            an=downstroke,
            cat=cathode,
            offst=self.p_i[2],
            thold=self.p_i[3],
            tcrise=self.p_i[4],
            tarise=self.p_i[5],
            center=self.p_i[6],
            gamma=self.p_i[7],
            skew=self.p_i[8]
        )

        # Get noise
        self.nt = np.loadtxt('null120v4v.isf',delimiter=',')

        #with open('noise_template.csv','r',newline='') as fnoise:
        #    for line in fnoise:
        #        self.nt.append(float(line.split(',')[1]))
        
    def trace_callback_14bit(self, var: tk.BooleanVar):
        print(f'Tracking digitization variable: {var.get()}')

    def trace_callback_lifetime(self, var: tk.StringVar):
        print(f'Tracking electron lifetime variable: {var.get()}')

    def trace_callback_cathode(self, var: tk.StringVar):
        print(f'Tracking cathode amplitude variable: {var.get()}')

    def calculate_msg(self):
        """ Calculates the message and returns as string """
        msg = ''
        lifetime = float(self.lifetime.get())
        exparg = self.deltaT / lifetime
        cathode = float(self.cathode.get())
        downstroke = np.exp(-exparg)*cathode

        print('lifetime:', lifetime, 'cathode:', cathode)

        self.preamble = self.preamble_16_bit if self.is_14bit else self.preamble_8_bit
        self.t = [ 1.0e6*(float(self.preamble.split(';')[8])*float(i)+float(self.preamble.split(';')[10])) for i in range(0,500) ]
        self.t = np.array(self.t)

        wavparams = self.wavmodel.make_params()
        wavparams['cat'].value = cathode
        wavparams['an'].value = downstroke
        wavparams['offst'].value = self.p_i[2]
        wavparams['thold'].value = self.p_i[3]
        wavparams['tcrise'].value = self.p_i[4]
        wavparams['tarise'].value = self.p_i[5]
        wavparams['center'].value = self.p_i[6]
        wavparams['gamma'].value = self.p_i[7]
        wavparams['skew'].value = self.p_i[8]

        # self.wavmodel = Model(self.smeared_func, nan_policy='raise')
        self.waveform = self.wavmodel.eval(
            x=self.t,
            an=downstroke,
            cat=cathode,
            offst=self.p_i[2],
            thold=self.p_i[3],
            tcrise=self.p_i[4],
            tarise=self.p_i[5],
            center=self.p_i[6],
            gamma=self.p_i[7],
            skew=self.p_i[8]
        )

        rdx = self.ct + self.randstart
        #baseline = np.array([self.nt[(((rdx+jj)+((rdx+jj)>=500))%500)*(-1)**((rdx+jj)>=500)] for jj in range(0,500)])
        baseline = [ bl in self.nt[2570+random.randint(4):4570:4,1] ]
        self.waveform = self.waveform + baseline

        print('**************** '+str(baseline[50])+' ***************')

        baseline = [ bl in self.nt[2570+random.randint(4):4570:4,1] ]
        for millivolt,bl in zip(self.waveform,baseline):
            #t_wall = t_wall + 8.0e-7
            #bl = int(16384.0*np.sin(2*np.pi*2500.0*t_wall - ct*np.pi*0.75 ))
            self.dl = float(self.preamble.split(';')[14]) + (millivolt/1.0e3 - float(self.preamble.split(';')[13]))/float(self.preamble.split(';')[12])
            self.dl = int(-1.0*self.dl) + 0

            bldl = float(self.preamble.split(';')[14]) + (bl/1.0e3 - float(self.preamble.split(';')[13]))/float(self.preamble.split(';')[12])
            bldl = int(-1.0*bldl) + 0
            # 14 bit operation
            if self.is_14bit.get():
                # throw away the 2 least significant bits so that 16 bit -> 14 bit
                self.dl = (self.dl >> 2)*4
                bldl = (bldl >> 2)*4
            # 8-bit operation
            else:
                self.dl = (self.dl >> 8)
                bldl = (bldl >> 8)
            msg = msg + str(self.dl*(self.ct%2) + bldl*(self.ct%2) )
            msg = msg + ','

        # Increment counter
        self.ct = self.ct + 1
        
        msg = msg[:-1] + '\n'
        print('msg:', msg)
        return msg

    def smeared_func(self, x, cat, an, offst, thold, tcrise, tarise, center, gamma, skew):
        y = 0.5*cat*erfc((-x+10.0+tcrise**2/thold)/(self.sqrt2*tcrise))*np.exp( -(x-10.0-tcrise**2/(2*thold))/thold )
        self.pars['amplitude'].value = an
        self.pars['sigma'].value = tarise
        self.pars['center'].value = center
        self.pars['gamma'].value = gamma
        self.pars['skew'].value = skew
        integrand = lambda xi: self.pkmodel.eval(self.pars, x=xi)
        norm = integrate.quad(integrand, -np.inf, np.inf)[0] 
        sv = np.array([integrate.quad( integrand, -np.inf,xi)[0] for xi in x]) / norm
        y = y-(an*sv)*np.exp(-(x-center)/thold) + offst

        return y 

class GraphFrame(tk.Frame):
    """ Shows the components of the graph! """
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        
        # Create parent, which is the class Simulator from down below 
        self.parent = parent 
        
        # Create class instance of Graph
        self.graph = Graph(self)
        self.plt = self.graph.plt
        self.canvas = self.graph.canvas

        # Pack each tkinter element
        self.graph.is_14bit_chk_butn.pack(side='top', pady=5)
        self.graph.lifetime_label.pack(side='top')
        self.graph.lifetime_entry.pack(side='top')
        self.graph.cathode_label.pack(side='top')
        self.graph.cathode_entry.pack(side='top')

        # Draw the canvas and show the navigation toolbar at the bottom 
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='top', expand=True, fill='both')
        self.graph.toolbar.update()
        self.canvas.get_tk_widget().pack(side='top', expand=True, fill='both')

        # Set the labels
        self.plt.set_title("XPM waveform")
        self.plt.set_xlabel('time (μs)')
        self.plt.set_ylabel('Amplitude (mV)')

    def plotit(self):
        """ Plot the data on the graph recursively. """
        milliseconds = 1000

        # Clear the graph before plotting the new dataset
        # Source: https://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib
        self.plt.clear()

        # Set labels again since plt.clear() removes them in the first place
        self.plt.set_title("XPM waveform")
        self.plt.set_xlabel('time (μs)')
        self.plt.set_ylabel('Amplitude (mV)')
        
        # Plot the points, then draw them on the canvas
        self.plt.plot(self.graph.t, self.graph.waveform, 'r-')
        self.canvas.draw()
        self.parent.after(milliseconds, self.plotit)

class Simulator(tk.Tk):
    """ Class instance of main/root window. Mainly responsible for showing the
        core components and setting other properties related to the main window."""
    def __init__(self):
        tk.Tk.__init__(self)

        # Set title and screen resolutions
        tk.Tk.wm_title(self, 'XPM Simulator')
        tk.Tk.minsize(self, width=640, height=320)
        # Optional TODO: Set a custom icon for the XPM application 
        # tk.Tk.iconbitmap(self, default="[example].ico")

        # Show window and control bar
        self.graph = GraphFrame(self)
        self.graph.pack(side='top', fill='both', expand=True)

# Set constants and global variables
HOST = 'localhost'
PORT = 5022
root = Simulator()
preamble = root.graph.graph.preamble
event = root.graph.graph.event

web_server = HTTPServer((HOST, PORT), MyServer)
print(f'Server started http://{HOST}:{PORT}')

def target_web_server():
    """ Target function that allows the server to run forever until user presses Ctrl-C. This
        will be used in a thread that handles the server itself. """
    try:
        web_server.serve_forever()
    except KeyboardInterrupt:
        pass

    web_server.server_close()
    print('Server stopped.')

# Create thread for server
server_thread = threading.Thread(target=target_web_server, daemon=True)
server_thread.start()

root.graph.plotit()
root.mainloop()
