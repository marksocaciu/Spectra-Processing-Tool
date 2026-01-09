from tkinter import  ttk, filedialog, messagebox
import pandas as pd
import re
import csv
import os
import matplotlib.pyplot as plt
import tkinter as tk
from scipy.signal import find_peaks
from sklearn import preprocessing
import numpy as np
from collections import defaultdict
import pywt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


class Measurement:
    def __init__(self, name, value, std, wave, alias):
        self.name = name
        self.value = value
        self.std = std
        self.wave = wave
        self.alias = alias
        self.afw=[]
        self.af=[]
        self.af_std=[]
    
    def find_max(self):
        try:
            max_index = np.argmax(self.value)
        except ValueError:
            return 0,0,0
        max_index = np.argmax(self.value)
        return self.value[max_index], self.wave[max_index], self.std[max_index]

class Fluorophor:
    def __init__(self,name,auto):
        self.name = name
        self.auto = auto

def extract_nume_label(filename):
    """Extracts the NUME_LABEL from a filename."""
    match = re.search(r"\_\_\[\[(.*?)\]\]\_", filename)
    if match:
        return match.group(1)
    else:
        return None

def is_control(filename):
    match = re.search(r"\_\_\[\[(.*?)\]\]\_", filename)
    if match:
        return match.group(1)
    else:
        return None

# Apply an optimal neighborhood filter (Savitzky-Golay filter for smoothing)
def savgol(data, window_size, poly_order):
    return savgol_filter(data, window_size, poly_order)

# Wavelet Denoising
def wavelet(data, wavelet='db4', level=None):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    threshold = np.std(coeffs[-1])  # Thresholding noise
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(data)]

def clear_input_files():
    input_files_var.set("")
    input_files_entry.delete(0, tk.END)  # Clear the entry field
    if len(alias_entries.values())> 0:
        for entry in alias_entries.values():
            entry.destroy()
        alias_entries.clear()
        submit_button.grid(row=7, column=0, columnspan=3, pady=10)

def clear_autofluorescence_files():
    autofluorescence_var.set("")
    autofluorescence_entry.config(state="disabled")
    autofluorescence_button.config(state="disabled")
    autofluorescence_entry.delete(0, tk.END)  # Clear the entry field

def delete_alias_entry(entry):
    # Remove the entry from the grid and the dictionary
    entry.destroy()
    print("the entry is ",entry.name)
    alias_entries.pop(entry.name)
    submit_button.grid(row=7+len(alias_entries), column=0, columnspan=3, pady=10)
    container.update_scrollbar_visibility(threshold_height=400)

    # Remove the entry from the input files variable
    input_files = input_files_var.get().split(',')
    print("input files before: ", input_files)
    input_files = [x for x in input_files if entry.name not in x.strip()]
    print("input files after: ", input_files)
    input_files_var.set(",".join(input_files)[:-2])
    print("input files after set: ", input_files_var.get())
    # file_paths = [x.strip() for x in input_files_var.get().split(',') if x.strip() != ""]
    # print("filepaths: ",file_paths)

# Select input files
def select_input_files():
    files = filedialog.askopenfilenames(title="Select Input Files",filetypes=[("Text files", "*.txt")])
    input_files_var.set( input_files_var.get()+ ", "+", ".join(files))
    file_paths = [x.strip() for x in input_files_var.get().split(',') if x.strip() != ""]
    print("filepaths: ",file_paths)
    
    
    # Group files by their basename prefix
    grouped_files = defaultdict(list)
    for file_path in file_paths:
        basename = "".join(file_path.split('/')[-1]).split('.')[0] # Extract basename (e.g., 'a', 'b')
        if re.search(r"_\d$", basename):
            basename = basename[:basename.rfind('_')]
        grouped_files[basename].append(file_path)
    try:grouped_files.pop('')  # Remove empty keys if any
    except: pass
    
    if len(alias_entries)> 0:
        for entry in alias_entries.values():
            entry.destroy()
        alias_entries.clear()

    for i,group in enumerate(grouped_files.keys()):
        entry_label = tk.Label(frame, text=f"{group.split('/')[-1]}: ")
        entry_label.grid(row=6+i, column=0, padx=10, pady=5, sticky="e")
        entry_entry = tk.Entry(frame, width=80)
        entry_entry.grid(row=6+i, column=1, padx=10, pady=5, columnspan=2)
        entry_delete = ttk.Button(frame, text="Delete")
        entry_delete.grid(row=6+i, column=3, padx=10, pady=5, sticky="w")
        
        # Store entry widget in dictionary with filename as key
        entry = AliasEntry(entry_label, entry_entry, entry_delete, group)
        entry.button.config(command=lambda e=entry: delete_alias_entry(e)) # Pass the newly created entry object to the command
        alias_entries[group] = entry

    submit_button.grid(row=7+len(grouped_files), column=0, columnspan=3, pady=10)
    container.update_scrollbar_visibility(threshold_height=400)
    on_resize()


# Select autofluorescence files
def select_autofluorescence_files():
    if fluorophors[solution_var.get()] == "True":
        files = filedialog.askopenfilenames(title="Select Autofluorescence Files",filetypes=[("Text files", "*.txt")])
        autofluorescence_var.set(", ".join(files))

# Update autofluorescence field based on solution selection
def on_solution_change(event):
    print(f"solution: {solution_var.get()} fd {fluorophors[solution_var.get()]}")
    if solution_var.get()!="":
        # Enable or disable autofluorescence field based on solution selection
        if fluorophors[solution_var.get()] == "True":
            autofluorescence_button.config(state="normal")
            autofluorescence_entry.config(state="readonly")
        else:
            autofluorescence_button.config(state="disabled")
            autofluorescence_entry.config(state="disabled")
        
# Save new fluorophor
def save_new(new_name_var,check_var):
    # Append-adds at last
    documents_path = os.path.join(os.path.expanduser("~"), "Documents")
    folder_path = os.path.join(documents_path, "Spectra Processing\Executable")
    if os.path.exists(folder_path):
        file1 = open(f"{folder_path}/fluorophor_data.txt", "a")  # append mode
        file1.write(f"\n{new_name_var.get()}, {check_var.get()} ")
        file1.close()

# Add new fluorophor
def add_fluorophor():
    # Toplevel object which will 
    # be treated as a new window
    newWindow = tk.Toplevel(root)
    newWindow.title("Add new Fluorophor")
    # Variables
    new_name_var = tk.StringVar()
    check_var = tk.BooleanVar()
 
    # Name field
    tk.Label(newWindow, text="Fluorophor Name:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    name_entry = tk.Entry(newWindow, textvariable=new_name_var, width=40)
    name_entry.grid(row=0, column=1, padx=10, pady=5, columnspan=2)
    
    # Checkbox for autofluorescence property
    checkbox = tk.Checkbutton(newWindow, text="Autofluorescence ",variable=check_var)
    checkbox.grid(row=1, column=0, padx=10, pady=5)
    
    # Submit button
    new_submit_button = ttk.Button(newWindow, text="Save", command=lambda: save_new(new_name_var,check_var))
    new_submit_button.grid(row=2, column=0, columnspan=3, pady=10)

class pHMeasurement:
    def __init__(self, name, alias, ph):
        self.name = name
        self.alias = alias
        self.ph = ph
    
# Select measurements to be compared 
def select_measurement_files(newWindow: tk.Toplevel,alias: dict, new_submit_button: ttk.Button, new_name_var: tk.StringVar):
    files = filedialog.askopenfilenames(title="Select Measurement Files",filetypes=[("CSV files", "*.csv")])
    file_paths = [x.strip() for x in files if x.strip() != ""]
    new_name_var.set(", ".join(files))
    
    for i,file_path in enumerate(file_paths):
        basename = "".join(file_path.split('/')[-1]).split('.')[0] # Extract basename (e.g., 'a', 'b')
        if re.search(r"_\d$", basename):
            basename = basename[:basename.rfind('_')]
        entry_label = tk.Label(newWindow, text=f"{basename}: ")
        entry_label.grid(row=3+i, column=0, padx=10, pady=5, sticky="e")
        entry_entry = tk.Entry(newWindow, width=80)
        entry_entry.grid(row=3+i, column=1, padx=10, pady=5, columnspan=2)
        entry_ph_label = tk.Label(newWindow, text="pH: ")
        entry_ph_label.grid(row=3+i, column=3, padx=10, pady=5, sticky="e")
        entry_ph_entry = tk.Entry(newWindow, width=10)
        entry_ph_entry.insert(0, "7.0")
        entry_ph_entry.grid(row=3+i, column=4, padx=10, pady=5)
        alias[basename] = pHMeasurement(file_path,entry_entry,entry_ph_entry)
    
    # new_submit_button.grid(row=3+len(files), column=0, pady=10)

def save_ph(newWindow: tk.Toplevel, alias: dict, output_name: str = "output"):
    base_dir = find_folder("Spectra Processing")
    base_dir=os.path.join(base_dir, output_name)
    try:
        os.mkdir(base_dir)
    except OSError:
        pass
    
    # aliases = {filename: entry.entry.get() for filename, entry in alias.items()}
    # iterate over the dictionary 
    max_norom=1
    md = defaultdict(list)
    for key, value in alias.items():
        print(f"{key} : {value.alias.get()} {value.ph.get()}")
        with open(value.name, "r") as f:
            lines = f.readlines()
            data_lines = lines[1:]
            if value.alias.get() == "normalize": max_norom = float(data_lines[0].strip().split(",")[2])
            for line in data_lines:
                data = line.strip().split(",")
                md[data[0]].append((data[2],data[3],value.ph.get()))
    
    # Add a button to select interpolation method
    interpolation_methods = ["none","polynomial_1st", "polynomial_2nd", "polynomial_3rd", "polynomial_4th", "polynomial_5th", "spline","logarithmic","exponential"]
    fits = {}
    displacement = 0
    for i in md.values():
        if len(i) > displacement:
            displacement = len(i)
            
    displacement += 3
    for i,key in  enumerate(md.keys()):
        tk.Label(newWindow, text=f"Fit {key}: ").grid(row=displacement + i, column=0, padx=10, pady=5, sticky="e")
        fit_dropdown = ttk.Combobox(newWindow, textvariable= tk.StringVar(), state="readonly")
        fit_dropdown["values"] = interpolation_methods
        fit_dropdown.grid(row=displacement + i, column=1, padx=10, pady=5)
        fit_dropdown.current(0)  # Set default value
        fits[key] = fit_dropdown
    tk.Button(newWindow, text="continue",command=lambda: save_ph_cont(md, fits, max_norom,base_dir, output_name)).grid(row=displacement+1+len(md.keys()), column=0, padx=10, pady=5)
    newWindow.update_idletasks()  # Update the window to ensure the new widgets are displayed

def save_ph_cont(md: dict, fits: dict, max_norom:float, base_dir:str, output_name: str = "output"):
    plt.figure(figsize=(20,14))
    f = open(f"{base_dir}/interpol_{output_name}.txt","w")
    # Iterate over the dictionary and plot the data
    for key, value in md.items():
        intensity = [float(x[0])/max_norom for x in value]
        std = [float(x[1])/max_norom for x in value]
        ph = [float(x[2]) for x in value]
        print(f"the values are {intensity} {std} {ph}")
        s = plt.scatter(ph,intensity,label=key)
        plt.errorbar(ph, intensity, yerr=std, fmt='o',  markersize=8, capsize=10)
        
        fit = fits[key].get()
        # setup the fit
        if fit == "polynomial_1st":
            # z = np.polyfit(ph, intensity, 1)
            z = curve_fit(lambda x, a, b: a*x + b, ph, intensity)
            f.write(f"{key} {z[0][0]} {z[0][1]}\n")
            plt.plot(ph, z[0][0]*np.array(ph) + z[0][1], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "polynomial_2nd":
            # z = np.polyfit(ph, intensity, 2)
            z = curve_fit(lambda x, a, b, c: a*x**2 + b*x + c, ph, intensity)
            f.write(f"{key} {z[0][0]} {z[0][1]} {z[0][2]}\n")
            plt.plot(ph, z[0][0]*np.array(ph)**2 + z[0][1]*np.array(ph) + z[0][2], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "polynomial_3rd":
            # z = np.polyfit(ph, intensity, 3)
            z = curve_fit(lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d, ph, intensity)
            f.write(f"{key} {z[0][0]} {z[0][1]} {z[0][2]} {z[0][3]}\n")
            plt.plot(ph, z[0][0]*np.array(ph)**3 + z[0][1]*np.array(ph)**2 + z[0][2]*np.array(ph) + z[0][3], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "polynomial_4th":
            # z = np.polyfit(ph, intensity, 4)
            z = curve_fit(lambda x, a, b, c, d, e: a*x**4 + b*x**3 + c*x**2 + d*x + e, ph, intensity)
            f.write(f"{key} {z[0][0]} {z[0][1]} {z[0][2]} {z[0][3]} {z[0][4]}\n")
            plt.plot(ph, z[0][0]*np.array(ph)**4 + z[0][1]*np.array(ph)**3 + z[0][2]*np.array(ph)**2 + z[0][3]*np.array(ph) + z[0][4], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "polynomial_5th":
            # z = np.polyfit(ph, intensity, 5)
            z = curve_fit(lambda x, a, b, c, d, e, f: a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f, ph, intensity)
            f.write(f"{key} {z[0][0]} {z[0][1]} {z[0][2]} {z[0][3]} {z[0][4]} {z[0][5]}\n")
            plt.plot(ph, z[0][0]*np.array(ph)**5 + z[0][1]*np.array(ph)**4 + z[0][2]*np.array(ph)**3 + z[0][3]*np.array(ph)**2 + z[0][4]*np.array(ph) + z[0][5], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "spline":
            pass
        elif fit == "logarithmic":
            # z = np.polyfit(np.log(ph), intensity, 1)
            z = curve_fit(lambda x, a, b: a*np.log(x) + b, ph, intensity)
            f.write(f"{key} {z[0][0]} {z[0][1]}\n")
            plt.plot(ph, z[0][0]*np.log(np.array(ph)) + z[0][1], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "exponential":
            # z = np.polyfit(ph, np.log(intensity), 1)
            z = curve_fit(lambda x, a, b: a*np.exp(b*x), ph, intensity)
            f.write(f"{key} {z[0][0]} {z[0][1]}\n")
            plt.plot(ph, z[0][0]*np.exp(z[0][1]*np.array(ph)), label=f"{key} fit", color=s.get_facecolor())
        else:
            pass
        
        plt.xlabel("pH", fontsize = 30)
        plt.ylabel("Intensity [a.u]", fontsize = 30)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.tick_params(axis='both', direction='in')
        leg = plt.legend(fontsize=32, frameon=True, framealpha=0.5, edgecolor="black")
        for l in leg.get_lines():
            l.set_linewidth(4.5)
    plt.savefig(f"{base_dir}/pH_plot_{output_name}.png")
    f.close()


def pH_plot():
    # Toplevel object which will 
    # be treated as a new window
    newWindow = tk.Toplevel(root)
    newWindow.title("Create pH plot")

    # Variables
    new_name_var = tk.StringVar()
    name_var = tk.StringVar()
    solution_var = tk.StringVar()
    alias = {}
 
    # Name field
    tk.Label(newWindow, text="Measurements to be compared:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    ttk.Button(newWindow, text="Select Files",command=lambda: select_measurement_files(newWindow,alias, new_submit_button, new_name_var)).grid(row=0, column=2, padx=10, pady=5)
    name_entry = tk.Entry(newWindow, textvariable=new_name_var, width=40, state="readonly")
    name_entry.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(newWindow, text="Output folder: ").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    output_name = tk.Entry(newWindow, textvariable=name_var, width=40)
    output_name.grid(row=1, column=1, padx=10, pady=5)

    # Add a button to select interpolation method
    # interpolation_methods = ["none","polynomial_1st", "polynomial_2nd", "polynomial_3rd", "polynomial_4th", "polynomial_5th", "spline","logarithmic","exponential"]
    # tk.Label(newWindow, text="Fit: ").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    # fit_dropdown = ttk.Combobox(newWindow, textvariable=solution_var, state="readonly")
    # fit_dropdown["values"] = interpolation_methods
    # fit_dropdown.grid(row=2, column=1, padx=10, pady=5)
    # fit_dropdown.current(0)  # Set default value
    
    # Submit button
    new_submit_button = ttk.Button(newWindow, text="Save", command=lambda: save_ph(newWindow,alias, output_name.get()))
    new_submit_button.grid(row=1, column=2, pady=10)


# Find the folder with the given name in the user's Pictures directory
def find_folder(folder_name):
  documents_path = os.path.join(os.path.expanduser("~"), "Pictures")
  folder_path = os.path.join(documents_path, folder_name)
  if os.path.exists(folder_path):
    return folder_path
  else:
    return None

# Retrieve data from UI
def submit():
    solution = solution_var.get()
    input_files = input_files_var.get()
    autofluorescence_files = autofluorescence_var.get()
    min_spectra = min_spectra_var.get()
    max_spectra = max_spectra_var.get()
    output_name = name_var.get()
    if output_name == "": output_name = "output"
    if min_spectra == "": min_spectra = 0.0
    else: min_spectra = float(min_spectra)
    if max_spectra == "": max_spectra = 0.0
    else: max_spectra = float(max_spectra)
    aliases = {filename: entry.entry.get() for filename, entry in alias_entries.items()}
    peaks = peak_display_var.get()
    normalize = normalize_var.get()
    normalize_all = normalize_all_var.get()
    denoise = denoise_var.get()

    base_dir = find_folder("Spectra Processing")
    base_dir=os.path.join(base_dir, output_name)
    try:
        os.mkdir(base_dir)
    except OSError:
        pass
    
    process_files(solution,input_files,autofluorescence_files,min_spectra, max_spectra,output_name,base_dir,aliases,peaks,normalize, normalize_all,denoise)

# Open a Tk window to select multiple txt files
def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])
    return file_paths

# Function to normalize a spectrum
def normalize_spectrum(intensities):
    max_intensity = max(intensities)
    if max_intensity == 0:
        return intensities
    return [intensity / max_intensity for intensity in intensities]

# Process files and perform tasks
def process_files(solution: str, input_files: str, autofluorescence_files: str, min_spectra: float, max_spectra: float, output_name: str, basedir: str, aliases: list, show_peaks: bool, normalize: bool, normalize_all: bool, denoise: bool):
    # Split the input files string into a list of file paths
    file_paths = input_files.split(',')
    file_paths = [x.strip() for x in file_paths if x.strip() != ""]
    file_paths_af = autofluorescence_files.split(',')
    file_paths_af = [x.strip() for x in file_paths_af]
    
    # Group files by their basename prefix
    grouped_files = defaultdict(list)
    grouped_files_baseline = defaultdict(list)
    for file_path in file_paths:
        basename = "".join(file_path.split('/')[-1]).split('.')[0] # Extract basename (e.g., 'a', 'b')
        if re.search(r"_\d$", basename):
            basename = basename[:basename.rfind('_')]
        grouped_files[basename].append(file_path)
    print("my gruofiles are:\n",grouped_files)
        
    for file_path in file_paths_af:
        basename = "".join(file_path.split('/')[-1]).split('.')[0] # Extract basename (e.g., 'a', 'b')
        if re.search(r"_\d$", basename):
            basename = basename[:basename.rfind('_')]
        grouped_files_baseline[basename].append(file_path)
    grouped_files_baseline.pop('', None)

    headers = []
    data = []
    max_ctr_nom = 1
    for group, files in grouped_files.items():
        print(f"Processing group: {group}")
        headers.append(group)
        wave = []
        samples = []
        skip_lines = 8
        delimiter = ';'
        skip_footer = 0
        if solution == "SERS_BWTeK": skip_lines = 1
        elif solution == "SERS_ReniShaw": 
            skip_lines = 1
            delimiter = '	'
        elif solution == "SERS_Avantes": skip_lines = 0
        elif solution == "FT-IR": 
            skip_lines = 19
            delimiter = '	'
            skip_footer = 42
        elif solution == "UV-Vis": 
            skip_lines = 19
            delimiter = '	'
            skip_footer = 42
        for file_path in files:
            # Read the file
            file_data = np.genfromtxt(file_path, encoding="UTF-8", dtype = np.float64, skip_header = skip_lines, delimiter = delimiter, skip_footer=skip_footer)
            # Extract data
            wave.append(file_data[:,0])
            sample = file_data[:,1]
            samples.append(sample)

        # Convert the list of lists to a NumPy array
        array = np.array(samples)
        arrayw = np.array(wave)
        # print(array,arrayw)

        # Calculate the mean and standard deviation along the first axis (rows)
        means = np.mean(array, axis=0)
        meansw = np.mean(arrayw, axis=0)
        stds = np.std(array, axis=0)

        # Combine the means and standard deviations into a list of tuples
        result = Measurement(group,means,stds,meansw,aliases[group])
        print("current group is ",result.alias)
        
        # check for autofluorescence
        print("af groups: ",len(grouped_files_baseline.keys()))
        if len(grouped_files_baseline.keys())>0: 
            gr = list(grouped_files_baseline.keys())[0]
            fl = grouped_files_baseline[gr]
            w = []
            s = []
            # print("the groups are ",gr,group)
            for file_path in fl:
                # Read the file and extract the second column using numpy
                file_data = np.genfromtxt(file_path, encoding="UTF-8", dtype=np.float64, skip_header=8,delimiter=';')

                # Extract data from the second column
                w.append(file_data[:,0])
                s.append(file_data[:,1])
                
            # Convert the list of lists to a NumPy array
            array = np.array(s)
            arrayw = np.array(w)
            result.af = np.mean(array, axis=0)
            result.afw = np.mean(arrayw, axis=0)
            result.af_std = np.std(array, axis=0)
        
        if len(result.af) == 0:
            if solution in ["SERS_BWTeK", "SERS_Avantes","SERS_ReniShaw"]:
                result.afw = result.wave
                result.af = np.polyval(np.polyfit(result.wave, result.value, 1), result.wave)
                result.af_std = np.zeros(len(result.value))
            else:
                result.afw = result.wave
                result.af = np.zeros(len(result.value))
                result.af_std = np.zeros(len(result.value))
        data.append(result)
    
    for measurement in data:
        if measurement.wave[0] > measurement.wave[-1]:
            measurement.wave = measurement.wave[::-1]
            measurement.value = measurement.value[::-1]
            measurement.std = measurement.std[::-1]
            measurement.af = measurement.af[::-1]
            measurement.afw = measurement.afw[::-1]
            measurement.af_std = measurement.af_std[::-1]
    
    # Find indexes for plotting
    for measurement in data:
        index_start = 0
        index_end = len(measurement.value)-1
        if min_spectra != 0.0:
            index_start = (np.abs(measurement.wave - min_spectra)).argmin()
        if max_spectra != 0.0:
            index_end = (np.abs(measurement.wave - max_spectra)).argmin()
        
        measurement.value = measurement.value[index_start:index_end]
        measurement.wave = measurement.wave[index_start:index_end]
        measurement.std = measurement.std[index_start:index_end]
        index_start = 0
        index_end = len(measurement.af)-1
        if min_spectra != 0.0:
            index_start = (np.abs(measurement.afw - min_spectra)).argmin()
        if max_spectra != 0.0:
            index_end = (np.abs(measurement.afw - max_spectra)).argmin()
        measurement.afw = measurement.afw[index_start:index_end]
        measurement.af = measurement.af[index_start:index_end]
        measurement.af_std = measurement.af_std[index_start:index_end]
        
    # Subtract autofluorescence and adjust std
    control = False
    for measurement in data:
        measurement.value -= measurement.af
        measurement.std += measurement.af_std
        if "_contr_" in measurement.name or "_control_" in measurement.name or "_ctr_" in measurement.name or "_ctrl_" in measurement.name: 
            max_ctr_nom,_,_ = measurement.find_max()
            control = True
    max_control = max_ctr_nom
    
    # Denoise the data
    if denoise:
        for measurement in data:
            # print(f"{type(measurement.std)} {type(np.std(wavelet(measurement.value)))}")
            measurement.value = wavelet(measurement.value)

    # Normalize the data
    if solution in ["SERS_BWTeK", "SERS_Avantes","SERS_ReniShaw"]:
        for measurement in data:
            intensities_array = np.array(measurement.value).reshape(-1, 1)  # Reshape to a column vector
            measurement.value = preprocessing.normalize(intensities_array, axis=0).flatten()
            intensities_array = np.array(measurement.std).reshape(-1, 1)  # Reshape to a column vector
            measurement.std = preprocessing.normalize(intensities_array, axis=0).flatten()
    # elif solution in ["UV-Vis","FT-IR"]: pass
    elif normalize:
        if not control:
            for measurement in data:
                maxi,_,_ = measurement.find_max()
                if maxi > max_ctr_nom: max_ctr_nom = maxi
        for measurement in data: 
            measurement.value = np.array([float(x)/max_ctr_nom for x in measurement.value])
            measurement.std = np.array([float(x)/max_ctr_nom for x in measurement.std])
    elif normalize_all:
        for measurement in data:
            max_ctr_nom,_,_ = measurement.find_max()
            measurement.value = np.array([float(x)/max_ctr_nom for x in measurement.value])
            measurement.std = np.array([float(x)/max_ctr_nom for x in measurement.std])
    
    # Transpose and export data to output.xls
    # Create a list of dictionaries to store the data for each measurement
    output = []
    for measurement in data:
        output.append({
        'Name': measurement.alias,
        **dict(zip(measurement.wave, measurement.value))
        })
        if normalize:
            output.append({
            'Name': f"{measurement.alias}_raw",
            **dict(zip(measurement.wave, measurement.std * max_ctr_nom))
            })
        output.append({
        'Name': f"{measurement.alias}_std",
        **dict(zip(measurement.wave, measurement.std))
        })

    # Create a DataFrame from the data
    df = pd.DataFrame(output)

    # Write the DataFrame to an Excel file
    df.to_excel(f'{basedir}/output_{output_name}.xlsx', index=False)
    print(f"Data exported to {basedir}/output_{output_name}.xlsx")

    # # Save relevant data to a CSV file
    # with open(f'{basedir}/max_values_{output_name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
    #     fieldnames = ['Name', 'Max Wave', 'Max Value', 'Max Std']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     # save data
    #     for measurement in data:
    #         max_value, max_wave, max_std = measurement.find_max()
    #         writer.writerow({'Name': measurement.alias,
    #                       'Max Value': max_value,
    #                       'Max Wave': max_wave,
    #                       'Max Std': max_std})
    
    # Plot the data
    maxlen=0
    cm = plt.cm.get_cmap('tab10')
    if len(data)>10:
        cm = plt.cm.get_cmap('tab20')
    elif len(data) > 20:
        cm = plt.cm.get_cmap('hsv')
    plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.hsv(np.linspace(0,1,len(data))))
    for d in data: 
        if len(d.wave)>maxlen: maxlen=len(d.wave)
    cumulative_height = np.zeros(maxlen)
    max_deplacement = 0
    colors = []
    plt.figure(figsize=(20,14))
    for measurement in data:
        try:
            line = plt.plot(measurement.wave, measurement.value+cumulative_height[:len(measurement.wave)], label=measurement.alias, linewidth=2.5)
            # print(f"line type {type(line)} \n the line is {line[0].get_color()}")
            color = line[0].get_color()
            colors.append(color)
            if show_peaks:
                # peaks = np.argmax(measurement.value)
                width = 30 # width of the peak
                if solution in ["SERS_BWTeK", "SERS_Avantes","SERS_ReniShaw"]: width = 45
                elif solution == "FT-IR": width = 25
                elif solution == "UV-Vis": width = 15
                peaks, _  = find_peaks(measurement.value, width=width) # peaks is ndarray
                label = [str(round(f)) for f in list(measurement.wave[peaks])]
                h = [measurement.value[p]+cumulative_height[p] for p in peaks]
                # print(f"the peaks are {peaks} {label}")
                # print(f"the lenghts are {len(measurement.wave[peaks])} {len(h)} {len(label)}")
                plt.scatter(measurement.wave[peaks], h, color=color)
                for i,p in enumerate(peaks):
                    plt.text(measurement.wave[p], h[i], label[i], fontsize=24, color=color)
        except ValueError:
            messagebox.showerror("Error", "Please adjust the domain to be plotted !")
        
        # Plot the upper and lower bounds (mean +/- standard deviation)
        try:
            plt.fill_between(measurement.wave, measurement.value + measurement.std + cumulative_height[:len(measurement.wave)], measurement.value - measurement.std + cumulative_height[:len(measurement.wave)], alpha=0.5)
        except ValueError:
            if len(measurement.value) > len(cumulative_height):
                cumulative_height = np.full(len(measurement.value),cumulative_height[0])
            plt.fill_between(measurement.wave, measurement.value + measurement.std + cumulative_height[:len(measurement.wave)], measurement.value - measurement.std + cumulative_height[:len(measurement.wave)], alpha=0.5)

        if solution in ["SERS_BWTeK","FT-IR", "SERS_Avantes","SERS_ReniShaw"]: 
            if max_deplacement < max(measurement.value):
                max_deplacement = max(measurement.value)
            cumulative_height += 1.2 * max_deplacement  # Update cumulative height for next spectrum
    
    if solution in ["SERS_BWTeK", "SERS_Avantes","SERS_ReniShaw"]:
        plt.xlabel("Raman Shift [cm-1]", fontsize = 35)
        plt.ylabel("Normalized intensity [a.u]", fontsize = 35)
        # plt.gca().set_yticklabels([])
    elif solution == "UV-Vis":
        plt.xlabel("Wavelength [nm]", fontsize = 35)
        plt.ylabel("Absorbance [a.u]", fontsize = 35)
        if normalize or normalize_all:
            plt.ylabel("Normalized intensity [a.u]", fontsize = 35)
    elif solution == "FT-IR":
        plt.xlabel("Wavenumber [cm-1]", fontsize = 35)
        plt.ylabel("Transmittance [a.u]", fontsize = 35)
        plt.yticks([])
    else:
        plt.xlabel("Wavelength [nm]", fontsize = 35)
        plt.ylabel("Normalized intensity [a.u]", fontsize = 35)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # plt.tight_layout()
    plt.tick_params(axis='both', direction='in')
    leg = plt.legend(fontsize=28, frameon=True, framealpha=0.5, edgecolor="black")
    # leg = plt.legend(fontsize=28, loc='upper right', frameon=True, framealpha=0.5, edgecolor="black")
    for l in leg.get_lines():
        l.set_linewidth(2.5)

    # plt.legend(fontsize=18, loc='upper right', frameon=True, framealpha=0.5, edgecolor="black",bbox_to_anchor=(1.1, 1))
    # plt.show()
    # plt.savefig(f"{basedir}/plot_{output_name}.png", bbox_inches="tight")
    plt.savefig(f"{basedir}/plot_{output_name}.png")
    plt.show()

    plt.close()

    # Save relevant data to a CSV file
    with open(f'{basedir}/max_values_{output_name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Name', 'Max Wave', 'Max Value', 'Max Std', 'Color']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # save data
        for measurement,color in zip(data,colors):
            max_value, max_wave, max_std = measurement.find_max()
            writer.writerow({'Name': measurement.alias,
                          'Max Value': max_value,
                          'Max Wave': max_wave,
                          'Max Std': max_std,
                          'Color': color})

    # if solution not in ("UV-Vis","FT-IR","SERS_BWTeK"):
    #     plt.clf()
    #     plt.cla()
    #     cumulative_height = np.zeros(len(data[0].value))
    #     # Plot unnormalized data
    #     plt.figure(figsize=(20,14))
    #     for measurement in data:
    #         measurement.value = measurement.value * max_ctr_nom
    #         measurement.std = measurement.std * max_ctr_nom
    #         # Plot the mean values
    #         try:
    #             plt.plot(measurement.wave, measurement.value+cumulative_height[:len(measurement.wave)], label=measurement.alias)
    #         except ValueError:
    #             messagebox.showerror("Error", "Please adjust the domain to be plotted !")
    #         # Plot the upper and lower bounds (mean +/- standard deviation)
    #         try:
    #             plt.fill_between(measurement.wave, measurement.value + measurement.std + cumulative_height[:len(measurement.wave)], measurement.value - measurement.std + cumulative_height[:len(measurement.wave)], alpha=0.5)
    #         except ValueError:
    #             if len(measurement.value) > len(cumulative_height):
    #                 cumulative_height = np.full(len(measurement.value),cumulative_height[0])
    #             plt.fill_between(measurement.wave, measurement.value + measurement.std + cumulative_height[:len(measurement.wave)], measurement.value - measurement.std + cumulative_height[:len(measurement.wave)], alpha=0.5)


def read_fluorophor(l):
    documents_path = os.path.join(os.path.expanduser("~"), "Documents")
    folder_path = os.path.join(documents_path, "Spectra Processing","Executable")
    if os.path.exists(folder_path):
        with open(f"{folder_path}/fluorophor_data.txt", "r") as file:
            for line in file:
                l[line.strip().split(',')[0].strip()] = line.strip().split(',')[1].strip()
    else: print("help!")

def updatelist():
    read_fluorophor(fluorophors)
    solution_dropdown["values"]=[f for f in fluorophors.keys()]

def replace_dots_in_filenames(directory):
    for root_dir, _, files in os.walk(directory):
        for filename in files:
            old_path = os.path.join(root_dir, filename)
            # Split filename and extension
            name, ext = os.path.splitext(filename)
            
            # Check for version suffix (e.g., p1.1.txt)
            match = re.search("([0-9].[0-9])$", name)
            print(name,match)
            if match:
                base_name = name[:match.start()]  # Extract the part before the version suffix
                new_name = base_name.replace('.', '_')
                new_name = new_name.replace(' ', '_')
                new_name += match.group(1)
                new_name += ext
            else:
                new_name = name.replace('.', '_')
                new_name = name.replace(' ', '_')
                new_name += ext
            
            new_path = os.path.join(root_dir, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f'Renamed: "{old_path}" -> "{new_path}"')

def rename_files():
    folder_path = filedialog.askdirectory()
    if folder_path:
        replace_dots_in_filenames(folder_path)
    
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        self.canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas
        self.scrollbar = scrollbar

    def update_scrollbar_visibility(self, threshold_height):
        self.update_idletasks()
        content_height = self.scrollable_frame.winfo_height()
        if content_height > threshold_height:
            self.scrollbar.pack(side="right", fill="y")
        else:
            self.scrollbar.pack_forget()

class AliasEntry():
    def __init__(self, label: tk.Label, entry: tk.Entry, button: ttk.Button, name: str):
        self.label = label
        self.entry = entry
        self.button = button
        self.name = name
    
    def destroy(self):
        self.label.destroy()
        self.entry.destroy()
        self.button.destroy()
    
def on_resize():
        container.update_scrollbar_visibility(threshold_height=400)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Spectra processing")
    root.geometry("1400x850")  # Starting size

    container = ScrollableFrame(root)
    container.pack(fill="both", expand=True)

    frame = container.scrollable_frame

    # Variables
    solution_var = tk.StringVar()
    input_files_var = tk.StringVar()
    autofluorescence_var = tk.StringVar()
    min_spectra_var = tk.StringVar()
    max_spectra_var = tk.StringVar()
    name_var = tk.StringVar()
    peak_display_var = tk.BooleanVar()
    normalize_var = tk.BooleanVar()
    normalize_all_var = tk.BooleanVar()
    normalize_by_group_var = tk.BooleanVar()
    denoise_var = tk.BooleanVar()
    fluorophors = {"": "False"}
    alias_entries = {}

    read_fluorophor(fluorophors)
    
    # Solution dropdown
    tk.Label(frame, text="Spectra:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    solution_dropdown = ttk.Combobox(frame, textvariable=solution_var, state="readonly", postcommand=updatelist)
    solution_dropdown["values"] = [f for f in fluorophors.keys()]
    solution_dropdown.grid(row=0, column=1, padx=10, pady=5)
    solution_dropdown.bind("<<ComboboxSelected>>", on_solution_change)
    
    # Insert new Fluorophor
    new_spectra_button = ttk.Button(frame, text="Add new Spectrometer", command=add_fluorophor)
    new_spectra_button.grid(row=0, column=2, pady=10)

    # Rename files to naming convention
    rename_files_button = ttk.Button(frame, text="Rename Files", command=rename_files)
    rename_files_button.grid(row=0, column=3, pady=10)

    # Input files field
    tk.Label(frame, text="Input Files:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    input_files_button = ttk.Button(frame, text="Select Files", command=select_input_files)
    # input_files_button = ttk.Button(frame, text="Select Files", command=select_input_files, postcommand=get_aliases)
    input_files_button.grid(row=1, column=1, padx=10, pady=5)
    input_files_entry = tk.Entry(frame, textvariable=input_files_var, state="readonly", width=40)
    input_files_entry.grid(row=1, column=2, padx=10, pady=5)
    input_files_button_clear = ttk.Button(frame, text="Clear", command=clear_input_files)
    input_files_button_clear.grid(row=1, column=3, padx=10, pady=5)

    # Autofluorescence field
    tk.Label(frame, text="Autofluorescence:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    autofluorescence_button = ttk.Button(frame, text="Select Files", command=select_autofluorescence_files)
    autofluorescence_button.grid(row=2, column=1, padx=10, pady=5)
    autofluorescence_entry = tk.Entry(frame, textvariable=autofluorescence_var, state="disable", width=40)
    autofluorescence_entry.grid(row=2, column=2, padx=10, pady=5)
    autofluorescence_files_button_clear = ttk.Button(frame, text="Clear", command=clear_autofluorescence_files)
    autofluorescence_files_button_clear.grid(row=2, column=3, padx=10, pady=5)

    # Spectra limits
    tk.Label(frame, text="Spectra Min:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    min_spectra_entry = tk.Entry(frame, textvariable=min_spectra_var, width=10)
    min_spectra_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

    tk.Label(frame, text="Spectra Max:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
    max_spectra_entry = tk.Entry(frame, textvariable=max_spectra_var, width=10)
    max_spectra_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

    # Peak display checkbox
    peak_display_var = tk.BooleanVar()
    peak_display_checkbox = tk.Checkbutton(frame, text="Show Peak Values", variable=peak_display_var)
    peak_display_checkbox.grid(row=3, column=2, padx=10, pady=5, sticky="w")

    # Normalize checkbox
    normalize_var = tk.BooleanVar()
    normalize_checkbox = tk.Checkbutton(frame, text="Normalize", variable=normalize_var)
    normalize_checkbox.grid(row=4, column=2, padx=10, pady=5, sticky="w")

    # Normalize_all_to_one checkbox
    normalize_all_var = tk.BooleanVar()
    normalize_all_checkbox = tk.Checkbutton(frame, text="Normalize all to 1", variable=normalize_all_var)
    normalize_all_checkbox.grid(row=4, column=3, padx=10, pady=5, sticky="w")

    # Normalize by group checkbox
    normalize_by_group_var = tk.BooleanVar()
    normalize_by_group_checkbox = tk.Checkbutton(frame, text="Normalize by group", variable=normalize_by_group_var)
    normalize_by_group_checkbox.grid(row=5, column=3, padx=10, pady=5, sticky="w")

    # Denoise checkbox
    denoise_var = tk.BooleanVar()
    denoise_checkbox = tk.Checkbutton(frame, text="Denoise", variable=denoise_var)
    denoise_checkbox.grid(row=3, column=3, padx=10, pady=5, sticky="w")

    # Name field
    tk.Label(frame, text="Output Name:").grid(row=5, column=0, padx=10, pady=5, sticky="e")
    name_entry = tk.Entry(frame, textvariable=name_var, width=40)
    name_entry.grid(row=5, column=1, padx=10, pady=5)

    # pH button
    pH_button = ttk.Button(frame, text="pH Plotting", command=pH_plot)
    pH_button.grid(row=5, column=2, padx=10, pady=5)

    # Submit button
    submit_button = ttk.Button(frame, text="Submit", command=submit)
    submit_button.grid(row=6, column=0, columnspan=3, pady=10)

    # GUI Layout
    # tk.Label(frame, text="Spectra:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    # solution_dropdown = ttk.Combobox(frame, textvariable=solution_var, state="readonly", postcommand=updatelist)
    # solution_dropdown["values"] = [f for f in fluorophors.keys()]
    # solution_dropdown.grid(row=0, column=1, padx=10, pady=5)
    # solution_dropdown.bind("<<ComboboxSelected>>", on_solution_change)

    # ttk.Button(frame, text="Add new Spectrometer", command=add_fluorophor).grid(row=0, column=2, pady=10)
    # ttk.Button(frame, text="Rename Files", command=rename_files).grid(row=0, column=3, pady=10)

    # tk.Label(frame, text="Input Files:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    # ttk.Button(frame, text="Select Files", command=select_input_files).grid(row=1, column=1, padx=10, pady=5)
    # tk.Entry(frame, textvariable=input_files_var, state="readonly", width=40).grid(row=1, column=2, padx=10, pady=5)
    # ttk.Button(frame, text="Clear", command=clear_input_files).grid(row=1, column=3, padx=10, pady=5)

    # tk.Label(frame, text="Autofluorescence:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    # ttk.Button(frame, text="Select Files", command=select_autofluorescence_files).grid(row=2, column=1, padx=10, pady=5)
    # tk.Entry(frame, textvariable=autofluorescence_var, state="disable", width=40).grid(row=2, column=2, padx=10, pady=5)
    # ttk.Button(frame, text="Clear", command=clear_autofluorescence_files).grid(row=2, column=3, padx=10, pady=5)

    # tk.Label(frame, text="Spectra Min:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    # tk.Entry(frame, textvariable=min_spectra_var, width=10).grid(row=3, column=1, padx=10, pady=5, sticky="w")
    # tk.Label(frame, text="Spectra Max:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
    # tk.Entry(frame, textvariable=max_spectra_var, width=10).grid(row=4, column=1, padx=10, pady=5, sticky="w")

    # tk.Checkbutton(frame, text="Show Peak Values", variable=peak_display_var).grid(row=3, column=2, padx=10, pady=5, sticky="w")
    # tk.Checkbutton(frame, text="Normalize", variable=normalize_var).grid(row=4, column=2, padx=10, pady=5, sticky="w")
    # tk.Checkbutton(frame, text="Normalize all to 1", variable=normalize_all_var).grid(row=4, column=3, padx=10, pady=5, sticky="w")
    # tk.Checkbutton(frame, text="Normalize by group", variable=normalize_by_group_var).grid(row=5, column=3, padx=10, pady=5, sticky="w")
    # tk.Checkbutton(frame, text="Denoise", variable=denoise_var).grid(row=3, column=3, padx=10, pady=5, sticky="w")

    # tk.Label(frame, text="Output Name:").grid(row=5, column=0, padx=10, pady=5, sticky="e")
    # tk.Entry(frame, textvariable=name_var, width=40).grid(row=5, column=1, padx=10, pady=5, columnspan=2)

    # ttk.Button(frame, text="Submit", command=submit).grid(row=6, column=0, columnspan=3, pady=10)

    on_solution_change(None)

    root.mainloop()
