"""
Usage:
Execute the following in a Jupyter cell:
from BioCARS_2025_04 import *
After making changes, press command-S to save.

Setup required:
nano ~/.ipython/profile_default/startup/startup.ipy
import IPython
IPython.get_ipython().run_line_magic("reload_ext", "autoreload")
IPython.get_ipython().run_line_magic("autoreload", "complete")
IPython.get_ipython().run_line_magic("matplotlib", "qt")
"""
from saxs_waxs import hdf5
from os import path, environ
from numpy import arange,array,argsort,zeros,ones,where
import numpy
import logging

numpy.seterr(invalid="ignore",divide="ignore")  # Suppress "invalid value encountered in ...""
 
home = environ["HOME"]
data_root = f'{home}/BioCARS_Data/'
analysis_root = f'{home}/Library/CloudStorage/OneDrive-SharedLibraries-NationalInstitutesofHealth/NIH Anfinrud-Lab - Documents/General/Analysis/'
beamtime = '2025-04'
data_dir = data_root+beamtime+'/WAXS/'
# analysis_dir = f'{home}/BioCARS_Data/2025.04/Analysis/WAXS/'
analysis_dir = analysis_root+beamtime +'/WAXS/'
datafile = analysis_dir+'2025.04.hdf5'
lag = 1.28
pixelsize = 0.1  # mm
detector_distance = 168.1  # mm
wavelength = 1.033  # A
hc_in_keV_A = 12.3985
# photon_energy_in_keV = hc_in_keV_A/wavelength  # 1.033A: 12.002 keV
photon_energy_in_keV = 11.648
T0_unknown = 0.71  # on-axis transmission of unknown filter
XmH_scale = 0.956  # Scape factor for subtracting helium from xenon scattering
readout_noise = 0.0  # counts

def chart_omit():
    from charting_functions import chart_vector
    datasets = find_datasets(data_dir)
    Z_datasets = [dataset for dataset in datasets if 'Tramp' in dataset]
    for dataset in Z_datasets:
        pathname = analysis_hdf5(dataset)
        filename = path.basename(pathname)
        omit = hdf5(pathname,'omit')
        IP = hdf5(pathname,'IP')
        IPn = IP/IP[omit==0].mean()
        M0 = hdf5(pathname,'M0')
        chart_vector(IPn/(omit==0),'IPn ({} blank images)\nMean IP: {:0.3e}\n{}'.format((omit>0).sum(),IP[omit==0].mean(),filename),x_label='image number')
        chart_vector(IP/(omit==0),'IP ({} blank images)\n{}'.format((omit>0).sum(),filename),x_label='image number')

def survey_parameters_across_beamtime():
    from charting_functions import chart_vector, chart_xy_symbol
    datasets = find_datasets(data_dir)
    IP = []
    omit = []
    M0 = []
    T_rtd = []
    T_target = []
    timestamps = []
    for dataset in datasets:
        pathname = analysis_hdf5(dataset)
        try:
            IP.extend(hdf5(pathname,'IP'))
            omit.extend(hdf5(pathname,'omit'))
            M0.extend(hdf5(pathname,'M0'))
            T_rtd.extend(hdf5(pathname,'T_rtd'))
            T_target.extend(hdf5(pathname,'T_target'))
            timestamps.extend(hdf5(pathname,'timestamps'))
        except:
            pass
    IP = array(IP)
    omit = array(omit)
    M0 = array(M0)
    T_rtd = array(T_rtd)
    T_target = array(T_target)
    timestamps = array(timestamps)

    chart_xy_symbol(timestamps-timestamps[0],T_rtd-T_target,'T_rtd-T_target',x_label='timestamps')
    chart_vector(T_rtd,'T_rtd',x_label='image number')
    chart_vector(T_target,'T_target',x_label='image number')
    chart_vector(T_rtd-T_target,'T_rtd-T_target',x_label='image number')


    chart_vector(IP,'IP',x_label='image number')
    chart_vector(IP/(omit==0),'IP',x_label='image number')
    chart_vector(M0/(omit==0),'M0 ({} blank images)'.format((omit>0).sum()),x_label='image number')

def survey_number_of_entries_in_logfile():
    datasets = find_datasets(data_dir)
    for dataset in datasets:
        pathname = analysis_hdf5(dataset)
        filename = path.basename(pathname)
        try:
            N_M0 = len(hdf5(pathname,'M0'))
            N_T_rtd = len(hdf5(pathname,'T_rtd'))
            if N_M0 != N_T_rtd:
                print('{} M0 and {} T_rtd in {}'.format(N_M0,N_T_rtd,filename))
        except:
            pass

def update_omit_to_ensure_empty_images_are_property_flagged():
    datasets = find_datasets(data_dir)
    for dataset in datasets:
        pathname = analysis_hdf5(dataset)
        filename = path.basename(pathname)
        try:
            IP = hdf5(pathname,'IP')
            omit = (IP < 0)
            hdf5(pathname,'omit',omit)
        except:
            pass

def read_logfiles():
    datasets = find_datasets(data_dir)
    for dataset in datasets:
        try:
            read_logfile(dataset)
        except:
            pass

def rename_photons_IP():
    from saxs_waxs import hdf5_rename
    datasets = find_datasets(data_dir)
    for dataset in datasets:
        pathname = analysis_hdf5(dataset)
        try:
            hdf5_rename(pathname,'photons','IP')
        except:
            pass

def untitled_code_snippet_1():
    from charting_functions import chart_vector
    datasets = find_datasets(data_dir)
    Tramp_datasets = [dataset for dataset in datasets if 'Tramp' in dataset and '_A' not in dataset and '_CH' not in dataset]

    NOP = zeros(len(Tramp_datasets))
    for i,dataset in enumerate(Tramp_datasets):
        pathname = analysis_hdf5(dataset)
        NOP[i] = hdf5(pathname,'OP_count').sum()

    chart_vector(NOP,'NOP for Tramp datasets',logy=True)
    dataset = Tramp_datasets[28]
    Txray_from_Tramp(dataset)

    stdev = zeros(len(Tramp_datasets))
    for i,dataset in enumerate(Tramp_datasets):
        pathname = analysis_hdf5(dataset)
        T_cap = hdf5(pathname,'T_cap')
        T_xray = hdf5(pathname,'T_xray')
        stdev[i] = (T_cap - T_xray).std()
    chart_vector(stdev,'stdev for T_cap - T_xray for all Tramp datasets')

    dataset = Tramp_datasets[10]
    pathname = analysis_hdf5(dataset)

    omit = hdf5(pathname,'omit')
    omit[:43] |= 1
    hdf5(pathname,'omit',omit)

def flag_differences_between_T_target_and_T_rtd_in_omit():
    from charting_functions import chart_vector
    datasets = find_datasets(data_dir)
    Tjump_datasets = [dataset for dataset in datasets if 'PumpProbe' in dataset]
    for dataset in Tjump_datasets:
        pathname = analysis_hdf5(dataset)
        filename = path.basename(pathname)
        T_target = hdf5(pathname,'T_target')
        T_rtd = hdf5(pathname,'T_rtd')
        omit = hdf5(pathname,'omit')
        omit[abs(T_rtd-T_target)> 0.15] |= 2
        hdf5(pathname,'omit',omit)
        chart_vector((T_rtd-T_target)/(omit==0),'T_rtd-T_target\n{}'.format(filename))


def chart_Tjump_temperature_differences():
    from charting_functions import chart_xy_symbol
    datasets = find_datasets(data_dir)
    Tjump_datasets = [dataset for dataset in datasets if 'PumpProbe' in dataset]
    for dataset in Tjump_datasets:
        pathname = analysis_hdf5(dataset)
        filename = path.basename(pathname)
        T_cap = hdf5(pathname,'T_cap')
        T_rtd = hdf5(pathname,'T_rtd')
        omit = hdf5(pathname,'omit')
        chart_xy_symbol(arange(len(T_cap)),array([T_cap-T_rtd,T_rtd])/(omit==0),'T_cap - T_rtd\n{}'.format(filename),x_label='image number',ms=3)

def generate_beamstop_images_across_Reference_datasets():
    from charting_functions import chart_image_mask
    datasets = find_datasets(data_dir)
    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    Reference_datasets = [dataset for dataset in datasets if 'Reference' in dataset]
    for dataset in Reference_datasets:
        pathname = analysis_hdf5(dataset)
        filename = path.basename(pathname)
        try:
            Pmean = hdf5(pathname,'Pmean')
            image = Pmean
            from numpy import zeros,sqrt
            from scipy.ndimage import binary_dilation,gaussian_gradient_magnitude
            iterations = 1
            if image.shape == (3840,3840):
                iterataions = 2
            x0i = int(X0)
            y0i = int(Y0)
            Isub = image[y0i-12:y0i+12,x0i-12:x0i+12]
            ggm_n = gaussian_gradient_magnitude(Isub,1)/Isub
            mask = ggm_n > 1/sqrt(2)
            mask = binary_dilation(mask,iterations=iterations)
            BSmask = zeros(image.shape,bool)
            BSmask[y0i-12:y0i+12,x0i-12:x0i+12] = mask
            chart_image_mask(Isub,mask,'Isub\n{}'.format(filename),vmin=0,vmax=130)
        except:
            pass

        chart_image_mask(ggm_n,mask,'ggm/Isub',vmin=0,vmax=2)


def bin_PB_Tramp(PBCH_dataset):
    """Determines S, sigS, zinger_image, IP; writes to hdf5 file. 
    Flags and replaces zingers with median of binned data. Performs 
    weighted linear fit of binned data to generate S and sigS."""
    from numpy import diff,int16,prod
    from saxs_waxs import mccd_read

    # Read relevant information from datafile
    PP6mask = hdf5(datafile,'PP6mask')
    BNKmask = hdf5(datafile,'BNKmask')
    SDWmask = hdf5(datafile,'SDWmask')
    DPmask = hdf5(datafile,'DPmask')
    BankID = hdf5(datafile,'BankID')

    sort_indices = hdf5(datafile,'sort_indices')
    reverse_indices = hdf5(datafile,'reverse_indices')
    POL = hdf5(datafile,'POL')
    GEO = hdf5(datafile,'GEO')
    CTn = hdf5(datafile,'CTn')
    STn = hdf5(datafile,'STn')
    FT = filter_transmittance(T0_unknown=T0_unknown)
    DR = detector_responsivity()

    DQE = hdf5(datafile,'DQE')
    Q = hdf5(datafile,'Q')
    Dvar = hdf5(datafile,'DV')
    Dvar[Dvar==0] = 1
    intercept = hdf5(datafile,'intercept')
    slope = hdf5(datafile,'slope')
    UCscx = hdf5(datafile,'UCscx')
    q_flat = hdf5(datafile,'q_flat').astype(float)
    qbin1 = hdf5(datafile,'qbin1')
    qbin2 = hdf5(datafile,'qbin2')
    q = hdf5(datafile,'q')
    # Calculate bkg given Tramp period
    pathname = index_pathname(datafile,PBCH_dataset)[1]
    period = diff(hdf5(pathname,'timestamps')).mean()
    bkg = intercept + period*slope  
    BSmask = beamstop_mask(Pc) 
    mask = BNKmask | (BankID==0) | SDWmask | BSmask | DPmask | PP6mask
    # Process PB_dataset
    print('Processing {}'.format(PBCH_dataset))
    # Extract M0 and nC from pathname and compute nCn
    M0 = hdf5(pathname,'M0')
    nC = hdf5(pathname,'nC')
    omit = hdf5(pathname,'omit')
    nCn = nC/nC[omit==0].mean()
    # Determine appropriate H_dataset, CH_dataset, and BCH_dataset for processing PB_dataset
    H_dataset,CH_dataset,BCH_dataset = H_CH_BCH_for_PBCH(PBCH_dataset)
    # Load bkg-corrected H, CH and corresponding M0
    H = hdf5(index_pathname(datafile,H_dataset)[1],'Imean')-bkg
    M0_H = hdf5(index_pathname(datafile,H_dataset)[1],'M0').mean()
    CH = hdf5(index_pathname(datafile,CH_dataset)[1],'Imean')-bkg
    M0_CH = hdf5(index_pathname(datafile,CH_dataset)[1],'M0').mean()
    # Extract appropriate UCb 
    UCb = hdf5(index_pathname(datafile,BCH_dataset)[1],'UCb')
    # Construct CHs and CORR
    CHs = CTn*CH/M0_CH + (1-CTn)*H/M0_H
    CORR = POL*GEO*FT*DR*STn*UCscx*UCb
    # Process all mccd images in dataset
    filenames = array(hdf5(pathname,'filenames'))
    N_images = len(filenames)
    S = zeros((N_images,len(q)))
    sigS = zeros((N_images,len(q)))
    IP = zeros(N_images)
    zingers_flat = -ones(prod(Q.shape),int16)
    mask_flat = mask.flatten()[sort_indices]
    for i,filename in enumerate(filenames):
        print('Processing image {} out of {}'.format(i,len(filenames)),end='\r')
        # Load image, subtract background, compute Ivar
        I = mccd_read(filename.astype(float))-bkg
        I_var = Dvar + I*VPC
        # Construct Ic, which is corrected I, and its variance
        Ic = M0G*(I/M0[i] - CHs)/CORR
        Vc = (M0G/(M0[i]*CORR))**2*I_var
        Ic_flat = Ic.flatten()[sort_indices]
        Vc_flat = Vc.flatten()[sort_indices]
        # Flag zingers and replace with median
        Ic_flat,zinger_flati = zinger_free_Ic_flat(Ic_flat,Vc_flat,mask_flat)
        # Generate one-dimensional S(q) and sigS(q)
        Si,sigSi,N,Si_flat = S_sigS(q_flat,Ic_flat,Vc_flat,mask_flat)
        S[i] = Si
        sigS[i] = sigSi
        zingers_flat[zinger_flati] = i
        IP[i] = I[~mask].sum()/DQE
    # Write results to hdf5 file
    zingers_image = zingers_flat[reverse_indices].reshape(Q.shape)
    hdf5(pathname,'IP',IP)
    hdf5(pathname,'S',S)
    hdf5(pathname,'sigS',sigS)
    hdf5(pathname,'zingers_image',zingers_image)


def H_CH_BCH_for_PBCH(PBCH_dataset):
    """Returns dataset names for CH, H, and UCb; needed to process PB_dataset."""
    datasets = hdf5(datafile,'datasets')
    # Find all H_datasets, CH_datasets, and B_datasets
    H_datasets = [dataset for dataset in datasets if 'Tramp_A' in dataset]
    CH_datasets = [dataset for dataset in datasets if 'Tramp_CH' in dataset]
    BCH_datasets = [dataset for dataset in datasets if 'Tramp_B' in dataset]
    # Find starting times for all H_datasets, CH_datasets, and B_datasets
    t0_H = [hdf5(index_pathname(datafile,dataset)[1],'timestamps')[0] for dataset in H_datasets]
    t0_CH = [hdf5(index_pathname(datafile,dataset)[1],'timestamps')[0] for dataset in CH_datasets]
    t0_B = [hdf5(index_pathname(datafile,dataset)[1],'timestamps')[0] for dataset in BCH_datasets]
    
    # Find starting time for PB_dataset
    pathname = index_pathname(datafile,PBCH_dataset)[1]
    t0 = hdf5(pathname,'timestamps')[0]
    # Find nearest prior CH_dataset and H_datasets, and first B_dataset after first pior H_dataset
    H_index = where(t0_H < t0)[0][-1]
    CH_index = where(t0_CH < t0)[0][-1]
    B_index = where(t0_B > t0_H[H_index])[0][0]
    CH_dataset = CH_datasets[CH_index]
    H_dataset = H_datasets[H_index]
    BCH_dataset = BCH_datasets[B_index]
    return H_dataset,CH_dataset,BCH_dataset

def CHs_from_CH_H(CH_dataset,H_dataset,CTn,bkg):
    """Returns CHs, which is generated from CH and H datasets and bkg."""
    pathname = index_pathname(datafile,CH_dataset)[1]
    CH = hdf5(pathname,'Imean') - bkg
    CH_M0 = hdf5(pathname,'M0').mean()
    pathname = index_pathname(datafile,H_dataset)[1]
    H = hdf5(pathname,'Imean') - bkg
    H_M0 = hdf5(pathname,'M0').mean()
    # Use M0 scaling to generate CHs
    CHs = CTn*CH/CH_M0 + (1-CTn)*H/H_M0
    return CHs

def Bs_from_BCH(BCH_dataset,CHs,bkg,scale=1):
    """Returns Bs, which is generated from BCH and CHs."""
    pathname = index_pathname(datafile,BCH_dataset)[1]
    BCH = hdf5(pathname,'Imean') - bkg
    BCH_M0 = hdf5(pathname,'M0').mean()
    # Use M0 scaling to generate Bs
    M0G = hdf5(pathname,'M0').mean()
    Bs = M0G*(BCH/BCH_M0 - scale*CHs)
    return Bs

def bkg_for_dataset(dataset):
    """Returns bkg for dataset."""
    from numpy import diff
    pathname = index_pathname(datafile,dataset)[1]
    period = diff(hdf5(pathname,'timestamps')).mean()
    intercept = hdf5(datafile,'intercept')
    slope = hdf5(datafile,'slope')
    return intercept + period*slope

def index_pathname(datafile,dataset):
    """Returns index and pathname for dataset given datafile."""
    from os import path
    datasets = hdf5(datafile,'datasets')
    i = datasets.index(dataset)
    analysis_dir = path.dirname(datafile) + '/'
    return i,analysis_dir + hdf5(datafile,'dataset_pathnames')[i]

def Process_2025_04_datasets():
    initialize_datafile()

    # Find datasets
    datasets = find_datasets(data_dir)

    # Process Reference datasets using Pmean_Pvar_dataset (results needed for UC)
    DPmask = hdf5(datafile,'DPmask')
    BankID = hdf5(datafile,'BankID')
    mask = (BankID==0) | DPmask
    for dataset in [dataset for dataset in datasets if 'Reference_' in dataset]:
        Pmean_Pvar_dataset(dataset,mask)

    # Gather datasets used to construct UC and generate UC
    FS_dataset = '/Users/philipa/BioCARS_Data/2025.04/WAXS/Reference/Reference_0.5mm-thick_FS-2/'
    GC_dataset = '/Users/philipa/BioCARS_Data/2025.04/WAXS/Reference/Reference_1.0mm-thick_GC-3/'
    A_dataset = '/Users/philipa/BioCARS_Data/2025.04/WAXS/Reference/Reference_A-5/'
    He_dataset = '/Users/philipa/BioCARS_Data/2025.04/WAXS/Reference/Reference_CH5-1/'
    Xe_dataset = '/Users/philipa/BioCARS_Data/2025.04/WAXS/Reference/Reference_CX5-1/'
    UC_from_datasets(FS_dataset,GC_dataset,A_dataset,Xe_dataset,He_dataset,XmH_scale=XmH_scale)

    # Process Reference_A datasets using process dataset with A_dataset set to None
    Z_datasets = [dataset for dataset in datasets if 'Reference_A' in dataset]
    for dataset in Z_datasets:
        process_dataset(dataset)

    # Process Reference datasets using process_dataset and A-5
    A_dataset = '/Users/philipa/BioCARS_Data/2025.04/WAXS/Reference/Reference_A-5/'
    Z_datasets = [dataset for dataset in datasets if 'Reference_' in dataset and '_A-' not in dataset]
    for dataset in Z_datasets:
        process_dataset(dataset,A_dataset)

    # Process Tramp and PumpProbe datasets using process_dataset with A-6
    A_dataset = '/Users/philipa/BioCARS_Data/2025.04/WAXS/Reference/Reference_A-6/'
    Z_datasets = [dataset for dataset in datasets if '_Tramp' in dataset or '_PumpProbe' in dataset and '_A-' not in dataset and '_RawData-' not in dataset]
    for dataset in Z_datasets:
        try:
            S = hdf5(analysis_hdf5(dataset),'S')
        except:
            process_dataset(dataset,A_dataset)

def chart_Tjump_dataset(dataset):
    from numpy import unique
    from charting_functions import chart_vector, chart_image_mask, chart_xy_rainbow
    
    q = hdf5(datafile,'q')
    
    PPmask = hdf5(datafile,'PPmask')
    pathname = analysis_hdf5(dataset) 
    filename = path.basename(pathname)
    Delay = hdf5(pathname,'Delay')
    T_target = hdf5(pathname,'T_target')
    T_rtd = hdf5(pathname,'T_rtd')
    Repeat1 = hdf5(pathname,'Repeat1')
    Repeat2 = hdf5(pathname,'Repeat2')
    S = hdf5(pathname,'S')
    M0 = hdf5(pathname,'M0')
    IP = hdf5(pathname,'IP')
    OP_count = hdf5(pathname,'OP_count')
    IP = hdf5(pathname,'IP')
    OP_count = hdf5(pathname,'OP_count')
    omit = hdf5(pathname,'omit')

    IPn = IP/IP.mean()
    Sn = (S.T/IPn).T
    T_unique = unique(T_target)
    D_unique = unique(Delay)
    
    STD = zeros((len(T_unique),len(D_unique),len(q)))
    for i,T in enumerate(T_unique):
        for j,D in enumerate(D_unique):
            select = (D == Delay) & (T == T_target) & (omit==0)
            STD[i,j] = Sn[select].mean(axis=0)
        DSTD = STD[i]-STD[i,1]
        chart_xy_rainbow(q,DSTD,'S(t) at T = {} vs. q\n{}'.format(T,filename),logx=True,x_label='q')
    
    chart_vector(IPn,'IPn\nmean IP: {:0.3e}\n{}'.format(IP.mean(),filename))
    chart_vector(M0/IPn,'M0/IPn\n{}'.format(filename))
    chart_vector(T_rtd,'T_rtd\n{}'.format(filename))
    chart_image_mask(OP_count,OP_count*~PPmask>10,'OPcount\n{}'.format(filename))

def chart_static_dataset(dataset):
    from charting_functions import chart_vector, chart_image_mask, chart_xy_rainbow
    
    pathname = analysis_hdf5(dataset) 
    filename = path.basename(pathname)
    q = hdf5(pathname,'q')
    S = hdf5(pathname,'S')
    M0 = hdf5(pathname,'M0')
    IP = hdf5(pathname,'IP')
    OP_count = hdf5(pathname,'OP_count')
    IPn = IP/IP.mean()
    Sn = (S.T/IPn).T
    chart_xy_rainbow(q,Sn,'Sn (normalized) vs. q\nscale [photons]: {:0.3e}\n{}'.format(IP.mean(),filename),logx=True,logy=True,x_label='q')
    chart_vector(IPn,'P_norm\nmean IP: {:0.3e}\n{}'.format(IP.mean(),filename))
    chart_vector(M0,'M0 [IP]\n{}'.format(filename))
    chart_image_mask(OP_count,OP_count>0,'OP_count\n{}'.format(filename))

def chart_Tramp_dataset(dataset):
    from numpy import argsort
    from charting_functions import chart_xy_rainbow, chart_UTsV, chart_vector, chart_image_mask
    
    q = hdf5(datafile,'q')

    pathname = analysis_hdf5(dataset) 
    filename = path.basename(pathname)
    S = hdf5(pathname,'S')
    M0 = hdf5(pathname,'M0')
    IP = hdf5(pathname,'IP')
    T_cap = hdf5(pathname,'T_cap')
    omit = hdf5(pathname,'omit')
    OP_count = hdf5(pathname,'OP_count')
    IPn = IP/IP.mean()
    Sn =(S.T/IPn).T
    select = omit==0
    sort_order = argsort(T_cap)
    UT,s,V = SVD((Sn.T*select).T)
    chart_UTsV(q,UT,s,V,'{}'.format(filename))
    Ssn = (Sn.T*select).T[sort_order]
    chart_xy_rainbow(q,Ssn,'Ssn (sorted, normalized) vs. q\nIP [photons]: {:0.3e}\n{}'.format(IP.mean(),filename),logx=True,x_label='q')
    chart_vector(IPn,'IPn\nmean IP: {:0.3e}\n{}'.format(IP.mean(),filename),x_label='image number')
    chart_vector(M0/IPn,'M0/IPn\n{}'.format(filename),x_label='image number')
    chart_image_mask(OP_count,OP_count>0,'OP_count ({} outliers)\n{}'.format(OP_count.sum(),filename))

def Txray_from_Tramp(dataset,chart=False):
    """Processes Tramp dataset to determine 'T_xray'. Assigns 'T_xray' 
    according to N_poly order polynomial fit of T_cap as a function 
    of VT[1]/VT[0], which is generated from SVD analysis of dataset. 
    The q region selected for the analysis is defined by qmin and qmax, 
    a region that is sensitive to temperature."""
    from numpy import array,arange,argsort,around
    from charting_functions import chart_xy_fit, chart_xy_symbol,chart_UTsV
    from numpy import polyfit,poly1d
    from time import time
    t0 = time()
    # Select restricted range of q
    N_poly=6
    qmin=1.5
    qmax=3.5
    pathname = analysis_hdf5(dataset)
    filename = path.basename(pathname)
    # Read q from ePix.hdf5
    q = hdf5(datafile,'q')
    # Read relevant data from dataset
    S = hdf5(pathname,'S')
    sigS = hdf5(pathname,'sigS')
    IP = hdf5(pathname,'IP')
    omit = hdf5(pathname,'omit')
    T_rtd = hdf5(pathname,'T_rtd')
    T_cap = hdf5(pathname,'T_cap')
    # Define qs and generate Ss
    q_select = (q > qmin) & (q < qmax)
    qs = q[q_select]
    Pn = IP/IP[omit==0].mean()
    Sn = (S.T/Pn).T
    sigSn = (sigS.T/Pn).T
    Sns = Sn[:,q_select]

    # Process Tramp data free of outliers; assign T_xray according to svd-based calibration
    select = omit == 0
    UT,s,V = SVD(Sns[select])
    V1 = V[1]/V[0]
        
    # Fit T_cap vs. V1 with polynomial and generate T_xray
    pf = polyfit(V1,T_cap[select],N_poly)
    T_xray = T_cap.copy()
    T_xray[select] = poly1d(pf)(V1)
    stdev = (T_cap - T_xray).std()

    hdf5(pathname,'T_xray',T_xray)
    print('\rProcessed {} in {:0.3f} seconds                             '.format(filename,time()-t0),end='\r')

    if chart:
        filename = path.basename(pathname)
        sort_order = argsort(T_xray)
        chart_xy_fit(V1,T_cap[select],T_xray[select],'Polynomial fit of T_cap vs. V1\nN_poly = {}; lag = {}; stdev = {:0.3f}\n{}'.format(N_poly,lag,stdev,filename),x_label='V1 (arb. units)',ms=2)
        chart_xy_symbol(arange(len(T_rtd)),array([T_rtd,T_cap]),'T_rtd,T_cap\n{}'.format(filename,ms=3))
        chart_xy_symbol(q,array([Sn[-1],sigSn[-1]]),'Sn and sigSn at {} Celsius\n{}'.format(around(T_xray[0],0),filename),x_label='q',y_label='Counts',logx=True,logy=True,ms=2)
        chart_UTsV(qs,UT,s,V,'{}'.format(filename))
        difference = T_cap[select] - T_xray[select]
        ri = arange(len(S))
        chart_xy_symbol(ri[select],difference,'{}\nT_cap - T_xray (N_poly = {})'.format(filename,N_poly),y_label='Temperature error [Celsius]',x_label='image index',ms=2)
        chart_xy_symbol(ri[select],array([T_cap[select],T_xray[select]]),'{}\nT_cap,T_xray (N_poly = {})'.format(filename,N_poly),x_label='image index',y_label='Temperature [Celsius]',ms=3)

def SVD(M):
    """Returns U.T,s,V given M. For convenience, U (and corresponding V) vectors 
    are negated if the sum of the first three values of U is negative. """
    from scipy.linalg import svd
    # Perform sigma-weighted singular value decomposition
    U,s,V = svd(M.T,full_matrices=0)
    # Negate U vectors (and corresponding VT vectors) if U[0][:4].mean() < 0.
    negate = U[:3].mean(axis=0) < 0
    U = U*~negate-U*negate
    V = (V.T*~negate-V.T*negate).T
    return U.T,s,V  

def UC_from_datasets(FS_dataset,GC_dataset,A_dataset,Xe_dataset,He_dataset,XmH_scale=XmH_scale,T0_unknown=T0_unknown):
    """Generates uniformity correction from scattering information contained
    in five datasets."""
    from numpy.linalg import lstsq 
    from numpy import stack,sqrt,median,nan_to_num, isnan, nan
    from saxs_waxs import FF_C
    from charting_functions import chart_image_mask  # chart_xy_symbol, chart_histogram

    qbin1 = hdf5(datafile,'qbin1')
    qbin2 = hdf5(datafile,'qbin2')
    PP6mask = hdf5(datafile,'PP6mask')
    BNKmask = hdf5(datafile,'BNKmask')
    SDWmask = hdf5(datafile,'SDWmask')
    DPmask = hdf5(datafile,'DPmask')
    BankID = hdf5(datafile,'BankID')
    POL = hdf5(datafile,'POL')
    GEO = hdf5(datafile,'GEO')
    Q = hdf5(datafile,'Q')
    q_flat = hdf5(datafile,'q_flat')
    sort_indices = hdf5(datafile,'sort_indices')
    reverse_indices = hdf5(datafile,'reverse_indices')
    UC = hdf5(datafile,'UC')
    q = hdf5(datafile,'q')
    qbin1 = hdf5(datafile,'qbin1')
    qbin2 = hdf5(datafile,'qbin2')
    q_flat = hdf5(datafile,'q_flat')
    POL = hdf5(datafile,'POL')
    GEO = hdf5(datafile,'GEO')
    CTn = hdf5(datafile,'CTn')
    STn = hdf5(datafile,'STn')
    FT = filter_transmittance(T0_unknown=T0_unknown)
    DR = detector_responsivity()

    # Calculate Xenon atomic scattring
    FF,C = FF_C(q,54)
    FF2C = (FF**2 + C)/54**2

    # Extract data needed
    FS_M0,FS_Pmean = M0_Pmean(FS_dataset)
    GC_M0,GC_Pmean = M0_Pmean(GC_dataset)
    GM0,A_Pmean = M0_Pmean(A_dataset)
    M0_CX5,CX5 = M0_Pmean(Xe_dataset)
    M0_CH5,CH5 = M0_Pmean(He_dataset)

    # Subract A scattering from FS and GC datasets
    FSmA = FS_Pmean - FS_M0/GM0 * A_Pmean
    GCmA = GC_Pmean - GC_M0/GM0 * A_Pmean
    FSmA_var = FSmA
    GCmA_var = GCmA
    # Combine and correct to generate Pc
    Pc = (FSmA + GCmA)/(POL*GEO)
    Pc_var = (FSmA_var + GCmA_var)/(POL*GEO)**2
    Pc_flat = nan_to_num(Pc.flatten()[sort_indices])
    Pc_var_flat = nan_to_num(Pc_var.flatten()[sort_indices])
    # Calculate corrected XHc (Xenon minus Helium scattering, corrected)
    # XHc = (CX5 - XmH_scale*CH5)/(POL*GEO*FT*DR)
    # XHc = GM0 * (CX5/M0_CX5/STn - XmH_scale * CH5/M0_CH5/CTn) / (POL*GEO*FT*DR)
    # XHc = GM0 * (CX5/M0_CX5 - XmH_scale * CTn*CH5/M0_CH5) / (STn * POL*GEO*FT*DR)
    CH = CTn*CH5/M0_CH5 + A_Pmean/GM0*(1-CTn)
    XHc = GM0 * (CX5/M0_CX5 - XmH_scale*CH) / (STn * POL*GEO*FT*DR)

    XHc_flat = XHc.flatten()[sort_indices]
    # Generate mask to select appropriate regions of detector
    BSmask = beamstop_mask(Pc)
    mask = BNKmask | (BankID==0) | SDWmask | BSmask | DPmask | PP6mask
    mask_flat = mask.flatten()[sort_indices]
    # Perform weighted line fit of data in each bin; generate UCi
    UC_flat = ones(len(q_flat))
    UC_var_flat = zeros(len(q_flat)) + nan
    S = zeros(len(q))
    for i in range(len(q)):
        first = qbin1[i]
        last  = qbin2[i]
        bin_select = arange(first,last)
        mask_select = ~mask_flat[bin_select]
        N = mask_select.sum()
        qi = q_flat[bin_select[mask_select]]
        Pi = Pc_flat[bin_select[mask_select]]
        Pi_median = median(Pi)
        M = stack((ones(N),qi))
        wi = nan_to_num(sqrt(Pi/Pi_median))
        coeffs = lstsq((M*wi).T,Pi*wi,rcond=None)[0]
        # Apply UC to XHc data and generate scale factor
        UCi = Pi/(coeffs.T@M)
        Pi = XHc_flat[bin_select[mask_select]]
        select = (UCi>0.5) & (UCi<1.5)
        if i == 0: FF2C_scale = Pi[select].mean()
        scale = FF2C_scale*FF2C[i]/Pi[select].mean()
        S[i] = (coeffs[0]+q[i]*coeffs[1])*scale
        # Generate UC for all pixels in bin
        qi = q_flat[bin_select]
        Pi = Pc_flat[bin_select]
        Pi_var = Pc_var_flat[bin_select]
        fit = coeffs[0]+q[i]*coeffs[1]
        UC_flat[bin_select] = Pi/fit/scale
        UC_var_flat[bin_select] = Pi_var/fit**2/scale**2

    UC = UC_flat[reverse_indices].reshape(mask.shape)
    UC_var = UC_var_flat[reverse_indices].reshape(mask.shape)

    # Ensure pixels near beamstop (q < 0.03) have same UC
    UCBC_mean = UC[(Q>0.03)&(Q<0.04)].mean()
    UC[(Q>0) & (Q<0.03)] = UCBC_mean
    hdf5(datafile,'UC',UC)
    # Chart results
    FF,C = FF_C(q_flat,54)
    FF2C = (FF**2 + C)/54**2
    BSmask = beamstop_mask(XHc)
    UC_flat = UC.flatten()[sort_indices]
    mask = (UC<0.6) | BSmask | DPmask
    mask_flat = mask.flatten()[sort_indices]

    valid = ~mask & ~isnan(UC) & ~isnan(UC_var) & ~(UC<0.6) & ~(UC>5)
    chisq = ((UC - 1)**2 / UC_var)[valid].mean()
    report = f"XmH_scale={XmH_scale}, T0_unknown={T0_unknown}: chisq={chisq:.6g}"

    # chart_xy_symbol(q_flat,array([XHc_flat/UC_flat/~mask_flat,FF2C_scale*FF2C]),'XHc/UC vs. Xe FF2C\nmask = (UC<0.6) | BSmask | DPmask',x_label='q',y_label='IP',ymin=0)
    # chart_xy_symbol(q,S,'S for FSpGC datasets',x_label='q',y_label='IP')
    # chart_histogram(UC,'UC',binsize=0.01,xmin=0,xmax=6)
    # chart_image_mask(UC,UC<0.6,f'UC (UC < 0.6)\n{report}',vmin=0.85,vmax=1.15)
    chart_image_mask(UC,UC>5,f'UC (UC > 5)\n{report}',vmin=0.85,vmax=1.15)

def M0_Pmean(dataset):
    pathname = analysis_hdf5(dataset) 
    M0 = hdf5(pathname,'M0').mean()
    Pmean = hdf5(pathname,'Pmean')
    return M0,Pmean
    
def trace_for_pixel(dataset,y,x):
    """Returns and charts trace for specified pixel."""
    import h5py
    from charting_functions import chart_vector
    pathname = data_hdf5(dataset)
    filename = path.basename(pathname)
    with h5py.File(pathname, 'r') as f:
        trace = f['images'][:,y,x]
    chart_vector(trace,'pixel[{},{}]\n{}'.format(y,x,filename))
    return trace

def beamstop_mask(image,chart=False):
    """Returns BSmask given an image. Selects a sub image centered on the 
    beamstop and sets the inner circle of specified radius to the minimium value.
    If Rayonix, calculates the normalized gaussian gradient magnitude of the
    pixel intensities, i.e. ggm_n = gaussian_gradient_magnitude(Isub,2)/Isub, 
    and flags pixels whose value is greater than 0.5*ggm_n.max(). If ePix, finds 
    pixels with less than half the max/min mean and enlarges the mask by one pixel
    via biary dilation."""
    from numpy import zeros,sqrt,indices
    from scipy.ndimage import binary_dilation,gaussian_gradient_magnitude

    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    x0i = int(X0)
    y0i = int(Y0)
    Isub = image[y0i-12:y0i+12,x0i-12:x0i+12]
    if image.shape == (3840,3840):
        x_indices,y_indices = indices(Isub.shape)-12
        r = sqrt(x_indices**2 + y_indices**2)
        mask = r < 4
        Isub[mask] = Isub.min()
        ggm_n = gaussian_gradient_magnitude(Isub,2)/Isub
        mask |= (ggm_n > 0.5*ggm_n.max())
    else:
        threshold = (Isub.max() + Isub.min())/2
        mask = binary_dilation(Isub < threshold)

    BSmask = zeros(image.shape,bool)
    BSmask[y0i-12:y0i+12,x0i-12:x0i+12] = mask
    if chart:
        from charting_functions import chart_image_mask
        chart_image_mask(Isub,mask,'Isub',vmin=0,vmax=1300)
        chart_image_mask(ggm_n,mask,'ggm/Isub')
    return BSmask

def beamstop_mask_old(image):
    """Returns BSmask given an image. Flags pixels with less than half the 
    median intensity in the sub image, thereby identifying the boundary of 
    shawdoed pixels. Uses binary_dilation to enlarge the mask by one pixel."""
    from numpy import zeros,sort,median
    from scipy.ndimage import binary_dilation

    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')

    x0i = int(X0)
    y0i = int(Y0)
    Isub = image[y0i-12:y0i+12,x0i-12:x0i+12]
    threshold = (median(sort(Isub)) + Isub.min())/2
    mask = binary_dilation(Isub < threshold)
    BSmask = zeros(image.shape,bool)
    BSmask[y0i-12:y0i+12,x0i-12:x0i+12] = mask
    return BSmask

def M0_from_image(image):
    """Returns background-free sum of photons in 2x2 pixel array. If
    any pixel is an overload, returns zero."""
    gain = hdf5(datafile,'gain')
    M0_mask= array([[0,0,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,1,1,0,0],
                    [0,0,1,1,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,0,0,0]])
    bkg_mask=array([[0,0,1,1,0,0],
                    [0,1,0,0,1,0],
                    [1,0,0,0,0,1],
                    [1,0,0,0,0,1],
                    [0,1,0,0,1,0],
                    [0,0,1,1,0,0]])
    counts = image[1448:1454,1407:1413]
    OVLD = (counts == 2**16-1).any()
    photons = counts/gain
    M0 = (photons*(M0_mask - bkg_mask/3)).sum()
    return 0 if OVLD else M0

def calculate_image_means():
    datasets = find_datasets(data_dir)
    H_datasets = [dataset for dataset in datasets if '_A' in dataset and "Tramp_A-1" not in dataset]
    calculate_image_mean_of_datasets(H_datasets, "H_mean")
    CH_datasets = [dataset for dataset in datasets if '_CH' in dataset]
    calculate_image_mean_of_datasets(CH_datasets, "CH_mean")
    BCH_datasets = [dataset for dataset in datasets if 'Buffer' in dataset]
    BCH_datasets = BCH_datasets[0:2]
    calculate_image_mean_of_datasets(BCH_datasets, "BCH_mean")

def calculate_image_mean_of_datasets(datasets, output_name):
    """Compute mean of images in mutiple datasets"""
    from numpy import sum, float32, sqrt
    total_N_images = sum([N_images_of_dataset(dataset) for dataset in datasets])
    usable_N_images = sum([len(usable_image_numbers_of_dataset(dataset)) for dataset in datasets])
    print(f'{output_name}: usable images: {usable_N_images} of {total_N_images}          ')
    BankID = hdf5(datafile,'BankID')
    pixel_mask = BankID>0
    gain = hdf5(datafile,'gain')
    shape = pixel_mask.shape
    Psum = zeros(shape)
    n = 0
    for dataset in datasets:
        for i in usable_image_numbers_of_dataset(dataset):
            print(f'{output_name}: processing image {n+1} of {usable_N_images}   ',end='\r')
            image = hdf5(data_hdf5(dataset),'images', index=i)
            image = (image.astype(float) - 100*pixel_mask)/gain
            Psum += image
            n += 1
    Pmean = Psum / usable_N_images
    var_Psum = Psum
    var_Pmean = var_Psum / usable_N_images ** 2

    # Conversion to 32-bit saves 50% storage space
    Pmean = Pmean.astype(float32)  
    Pmean = Pmean.astype(float32)
    var_Pmean = var_Pmean.astype(float32)
    hdf5(datafile,f"{output_name}",Pmean)
    hdf5(datafile,f"var_{output_name}",var_Pmean)

    hdf5(datafile,f"var_{output_name}",var_Pmean)

def N_images_of_dataset(dataset): 
    return hdf5(data_hdf5(dataset),'images',attribute='len')

def usable_image_numbers_of_dataset(dataset):
    from numpy import where
    return where(hdf5(analysis_hdf5(dataset),'omit') == 0)[0]

def image_mean_of_dataset(dataset):
    """Compute mean of images in dataset."""
    from numpy import float32
    images = hdf5(data_hdf5(dataset),'images')
    BankID = hdf5(datafile,'BankID')
    pixel_mask = BankID>0
    gain = hdf5(datafile,'gain')
    shape = pixel_mask.shape
    Psum = zeros(shape)
    N_images = len(images)
    for i in range(N_images):
        print(f'processing image {i+1} of {N_images}   ',end='\r')
        image = (images[i].astype(float) - 100*pixel_mask)/gain
        Psum += image
    Pmean = Psum / N_images
    hdf5(analysis_hdf5(dataset),'Pmean',Pmean.astype(float32))

def Pmean_Pvar_dataset(dataset,mask):
    """Generates zinger-free, offset-subtracted Pmean and Pvar and writes to 
    hdf5 file. Normalizes images according to integrated photons with mask pixels
    excluded to avoid Pvar inflation due to x-ray intensity variations. 
    To calculate Pvar, employs double-pass approach to address numerical instability.
    Also writes Pmax/Pmin. Blank images are omitted from the Pmean and Pvar 
    calculations. Pmean and Pvar are set to zero for OVLD pixels."""
    from numpy import zeros,ones,where,int32,int16
    from os import path
    import h5py
    
    BankID = hdf5(datafile,'BankID')

    gain = hdf5(datafile,'gain')

    pathname = data_hdf5(dataset)
    filename = path.basename(pathname)
    print('Processing {:0.2f} GB in {}'.format(path.getsize(pathname)/1024**3,filename))
    with h5py.File(pathname, 'r') as f:
        N_images = len(f['images'])
        # Determine Cmean and Cvar
        select = ~mask
        N = select.sum()
        shape = mask.shape
        pixel_mask = BankID>0
        Psum = zeros(shape)
        P2sum = zeros(shape)
        Pmax = -(2**16-1)*ones(shape)
        Pmin = (2**16-1)*ones(shape)
        IP = zeros(N_images)
        M0 = zeros(N_images)
        OVLD = zeros(shape,int32)
        for i in range(N_images):
            image = f['images'][i]
            OVLD += image==2**16-1
            Pi = (image.astype(float) - 100*pixel_mask)/gain
            M0[i] = M0_from_image(image)
            IP[i] = Pi[select].sum()
            if M0[i]>0:
                Pni = Pi/IP[i]
                Psum += Pni
                Pmax = where(Pni>Pmax,Pni,Pmax)
                Pmin = where(Pni<Pmin,Pni,Pmin)
            print('processing image {} of {}   '.format(i+1,N_images),end='\r')
        Nz = (M0==0).sum()
        Pmean = (Psum - Pmax - Pmin)/(N_images-Nz-2) 
        for i in range(N_images):
            if M0[i]>0:
                image = f['images'][i]
                Pni = ((image.astype(float) - 100*pixel_mask)/gain)/IP[i]
                P2sum += (Pni - Pmean)**2
            print('processing image {} of {}   '.format(i+1,N_images),end='\r')
    Pvar = (P2sum - (Pmax-Pmean)**2 - (Pmin-Pmean)**2)/(N_images-Nz-3)
    # Rescale results; set OVLD to zero
    OVLD_bool = OVLD > 0
    Pmean *= IP.mean()
    Pmean[OVLD_bool] = 0
    Pvar *= IP.mean()**2
    Pvar[OVLD_bool] = 0
    Pmax *= IP.mean()
    Pmin *= IP.mean()
    # Write results to hdf5 file
    pathname = analysis_hdf5(dataset)
    hdf5(pathname,'IP',IP)
    hdf5(pathname,'M0',M0)
    hdf5(pathname,'Pmean',Pmean)
    hdf5(pathname,'Pvar',Pvar)
    hdf5(pathname,'Pmax',Pmax.astype(int16))
    hdf5(pathname,'Pmin',Pmin.astype(int16))
    hdf5(pathname,'mask',mask)

def process_dataset(dataset,H_dataset=None,CH_dataset=None,G_dataset=None):
    """Process data in dataset and generates S(q). If provided, A_dataset
    is first subtracted from the images. The corresponding logfile is read 
    relevant parameters included in the analysis hdf5 file."""
    from numpy import uint8,uint16,zeros,sqrt
    import h5py

    # Read relevant parameters from ePix.hdf5
    q = hdf5(datafile,'q')
    q_flat = hdf5(datafile,'q_flat')
    qbin1 = hdf5(datafile,'qbin1')
    qbin2 = hdf5(datafile,'qbin2')
    gain = hdf5(datafile,'gain')
    POL = hdf5(datafile,'POL')
    GEO = hdf5(datafile,'GEO')
    UC = hdf5(datafile,'UC2')
    BankID = hdf5(datafile,'BankID')
    DPmask = hdf5(datafile,'DPmask')
    BNKmask = hdf5(datafile,'BNKmask')
    sort_indices = hdf5(datafile,'sort_indices')
    reverse_indices = hdf5(datafile,'reverse_indices')
    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    CTn = hdf5(datafile,'CTn')
    STn = hdf5(datafile,'STn')
    FT = filter_transmittance(T0_unknown=T0_unknown)
    DR = detector_responsivity()

    CTn_flat = CTn.flatten()[sort_indices]
    STn_flat = STn.flatten()[sort_indices]
    FT_flat = FT.flatten()[sort_indices]
    DR_flat = DR.flatten()[sort_indices]

    # Try execute read_logfile(), which copies relevant data to analysis hdf5 file.
    try:
        read_logfile(dataset)
    except:
        pass
    
    pathname = data_hdf5(dataset)
    filename = path.basename(pathname)
    print('Processing {:0.2f} GB in {}'.format(path.getsize(data_hdf5(dataset))/1024**3,filename))
    with h5py.File(pathname,'r') as f:
        # Generate BSmask for dataset
        x0i = int(X0)
        y0i = int(Y0)
        sub = f['images'][:,y0i-12:y0i+12,x0i-12:x0i+12]-100
        BS = zeros(UC.shape)
        BS[y0i-12:y0i+12,x0i-12:x0i+12] = sub.mean(axis=0)
        BSmask = beamstop_mask(BS)

        # Generate mask and correction factor needed to process dataset
        pixel_mask = BankID>0
        mask = (DPmask | BNKmask | BSmask | (UC < 0.6) | (BankID==0))
        PGUC = (POL*GEO*UC)
        PGUC[mask] = 1
        PGUC_flat = PGUC.flatten()[sort_indices]

        # If omit uint8 array doens't exist, create it.
        N_images = len(sub)
        try:
            omit = hdf5(analysis_hdf5(dataset),'omit')
        except:
            omit = zeros(N_images,uint8)

        M0_BCH = hdf5(analysis_hdf5(dataset),'M0')

        if H_dataset is not None:
            H = hdf5(analysis_hdf5(H_dataset),'Pmean')
            H_flat = H.flatten()[sort_indices]
            M0_H = hdf5(analysis_hdf5(H_dataset),'M0')
            M0_H = M0_H[M0_H>0].mean()
        else:
            H_flat = zeros(len(q_flat))
            M0_H = 1.0

        if CH_dataset is not None:
            CH = hdf5(analysis_hdf5(CH_dataset),'Pmean')
            CH_flat = CH.flatten()[sort_indices]
            M0_CH = hdf5(analysis_hdf5(CH_dataset),'M0')
            M0_CH = M0_CH[M0_CH>0].mean()
        else:
            CH_flat = zeros(len(q_flat))
            M0_CH = 1.0

        if G_dataset is not None:
            M0_G = hdf5(analysis_hdf5(G_dataset),'M0')
            M0_G = M0_G[M0_G>0].mean()
        else:
            M0_G = M0_BCH[M0_BCH>0].mean()


        # Create arrays needed to process dataset
        S = zeros((N_images,len(q)))
        sigS = zeros((N_images,len(q)))
        OP_flat = zeros(len(q_flat),uint16)
        IP = zeros(N_images)
        OVLD = zeros(mask.shape,bool)
        for i in range(N_images):
            print('processing image {} of {}'.format(i+1,N_images),end='\r')
            image = f['images'][i]
            OVLD = image==2**16-1
            BCH = (image.astype(float) - 100*pixel_mask)/gain*~(mask | OVLD)
            IP[i] = BCH.sum()
            omit[i] |= (IP[i] < 0)
            # Subtract scaled A_flat and apply POL, GEO, and UC corrections
            Mask_flat = (mask | OVLD).flatten()[sort_indices]
            BCH_flat = BCH.flatten()[sort_indices]

            # Bs_flat = (BCH_flat - (M0_BCH[i]/M0_H)*H_flat)/PGUC_flat
                    
            CHs_flat = CTn_flat*CH_flat/M0_CH + H_flat/M0_H*(1-CTn_flat)
            Bs_flat = M0_G * (BCH_flat/M0_BCH[i] - CHs_flat) / (STn_flat * PGUC_flat * FT_flat * DR_flat)
            Bs_flat[Mask_flat] = 0
            
            sigBCH_flat = sqrt(1.35*abs(BCH_flat)+3)
            # sigBs_flat = sigBCH_flat/PGUC_flat
            sigBs_flat = M0_G * sigBCH_flat/M0_BCH[i] / (STn_flat * PGUC_flat * FT_flat * DR_flat)

            # Find outliers and zero corresponding pixels in Pci_flat
            outliers_flat = outlier_pixels(q,qbin1,qbin2,Bs_flat,sigBs_flat)
            OP_flat += outliers_flat
            Bs_flat[outliers_flat] = 0
            # Generate weights W for Pci_flat
            W_flat = PGUC_flat*(Bs_flat>0)
            # Compute S(q) and sigS(q)
            S[i],sigS[i],Si_flat = integrate(Bs_flat,W_flat,q_flat,qbin1,qbin2,q)
            # Update outlier pixel count, excluding Gmask
            
    OP_count = OP_flat[reverse_indices].reshape(image.shape)
    pathname = analysis_hdf5(dataset)
    hdf5(pathname,'q',q)
    hdf5(pathname,'S',S)
    hdf5(pathname,'sigS',sigS)
    hdf5(pathname,'OP_count',OP_count)
    hdf5(pathname,'M0',M0_BCH)
    hdf5(pathname,'IP',IP)
    hdf5(pathname,'omit',omit)

def read_logfile(dataset):
    """Read logfile assigned to dataset and write results to
    hdf5 file in analysis_dir."""
    from numpy import loadtxt,int8
    import warnings
    warnings.filterwarnings('ignore', 'no explicit representation of timezones available')
    warnings.filterwarnings('ignore', 'Input line 1 contained no data')

    from scipy.interpolate import UnivariateSpline
    logfile = dataset_logfile(dataset)
    pathname = analysis_hdf5(dataset)
    if 'PumpProbe' in logfile:
        timestamps = loadtxt(logfile,usecols = (0),delimiter='\t',dtype='datetime64[s]').astype(float)
        hdf5(pathname,'timestamps',timestamps)
        data = loadtxt(logfile,usecols = (2,3,4,5,8),delimiter='\t')
        Delay = data[:,0]
        Repeat1 = data[:,1].astype(int8)
        T_target = data[:,2]
        Repeat2 = data[:,3].astype(int8)
        T_rtd = data[:,4]
        hdf5(pathname,'Delay',Delay)
        hdf5(pathname,'Repeat1',Repeat1)
        hdf5(pathname,'T_target',T_target)
        hdf5(pathname,'Repeat2',Repeat2)
        hdf5(pathname,'T_rtd',T_rtd)
    elif 'Tramp' in logfile:
        timestamps = loadtxt(logfile,usecols=0,delimiter='\t',dtype='datetime64[us]').astype(float)/1e6
        filenames = loadtxt(logfile,usecols=1,delimiter='\t',dtype=str)
        T_target = loadtxt(logfile,usecols=2,delimiter='\t')
        Repeat1 = loadtxt(logfile,usecols=3,delimiter='\t')
        ring_current = loadtxt(logfile,usecols=4,delimiter='\t')
        bunch_current = loadtxt(logfile,usecols=5,delimiter='\t')
        T_rtd = loadtxt(logfile,usecols=6,delimiter='\t')
        hdf5(pathname,'timestamps',timestamps)
        hdf5(pathname,'filenames',filenames)
        hdf5(pathname,'T_rtd',T_rtd)
        hdf5(pathname,'Repeat1',Repeat1)
        hdf5(pathname,'ring_current',ring_current)
        hdf5(pathname,'bunch_current',bunch_current)
        hdf5(pathname,'T_target',T_target)
        UP = zeros(len(T_target),bool)
        DOWN = zeros(len(T_target),bool)
        UP[1:] = (T_target[1:] - T_target[:-1]) > 0
        DOWN[1:] = (T_target[1:] - T_target[:-1]) < 0
        hdf5(pathname,'UP',UP)
        hdf5(pathname,'DOWN',DOWN)
        # Fit T_rtd with univariate spline to smooth ramping phase
        x = arange(len(T_rtd))
        w = ones(len(T_rtd))
        w[~UP & ~DOWN] = 4
        s_scale = 0.04
        s = s_scale*0.5*(w[1:]*w[:-1]*(T_rtd[1:] - T_rtd[:-1])**2).sum()
        us = UnivariateSpline(x,T_rtd,w=w,s=s)
        N_knots = len(us.get_knots())
        # Determine T_cap from us fit of T_rtd
        T_cap = us(x+lag)
        hdf5(pathname,'T_cap',T_cap)
        filename = path.basename(pathname)
        #chart_xy_fit(x,T_rtd,us(x),'univariate spline fit of T_rtd\n{}\ns_scale = {}; N_knots = {}'.format(filename,s_scale,N_knots),ms=3)
    else:
        timestamps = loadtxt(logfile,usecols = (0),delimiter='\t',dtype='datetime64[s]').astype(float)
        hdf5(pathname,'timestamps',timestamps)

def z_statistic(N):
    """Returns z, which corresponds to the threshold in units of sigma where
    the probability of measuring a value greater than z*sigma above the mean
    is 1 out of N."""
    from numpy import sqrt
    from scipy.special import erfinv
    return sqrt(2)*erfinv(1-2/N)

def outlier_pixels(q,qbin1,qbin2,Ic_flat,sigma_flat):
    """Flags outliers in Ic_flat. Returns OP_flat."""
    from numpy import zeros,median,sqrt,arange
    OP_flat = zeros(len(Ic_flat),bool)
    mask_flat = Ic_flat == 0
    # Allow one false positive per image
    Zstat = z_statistic((~mask_flat).sum()) 
    # Determine sigma_flat and OP_flat relative to median from binned values of q
    for j in range(len(q)):
        first = qbin1[j]
        last  = qbin2[j]
        bin_select = arange(first,last)
        mask_select = ~mask_flat[bin_select]
        N = mask_select.sum()
        if N > 2:
            Ics = Ic_flat[bin_select[mask_select]]
            Ics_median = median(Ics)
            outlier = abs((Ics - Ics_median)/sigma_flat[bin_select[mask_select]]) > Zstat
            OP_flat[bin_select[mask_select]] = outlier
    return OP_flat

def integrate(I_flat,W_flat,q_flat,qbin1,qbin2,q):
    """Returns the weighted mean (S) and standard deviation of the mean (sigS)
    for each q bin defined by qbin1 and qbin2. Requires as input I_flat [photons], 
    W_flat (weights), and q_flat. The weights must be zeroed for defective pixels 
    and outliers. The data in each bin are fitted to a straight line (Ii = a + b*qi)
    with S representing the value of the line at the corresonding value of q, and 
    sigS representing the estimated standard deviation of S. Also returns S_flat, 
    the weighted least-squares linear fit evaluated across q_flat."""
    from numpy import cumsum,sqrt,zeros,ones_like,nan_to_num
    S = zeros(len(q))
    sigS = zeros(len(q))
    S_flat = ones_like(q_flat)
    # Copy q_flat, I_flat, V_flat, W_flat; convert to float64 to compute statistics accurately
    qc_flat = q_flat.copy().astype(float)
    Ic_flat = nan_to_num(I_flat).astype(float)
    Wc_flat = W_flat.copy().astype(float)
    # Calculate S and sigS from weighted linear least-squares fit of values in each q bin
    cs_I  = cumsum(Ic_flat*Wc_flat)
    cs_I2 = cumsum(Ic_flat**2*Wc_flat)
    cs_q  = cumsum(qc_flat*Wc_flat)
    cs_q2 = cumsum(qc_flat**2*Wc_flat)
    cs_qI = cumsum(qc_flat*Ic_flat*Wc_flat)
    cs_W  = cumsum(Wc_flat)
    cs_N =  cumsum(Wc_flat>0)
    first = 0
    for j in range(len(q)):
        if qbin1[j] > 0:
            first = qbin1[j] - 1
        last  = qbin2[j] - 1
        N = cs_N[last] -  cs_N[first]
        if N > 2:
            sI  =  cs_I[last] -  cs_I[first]
            sI2 = cs_I2[last] - cs_I2[first]
            sq  =  cs_q[last] -  cs_q[first]
            sq2 = cs_q2[last] - cs_q2[first]
            sqI = cs_qI[last] - cs_qI[first]
            sW  =  cs_W[last] -  cs_W[first]
            delta = sW*sq2 - sq**2
            a =  (1/delta)*(sq2*sI - sq*sqI)
            b =  (1/delta)*(sW*sqI - sq*sI)
            S[j] = a + b*q[j]
            variance = (sI2 - 2*a*sI - 2*b*sqI + 2*a*b*sq + b*b*sq2 + a*a*sW)/sW/N
            sigS[j] = sqrt(variance)
            if j == len(q)-1:
                S_flat[first:] = a + b*q_flat[first:]
            else:
                S_flat[first:last] = a + b*q_flat[first:last]
    return S,sigS,S_flat

def find_datasets(data_dir,term='ePIX_detector',exclude_folders=['xray_traces','laser_traces']):
    """Returns array of datasets in data_dir in which 'term' is found; 
    sorted according to getmtime for the corresponding log file."""
    from os import walk
    from os.path import getmtime
    from numpy import argsort
    datasets = []
    for root, dirs, files in walk(data_dir):
        # Modify dirs in-place to exclude specified folders
        dirs[:] = [d for d in dirs if d not in exclude_folders]
        if (term in root) & ('Dark' not in root):
            datasets.append(root.split(term)[0])
    timestamps = [getmtime(dataset_logfile(dataset)) for dataset in datasets]
    sort_order = argsort(timestamps)
    return array(datasets)[sort_order]

def dataset_logfile(dataset):
    return dataset + (dataset.split('/')[-2])+'.log'

def laser_tracefiles_sorted(dataset):
    from os import listdir
    from os.path import getmtime
    from numpy import argsort
    pathname = dataset + 'laser_traces/'
    trace_files = array([pathname+name for name in listdir(pathname)])
    sort_order = argsort(array([getmtime(file) for file in trace_files]))
    return array(trace_files)[sort_order]

def xray_tracefiles_sorted(dataset):
    from os import listdir
    from os.path import getmtime
    from numpy import argsort
    pathname = dataset + 'xray_traces/'
    trace_files = array([pathname+name for name in listdir(pathname)])
    sort_order = argsort(array([getmtime(file) for file in trace_files]))
    return array(trace_files)[sort_order]

def data_hdf5(dataset):
    return dataset + 'ePIX_detector/' + (dataset.split('/')[-2]) +'.hdf5'

def analysis_hdf5(dataset):
    from os.path import relpath, basename
    dataset = dataset.rstrip("/")
    pathname = f"{analysis_root}{relpath(dataset, data_root)}/{basename(dataset)}.hdf5"
    return pathname

def initialize_datafile(X0=1409.5,Y0=1450.5,detector_distance=detector_distance,gain=48.6*12/9.5/16):
    """Generates information needed to process scattering data."""
    from saxs_waxs import hdf5
    from saxs_waxs import capillary_transmittance, sample_transmittance
    from numpy import pi,sin,cos,sqrt,arctan,arctan2,fromfunction
    
    # Needed parameters
    image_shape = (1675, 1675)

    def frame_to_image_lut():
        """Returns lookup table that maps frame of stacked modules to 
        an image. The corner indices (yi,xi) for each module
        position the midpoint of the modules at the same pixel 
        position determined from metrology on the BioCARS ePix detector."""
        from numpy import zeros,arange,uint32,rot90,argsort
        na,nx,ny = 16,352,384
        iframe = arange(na*nx*ny).reshape((na*ny,nx)).astype(uint32)+1
        quad = zeros((ny+2,nx+2))
        image = zeros((1675,1675),uint32)
        yi = array([1251,1251,832,834,1286,880,1282,880,73,69,489,489,4,408,5,409])
        xi = array([2,409,4,409,1252,1252,830,830,1284,881,1281,879,73,71,492,490])
        for i in range(na):
                module = iframe[i*ny:(i+1)*ny]
                quad[:192,:176] = module[:192,:176]
                quad[:192,-176:] = module[:192,-176:]
                quad[-192:,:176] = module[-192:,:176]
                quad[-192:,-176:] = module[-192:,-176:]
                rquad = rot90(quad,i//4+1)
                ys,xs = rquad.shape
                image[yi[i]:yi[i]+ys,xi[i]:xi[i]+xs] = rquad
        image_flat = image.ravel()
        N_zeros = (image_flat==0).sum()
        return argsort(image_flat)[N_zeros:]

    def Bank_mask():
        IDs = [9,10,11,12,25,26,27,28]
        return array([BankID == ID for ID in IDs]).any(axis=0)

    def BS_mask():
        """Returns beamstop pixel mask."""
        from numpy import indices,sqrt
        BSmask = zeros(image_shape,bool)
        y_indices,x_indices = indices(image_shape)
        r = 4.5
        BSmask[sqrt((y_indices-Y0)**2+(x_indices-X0)**2) < r] = True
        return BSmask
    
    def defective_pixel_mask():
        """Returns defective pixel mask as hardcoded here."""
        from numpy import zeros
        frame = zeros((16*384,352),bool)
        frame[16*384-4528,176:] = True
        frame[16*384-4529,176:] = True
        frame[16*384-5559,176:] = True
        frame[16*384-5560,176:] = True
        image = frame_to_image(frame)
        image[140,89] = True
        image[139,90] = True
        image[138,90] = True
        image[25,503] = True    # Erratic
        image[1473,1005] = True # Corner pixel
        image[1480,1427] = True # Corner pixel
        image[1480,1430] = True # Corner pixel
        image[1477,1427] = True # Corner pixel
        image[1477,1430] = True # Corner pixel
        return image
    
    def shadow_mask():
        """Mask shadowed region near the center of the detector."""
        from numpy import indices,sqrt
        SWmask = zeros(image_shape,bool)
        # Shadow from tube near center of detector
        y_indices,x_indices = indices(image_shape)
        xc,yc = (818,817)
        r = 40
        SWmask[sqrt((y_indices-yc)**2+(x_indices-xc)**2) < r] = True
        return SWmask
    
    def qbin(q_flat):
        """Generates q, qbin1, and qbin2 given q_flat and dq, where
        [qbin1[i]:qbin2[i]] specifies the range of q_flat assigned
        to each bin i. """
        from numpy import arange,rint,roll,where
        # Define qmin, and qmax
        dq = 0.0025
        qmin = 0.0175
        qmax = q_flat.max()-11*dq/2
        # Define q, values between qmin and qmax in integer multiples of dq
        q0 = round((qmin + dq/2)/dq)*dq
        q1 = round((qmax - dq/2)/dq)*dq
        q = arange(q0,q1,dq)
        # Define qbin1 and qbin2, the start and stop indices for dq bins along q_flat
        q_boundary = rint(q_flat/dq)
        qbin2 = (where(q_boundary != roll(q_boundary,-1))[0]+1)[int(q0/dq):int(q1/dq)+1][:len(q)]
        qbin1 = roll(qbin2,1)
        qbin1[0] = where(q_flat > qmin-dq/2)[0][0]
        qbin2[-1] = len(q_flat)-1
        return q,qbin1,qbin2

    def frame_to_image(frame):
        """Converts frame to image."""
        from numpy import zeros
        image = zeros((1675,1675),frame.dtype)
        image0 = image.ravel()
        image0[Image_lut] = frame.ravel()
        return image.copy()
    
    def bank_ID():
        """ Returns BankID image (int16), which contains unique IDs
        for each bank of all four quadrants. Each bank corresponds
        to a 176x48 array of pixels and their IDs are numbered in the 
        order they appear in the datastream (DS) starting at 1."""
        from numpy import tile,arange,vstack,hstack,uint16
        q3 = tile(tile(arange(16)[::-1],(48,1)).T.flatten() + 1,(176,1)).T
        q2 = tile(tile(arange(16,32)[::-1],(48,1)).T.flatten() + 1,(176,1)).T
        q0 = tile(tile(arange(32,48)[::-1],(48,1)).T.flatten() + 1,(176,1)).T
        q1 = tile(tile(arange(48,64)[::-1],(48,1)).T.flatten() + 1,(176,1)).T
        Qframe = vstack((hstack((q1,q0)),hstack((q2,q3))))
        frame = vstack((Qframe+192,Qframe+128,Qframe+64,Qframe)).astype(uint16)
        return frame_to_image(frame)    
    
    def module_ID():
        """ Returns ModuleID image (uint8), which contains unique IDs
        for all sixteen modules. Each module corresponds
        to a 384x352 array of pixels and their IDs are numbered in the 
        order they appear in the datastream starting at 1."""
        from numpy import array,uint8,ones
        module = ones((384,352))
        frame = []
        for i in range(1,17)[::-1]:
            frame.append(module*i)
        frame = array(frame).reshape(-1,352).astype(uint8)
        return frame_to_image(frame) 
    
    def perimeter_pixel_mask():
        """Returns perimeter pixel mask."""
        from numpy import zeros
        frame = zeros((16*384,352),bool)
        frame[:,0] = True
        frame[:,-1] = True
        frame[0,:] = True
        frame[-1,:] = True
        frame[:,175] = True
        frame[:,176] = True
        for i in range(16):
            frame[i*384-1:i*384+1,:] = True
            frame[i*384+191:i*384+193,:] = True
        return frame_to_image(frame) 

    def perimeter_2pixel_mask():
        """Returns 2-pixel wide perimeter pixel mask."""
        from numpy import zeros
        frame = zeros((16*384,352),bool)
        frame[:,:2] = True
        frame[:,-2:] = True
        frame[:2,:] = True
        frame[-2:,:] = True
        frame[:,175:177] = True
        for i in range(16):
            frame[i*384-2:i*384+2,:] = True
            frame[i*384+191:i*384+193,:] = True
        return frame_to_image(frame) 
    
    def perimeter_4pixel_mask():
        """Returns 4-pixel wide perimeter pixel mask."""
        from numpy import zeros
        frame = zeros((16*384,352),bool)
        frame[:,:4] = True
        frame[:,-4:] = True
        frame[:4,:] = True
        frame[-4:,:] = True
        frame[:,175:177] = True
        for i in range(16):
            frame[i*384-4:i*384+4,:] = True
            frame[i*384+191:i*384+193,:] = True
        return frame_to_image(frame) 

    def perimeter_6pixel_mask():
        """Returns 6-pixel wide perimeter pixel mask."""
        from numpy import zeros
        frame = zeros((16*384,352),bool)
        frame[:,:6] = True
        frame[:,-6:] = True
        frame[:6,:] = True
        frame[-6:,:] = True
        frame[:,175:177] = True
        for i in range(16):
            frame[i*384-6:i*384+6,:] = True
            frame[i*384+191:i*384+193,:] = True
        return frame_to_image(frame) 

    def f_q(iy, ix):
        theta = f_phi(iy, ix) / 2
        q = 4 * pi * sin(theta) / wavelength
        return q

    def f_phi(iy, ix):
        x = (ix - X0) * pixelsize
        y = (iy - Y0) * pixelsize
        r = sqrt(x ** 2 + y ** 2)
        phi = arctan(r / detector_distance)
        return phi
    
    def f_psi(iy, ix):
        x_rot = (ix - X0)*cos(pi/4) - (iy - Y0)*sin(pi/4)
        y_rot = (ix - X0)*sin(pi/4) + (iy - Y0)*cos(pi/4)
        return arctan2(y_rot, x_rot)

    # Generate q,phi, and psi
    Q = fromfunction(f_q, image_shape)
    PHI = fromfunction(f_phi, image_shape)
    PSI = fromfunction(f_psi, image_shape)

    # Generate image lookup table
    Image_lut = frame_to_image_lut()

    # Generate BankID and ModuleID
    BankID = bank_ID()
    ModuleID = module_ID()

    # Generate bool masks
    PPmask = perimeter_pixel_mask()
    PP2mask = perimeter_2pixel_mask()
    PP4mask = perimeter_4pixel_mask()
    PP6mask = perimeter_6pixel_mask()
    DPmask = defective_pixel_mask()
    SDWmask = shadow_mask()
    BSmask = BS_mask()
    BNKmask = Bank_mask()
    
    # Calculate geometry and polarization corrections and combine with Gain
    pixel_mask = BankID > 0
    Q *= pixel_mask
    PHI *= pixel_mask
    PSI *= pixel_mask
    POL = 0.5*(1+cos(PHI)**2-cos(2*PSI)*sin(PHI)**2)*pixel_mask
    GEO = cos(PHI)**3*pixel_mask
    PG = POL*GEO

    # Generate sort_indices and reverse_indices
    sort_indices = argsort(Q.flatten())
    reverse_indices = argsort(sort_indices)
    q_flat = Q.flatten()[sort_indices]
    psi_flat = PSI.flatten()[sort_indices]

    # Generate q, qbin1 and qbin2
    q,qbin1,qbin2 = qbin(q_flat)

    # Absorbance corrections
    t = 0.024  # capillary wall thickness in mm
    R = 0.136  # capillary inner radius in mm
    THETA = PHI / 2
    mu_Xe = 0.2913
    mu_c = 2.30835223
    CT = capillary_transmittance(THETA,PSI,R,t,mu_s=mu_Xe,mu_c=mu_c)
    ST = sample_transmittance(THETA,PSI,R,t,mu_s=mu_Xe,mu_c=mu_c)
    CTn = CT / CT.max()
    STn = ST / ST.max()

    # Save information in hdf5 file
    hdf5(datafile,'PPmask',PPmask)
    hdf5(datafile,'PP2mask',PP2mask)
    hdf5(datafile,'PP4mask',PP4mask)
    hdf5(datafile,'PP6mask',PP6mask)
    hdf5(datafile,'DPmask',DPmask)
    hdf5(datafile,'BSmask',BSmask)
    hdf5(datafile,'Q',Q)
    hdf5(datafile,'PHI',PHI)
    hdf5(datafile,'PSI',PSI)
    hdf5(datafile,'POL',POL)
    hdf5(datafile,'GEO',GEO)
    hdf5(datafile,'PG',PG)
    hdf5(datafile,'q',q)
    hdf5(datafile,'qbin1',qbin1)
    hdf5(datafile,'qbin2',qbin2)
    hdf5(datafile,'sort_indices',sort_indices)
    hdf5(datafile,'reverse_indices',reverse_indices)
    hdf5(datafile,'Image_lut',Image_lut)
    hdf5(datafile,'q_flat',q_flat)
    hdf5(datafile,'psi_flat',psi_flat)
    hdf5(datafile,'BankID',BankID)
    hdf5(datafile,'BNKmask',BNKmask)
    hdf5(datafile,'SDWmask',SDWmask)
    hdf5(datafile,'ModuleID',ModuleID)
    hdf5(datafile,'X0',X0)
    hdf5(datafile,'Y0',Y0)
    hdf5(datafile,'gain',gain)
    hdf5(datafile,'CT',CT)
    hdf5(datafile,'ST',ST)
    hdf5(datafile,'CTn',CTn)
    hdf5(datafile,'STn',STn)

    pathnames = find_datasets(data_dir)
    pathnames = [pathname.rstrip("/") for pathname in pathnames]
    dataset_pathnames = [f"{path.relpath(pathname, data_dir)}/{path.basename(pathname)}.hdf5" for pathname in pathnames]
    datasets = [path.basename(pathname) for pathname in pathnames]
    hdf5(datafile,'dataset_pathnames', dataset_pathnames)
    hdf5(datafile,'datasets', datasets)


def dataset_trc(dataset):
    """Process x-ray trace files and generate nC, the integrated 
    electron charge in nano coulombs (assuming 50 ohm load). As the 
    number of trigger pulses can be fewer than 40, the integral is 
    extrapolated to 40 trigger pulses to ensure all trace files are 
    on the same scale. Results are sorted according to trc_timestamps.
    Writes trc_filenames, trc_timestamps, trc_nC, and trc_N in dataset."""
    from numpy import zeros,argsort,sort
    from os import path,listdir
    from time import time
    def V_integrate(V):
        """Determine integral of trace up to zero crossing. Use statisical 
        criterion to define range of background."""
        from numpy import cumsum,arange,where,sqrt
        N = arange(len(V))+1
        IV = cumsum(V)
        IV2 = cumsum(V**2)
        M1 = IV/N
        M2 = (IV2/N-M1**2)
        try:
            i1 = where((N > 100) & (abs(V-M1) > 5*sqrt(M2)))[0][0]
            bkg = M1[i1]
            i2 = where((N > i1) & (V > bkg))[0][-1]
            integral = V[i1:i2].sum() - bkg*(i2-i1)
        except:
            integral = 0
        return integral
    t0 = time()
    # Find trc_filenames for dataset
    pathname = analysis_hdf5(dataset)

    # prefix = '/Volumes/mirrored_femto-data2'
    # xray_filename = prefix + xray_filename.split('/net/femto-data2.niddk.nih.gov')[-1]
    
    dirname = dataset + 'xray_traces'
    try:
        filenames = sort(listdir(dirname))
        trc_filenames = [path.join(dirname,filename) for filename in filenames]
        # Extract information from trc_filenames
        trc_timestamps = zeros(len(trc_filenames))
        trc_nC = zeros(len(trc_filenames))
        trc_N = zeros(len(trc_filenames),'uint16')
        for j,filename in enumerate(trc_filenames):
            trc_dict = trc_read(filename)
            trc_timestamps[j] = trc_dict['timestamp_h']
            trig_times = trc_dict['trig_times']
            V_gain = trc_dict['vertical_gain']
            traces = trc_dict['data2D']
            N = len(trig_times)
            trc_N[j] = N
            trc_nC[j] = V_gain*V_integrate(traces.sum(axis=0))/50
        # Sort according to trc_timestamps
        sort_order = argsort(trc_timestamps)
        trc_filenames = [sort_order]
        trc_timestamps = trc_timestamps[sort_order]
        trc_nC = trc_nC[sort_order]
        trc_N = trc_N[sort_order]
        rn = range(len(sort_order))
        if (rn != sort_order).any():
            print('Out-of-order trace files found in {}'.format(dataset))

        # Write results
        hdf5(pathname,'trc_filenames',trc_filenames)
        hdf5(pathname,'trc_timestamps',trc_timestamps)
        hdf5(pathname,'trc_nC',trc_nC)
        hdf5(pathname,'trc_N',trc_N)
        print(f'{time()-t0:0.3f} seconds to process {len(trc_N)} trace files in {dataset}')
    except Exception as x:
        logging.exception(f'{dirname}: {x}')

def trc_read(trc_file):
    """Read trc file and return dictionary containing relevant information:
        timestamp_h (header timestamp shifted to correspond to last trigger pulse)
        timestamp_f (file timestamp)
        trig_times (relative to first trigger pulse in seconds)
        trig_offsets (sampling offsets for each trigger pulse in seconds)
        tj (timebase for traces in nanoseconds)
        data2D (2D array of counts)."""
    from numpy import frombuffer,int8,int16,around,float64,around,array,arange
    from struct import unpack
    from time import mktime
    from datetime import datetime
    from os.path import getmtime
    # Read trc_file and extract header
    with open(trc_file, mode='rb') as f:
        raw = f.read() 
    startpos = raw.find(b'WAVEDESC')
    header = raw[startpos:startpos+346]

    # Determine file timestamp
    timestamp_f = getmtime(trc_file)

    # Determine if data are stored as 1 or 2 byte integers
    comm_type, = unpack('<h',header[32:34])
    if comm_type:
        number_type = int16 
        byte_length = 2 
    else:
        number_type = int8
        byte_length = 1

    # Horizontal (tj)
    horiz_interval, = unpack("<f", header[176:180])
    horiz_offset, = unpack("<d", header[180:188])
    wave_array_count, = unpack("<i", header[116:120])
    subarray_count, = unpack("<i", header[144:148]) # number of trigger events
    N_traces = subarray_count
    N_pts = wave_array_count//subarray_count
    dt = horiz_interval*1e9 # in nanoseconds
    tj = horiz_offset*1e9 + arange(N_pts)*dt

    # Vertical (Vij)
    vertical_gain, = unpack("<f", header[156:160])
    vertical_offset, = unpack("<f", header[160:164])
    data = raw[len(raw)-byte_length*N_traces*N_pts:]
    data2D = frombuffer(data,number_type).reshape(N_traces,N_pts)

    # Determine timestamp for first trigger
    seconds, = unpack("<d", header[296:304]) # double (8-byte real number)
    minutes, = unpack("<b", header[304:305])
    hours, = unpack("<b", header[305:306])
    days, = unpack("<b", header[306:307])
    month, = unpack("<b", header[307:308])
    year, = unpack("<h", header[308:310]) # short (2-byte signed integer)

    date_time = datetime(year, month, days, hours, minutes, int(seconds), int(1e6*(seconds-int(seconds))))
    timestamp_h = mktime(date_time.timetuple()) + date_time.microsecond*1.e-6 

    # Use timestamp_f as reference to convert timestamp_h to universal time
    timestamp_h += 3600*around((timestamp_f - timestamp_h)/3600)

    # Shift timestamp_h to correspond to last trigger pulse
    trig_times = array([0])
    if N_traces > 1:
        wave_descriptor, = unpack("<i", header[36:40]) # int (4-byte signed integer)
        user_text, = unpack("<i", header[40:44])
        trig_time_array, = unpack("<i", header[48:52])
        trigtimes_startpos = startpos + wave_descriptor + user_text
        trig_data = raw[trigtimes_startpos:trigtimes_startpos+trig_time_array]
        trig_stats = frombuffer(trig_data, dtype=float64, count=trig_time_array//8).reshape(2, -1, order='F')
        trig_times = trig_stats[0]
        trig_offsets = trig_stats[1]
        # Shift timestamp_h to coincide with last trigger
        timestamp_h += trig_times[-1]
    
    # Construct and return dictionary of relevant parameters
    trc_dict = {'date_time':date_time,'timestamp_f':timestamp_f,'timestamp_h':timestamp_h,'trig_times':trig_times,'trig_offsets':trig_offsets,'data2D':data2D,'vertical_gain':vertical_gain,'vertical_offset':vertical_offset,'tj':tj}
    return trc_dict

def beamtime_PumpProbe_temp(chart=False):
    """Assigns temp in PumpProbe datasets, but requires a matching Tramp dataset.
    If there are multiple matching Tramp datasets, selects the nearest one."""
    from numpy import array, where
    datasets = find_datasets(data_dir)
    PP_datasets = array([dataset for dataset in datasets if 'PumpProbe' in dataset])
    TR_datasets = array([dataset for dataset in datasets if 'Tramp' in dataset])
    for PP_dataset in PP_datasets:
        i = where(datasets == PP_dataset)[0][0]
        # Find nearest suitable Tramp dataset
        name = PP_dataset.replace('PumpProbe','Tramp')[:-2] # Strip repeat number
        matches = array([TR_dataset for TR_dataset in TR_datasets if name in TR_dataset])
        if len(matches) > 0:
            di = array([abs(i-where(datasets == match)[0][0]) for match in matches])
            TR_dataset = matches[di==di.min()][0]
            print(f'PumpProbe_temp("{PP_dataset}","{TR_dataset}",{chart})')
            # PumpProbe_temp(PP_dataset,TR_dataset,chart)
        else:
            print(f'No Tramp match found for {PP_dataset}')
        
def PumpProbe_temp(PP_dataset,TR_dataset,chart=False):
    """Processes PP_dataset to determine 'temp'. Assumes 'temp' 
    already exists in TR_dataset. Performs SVD on combined PumpProbe 
    and corresponding Tramp data; generates a temperature calibration 
    curve via a polynomial fit of TR_dataset 'temp' as a function of 
    corresponding 'Vn' vectors; and then uses this function to assign 
    'temp' to PP_dataset."""
    from numpy import zeros,polyfit,poly1d,nan,concatenate,argsort,arange
    from time import time
    t0 = time()

    # Determine pathnames for PP_dataset and TR_dataset
    PP_pathname = analysis_hdf5(PP_dataset)
    TR_pathname = analysis_hdf5(TR_dataset)

    # Select restricted range of q
    N_poly = 6
    qmin=1.4
    qmax=3.5
    q = hdf5(PP_pathname,'q')
    q_select = (q > qmin) & (q < qmax)
    qs = q[q_select]

    try:
        PP_S = hdf5(PP_pathname,'S')
        TR_S = hdf5(TR_pathname,'S')
        TR_temp = hdf5(TR_pathname,'T_cap')
        RTD_temp = hdf5(PP_pathname,'T_rtd')
        PP_select = hdf5(PP_pathname,'omit') == 0
        TR_select = hdf5(TR_pathname,'omit') == 0
        # Concatenate q-selected S and analyze with SVD
        S = concatenate((TR_S[TR_select],PP_S[PP_select]))[:,q_select]
        UT,s,V = SVD(S)
        # Normalize VT[1] and use polynomial to generate temperature calibration using TR_select
        Vn = V[1]/V[0]
        NTR = TR_select.sum()
        pf = polyfit(Vn[:NTR],TR_temp[TR_select],N_poly)
        # Assign temp to PP_select
        PP_temp = nan*zeros(len(PP_select))
        PP_temp[PP_select] = poly1d(pf)(Vn[NTR:])
        hdf5(PP_pathname,'T_cap',PP_temp)
        print(f'processed {PP_dataset} in {time()-t0:0.3f} seconds')
        if chart:
            from charting_functions import chart_UTsV,chart_xy_symbol,chart_xy_rainbow
            chart_UTsV(qs,UT,s,V,PP_dataset+' combined with ' + TR_dataset)
            chart_xy_symbol(arange(len(PP_S)),PP_temp - RTD_temp,'PP_temp - RTD_temp\n{}'.format(PP_dataset),x_label='image index')
            chart_xy_rainbow(q,TR_S[argsort(TR_temp)],'S\n{}'.format(TR_dataset),x_label='q',logx=True,logy=True)
    except Exception as x:
        logging.exception(f'Failed to process {PP_dataset}: {x}')

def Ss_to_Ssq(Ss,sigSs,qs,qb):
    """Returns Ssq and sigSsq, which represent merged/averaged
    Ss evaluated at qb."""
    from numpy import array
    from saxs_waxs import linefit_weighted
    dq = qb[1]-qb[0]
    Ssq = []
    sigSsq = []
    for q in qb:
        select = (qs > q - 2*dq) & (qs < q + 2*dq)
        a,b,stdev = linefit_weighted(qs[select],Ss[:,select].T,sigSs[:,select].T)
        Ssq.append(a + q*b)
        sigSsq.append(stdev)
    Ssq = array(Ssq).T
    sigSsq = array(sigSsq).T
    return Ssq,sigSsq

def S_to_ST(S,sigS,T,Tb,bin_range=4.2):
    """Returns ST and sigST, interpolated S and sigS at Tb.
    Uses weighted least squares fit of data selected within 
    bin_range of each value in Tb."""
    from numpy import array
    from saxs_waxs import linefit_weighted
    ST = []
    sigST = []
    dT = bin_range/2
    for Ti in Tb:
        select = (T > Ti - dT) & (T < Ti + dT)
        a,b,stdev = linefit_weighted(T[select],S[select],sigS[select])
        ST.append(a + Ti*b)
        sigST.append(stdev)  
    ST = array(ST)
    sigST = array(sigST)
    return ST,sigST 

def beamtime_M0_M1X_M1Y():
    datasets = find_datasets(data_dir)
    for dataset in datasets:
        try:
            dataset_M0_M1X_M1Y(dataset)
        except Exception as x:
            logging.exception(f"{dataset}: {x}")

def dataset_M0_M1X_M1Y(dataset):
    import h5py
    dataset_basename = path.basename(dataset.strip("/"))
    pathname = data_hdf5(dataset)
    with h5py.File(pathname,'r') as f:
        N_images = len(f['images'])
    M0, M1X, M1Y = zeros(N_images), zeros(N_images), zeros(N_images)
    for i in range(N_images):
        print(f'{dataset_basename}: image {i+1} of {N_images}', end=' '*40 + '\r')
        M0[i], M1X[i], M1Y[i] = M0_M1X_M1Y_of_dataset_image_number(dataset, i)
    pathname = analysis_hdf5(dataset)
    hdf5(pathname,'M0',M0)
    hdf5(pathname,'M1X',M1X)
    hdf5(pathname,'M1Y',M1Y)

def M0_M1X_M1Y_of_dataset_image_number(dataset, i_image):
    image = hdf5(data_hdf5(dataset),'images', index=i_image)
    return M0_M1X_M1Y_of_image(image)

def M0_M1X_M1Y_of_image(image):
    from numpy import rint, radians

    X0 = hdf5(datafile,'X0')  # 1409.5
    Y0 = hdf5(datafile,'Y0')  # 1450.5
    rotation_angle = 135

    r = 3
    x1, x2, y1, y2 = rint([X0 + 0.5 - r, X0 + 0.5 + r, Y0 + 0.5 - r, Y0 + 0.5 + r]).astype(int)
    gain = hdf5(datafile,'gain')

    ROI = image.T[x1:x2, y1:y2]
    M0, M1X, M1Y = M0_M1X_M1Y_from_roi(ROI)
    M0 = M0 / gain
    M1X, M1Y = rotate([M1X, M1Y], radians(rotation_angle)) * pixelsize * 1000
    return M0, M1X, M1Y

def M0_M1X_M1Y_from_roi(roi):
    """Given roi, calculates M0, the background-subtracted integrated
    number of counts in pixels near the center, as well as (M1X,M1Y),
    center-of-mass coordinates relative to the center. """
    # Based on: /net/femto/C/SAXS-WAXS Analysis/SAXS_WAXS_Analysis.py, M0_M1X_M1Y_from_roi
    from numpy import seterr, array, indices
    seterr(invalid="ignore", divide="ignore")  # invalid value encountered in double_scalars

    # Define masks needed to determine M0, M1X, and M1Y
    spot = array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    bkg = array([[0, 0, 1, 1, 0, 0],
                 [0, 1, 0, 0, 1, 0],
                 [1, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 1],
                 [0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 1, 0, 0]])
    s = (spot.shape[0] - 1) / 2
    roi_bs = (roi - (roi * bkg).sum() / bkg.sum()) * spot
    M0 = roi_bs.sum()
    # Compute X1 and Y1 beam positions
    x_mask, y_mask = indices(spot.shape) - s
    M1X = (x_mask * roi_bs).sum() / M0
    M1Y = (y_mask * roi_bs).sum() / M0
    return M0, M1X, M1Y

def rotate(v, theta):
    from numpy import asarray, sin, cos, array

    v = asarray(v)
    c, s = cos(theta), sin(theta)
    R = array([[c, -s], [s, c]])
    vr = R @ v
    return vr

def chart_Xe_dataset(CX_dataset, CH_dataset):
    from charting_functions import chart_xy_symbol
    from numpy import sum

    q = hdf5(datafile,'q')

    X_pathname = analysis_hdf5(CX_dataset) 
    X_filename = path.basename(X_pathname)
    S_X = hdf5(X_pathname,'S')
    S_X = sum(S_X, axis=0)
    M0_X = hdf5(X_pathname,'M0')

    C_pathname = analysis_hdf5(CH_dataset) 
    C_filename = path.basename(C_pathname)
    S_C = hdf5(C_pathname,'S')
    S_C = sum(S_C, axis=0)
    M0_C = hdf5(C_pathname,'M0')

    # Subtract Capillary scattering
    S = S_X - sum(M0_X)/sum(M0_C) * S_C

    chart_xy_symbol(q,S,f'CX - CH\n{X_filename},{C_filename}',logx=True,logy=False,x_label='q',y_label='S')

def process_Xe_dataset(Xe_dataset, He_dataset, A_dataset, XmH_scale=XmH_scale, T0_unknown=T0_unknown):
    import h5py
    from numpy import zeros
    from saxs_waxs import FF_C
    from charting_functions import chart_xy_symbol

    q = hdf5(datafile,'q')
    q_flat = hdf5(datafile,'q_flat')
    qbin1 = hdf5(datafile,'qbin1')
    qbin2 = hdf5(datafile,'qbin2')
    UC = hdf5(datafile,'UC')
    POL = hdf5(datafile,'POL')
    GEO = hdf5(datafile,'GEO')
    BankID = hdf5(datafile,'BankID')
    DPmask = hdf5(datafile,'DPmask')
    BNKmask = hdf5(datafile,'BNKmask')
    sort_indices = hdf5(datafile,'sort_indices')
    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    CTn = hdf5(datafile,'CTn')
    STn = hdf5(datafile,'STn')
    FT = filter_transmittance(T0_unknown=T0_unknown)
    DR = detector_responsivity()

    GM0,H = M0_Pmean(A_dataset)
    M0_CX5,CX5 = M0_Pmean(Xe_dataset)
    M0_CH,CH = M0_Pmean(He_dataset)

    # Generate beam stop mask for dataset
    x0i = int(X0)
    y0i = int(Y0)
    with h5py.File(data_hdf5(Xe_dataset),'r') as f:
        sub = f['images'][:,y0i-12:y0i+12,x0i-12:x0i+12]-100
    BS = zeros(UC.shape)
    BS[y0i-12:y0i+12,x0i-12:x0i+12] = sub.mean(axis=0)
    BSmask = beamstop_mask(BS)

    # Generate mask
    BSmask = beamstop_mask(BS)
    mask = (DPmask | BNKmask | BSmask | (UC < 0.6) | (BankID==0))

    # Subtract scaled capillary scattering and apply POL, GEO, and UC corrections
    # 2025 JSR Absolute Scaling - 3.9 Sample Absorbance corrections
    # XHc = GM0 * (CX5/M0_CX5/STn - XmH_scale * CH5/M0_CH5/CTn) / (POL*GEO*UC*FT*DR)
    # XHc = GM0 * (CX5/M0_CX5 - XmH_scale * CTn*CH5/M0_CH5) / (STn * POL*GEO*UC*FT*DR)
    CHs = CTn*CH/M0_CH + (1-CTn)*H/GM0
    XHc = GM0 * (CX5/M0_CX5 - XmH_scale*CHs) / (STn * POL*GEO*UC*FT*DR)
    XHc[mask] = 0
    # Generate weights
    W = (POL*GEO*UC)*(XHc>0)

    # Compute S(q) and sigS(q)
    Xe_flat = XHc.flatten()[sort_indices]
    W_flat = W.flatten()[sort_indices]
    S,sigS,Si_flat = integrate(Xe_flat,W_flat,q_flat,qbin1,qbin2,q)

    CX_filename = path.basename(analysis_hdf5(Xe_dataset))
    CH_filename = path.basename(analysis_hdf5(He_dataset))
    A_filename = path.basename(analysis_hdf5(A_dataset))

    # Calculate normalized Xenon scattering (atomic form factor squared plus Compton scattering)
    FF,C = FF_C(q,54)
    FF2C = (FF**2 + C)/54**2

    q_scale = 0.2
    dq_scale = 0.05
    scale_range = abs(q - q_scale) < dq_scale
    FF2C_scaled = FF2C / FF2C[scale_range].mean() * S[scale_range].mean()
    title = f'CX - CH\n{CX_filename},{CH_filename},{A_filename}\nXmH_scale={XmH_scale}'
    chart_xy_symbol(q,array([S, FF2C_scaled]),title,logy=False,x_label='q',y_label='S',legend_labels=['S', 'FF'])

def process_image_mean(BCH_dataset, CH_dataset, H_dataset, G_dataset, XmH_scale=1.0, T0_unknown=T0_unknown):
    import h5py
    from numpy import zeros
    from saxs_waxs import image_psi

    UC = hdf5(datafile,'UC')
    POL = hdf5(datafile,'POL')
    GEO = hdf5(datafile,'GEO')
    BankID = hdf5(datafile,'BankID')
    DPmask = hdf5(datafile,'DPmask')
    BNKmask = hdf5(datafile,'BNKmask')
    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    CTn = hdf5(datafile,'CTn')
    STn = hdf5(datafile,'STn')
    FT = filter_transmittance(T0_unknown=T0_unknown)
    DR = detector_responsivity()

    M0_H,H = M0_Pmean(H_dataset)
    M0_CH,CH = M0_Pmean(CH_dataset)
    M0_BCH,BCH = M0_Pmean(BCH_dataset)
    M0_G,_ = M0_Pmean(G_dataset)

    # Generate beam stop mask for dataset
    x0i = int(X0)
    y0i = int(Y0)
    with h5py.File(data_hdf5(BCH_dataset),'r') as f:
        sub = f['images'][:,y0i-12:y0i+12,x0i-12:x0i+12]-100
    BS = zeros(UC.shape)
    BS[y0i-12:y0i+12,x0i-12:x0i+12] = sub.mean(axis=0)
    BSmask = beamstop_mask(BS)

    # Generate mask
    BSmask = beamstop_mask(BS)
    mask = (DPmask | BNKmask | BSmask | (UC < 0.6) | (BankID==0))

    # Subtract scaled capillary scattering and apply POL, GEO, and UC corrections
    # 2025 JSR Absolute Scaling - 3.9 Sample Absorbance corrections
    CHs = CTn*CH/M0_CH + H/M0_H*(1-CTn)
    Bs = M0_G * (BCH/M0_BCH - XmH_scale*CHs) / (STn * POL*GEO*UC*FT*DR)
    Bs[mask] = 0

    H_dataset_name = H_dataset.split("/")[-2]
    CH_dataset_name = CH_dataset.split("/")[-2]
    BCH_dataset_name = BCH_dataset.split("/")[-2]
    title = f"{BCH_dataset_name}, {CH_dataset_name}, {H_dataset_name}"
    image_psi(datafile,Bs,mask,title,vmin=0.96,vmax=1.04,chart=True,font_size=12)

def process_image_means(XmH_scale=1.0, T0_unknown=T0_unknown):
    from numpy import zeros
    from saxs_waxs import image_psi

    UC = hdf5(datafile,'UC')
    POL = hdf5(datafile,'POL')
    GEO = hdf5(datafile,'GEO')
    BankID = hdf5(datafile,'BankID')
    DPmask = hdf5(datafile,'DPmask')
    BNKmask = hdf5(datafile,'BNKmask')
    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    CTn = hdf5(datafile,'CTn')
    STn = hdf5(datafile,'STn')
    FT = filter_transmittance(T0_unknown=T0_unknown)
    DR = detector_responsivity()

    H = hdf5(datafile,'H_mean')
    CH = hdf5(datafile,'CH_mean')
    BCH = hdf5(datafile,'BCH_mean')
    M0_H = M0_M1X_M1Y_of_image(H)[0]
    M0_CH = M0_M1X_M1Y_of_image(CH)[0]
    M0_BCH = M0_M1X_M1Y_of_image(BCH)[0]
    M0_G = M0_H

    # Generate beam stop mask for dataset
    x0i = int(X0)
    y0i = int(Y0)
    sub = BCH[y0i-12:y0i+12,x0i-12:x0i+12]-100
    BS = zeros(UC.shape)
    BS[y0i-12:y0i+12,x0i-12:x0i+12] = sub
    BSmask = beamstop_mask(BS)

    # Generate mask
    BSmask = beamstop_mask(BS)
    mask = (DPmask | BNKmask | BSmask | (UC < 0.6) | (BankID==0))

    # Subtract scaled capillary scattering and apply POL, GEO, and UC corrections
    # 2025 JSR Absolute Scaling - 3.9 Sample Absorbance corrections
    CHs = CTn*CH/M0_CH + H/M0_H*(1-CTn)
    Bs = M0_G * (BCH/M0_BCH - XmH_scale*CHs) / (STn * POL*GEO*UC*FT*DR)
    Bs[mask] = 0

    title = "BCH_mean,CH_mean,H_mean"
    image_psi(datafile,Bs,mask,title,vmin=0.96,vmax=1.04,chart=True,font_size=12)

def UC_from_image_means_manual_weights(XmH_scale=1.0, T0_unknown=T0_unknown):
    from numpy import zeros, median, arange, isnan
    from charting_functions import chart_xy_fit, chart_image
    from scipy.interpolate import UnivariateSpline

    font_size = 12

    UC = hdf5(datafile,'UC')
    POL = hdf5(datafile,'POL')
    GEO = hdf5(datafile,'GEO')
    BankID = hdf5(datafile,'BankID')
    DPmask = hdf5(datafile,'DPmask')
    BNKmask = hdf5(datafile,'BNKmask')
    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    CTn = hdf5(datafile,'CTn')
    STn = hdf5(datafile,'STn')
    FT = filter_transmittance(T0_unknown=T0_unknown)
    DR = detector_responsivity()
    Q = hdf5(datafile,'Q')

    H = hdf5(datafile,'H_mean')
    CH = hdf5(datafile,'CH_mean')
    BCH = hdf5(datafile,'BCH_mean')
    M0_H = M0_M1X_M1Y_of_image(H)[0]
    M0_CH = M0_M1X_M1Y_of_image(CH)[0]
    M0_BCH = M0_M1X_M1Y_of_image(BCH)[0]
    M0_G = M0_H

    # Generate beam stop mask for dataset
    x0i = int(X0)
    y0i = int(Y0)
    sub = BCH[y0i-12:y0i+12,x0i-12:x0i+12]-100
    BS = zeros(UC.shape)
    BS[y0i-12:y0i+12,x0i-12:x0i+12] = sub
    BSmask = beamstop_mask(BS)

    # Generate mask
    BSmask = beamstop_mask(BS)
    mask = (DPmask | BNKmask | BSmask | (UC < 0.6) | (BankID==0))

    # Subtract scaled capillary scattering and apply POL, GEO, and UC corrections
    # 2025 JSR Absolute Scaling - 3.9 Sample Absorbance corrections
    CHs = CTn*CH/M0_CH + H/M0_H*(1-CTn)
    Bs = M0_G * (BCH/M0_BCH - XmH_scale*CHs) / (STn * POL*GEO*UC*FT*DR)
    Bs[mask] = 0

    # Chart image normalized to median in each q bin.
    # Exclude and zero masked pixels.
    q = hdf5(datafile,'q')
    qbin1 = hdf5(datafile,'qbin1')
    qbin2 = hdf5(datafile,'qbin2')
    sort_indices = hdf5(datafile,'sort_indices')
    reverse_indices = hdf5(datafile,'reverse_indices')
    I_flat = Bs.flatten()[sort_indices]
    mask_flat = mask.flatten()[sort_indices]
    In_flat = zeros(len(I_flat))
    Iq = zeros(len(q))
    for i in range(len(q)):
        first = qbin1[i]
        last  = qbin2[i]
        bin_select = arange(first,last)
        mask_select = ~mask_flat[bin_select]
        Iq[i] = median(I_flat[bin_select[mask_select]])
        In_flat[bin_select] = I_flat[bin_select]/Iq[i]
    In = In_flat[reverse_indices].reshape(Bs.shape)

    title = "BCH_mean,CH_mean,H_mean"
    vmin, vmax = 0.96, 1.04
    chart_image(In,f'I/I_median for each q-bin\n{title}',vmin=vmin,vmax=vmax,font_size=font_size)

    # Select valid data and fit Iq with univariate spline
    select = ~isnan(Iq) & (q>0.02)
    Iqs = Iq[select]
    qs = q[select]
    s_scale = 0.88
    s = s_scale*((Iqs[1:-1] - 0.5*(Iqs[:-2] + Iqs[2:]))**2).sum()
    w = ones(len(qs))
    w[qs<0.9] = 3
    w[(qs>3.6)&(qs<4.6)] = 2
    w[qs>4.6] = 0.5
    us = UnivariateSpline(qs,Iqs,w=w,s=s)
    N_knots = len(us.get_knots())
    title = f'Bs median with us fit\ns_scale: {s_scale}; N_knots: {N_knots}'
    chart_xy_fit(qs,Iqs,us(qs),title,ymin=0,x_label='q',font_size=font_size)

    UCb = Bs/us(Q)
    chart_image(UCb,'UCb',vmin=vmin,vmax=vmax,font_size=font_size)

    UC2 = UC * In
    hdf5(datafile,'UC2', UC2)

def UC_from_image_means(XmH_scale=1.0, T0_unknown=T0_unknown):
    from numpy import zeros, median, arange, isnan
    from charting_functions import chart_xy_fit, chart_image, chart_histogram
    from scipy.interpolate import UnivariateSpline

    font_size = 12

    UC = hdf5(datafile,'UC')
    POL = hdf5(datafile,'POL')
    GEO = hdf5(datafile,'GEO')
    BankID = hdf5(datafile,'BankID')
    DPmask = hdf5(datafile,'DPmask')
    BNKmask = hdf5(datafile,'BNKmask')
    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    CTn = hdf5(datafile,'CTn')
    STn = hdf5(datafile,'STn')
    FT = filter_transmittance(T0_unknown=T0_unknown)
    DR = detector_responsivity()
    Q = hdf5(datafile,'Q')

    H = hdf5(datafile,'H_mean')
    var_H = hdf5(datafile,'var_H_mean')
    CH = hdf5(datafile,'CH_mean')
    var_CH = hdf5(datafile,'var_CH_mean')
    BCH = hdf5(datafile,'BCH_mean')
    var_BCH = hdf5(datafile,'var_BCH_mean')
    M0_H = M0_M1X_M1Y_of_image(H)[0]
    M0_CH = M0_M1X_M1Y_of_image(CH)[0]
    M0_BCH = M0_M1X_M1Y_of_image(BCH)[0]
    M0_G = M0_H

    # Generate beam stop mask for dataset
    x0i = int(X0)
    y0i = int(Y0)
    sub = BCH[y0i-12:y0i+12,x0i-12:x0i+12]-100
    BS = zeros(UC.shape)
    BS[y0i-12:y0i+12,x0i-12:x0i+12] = sub
    BSmask = beamstop_mask(BS)

    # Generate mask
    BSmask = beamstop_mask(BS)
    mask = (DPmask | BNKmask | BSmask | (UC < 0.6) | (BankID==0))

    # Subtract scaled capillary scattering and apply POL, GEO, and UC corrections
    # 2025 JSR Absolute Scaling - 3.9 Sample Absorbance corrections
    CHs = CTn * CH / M0_CH + H / M0_H * (1-CTn)
    var_CHs = CTn**2 * var_CH / M0_CH**2 + var_H**2 / M0_H**2 * (1-CTn)**2
    Bs = M0_G * (BCH / M0_BCH - XmH_scale * CHs) / (STn*POL*GEO*UC*FT*DR)
    var_Bs = M0_G**2 * (var_BCH**2 / M0_BCH**2 + XmH_scale**2 * var_CHs**2) / (STn*POL*GEO*UC*FT*DR)**2
    Bs[mask] = 0
    var_Bs[mask] = 0

    # Chart image normalized to median in each q bin.
    # Exclude and zero masked pixels.
    q = hdf5(datafile,'q')
    qbin1 = hdf5(datafile,'qbin1')
    qbin2 = hdf5(datafile,'qbin2')
    sort_indices = hdf5(datafile,'sort_indices')
    reverse_indices = hdf5(datafile,'reverse_indices')
    Bs_flat = Bs.flatten()[sort_indices]
    var_Bs_flat = var_Bs.flatten()[sort_indices]
    mask_flat = mask.flatten()[sort_indices]
    Bsn_flat = zeros(len(Bs_flat))
    var_Bsn_flat = zeros(len(Bs_flat))
    Bsq = zeros(len(q))
    var_Bsq = zeros(len(q))
    for i in range(len(q)):
        first = qbin1[i]
        last  = qbin2[i]
        bin_select = arange(first,last)
        mask_select = ~mask_flat[bin_select]
        select = bin_select[mask_select]
        Bsq[i] = median(Bs_flat[select])
        var_Bsq[i] = median(var_Bs_flat[select]) / len(var_Bs_flat[select])
        Bsn_flat[bin_select] = Bs_flat[bin_select] / Bsq[i]
        var_Bsn_flat[bin_select] = (1 / Bsq[i])**2 * var_Bs_flat[bin_select] + (-Bs_flat[bin_select] / Bsq[i]**2)**2 * var_Bsq[i] 
    Bsn = Bsn_flat[reverse_indices].reshape(Bs.shape)

    vmin, vmax = 0.96, 1.04
    title = "BCH_mean,CH_mean,H_mean"
    chart_image(Bsn,f'Bs/Bs_median for each q-bin\n{title}',vmin=vmin,vmax=vmax,font_size=font_size)

    # Select valid data and fit Iq with univariate spline
    select = ~isnan(Bsq) & (q>0.02)
    Bsqs = Bsq[select]
    var_Bsqs = var_Bsq[select]
    qs = q[select]
    w = 1/var_Bsqs
    w /= 2000 * 4.5
    us = UnivariateSpline(qs,Bsqs,w=w)
    N_knots = len(us.get_knots())
    title = f'Bs median with us fit\nsN_knots: {N_knots}'
    chart_xy_fit(qs,Bsqs,us(qs),title,ymin=0,x_label='q',font_size=font_size)
    
    UCb = Bs/us(Q)
    chart_image(UCb,'UCb',vmin=vmin,vmax=vmax,font_size=font_size)
    chart_histogram(UCb[~mask],'UCb',binsize=0.001,xmin=0.9,xmax=1.1)

    UC2 = UC * UCb
    hdf5(datafile,'UC2', UC2)

def filter_transmittance(T0_unknown=T0_unknown):
    from numpy import exp, cos, log
    from saxs_waxs import mu_Air, mu_Kapton, mu_for_Z

    PHI = hdf5(datafile,'PHI')

    # Filter transmittance (0.980 at q_max)
    # Air in space between He cone and detector
    beam_stop_distance = 110 # mm
    x_A = detector_distance - beam_stop_distance
    mu_A = mu_Air(keV=photon_energy_in_keV)
    T_A = exp(-mu_A * x_A / cos(PHI))
    # Windows (He cone entrance + detector entrance)
    x_W = 0.0085 + 0.0254  # mm
    mu_W = mu_Kapton(keV=photon_energy_in_keV)
    T_W = exp(-mu_W * x_W / cos(PHI))
    mu_Be = mu_for_Z(Z=4, keV=photon_energy_in_keV)
    x_Be = 0.2
    T_Be = exp(-mu_Be * x_Be / cos(PHI))
    # Unknown filter
    T_unknown = exp(log(T0_unknown)/cos(PHI))

    T = T_A * T_W * T_Be * T_unknown
    T *= 1 / T.min()
    return T

def detector_responsivity(): 
    # correction: 1.238 at phi_max
    from numpy import exp, cos
    from saxs_waxs import mu_for_Z

    PHI = hdf5(datafile,'PHI')
    mu = mu_for_Z(Z=14,keV=photon_energy_in_keV)

    thickness = 0.5 # mm

    DR = (1 - exp(-mu * thickness / cos(PHI))) / (1 - exp(-mu * thickness))
    return DR

    """Writes to, or reads from hdf5 file specified by path_name. 
    If keyword is None and path_name exists, prints keys. If val is None, 
    reads keyword and if a byte string, decodes and converts it
    to string; else, writes val into keyword. If path_name doesn't exist, 
    creates it.
    Use keyword="images", index=0 to retreive a single image.
    Use keyword="images", attribute="len" to get the number of images.
    """
    from numpy import isscalar, array
    import h5py
    from os.path import exists,dirname
    from os import makedirs
    # Assign val to data; if string, explicitly assign string datatype before writing to hdf5 file
    if val is not None:
        if isscalar(val):
            if isinstance(val, str):
                data = str(val)
            else:
                data = val
        else:
            if isinstance(val[0], str):
                data = array(val).astype('S')
            else:
                data = val 
    if exists(path_name):
        if keyword is None: # Print keywords
            with h5py.File(path_name, 'r') as f:
                print(f.keys())
        elif val is None: # Read data (decode if string)
            with h5py.File(path_name, 'r') as f:
                data = f[keyword]
                if attribute == "len":
                    data = len(data)
                elif attribute:
                    data = getattr(data, attribute)
                elif index is not None:
                    data = data[index]
                else:
                    data = data[()]
            return data
        else:   
            # Write data
            with h5py.File(path_name, 'r+') as f:
                if keyword not in f.keys():
                    f[keyword] = data
                else:
                    try:
                        f[keyword][...] = data
                    except:
                        del f[keyword]
                        f[keyword] = data
    else:
        try:
            makedirs(dirname(path_name))
        except FileExistsError:
            pass
        with h5py.File(path_name, 'w') as f:
            f[keyword] = data
