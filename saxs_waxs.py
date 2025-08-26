from matplotlib.pyplot import close
from numpy import ndarray
exclude=['AppleDouble','original','backup','trash','Trash','overlap']

def process_beamtime(datafile,params,N_proc=16,chart=False):
    """Process datasets according to params."""

    # Create hdf5 datafile and copy params into it
    params_to_datafile(datafile,params)
    # Generate hdf5 files containing filenames for each dataset found below data_dir
    datafile_datasets(datafile)
    # Extract attributes for each dataset and write to corresponding hdf5 file
    multiprocess_attributes(datafile,N_proc)
    # Identify low intensity images in omit (based on ICs)
    datafile_omit0(datafile,chart)
    # Generate beamstop masks, BSm, bkg, GC_VPC, GC_UCp, and GCn from from glassy carbon data
    datafile_glassy_carbon(datafile,chart)
    # Determine M0, M1X, M1Y, and Mbs
    datafile_M0_M1X_M1Y_Mbs(datafile,chart)
    # Process saxs data 
    datafile_process_saxs(datafile,chart)
    # Generate timestamps from acquire and header timestamps
    datafile_timestamps(datafile,chart)
    # Assign RTD_temp from temperature_archive according to image timestamps
    datafile_RTD_temp(datafile,chart)
    # Generate geometry data needed for WAXS analysis and write to datafile
    datafile_geometry(datafile,chart)
    # Process dark datasets to generate slope and intercept
    datafile_dark(datafile)
    # Generate zinger-free statistics of datasets needed for subsequent analysis
    multiprocess_zfs(datafile,N_proc)
    # Generate VPC
    datafile_VPC(datafile,chart)
    # Generate UCp
    datafile_UCp(datafile,chart)
    # Assess UCq
    #datafile_UCq(datafile,VG_amp=0.05)
    # Process xray_trace files
    beamtime_trc(datafile)
    beamtime_nC(datafile)
    # Process Tramp data to determine temp
    beamtime_Tramp_temp(datafile)
    # Process PumpProbe data to determine temperature
    beamtime_PumpProbe_temp(datafile)
    # Generate An, which is needed for integration
    datafile_An(datafile,chart)
    # Integrate data needed to define global mask
    multiprocess_integrate(datafile,'Tramp_B-2',N_proc)
    # Define global mask
    datafile_global_mask(datafile,term='Tramp_B-2')
    # Integrate all data (execute twice)
    multiprocess_integrate(datafile,term=None,N_proc=N_proc)
    multiprocess_integrate(datafile,term=None,N_proc=N_proc)

def us_fit(y,s_scale=1,sig=None,chart=False):
    """Fit y with univariate spline. Increasing s_scale beyond its default value of 1
    increases the number of knots employed in the fit. If sig is entered, estimates
    weights from sigma, else assumes weights are one. If chart is True, plots the 
    fit and the residuals."""
    from numpy import arange,median
    from scipy.interpolate import UnivariateSpline
    x = arange(len(y))
    s = 3.33*median((y[1:] - y[:-1])**2)*len(y)/s_scale
    if sig is None:
        w = None
    else:
        w = sig.mean()/sig
    us = UnivariateSpline(x,y,w,s=s)
    N_knots = len(us.get_knots())
    if chart:
        from charting_functions import chart_xy_fit,chart_xy_symbol
        chart_xy_fit(x,y,us(x),'univariate spline fit\ns_scale = {}; N_knots = {}'.format(s_scale,N_knots))
        chart_xy_symbol(x,y-us(x),'residuals\ns_scale = {}; N_knots = {}'.format(s_scale,N_knots),ms=2)
    return us(x)
    
def I0_Rg_from_polynomial(q,Sj,sigSj,qmax=0.15):
    """Performs statistically-weighted least squares fit of Sj with a 
    eighth-order even polynomial. qmax defines the upper limit of the 
    fitting range. Returns I0, Rg, qs, and residual. Omits the first 
    point in Sj, which may suffer from systematic error."""
    from numpy import sqrt,vstack,ones
    from numpy.linalg import lstsq 
    qs = q < qmax
    w = 1/sigSj[:,qs].mean(axis=0)
    M = vstack((ones(qs.sum()),q[qs]**2,q[qs]**4,q[qs]**6,q[qs]**8))
    coeffs = lstsq((M*w).T,(Sj[:,qs]*w).T,rcond=None)[0] 
    fit = (M.T@coeffs).T
    residual = Sj[:,qs] - fit
    I0 = coeffs[0]
    Rg = sqrt(-3*coeffs[1]/coeffs[0])
    return I0,Rg,q[qs],residual

def even_polynomial_fit(q,Sj,sigSj,N_terms=2,qmax=0.25):
    """Performs statistically-weighted least-squares fit of Sj with 
    N_term even polynomial (up to 8 terms) plus an offset. qmax defines 
    the upper limit of the fitting range. Returns coeffs, q, and residual."""
    from numpy import vstack,ones,zeros_like
    from numpy.linalg import lstsq 
    qs = (q < qmax)
    w = 1/sigSj[:,qs].mean(axis=0)
    M = vstack((ones(qs.sum()),q[qs]**2,q[qs]**4,q[qs]**6,q[qs]**8,q[qs]**10,q[qs]**12,q[qs]**14,q[qs]**16))
    coeffs = lstsq((M[:N_terms+1]*w).T,(Sj[:,qs]*w).T,rcond=None)[0] 
    fit = (M[:N_terms+1].T@coeffs).T
    residual = Sj[:,qs] - fit
    return coeffs,q[qs],residual
    
def Xn_Polynomial_fit(q,Sj,sigSj,Xn=None,N_terms=2,qmax=0.25):
    """Performs statistically-weighted least-squares fit of Sj with 
    Xn plus an N_term even polynomial. Xn represents normalized 
    excess scattering signal in the SAXS region. qmax defines the upper 
    limit of the fitting range. Returns Xa, Pn, q, and residual."""
    from numpy import vstack,ones,zeros_like
    from numpy.linalg import lstsq 
    qs = (q < qmax)
    w = 1/sigSj[:,qs].mean(axis=0)
    if Xn is None:
        Xn = zeros_like(q)
    M = vstack((Xn[:qs.sum()],ones(qs.sum()),q[qs]**2,q[qs]**4,q[qs]**6,q[qs]**8,q[qs]**10,q[qs]**12,q[qs]**14,q[qs]**16))
    coeffs = lstsq((M[:N_terms+1]*w).T,(Sj[:,qs]*w).T,rcond=None)[0] 
    fit = (M[:N_terms+1].T@coeffs).T
    residual = Sj[:,qs] - fit
    Pn = coeffs[1:]
    Xa = coeffs[0]
    return Xa,Pn,q[qs],residual

def IC_scale(datafile,dataset,chart=False):
    """Determine IC_scale, which is used to normalize S. Fits IC with 
    second order polynomial plus water density, wd, and gas density, gd. 
    Uses nC and temp if available, and substitutes RTD_temp if temp 
    is not available. Writes IC_scale and IC_coeffs to dataset."""
    from numpy import ones,vstack,isnan,zeros,arange,array
    from numpy.linalg import lstsq 
    i,pathname = index_pathname(datafile,dataset)
    IC = hdf5(pathname,'IC')
    RTD_temp = hdf5(pathname,'RTD_temp')
    try:
        nC = hdf5(pathname,'nC')
    except:
        nC = ones(len(IC))
    try:
        temp = hdf5(pathname,'temp')
        temp[isnan(temp)] = RTD_temp
    except:
        temp = RTD_temp
    select = (hdf5(pathname,'omit') == 0) & ~isnan(nC)
    nC_mean = nC[select].mean()
    ICn = zeros(len(IC))
    ICn[select] = IC[select]*nC_mean/nC[select]
    ri = arange(len(IC))
    wd = water_density(temp)
    gd = (273 + 4)/(273 + temp - 4)
    M = vstack((ones(len(ri)),ri,ri**2,gd,wd))
    coeffs = lstsq(M[:,select].T,ICn[select],rcond=None)[0] 
    IC_fit = (M[:3].T@coeffs[:3]).mean() + M[3:].T@coeffs[3:]
    IC_scale = IC/IC_fit

    if chart:
        from charting_functions import chart_vector
        chart_vector(array([IC,ICn,IC_fit]),'IC, ICn and IC_fit\n{}'.format(dataset),x_label='image index',y_label='Counts')
        ICn_fit = M.T@coeffs
        residual =  ICn - ICn_fit
        relative_error = residual/ICn_fit
        chart_vector(relative_error,'relative error when fitting ICn\n{}'.format(dataset),x_label='image index',y_label='Counts')
        chart_vector(IC_scale,'IC_scale\n{}'.format(dataset),x_label='image index',y_label='Counts')
        Sm = hdf5(pathname,'S').mean(axis=1)
        chart_vector(array([Sm,Sm/IC_scale]),'S_mean(axis=1) before and after IC_scale normalization\n{}'.format(dataset),x_label='image index',y_label='Counts')
        # Generate M0s, M0 normalized using IC_scale
        M0 = hdf5(pathname,'M0')
        M0s = M0/IC_scale
        M0s[~select] = 0
        # Generate M0c, corrected M0s according to least-squares fit of M0s that includes gd and wd
        M = vstack((ones(len(ri)),gd,wd))
        M0_coeffs = lstsq(M[:,select].T,M0s[select],rcond=None)[0] 
        M0s_fit = M.T@M0_coeffs
        M0s_scale = M0s_fit/M0s_fit.mean()
        M0c = M0s/M0s_scale
        M0c[~select] = 0
        chart_vector(array([M0,M0s,M0c,M0s_fit]),'M0, M0s, M0c, M0s_fit\n{}'.format(dataset),x_label='image index',y_label='Counts')

    hdf5(pathname,'IC_scale',IC_scale)
    hdf5(pathname,'IC_coeffs',coeffs)

def beamtime_PumpProbe_temp(datafile,chart=False):
    """Assigns temp in PumpProbe datasets, but requires a matching Tramp dataset.
    If there are multiple matching Tramp datasets, selects the nearest one."""
    from numpy import array
    datasets = hdf5(datafile,'datasets')
    PP_datasets = array([dataset for dataset in datasets if 'PumpProbe' in dataset])
    TR_datasets = array([dataset for dataset in datasets if 'Tramp' in dataset])
    for PP_dataset in PP_datasets:
        i = index_pathname(datafile,PP_dataset)[0]
        # Find nearest suitable Tramp dataset
        name = PP_dataset.replace('PumpProbe','Tramp')[:-2] # Strip repeat number
        matches = array([TR_dataset for TR_dataset in TR_datasets if name in TR_dataset])
        if len(matches) > 0:
            di = array([abs(i-index_pathname(datafile,match)[0]) for match in matches])
            TR_dataset = matches[di==di.min()][0]
            PumpProbe_temp(datafile,PP_dataset,TR_dataset,chart)
        else:
            print('No Tramp match found for {}'.format(PP_dataset))
        
def PumpProbe_temp(datafile,PP_dataset,TR_dataset,chart):
    """Processes PP_dataset to determine 'temp'. Assumes 'temp' 
    already exists in TR_dataset. Performs SVD on combined PumpProbe 
    and corresponding Tramp data; generates a temperature calibration 
    curve via a polynomial fit of TR_dataset 'temp' as a function of 
    corresponding 'Vn' vectors; and then uses this function to assign 
    'temp' to PP_dataset."""
    from numpy import zeros,polyfit,poly1d,nan,concatenate,argsort,arange
    from time import time
    t0 = time()
    # Select restricted range of q
    N_poly = 6
    qmin=1.4
    qmax=3.5
    q = hdf5(datafile,'q')
    q_select = (q > qmin) & (q < qmax)
    qs = q[q_select]

    # Determine pathnames for PP_dataset and TR_dataset
    PP_pathname = index_pathname(datafile,PP_dataset)[1]
    TR_pathname = index_pathname(datafile,TR_dataset)[1]
    try:
        PP_S = hdf5(PP_pathname,'S')
        TR_S = hdf5(TR_pathname,'S')
        TR_temp = hdf5(TR_pathname,'temp')
        RTD_temp = hdf5(PP_pathname,'RTD_temp')
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
        hdf5(PP_pathname,'temp',PP_temp)
        print('processed {} in {:0.3f} seconds'.format(PP_dataset,time()-t0))
        if chart:
            from charting_functions import chart_UTsV,chart_xy_symbol,chart_xy_rainbow
            chart_UTsV(qs,UT,s,V,PP_dataset+' combined with ' + TR_dataset)
            chart_xy_symbol(arange(len(PP_S)),PP_temp - RTD_temp,'PP_temp - RTD_temp\n{}'.format(PP_dataset),x_label='image index')
            chart_xy_rainbow(q,TR_S[argsort(TR_temp)],'S\n{}'.format(TR_dataset),x_label='q',logx=True,logy=True)
    except:
        print('Failed to process {}'.format(PP_dataset))

def beamtime_Tramp_temp(datafile):
    # Define args for multiprocess integration of each dataset
    exclude_terms = ['Tramp_A','Tramp_C']
    datasets = hdf5(datafile,'datasets')
    datasets = [dataset for dataset in datasets if 'Tramp' in dataset and not any([exclude_term in dataset for exclude_term in exclude_terms])]
    for dataset in datasets:
        Tramp_temp(datafile,dataset)

def Tramp_temp(datafile,dataset,chart=False):
    """Processes Tramp dataset to determine 'temp'. Assigns 'temp' 
    according to N_poly order polynomial fit of RTD_temp as a function 
    of VT[1]/VT[0], which is generated from SVD analysis of dataset. 
    The q region selected for the analysis is defined by qmin and qmax, 
    a region that is sensitive to temperature."""
    from numpy import array,arange,argsort,around
    from charting_functions import chart_xy_symbol,chart_UTsV,chart_xy_rainbow
    from numpy import polyfit,poly1d
    from time import time
    t0 = time()
    # Select restricted range of q
    N_poly=6
    qmin=1.4
    qmax=3.5
    i,pathname = index_pathname(datafile,dataset)
    # Read relevant data from datafile
    q = hdf5(datafile,'q')
    # Read relevant data from dataset
    S = hdf5(pathname,'S')
    sigS = hdf5(pathname,'sigS')
    IC = hdf5(pathname,'IC')
    omit = hdf5(pathname,'omit')
    RTD_temp = hdf5(pathname,'RTD_temp')
    # Define qs and generate Ss
    q_select = (q > qmin) & (q < qmax)
    qs = q[q_select]
    ICn = IC/IC[omit==0].mean()
    Sg = (S.T/ICn).T
    sigSg = (sigS.T/ICn).T
    Ss = Sg[:,q_select]
    # Determine temp 
    temp = RTD_temp.copy()
    # Process Tramp data free of outliers; assign temp according to svd-based calibration
    select = omit == 0
    UT,s,V = SVD(Ss[select])
    V1 = V[1]
    pf_inv = polyfit(RTD_temp[select],V1,N_poly)
    V1_fit = poly1d(pf_inv)(RTD_temp[select])
    pf = polyfit(V1,RTD_temp[select],N_poly)
    temp[select] = poly1d(pf)(V1)
    # Write temp to dataset
    hdf5(pathname,'temp',temp)
    print('\rProcessed {}: {} in {:0.3f} seconds                             '.format(i,dataset,time()-t0),end='\r')

    if chart:
        sort_order = argsort(temp)
        chart_xy_rainbow(q,Sg[sort_order],'Sg\n{}'.format(dataset),x_label='q',y_label='Counts',logx=True)
        chart_xy_symbol(q,array([Sg[-1],sigSg[-1]]),'Sg and sigSg at {} Celsius\n{}'.format(around(temp[-1],0),dataset),x_label='q',y_label='Counts',logx=True,logy=True,ms=2)
        chart_UTsV(qs,UT,s,V,'{}'.format(dataset))
        difference = RTD_temp[select] - temp[select]
        ri = arange(len(S))
        chart_xy_symbol(ri[select],difference,'{}\nRTD_temp - temp (N_poly = {})'.format(dataset,N_poly),y_label='Temperature error [Celsius]',x_label='image index',ms=2)
        chart_xy_symbol(temp[select],array([V1,V1_fit]),'{}\npolynomial fit of VT1n vs. RTD_temp (N_poly = {})'.format(dataset,N_poly),x_label='RTD_temp',ms=2)
        chart_xy_symbol(ri[select],array([RTD_temp[select],temp[select]]),'{}\nRTD_temp,temp (N_poly = {})'.format(dataset,N_poly),x_label='image index',y_label='Temperature [Celsius]')

def datafile_An(datafile,chart):
    """From Tramp air datasets generate normalized air scattering, An, which 
    is the zinger-free average of air scattering datasets minus spurious 
    scattering divided by the mean M0 for images not flagged by omit."""
    from numpy import array
    # Find pathnames for air scattering datasets
    datasets = hdf5(datafile,'datasets')
    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    saxs_boxsize = hdf5(datafile,'saxs_boxsize')
    s = int((saxs_boxsize - 1)/2)
    A_pathnames = array([index_pathname(datafile,dataset)[1] for dataset in datasets if ('Tramp_A-' in dataset)])
    M0i = []
    Ai = []
    for pathname in A_pathnames:
        spurious = hdf5(pathname,'spurious')
        Imean = hdf5(pathname,'Imean')
        M0 = hdf5(pathname,'M0')
        omit = hdf5(pathname,'omit')
        Imean[Y0-s:Y0+s+1,X0-s:X0+s+1] -= spurious
        Ai.append(Imean)
        M0i.append(M0[omit==0].mean())
    M0i = array(M0i)
    Ai = array(Ai)
    An = (Ai.mean(axis=0)/M0i.mean()).astype('float32')
    hdf5(datafile,'An',An)
    if chart:
        from charting_functions import chart_image
        BSm = hdf5(datafile,'BSm')
        chart_image(An[Y0-s:Y0+s+1,X0-s:X0+s+1]*~BSm[0]*M0i.mean(),'An*<M0>\nAverage of {} Tramp datasets; spurious-corrected'.format(len(A_pathnames)))

def datafile_UCq(datafile,VG_amp=None):
    """Given zinger-free statistics for Xe and He, assess parameters used
    to generate PGFP (e.g., VG_amp and phosphor_FF)."""
    from charting_functions import chart_xy_fit,chart_image
    from numpy import array,where,ones_like,indices
    from numpy.linalg import lstsq  
    from scipy.interpolate import UnivariateSpline
    datasets = hdf5(datafile,'datasets')
    X5_dataset = [dataset for dataset in datasets if '_CX5' in dataset][0]
    H5_dataset = [dataset for dataset in datasets if '_CH5' in dataset][0]

    i,X5_pathname = index_pathname(datafile,X5_dataset)
    i,H5_pathname = index_pathname(datafile,H5_dataset)

    # Find A_pathname for nearest, prior air scattering dataset
    A_indicies = array([index_pathname(datafile,dataset)[0] for dataset in datasets if ('Reference_A' in dataset)])
    Ai = A_indicies[where(A_indicies < i)[0][-1]]
    A_dataset = datasets[Ai]
    i,A_pathname = index_pathname(datafile,A_dataset)
    A = hdf5(A_pathname,'Imean')
    Avar = hdf5(A_pathname,'Ivar')
    A_M0 = hdf5(A_pathname,'M0').mean()

    # Load zinger-free stats for FS and A
    X5 = hdf5(X5_pathname,'Imean')
    X5var = hdf5(X5_pathname,'Ivar')
    X5_M0 = hdf5(X5_pathname,'M0').mean()
    X5_N = len(hdf5(X5_pathname,'M0'))
    H5 = hdf5(H5_pathname,'Imean')
    H5var = hdf5(H5_pathname,'Ivar')
    H5_M0 = hdf5(H5_pathname,'M0').mean()

    # Load needed information from datafile
    shape = hdf5(datafile,'shape')
    UCp = hdf5(datafile,'UCp')
    PGFP = hdf5(datafile,'PGFP')
    CXT = hdf5(datafile,'CXT')
    CHT = hdf5(datafile,'CHT')
    W = hdf5(datafile,'W')
    RPm = hdf5(datafile,'RPm')
    Y0 = hdf5(datafile,'Y0')
    Q = hdf5(datafile,'Q')
    sort_indices = hdf5(datafile,'sort_indices')
    phosphor_FF = hdf5(datafile,'phosphor_FF')

    # Generate vertical gradient for correction
    if VG_amp == None:
        VG_amp = hdf5(datafile,'VG_amp')
    y_indices = indices(shape)[0] 
    VG = ones_like(y_indices) + VG_amp*(Y0 - y_indices)/shape[0]

    # Combine uniformity corrections to generate UCX and UCH
    UCX = UCp*VG*PGFP*CXT
    UCH = UCp*VG*PGFP*CHT

    # Calculate Atomic scattering for Xenon (Z=54)
    FF,C = FF_C(Q,54)
    FF2C = (FF**2 + C)/54**2

    # Construct air-subtracted, uniformity-corrected scattering intensity for He and Xe 
    H5c = (H5 - A*(H5_M0/A_M0))/UCH
    X5c = (X5 - A*(X5_M0/A_M0))/UCX

    # Construct air-subtracted, uniformity-corrected variance for He and Xe scattering
    VH = (H5var + Avar*(H5_M0/A_M0)**2)/UCH**2
    VX = (X5var + Avar*(X5_M0/A_M0)**2)/UCX**2
    V = VX + VH

    # Estimate weights for least-squares fit; zero pixels that are defective or masked
    w = W*(UCp*PGFP)
    DPm = RPm > 1
    mask = ones_like(w,bool)
    mask[240:,80:] = False
    mask[DPm] = True
    w[mask] = 0
    #w[(Q < 1) | (Q > 3)] = 0

    # Flatten arrays and perform weighted least squares fit of H5 and X5 to FF2C over unmasked range
    FF2Cf = FF2C.flatten()[sort_indices]
    Xf = X5c.flatten()[sort_indices]
    Hf = H5c.flatten()[sort_indices]
    wf = w.flatten()[sort_indices]
    M = array([Xf,Hf])
    sX,sH = lstsq((M*wf).T,FF2Cf*wf,rcond=None)[0] 

    # Construct Xenon as linear combination of H5c and X5c
    Xenon = (sH*H5c + sX*X5c)

    # Construct Xenon_UCp
    Xenon_UCp = Xenon/FF2C
    Xenon_UCp[DPm] = 1
    chart_image(Xenon_UCp,'Xenon_UCp\n vertical gradient: {}'.format(VG_amp),vmin=0.96,vmax=1.04)
    chart_xy_fit(Q[~mask],Xenon[~mask],FF2C[~mask],'fit of Xenon[~mask] vs. Q (exclude DPm and irregular regions))\nphosphor_FF = {}'.format(phosphor_FF),x_label='q')

def datafile_UCp(datafile,chart=False,verbose=False):
    """Calculates the uniformity correction for each pixel, UCp, assuming circular symmetry
    in phi, given FS_stats and A_stats. Writes UCp and CCD_scale in datafile, where CCD_scale 
    is an image in which pixels assigned to each module are set to the scale factor that 
    minimizes scaling differences between the modules. The weighted average of the scattering 
    intensity as a function of q is determined using the inner four and top left three 
    modules, which is then used to determine UCp for the entire detector. The variance
    used to identify outliers has contributions from both the variance found in FS_stats
    as well as A_stats."""
    from numpy import array,zeros,ones,zeros,seterr,sqrt,where,polyfit,polyval,prod,sqrt,indices,ones_like,zeros_like
    from scipy.interpolate import UnivariateSpline
    seterr(divide='ignore',invalid='ignore')
    
    def outlier_pixels():
        """Uses Ic_flat, Vc_flat, and Wc_flat to flag outlier pixels. 
        Returns OP_flat,N_pixels."""
        from numpy import zeros,median,sqrt,arange
        sigma_flat = zeros(len(q_flat))
        OP_flat = zeros(len(q_flat),bool)
        N_pixels = zeros(len(q),'uint16')
        stdev = zeros(len(q))
        # Determine sigma_flat and OP_flat relative to median from binned values of q
        select = Wc_flat > 0
        for j in range(len(q)):
            first = qbin1[j]
            last  = qbin2[j]
            selected = arange(first,last)[select[first:last]]
            N = len(selected)
            if N > 2:
                N_pixels[j] = N
                I_median = median(Ic_flat[selected])
                sigma_flat[selected] = (Ic_flat[selected] - I_median)/sqrt(Vc_flat[selected])
                stdev[j] = 1/sqrt((1/Vc_flat[selected]).sum())
        OP_flat = abs(sigma_flat) > Zstat*sigma_flat[select].std()
        return OP_flat,N_pixels

    # Find FS_pathname
    FS_dataset = hdf5(datafile,'FS_dataset')
    i,FS_pathname = index_pathname(datafile,FS_dataset)
    # Find A_pathname for nearest, prior air scattering dataset
    datasets = hdf5(datafile,'datasets')
    A_indicies = array([index_pathname(datafile,dataset)[0] for dataset in datasets if ('Reference_A' in dataset)])
    Ai = A_indicies[where(A_indicies < i)[0][-1]]
    A_dataset = datasets[Ai]
    i,A_pathname = index_pathname(datafile,A_dataset)

    # Load zinger-free stats for FS and A
    FS = hdf5(FS_pathname,'Imean')
    M0 = hdf5(FS_pathname,'M0')
    M0_FS = M0.mean()
    FSvar = hdf5(FS_pathname,'Ivar')/len(M0)
    A = hdf5(A_pathname,'Imean')
    M0 = hdf5(A_pathname,'M0')
    M0_A = M0.mean()
    Avar = hdf5(A_pathname,'Ivar')/len(M0)


    # Define flagged pixel mask, FPm, consising of defective_ROR, perimeter, BS, and boundary pixels
    RPm = hdf5(datafile,'RPm')
    shape = hdf5(datafile,'shape')
    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    PGFP = hdf5(datafile,'PGFP')
    PST = hdf5(datafile,'PST')
    W = hdf5(datafile,'W')
    CCD_id = hdf5(datafile,'CCD_id')
    sort_indices = hdf5(datafile,'sort_indices')
    reverse_indices = hdf5(datafile,'reverse_indices')
    q_flat = hdf5(datafile,'q_flat')
    qbin1 = hdf5(datafile,'qbin1')
    qbin2 = hdf5(datafile,'qbin2')
    q = hdf5(datafile,'q')
    Q = hdf5(datafile,'Q')
    VG_amp = hdf5(datafile,'VG_amp')

    # Generate vertical gradient for correction
    y_indices = indices(shape)[0] 
    VG = ones_like(y_indices) + VG_amp*(Y0 - y_indices)/shape[0]

    # Map GC_UCp correction (from glassy carbon) onto full size UCgc 
    UCgc = ones_like(W)
    GC_UCp = hdf5(datafile,'GC_UCp')[10:-10,10:-10]
    offset = int((GC_UCp.shape[0]-1)/2)
    UCgc[Y0-offset:Y0+offset+1,X0-offset:X0+offset+1] = GC_UCp

    FS_spurious = zeros_like(W)
    spurious = hdf5(FS_pathname,'spurious')
    FS_spurious[Y0-offset:Y0+offset+1,X0-offset:X0+offset+1] = spurious[10:-10,10:-10]

    A_spurious = zeros_like(W)
    spurious = hdf5(A_pathname,'spurious')
    A_spurious[Y0-offset:Y0+offset+1,X0-offset:X0+offset+1] = spurious[10:-10,10:-10]

    # Generate uniformity correction; use to generate Ic and Vc
    UC = UCgc*PST*VG*PGFP/PST.max()
    Ic = (FS - FS_spurious - (A - A_spurious)*(M0_FS/M0_A))/UC
    Vc = (FSvar + Avar*(M0_FS/M0_A)**2)/UC**2 
    Wz = W*(RPm == 0)
    # Flatten Ic, Vc, and Wz
    Ic_flat = Ic.flatten()[sort_indices]
    Vc_flat = Vc.flatten()[sort_indices]
    Wz_flat = Wz.flatten()[sort_indices]
    # Determine Zstat
    Zstat = z_statistic(prod(W.shape))
    # Integrate data from each CCD chip and assemble results into arrays
    print('Characterizing each CCD in mosaic ... please wait')
    S_CCD = []
    sigS_CCD = []
    outliers_CCD = []
    select_CCD = []
    N_CCD = []
    id_flat = CCD_id.flatten()[sort_indices]
    for chip_id in range(16):
        print('Processing CCDid = {}'.format(chip_id))
        select = (id_flat==chip_id)
        Wc_flat = Wz_flat*select
        OP_flat,N_pixels = outlier_pixels()
        Wc_flat *= ~OP_flat
        S,sigS,sigSV,S_flat = integrate(Ic_flat,Vc_flat,Wc_flat,q_flat,qbin1,qbin2,q)
        S_CCD.append(S)
        sigS_CCD.append(sigS)
        select_CCD.append(select)
        outliers_CCD.append(select*OP_flat)
        N_CCD.append(N_pixels)
    S_CCD = array(S_CCD)
    sigS_CCD = array(sigS_CCD)
    select_CCD = array(select_CCD)
    outliers_CCD = array(outliers_CCD)
    N_CCD = array(N_CCD)

    scale = ones(16)
    # Determine relative scale for inner modules
    inner_select = array([5,6,9,10])
    inner_q = (S_CCD[inner_select]>0).all(axis=0)
    inner_sum = S_CCD[inner_select][:,inner_q].sum(axis=1)
    scale[inner_select] = inner_sum/inner_sum[-1]
    Ss_CCD = (S_CCD.T/scale).T

    # Determine relative scale for neighbor modules
    neighbor_select = array([1,2,4,7,8,11,13,14])
    neighbor_q = (S_CCD[neighbor_select]>0).all(axis=0) & (q < 4.8) & (q > 0.025)
    neighbor_sum = S_CCD[neighbor_select][:,neighbor_q].sum(axis=1)
    scale[neighbor_select] = neighbor_sum/neighbor_sum.mean()
    Ss_CCD = (S_CCD.T/scale).T

    # Determine relative scale for corner modules
    corner_select = array([0,3,12,15])
    corner_q = (S_CCD[corner_select]>0).all(axis=0) & (q < 4.8) & (q > 0.025)
    corner_sum = S_CCD[corner_select][:,corner_q].sum(axis=1)
    scale[corner_select] = corner_sum/corner_sum.mean()
    Ss_CCD = (S_CCD.T/scale).T

    # Determine weighted averages for inner and neighbor and rescale neighbor accordingly 
    inner_neighbor_select = array([1,2,4,5,6,7,8,9,10,11,13,14])
    inner_neighbor_q = (S_CCD[inner_neighbor_select]>0).all(axis=0) & (q < 4.8) & (q > 0.025)
    inner_ave = (Ss_CCD[inner_select]*N_CCD[inner_select]).sum(axis=0)/N_CCD[inner_select].sum(axis=0)
    neighbor_ave = (Ss_CCD[neighbor_select]*N_CCD[neighbor_select]).sum(axis=0)/N_CCD[neighbor_select].sum(axis=0)
    neighbor_inner_scale = neighbor_ave[inner_neighbor_q].sum()/inner_ave[inner_neighbor_q].sum()
    scale[neighbor_select] *= neighbor_inner_scale
    Ss_CCD = (S_CCD.T/scale).T

    # Determine weighted averages for neighbor and corner and rescale corner accordingly 
    neighbor_corner_select = array([0,1,2,3,4,7,8,11,12,13,14,15])
    neighbor_corner_q = (S_CCD[neighbor_corner_select]>0).all(axis=0) & (q < 4.8) & (q > 0.025)
    neighbor_ave = (Ss_CCD[neighbor_select]*N_CCD[neighbor_select]).sum(axis=0)/N_CCD[neighbor_select].sum(axis=0)
    corner_ave = (Ss_CCD[corner_select]*N_CCD[corner_select]).sum(axis=0)/N_CCD[corner_select].sum(axis=0)
    corner_neighbor_scale = corner_ave[neighbor_corner_q].sum()/neighbor_ave[neighbor_corner_q].sum()
    scale[corner_select] *= corner_neighbor_scale
    Ss_CCD = (S_CCD.T/scale).T
    sigSs_CCD = (sigS_CCD.T/scale).T

    # Generate weighted average with selected modules (along upper left diagonal)
    M_select = array([0,1,4,5,6,9,10])
    SMwa = (Ss_CCD[M_select]*N_CCD[M_select]).sum(axis=0)/(N_CCD[M_select]).sum(axis=0)

    # Determine sigSwa
    sigSMwa = sqrt((sigSs_CCD[M_select]**2*N_CCD[M_select]).sum(axis=0)/N_CCD[M_select].sum(axis=0))
    
    # Generate SMwae, where the high q range is an extrapolation based upon linear fit of the last 32 values of q 
    q_polyfit = (q > q[-32]) & (q < q[-8])
    p = polyfit(q[q_polyfit], SMwa[q_polyfit], 1)
    SMwae = SMwa*(q < q[-8]) + polyval(p, q)*(q >= q[-8])

    # Fit SMwae, extrapolated SMwa, with a univariate spline using weights to determine number of knots
    us = UnivariateSpline(q,SMwae,w=1/sigSMwa) 

    # Generate UCp_flat and UCp; set UCp for DPm to 1.0 
    XPm = RPm > 1
    UCp = Ic/us(Q)
    UCp[XPm] = 1.0

    # Generate CCD_scale
    scale_flat = ones(len(q_flat))
    for j,select in enumerate(select_CCD):
        scale_flat[select] = scale[j]
    CCD_scale = scale_flat[reverse_indices].reshape(shape)

    # Replace low q region of UCp with GC_UCp
    UCp[Y0-offset:Y0+offset+1,X0-offset:X0+offset+1] = GC_UCp

    # Write UCp and CCD_scale in datafile
    hdf5(datafile,'UCp',UCp.astype('float32'))
    hdf5(datafile,'CCD_scale',CCD_scale.astype('float32'))

    seterr(divide='warn',invalid='warn')
    if chart:
        from os import path
        from charting_functions import chart_image,chart_xy_symbol,chart_image_mask,chart_xy_fit,chart_histogram
        beamtime = path.basename(datafile).split('.hdf5')[0]
        chart_image(spurious,'spurious scattering\n{}: {}'.format(beamtime,FS_dataset))

        chart_image(scale.reshape(4,-1),'scale\n{}: {}'.format(beamtime,FS_dataset))
        chart_xy_symbol(q,Ss_CCD/us(q),'Ss_CCD/us[q]\n{}: {}'.format(beamtime,FS_dataset),ymin=0.96,ymax=1.04)
        chart_xy_symbol(q,S_CCD/us(q),'S_CCD/us[q]\n{}: {}'.format(beamtime,FS_dataset),ymin=0.96,ymax=1.08)
        chart_image(UCp,'UCp\nVertical gradient: {}\n{}: {}'.format(VG_amp,beamtime,FS_dataset),vmin=0.96,vmax=1.04)
        chart_image(UCp/CCD_scale,'UCp/CCD_scale\n{}: {}'.format(beamtime,FS_dataset),vmin=0.96,vmax=1.04)
        if verbose:
            knots = us.get_knots()
            knot_indices = [where(knot == q)[0][0] for knot in knots]
            knot_select = zeros(len(q),bool)
            knot_select[knot_indices] = True
            chart_xy_symbol(q[inner_q],Ss_CCD[inner_select][:,inner_q],'Ss_CCD[inner]',x_label='q')
            chart_xy_symbol(q[neighbor_q],Ss_CCD[neighbor_select][:,neighbor_q],'Ss_CCD[neighbor]',x_label='q')
            chart_xy_symbol(q[corner_q],Ss_CCD[corner_select][:,corner_q],'Ss_CCD[corner]',x_label='q')
            chart_xy_symbol(q[inner_neighbor_q],array([inner_ave[inner_neighbor_q],neighbor_ave[inner_neighbor_q]]),'inner_ave and neighbor_ave',x_label='q')
            chart_xy_symbol(q[neighbor_corner_q],array([neighbor_ave[neighbor_corner_q],corner_ave[neighbor_corner_q]]),'neighbor_ave and corner_ave  ',x_label='q')
            chart_xy_symbol(q,S_CCD,'S_CCD',x_label='q',logy=True,ymin=100)
            chart_xy_symbol(q[inner_neighbor_q],Ss_CCD[inner_neighbor_select][:,inner_neighbor_q],'Ss_CCD[inner-neighbor]',x_label='q')
            chart_xy_symbol(q[neighbor_corner_q],Ss_CCD[neighbor_corner_select][:,neighbor_corner_q],'Ss_CCD[neighbor-corner]',x_label='q')
            chart_xy_symbol(q,Ss_CCD,'Ss_CCD',x_label='q',logy=True,ymin=100)
            chart_xy_symbol(q,Ss_CCD,'Ss_CCD',x_label='q')
            chart_xy_symbol(q,N_CCD,'N_CCD',x_label='q')
            outliers = outliers_CCD.any(axis=0)[reverse_indices].reshape(shape)
            chart_image_mask(UCp,outliers,'UCp with Z = {:0.3f} sigma\n{}:{}'.format(Zstat,beamtime,FS_dataset),vmin=0.96,vmax=1.04)
            chart_xy_fit(q,SMwa,polyval(p, q),'Linear fit of SMwa from {:0.3f} < q < {:0.3f}'.format(q[-32],q[-8]),x_label='q')
            chart_xy_fit(q,SMwa,us(q),'Univariate Spline fit of SMwa with {} knots\n{}:{}'.format(len(knots),beamtime,FS_dataset),x_label='q')
            chart_xy_symbol(q,SMwa-us(q),'SMwa minus univariate spline fit with {} knots\n{}:{}'.format(len(knots),beamtime,FS_dataset),ymin=-2,ymax=2,x_label='q')
            chart_xy_symbol(q,array([SMwa,SMwa/knot_select]),'SMwa fit with {} knots\n{}:{}'.format(len(knots),beamtime,FS_dataset),x_label='q',ms=2)
            chart_xy_symbol(q,array([SMwa,sigSMwa]),'[SMwa,sigSMwa]\n{}:{}'.format(beamtime,FS_dataset),xmin= 0.025,logx=True,logy=True,x_label='q')
            chart_histogram(UCp/CCD_scale,'UCp/CCD_scale',binsize=0.005,xmin=0.4,xmax=1.3)

def datafile_VPC(datafile,chart=False):
    """Process specified scattering dataset to generate VPC."""
    from numpy import zeros_like
    # Extract dataset name and DV from datafile
    FS_dataset = hdf5(datafile,'FS_dataset')
    DV = hdf5(datafile,'DV')
    # Extract Imean and Ivar from FS_dataset
    i,pathname = index_pathname(datafile,FS_dataset)
    Imean = hdf5(pathname,'Imean')
    Ivar = hdf5(pathname,'Ivar')
    # Determine VPC from Imean, Ivar, and DV; write to datafile
    VPC = zeros_like(DV)
    VPC[Imean>0] = (Ivar-DV)[Imean>0]/Imean[Imean>0]
    hdf5(datafile,'VPC',VPC)
    if chart:
        from os import path
        from charting_functions import chart_histogram,chart_image_mask
        basename = path.splitext(path.basename(datafile))[0]
        chart_histogram(VPC,'VPC\n{}'.format(basename),binsize=0.01,xmax=2)
        chart_image_mask(VPC,VPC>0.8,'VPC (flagged VPC > 0.8)\n{}'.format(basename),vmax=0.5)
    
def datafile_dark(datafile,chart=False):
    """Dark datasets acquired with different integration times are 
    evaluated to generate slope and intercept, which accounts for 
    leakage current in some pixels. These datasets must first be
    analyzed by 'zinger_free_stats' to generate relevant statistics. 
    Given Imean, Ivar, and period, Imean is fitted to a straight line 
    as a function of period. The weighted least squares fit generates
    'intercept' and 'slope', which are written to datafile. 'DV',
    the readout dark variance, is also written to datafile. 

    From 'intercept' and 'slope', background counts 
    can be calculated given integration time 'T' according to:
        background(T) = intercept + T*slope
    """
    from numpy import array,sqrt,ones_like,sort
    datasets = hdf5(datafile,'datasets')
    dark_datasets = [dataset for dataset in datasets if 'Dark' in dataset]
    args = []
    for dataset in dark_datasets:
        i,pathname = index_pathname(datafile,dataset)
        try:
            Imean = hdf5(pathname,'Imean')
        except:
            args.append((datafile,dataset))
    if len(args) > 0:
        N_proc = len(args)
        import multiprocessing
        multiprocessing.set_start_method('fork', force=True)
        with multiprocessing.Pool(N_proc) as pool:
            pool.starmap(zinger_free_stats,args)

    # Read 'dark' data and append relevant stats
    #   T period in seconds
    #   C counts
    #   V variance
    #   W weights 
    T = []
    C = []
    V = []
    for dataset in dark_datasets:
        i,pathname = index_pathname(datafile,dataset)
        Imean = hdf5(pathname,'Imean')
        Ivar = hdf5(pathname,'Ivar')
        period = hdf5(pathname,'period')
        T.append(period)
        C.append(Imean)
        V.append(Ivar)       
    T = array(T)
    C = array(C)
    V = array(V)

    # Calculate weights as 1/V, avoiding nan when V = 0
    W = ones_like(V)
    W[V>0] = 1/V[V>0]
    Ts = (ones_like(W).T*T).T

    # Perform weighted linear least-squares fit of C vs. T; generate intercept and slope
    Wsum = W.sum(axis=0)
    WT2sum = (W*Ts*Ts).sum(axis=0)
    WTsum = (W*Ts).sum(axis=0)
    WCsum = (W*C).sum(axis=0)
    WTCsum = (W*Ts*C).sum(axis=0)
    delta = Wsum*WT2sum - WTsum**2

    intercept = (WT2sum*WCsum - WTsum*WTCsum)/delta
    slope = (Wsum*WTCsum - WTsum*WCsum)/delta

    # Compute average variance
    DV = V.mean(axis=0)

    # Add DV == 0 to defective pixel mask
    RPm = hdf5(datafile,'RPm')
    RPm |= 4*(DV==0).astype('uint8')
    
    # Write results in datafile
    hdf5(datafile,'slope',slope.astype('float32'))
    hdf5(datafile,'intercept',intercept.astype('float32'))
    hdf5(datafile,'DV',DV.astype('float32'))
    hdf5(datafile,'RPm',RPm)

    if chart:
        from os import path
        from charting_functions import chart_histogram,chart_image_mask
        beamtime = path.splitext(path.basename(datafile))[0]
        chart_histogram(DV,'DVPm: {}'.format(beamtime),binsize=0.1,xmax=10)
        DV100_threshold = sort(DV.flatten())[-100]
        chart_image_mask(DV,DV>DV100_threshold,'DV: {}\n largest 100 (DV > {:0.3f})'.format(beamtime,DV100_threshold),vmin=0,vmax=2)
        
        chart_histogram(intercept,'intercept: {}'.format(beamtime),binsize=0.1,xmax=20)
        intercept_threshold = sort(intercept.flatten())[-50]
        chart_image_mask(intercept,intercept>intercept_threshold,'intercept: {} \nlargest 50 (intercept > {:0.2f}) \n'.format(beamtime,intercept_threshold),vmin=8,vmax=12)
        
        chart_histogram(slope,'slope: {}'.format(beamtime),binsize=0.1,xmin=-1)
        slope_threshold = sort(slope.flatten())[-50]
        chart_image_mask(slope,slope>slope_threshold,'slope: {} \nlargest 50 (slope > {:0.2f}) \n'.format(beamtime,slope_threshold),vmin=-0.2,vmax=0.2)
        
        # Calculate fit and rms fit error.
        fit = intercept + slope*Ts
        error = sqrt(((C - fit)**2).sum(axis=0)/len(dark_datasets))    
        chart_histogram(error,'fit error (rms): {}'.format(beamtime),binsize=0.01)
        chart_histogram(C - fit,'fit error (absolute): {}'.format(beamtime),binsize=0.01)

def datafile_global_mask(datafile,term='Tramp_B-2',chart=None):
    """Generates Global mask from UCd0 extracted from Tramp_B datasets.
    Write result to bit 3 of RPm. Tramp datasets must be integrated before
    calling this function."""
    from scipy.ndimage import binary_dilation,binary_erosion
    from numpy import ones,zeros,minimum
    # Load datasets and RPm
    datasets = hdf5(datafile,'datasets')
    RPm = hdf5(datafile,'RPm')
    # Reset global mask (bit3) in RPm
    RPm -= 8*(RPm&8 == 8).astype('uint8')
    # Extract UCd0 from Tramp_B datasets and find minimum (as well as mean)
    B_datasets = [dataset for dataset in datasets if term in dataset]
    UCd_min = ones(RPm.shape)
    UCd_mean = zeros(RPm.shape)
    N = len(B_datasets)
    for dataset in B_datasets:
        i,pathname = index_pathname(datafile,dataset)
        UCd = hdf5(pathname,'UCd')
        UCd_mean += UCd/N
        UCd_min = minimum(UCd_min,UCd)
    # Generate mask from anomalously low values of UCd_min
    mask = (UCd_min < 0.98)*~(RPm>1)
    # Dilation with two iterations to connect gaps
    mask = binary_dilation(mask,iterations = 2)
    # Erosion with four iterations to eliminate islands
    mask = binary_erosion(mask,iterations = 4)
    # Dilation with eight iterations to expand edge toward unaffected pixels
    mask = binary_dilation(mask,structure = ones((3,3)),iterations = 8)
    RPm += 8*mask.astype('uint8')
    # Write RPm to datafile
    hdf5(datafile,'RPm',RPm)
    # Chart results
    if chart:
        from charting_functions import chart_image
        beamtime = datafile.split('/')[-1].split('.hdf5')[0]
        mask = RPm&8 == 8
        chart_image(UCd_mean,'UCd_mean\n{}'.format(beamtime),vmin=0.96,vmax=1.04)
        chart_image(UCd_mean/~mask,'UCd_mean after masking\n{}'.format(beamtime),vmin=0.96,vmax=1.04)
        chart_image(UCd_min/~mask,'UCd_min after masking\n{}'.format(beamtime),vmin=0.96,vmax=1.04)
        chart_image(RPm,'RPm\n{}'.format(beamtime))
    
def multiprocess_zfs(datafile,N_proc=6):
    """Uses multiprocessor approach to generate zinger-free stats of relevant files."""
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    print('Multiprocessing with zinger_free_stats')
    datasets = hdf5(datafile,'datasets')
    zfs_datasets = [dataset for dataset in datasets if ('Reference_A' in dataset)| ('Tramp_A' in dataset) | ('_FS' in dataset) | ('_CH5' in dataset) | ('_CX5' in dataset)]
    # Find datasets that have not yet been processed
    args = []
    for dataset in zfs_datasets:
        i,pathname = index_pathname(datafile,dataset)
        try:
            Imean = hdf5(pathname,'Imean')
        except:
            args.append((datafile,dataset))
    with multiprocessing.Pool(N_proc) as pool:
        pool.starmap(zinger_free_stats,args)

def zinger_free_stats(datafile,dataset):
    """Generates zinger-free statistics for dataset and returns dictionary.
    If not a 'Dark' dataset, the background-subtracted image 
    is normalized to unit area before computing the zinger-free average. 
    This step is necessary to accurately compute the pixel-to-pixel variance,
    Ivar, in the presence of varying x-ray intensities. This function 
    determines Imin and Imax, which represent the four smallest and four largest
    values found for each pixel in the dataset. Outliers are excluded from the 
    calculation of Imean and Ivar. The threshold for flagging outliers is defined 
    according to the Z statistic that would be expected to generate one
    false positive per image. The frames in which the largest Imax and smallest 
    Imin are found correspond to fmax and fmin, respectively.  'scale' represents 
    the integrated number of counts for each image in the series after background 
    subtraction. """
    from numpy import zeros,uint16,float64,where,ones,sort
    from time import time
    i,pathname = index_pathname(datafile,dataset)
    filenames = hdf5(pathname,'filenames')
    if  'Dark' not in dataset:
        period = hdf5(pathname,'period')
        intercept = hdf5(datafile,'intercept')
        slope = hdf5(datafile,'slope')
        background = intercept + period*slope
        RPm = hdf5(datafile,'RPm')
        DPm = (RPm & 4) == 4
    for j,filename in enumerate(filenames):
        t0 = time()
        image = mccd_read(filename).astype(float64)
        if j == 0:
            Nf = len(filenames)
            Nr,Nc = image.shape
            Imax = zeros((4,Nr,Nc),float64)
            fmax = zeros((Nr,Nc),uint16)
            Imin = ones((4,Nr,Nc),float64)*2**15
            fmin = zeros((Nr,Nc),uint16)
            Isum1 = zeros((Nr,Nc),float64)
            Isum2 = zeros((Nr,Nc),float64)
            scale = ones(Nf)
        # If not 'Dark' dataset, calculate background and normalize
        if  'Dark' not in dataset:
            image -= background
            scale[j] = image[~DPm].sum() 
            image /= scale[j]
        # Update Imax, Imin, and the frames in which the max/min were found
        Imax[0] = where(image > Imax[0],image,Imax[0])
        imax = Imax[0] > Imax[-1]
        fmax[imax] = j
        Imax = sort(Imax,axis=0)
        Imin[-1] = where(image < Imin[-1],image,Imin[-1])
        imin = Imin[-1] < Imin[0]
        fmin[imin] = j
        Imin = sort(Imin,axis=0)
        # Accumulate Isum1, Isum2
        Isum1 += image
        Isum2 += image**2
        print('\r{:0.3f} seconds to process image {} in {}                         '.format(time()-t0,j,dataset),end='\r')
    # Estimate mean and stdev given N (ZN: Z for 3 out of N)
    N = len(filenames)
    ZN = z_statistic(N/3)
    mean = (Imin[-1] + Imax[0])/2
    stdev = 0.5*((Imax[0] - Imin[-1]))/ZN
    # Find Imax and Imin outliers according to Z statistic that corresponds to 1 false positive per image (Nr*Nc)
    Z = z_statistic(Nr*Nc)
    Imax_outliers = Imax > mean + Z*stdev
    Imin_outliers = Imin < mean - Z*stdev
    # Compute Imean and Ivar afer omitting Imax and Imin outliers
    Nmo = (N - Imax_outliers.sum(axis=0) - Imin_outliers.sum(axis=0))
    Imean = (Isum1 - (Imax_outliers*Imax).sum(axis=0) - (Imin_outliers*Imin).sum(axis=0))/Nmo
    Ivar = (Isum2 - (Imax_outliers*Imax**2).sum(axis=0) - (Imin_outliers*Imin**2).sum(axis=0))/Nmo - Imean**2
    # Rescale data according to the mean of scale.
    scale_mean = scale.mean()
    Imean *= scale_mean
    Ivar *= scale_mean**2
    Imax *= scale_mean
    Imin *= scale_mean

    zfs_dict = {'Imean':Imean,'Ivar':Ivar,'Imax':Imax,'Imin':Imin,'scale':scale,'fmin':fmin,'fmax':fmax,'N':N}
    hdf5(pathname,'Imean',Imean.astype('float32'))
    hdf5(pathname,'Ivar',Ivar.astype('float32'))
    hdf5(pathname,'Imax',Imax[-1].astype('float32'))
    hdf5(pathname,'Imin',Imin[0].astype('float32'))
    return zfs_dict

def multiprocess_integrate(datafile,term=None,N_proc=4):
    """Uses multiprocessing approach with tasks assigned according to dataset."""
    import multiprocessing
    from time import time
    multiprocessing.set_start_method('fork', force=True)
    t0 = time()
    # Define args for multiprocess integration of each dataset
    exclude_terms = ['Dark','PSF']
    datasets = hdf5(datafile,'datasets')
    if term is not None:
        datasets = [dataset for dataset in datasets if term in dataset]
    else:
        datasets = [dataset for dataset in datasets if not any([exclude_term in dataset for exclude_term in exclude_terms])]
    args = [(datafile,dataset) for dataset in datasets]
    with multiprocessing.Pool(N_proc) as pool:
        pool.starmap(integrate_dataset,args)
    print('\nProcessed {} datasets in {:0.3f} seconds'.format(len(datasets),time()-t0))

def integrate_dataset(datafile,dataset,chart=False):
    """Process image files in dataset according to UCd_status, whose 
    bit0 defines the 'first_pass' boolean, and bit1 defines the 
    'global_mask' boolean. UCd corresponds to the uniformity
    correction for dataset. If integrated a second time, UCd is 
    combined with UCp to account for dataset-specific
    sources of systematic error. After the second pass, sigS should be 
    very close to sigSV, the standard deviation predicted assuming perfect
    uniformity correction."""
    from numpy import array,where,indices,ones_like,zeros_like,ones
    from time import time
    from os import path

    def An_prescale_from_dataset(dataset):
        """Returns A and prescale according to dataset name."""
        from numpy import zeros_like,ones_like
        An = hdf5(datafile,'An')
        if ('_A-' in dataset) | ('_GC' in dataset):
            prescale = ones_like(An)
            An = zeros_like(An)
        elif '_CH' in dataset:
            prescale = hdf5(datafile,'CHT')
        elif '_CX' in dataset:
            prescale = hdf5(datafile,'CXT')
        elif '_FS' in dataset:
            prescale = hdf5(datafile,'PST')
        else:
            prescale = hdf5(datafile,'CST')
        return An,prescale

    def outlier_pixels():
        """Uses Ic_flat, Vc_flat, and Wz_flat to flag outlier pixels. 
        Returns OP_flat, sigma_flat, N_pixels."""
        from numpy import zeros,median,sqrt,arange
        sigma_flat = zeros(len(q_flat))
        OP_flat = zeros(len(q_flat),bool)
        N_pixels = zeros(len(q),'uint16')
        # Determine sigma_flat and OP_flat relative to median from binned values of q
        select = Wz_flat > 0
        for j in range(len(q)):
            first = qbin1[j]
            last  = qbin2[j]
            selected = arange(first,last)[select[first:last]]
            N = len(selected)
            if N > 2:
                N_pixels[j] = N
                Ics = Ic_flat[selected]
                Vcs = Vc_flat[selected]
                sigma = (Ics - median(Ics))/sqrt(Vcs)
                sigma_flat[selected] = sigma
                outlier = abs(sigma) > sigma.std()*Zstat
                OP_flat[selected] = outlier
        # Generate unique q from q_flat in SAXS region (q<0.2); needed for outlier_pixels()
        # qu,N = unique(q_flat[q_flat<0.2],return_counts=True)
        # Nu = cumsum(N)
        # Determine sigma_flat and OP_flat relative to median for every unique value of q in the SAXS region
        # for j in range(len(qu)-1):
        #     first = Nu[j]
        #     last = Nu[j+1]
        #     Ij = Ic_flat[first:last][select[first:last]]
        #     Vj = Vc_flat[first:last]
        #     Nj = len(Ij)
        #     sigma_flat[first:last] = 0
        #     OP_flat[first:last] = False
        #     if Nj == 2:
        #         sigma = (Ij.max()-Ij.min())/sqrt(Vj.sum())
        #         if sigma > Zstat: 
        #             zinger = Ij == Ij.max()
        #             OP_flat[first:last] = zinger
        #             sigma_flat[first:last] = zinger*sigma
        #     elif Nj > 2:
        #         I_median = median(Ij)
        #         sigma = (Ic_flat[first:last] - I_median)/sqrt(Vc_flat[first:last])
        #         zinger = sigma > sigma.std()*Zstat 
        #         OP_flat[first:last] = zinger  # flag only positive outliers              
        #         sigma_flat[first:last] = sigma
        return OP_flat,sigma_flat,N_pixels

    # Determine if global mask is set
    RPm = hdf5(datafile,'RPm')
    global_mask = (RPm&8==8).any()

    # Determine pathname for dataset
    i,pathname = index_pathname(datafile,dataset)

    # Try to load UCd and UCd_status; force first_pass if global mask has changed state 
    try:
        UCd_status = hdf5(pathname,'UCd_status')
        UCd = hdf5(pathname,'UCd')
        first_pass = (UCd_status&2==2) != global_mask
    except:
        first_pass = True
    if first_pass:
        UCd_status = (2*global_mask).astype('uint8')
        UCd = ones_like(RPm).astype('float32')
    # Exclude global pixel mask from DPm when processing FS datasets
    if '_FS' in dataset:
        DPm = (RPm==2) | (RPm==4)
    else:
        DPm = RPm>1

    # If bit0 (integration count) has not been set, integrate the data
    if UCd_status&1==0:
        # Load data needed from datafile 
        sort_indices = hdf5(datafile,'sort_indices')
        reverse_indices = hdf5(datafile,'reverse_indices')
        W = hdf5(datafile,'W')
        VPC = hdf5(datafile,'VPC')
        DV = hdf5(datafile,'DV')
        intercept = hdf5(datafile,'intercept')
        slope = hdf5(datafile,'slope')
        q_flat = hdf5(datafile,'q_flat')
        qbin1 = hdf5(datafile,'qbin1')
        qbin2 = hdf5(datafile,'qbin2')
        q = hdf5(datafile,'q')
        Q = hdf5(datafile,'Q')
        psi = hdf5(datafile,'psi')
        UCp = hdf5(datafile,'UCp')
        PGFP = hdf5(datafile,'PGFP')
        shape = hdf5(datafile,'shape')
        X0 = hdf5(datafile,'X0')
        Y0 = hdf5(datafile,'Y0')
        saxs_boxsize = hdf5(datafile,'saxs_boxsize')
        
        # Generate vertical gradient for correction
        VG_amp = hdf5(datafile,'VG_amp')
        y_indices = indices(shape)[0] 
        VG = ones_like(y_indices) + VG_amp*(Y0 - y_indices)/shape[0]

        # Generate combined uniformity correction 
        An,prescale = An_prescale_from_dataset(dataset)
        UC = prescale*VG*PGFP*UCp*UCd

        # Zero DPm in Wz and flatten
        Wz = W.copy()
        Wz[DPm] = 0
        Wz_flat = Wz.flatten()[sort_indices]

        # Set Zstat to one false positive per image
        Zstat = z_statistic((~DPm).sum())

        # Load needed parameters from pathname
        M0 = hdf5(pathname,'M0')
        omit = hdf5(pathname,'omit')
        period = hdf5(pathname,'period')
        filenames = hdf5(pathname,'filenames')
        prefix = '/Volumes/mirrored_femto-data2'
        filenames = [prefix + filename.split('/net/femto-data2.niddk.nih.gov')[-1] for filename in filenames]
        spurious = hdf5(pathname,'spurious')

        # Add normalized spurious from dataset to spurious-free, normalized air scattering (An)
        s = int((saxs_boxsize-1)/2)
        An[Y0-s:Y0+s+1,X0-s:X0+s+1] += spurious/M0[omit==0].mean()

        # Determine background given period
        bkg = intercept + period*slope

        # Integrate images found in dataset
        IC = []
        S = []
        sigS = []
        sigSV = []
        OP_frame = []
        OP_index = []
        OP_sigma = []
        N_flat = zeros_like(q_flat)
        Sfit_flat = zeros_like(q_flat)
        Sexp_flat = zeros_like(q_flat)
        for i,filename in enumerate(filenames):
            t0 = time()
            # Load image as float32 and subtract background
            image_bkg = mccd_read(filename).astype('float32') - bkg
            # Calculate IC, background-subtracted integrated counts with defective pixels omitted
            IC.append(image_bkg[W>0].sum())
            # Subtract M0-scaled, air-scattering plus spurious contribution to image and divide result by UC
            Ic = ((image_bkg - M0[i]*An)/UC)
            Vc = ((VPC*abs(image_bkg) + DV)/UC**2) + 1
            # Flatten Ic and Vc
            Ic_flat = Ic.flatten()[sort_indices]
            Vc_flat = Vc.flatten()[sort_indices]
            # Identify outlier pixels and zero weights accordingly
            OP_flat,sigma_flat,N_pixels = outlier_pixels()
            Wc_flat = Wz_flat*~OP_flat
            # Generate Si and sigSi
            Si,sigSi,sigSVi,Si_flat = integrate(Ic_flat,Vc_flat,Wc_flat,q_flat,qbin1,qbin2,q)
            # Assemble Sfit_flat, Sexp_flat and OP information for images not omitted
            if omit[i] == 0:
                # Construct sum of Si_flat and Ic_flat (omitting outlier images)
                Sfit_flat += Si_flat
                zingers = OP_flat & (sigma_flat>0)
                Sexp_flat += Ic_flat*~zingers
                N_flat += ~zingers
            # Generate outlier pixel lists (frame, index, sigma)
            OPi = where(OP_flat)[0]
            OP_frame.extend(array(len(OPi)*[i]))
            OP_index.extend(OPi)
            OP_sigma.extend(sigma_flat[OPi])       
            # Append Si, sigSi, sigSVi
            S.append(Si)
            sigS.append(sigSi)
            sigSV.append(sigSVi)
            basename = path.basename(filename)
            print('\rProcessed {} in {:0.3f} seconds                   '.format(basename,time()-t0),end='\r')
        # Generate Sfit image averaged across the dataset (omitting outlier images)
        Sfit = (Sfit_flat/(omit==0).sum())[reverse_indices].reshape(shape)
        # Generate Sexp image averaged across the dataset (if always outlier, replace with last known value)
        N_flat[N_flat==0] = 1 # Avoid divide by zero error
        Sexp = (Sexp_flat/N_flat)[reverse_indices].reshape(shape)
        Sexp[Sexp==0] = Ic[Sexp==0]
        # Compute UCd 
        UCd = (Sexp/Sfit).astype('float32')

        # Define perimeter pixel mask and set corresponding pixels in UCd to unity
        PPm = ones(shape,bool)
        PPm[1:-1,1:-1] = False
        UCd[PPm] = 1
    
        # Convert lists to arrays
        IC = array(IC)
        S = array(S).astype('float32')
        sigS = array(sigS).astype('float32')
        sigSV = array(sigSV).mean(axis=0)
        OP_frame = array(OP_frame).astype('uint16')
        OP_index = array(OP_index).astype('uint32')
        OP_sigma = array(OP_sigma).astype('float32')
        # Write results
        hdf5(pathname,'IC',IC)
        hdf5(pathname,'S',S)
        hdf5(pathname,'sigS',sigS)
        hdf5(pathname,'sigSV',sigSV)
        hdf5(pathname,'OP_frame',OP_frame)
        hdf5(pathname,'OP_index',OP_index)
        hdf5(pathname,'OP_sigma',OP_sigma)
        # If first_pass, write UCd; else update and then write UCd_status
        if first_pass:
            hdf5(pathname,'UCd',UCd)
        else:
            UCd_status += 1
        hdf5(pathname,'UCd_status',UCd_status)
        hdf5_repack(pathname)


        if chart:
            from charting_functions import chart_xy_scatter,chart_xy_symbol,chart_image,chart_xy_rainbow
            from numpy import argsort
            qmax = 0.5
            chart_xy_scatter(Q[Q<qmax],(Sexp/(RPm<1))[Q<qmax],psi[Q<qmax],'Sexp in SAXS region\n{}'.format(dataset),logy=True,x_label='q')
            chart_xy_symbol(Q[Q<qmax],(Sfit/(RPm<1))[Q<qmax],'Sfit in SAXS region\n{}'.format(dataset),logy=True,x_label='q')
            chart_xy_symbol(q,array([Si,sigSi,sigSVi]),'Si,sigSi,sigSVi\n{}'.format(dataset),logy=True,logx=True,x_label='q')
            chart_image(UCd,'UCd\n{}'.format(dataset),vmin=0.96,vmax=1.04)
            RTD_temp = hdf5(pathname,'RTD_temp')
            chart_xy_rainbow(q,S[argsort(RTD_temp)],'S\n{}'.format(dataset),logx=True,logy=True,x_label='q')
            chart_xy_rainbow(q,sigS[argsort(RTD_temp)],'sigS\n{}'.format(dataset),logx=True,logy=True,x_label='q')

def integrate(I_flat,V_flat,W_flat,q_flat,qbin1,qbin2,q):
    """Returns the weighted mean (S) and standard deviation of the mean (sigS)
    for each q bin defined by qbin1 and qbin2. Requires as input I_flat (counts), 
    V_flat (variance), W_flat (weights), and q_flat. The weights must be zeroed 
    for defective pixels and zingers. The data in each bin are fitted to a straight 
    line (Ii = a + b*qi) with S representing the value of the line at the corresonding value 
    of q, and sigS representing the estimated standard deviation of S. 
    Also returns S_flat, the weighted least-squares linear fit evaluated across q_flat."""
    from numpy import cumsum,sqrt,zeros,ones_like
    S = zeros(len(q))
    sigS = zeros(len(q))
    sigSV = zeros(len(q))
    S_flat = ones_like(q_flat)
    # Copy q_flat, I_flat, V_flat, W_flat; convert to float64 to compute statistics accurately
    qc_flat = q_flat.copy().astype(float)
    Ic_flat = I_flat.copy().astype(float)
    Vc_flat = V_flat.copy().astype(float)
    Wc_flat = W_flat.copy().astype(float)
    # Calculate S and sigS from weighted linear least-squares fit of values in each q bin
    cs_I  = cumsum(Ic_flat*Wc_flat)
    cs_I2 = cumsum(Ic_flat**2*Wc_flat)
    cs_q  = cumsum(qc_flat*Wc_flat)
    cs_q2 = cumsum(qc_flat**2*Wc_flat)
    cs_qI = cumsum(qc_flat*Ic_flat*Wc_flat)
    cs_W  = cumsum(Wc_flat)
    cs_N =  cumsum(Wc_flat>0)
    cs_WV  = cumsum(Wc_flat/Vc_flat)
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
            sWV = cs_WV[last] -  cs_WV[first]
            delta = sW*sq2 - sq**2
            a =  (1/delta)*(sq2*sI - sq*sqI)
            b =  (1/delta)*(sW*sqI - sq*sI)
            S[j] = a + b*q[j]
            variance = (sI2 - 2*a*sI - 2*b*sqI + 2*a*b*sq + b*b*sq2 + a*a*sW)/sW/N
            sigS[j] = sqrt(variance)
            sigSV[j] = sqrt(sW/sWV/N)
            if j == len(q)-1:
                S_flat[first:] = a + b*q_flat[first:]
            else:
                S_flat[first:last] = a + b*q_flat[first:last]
    return S,sigS,sigSV,S_flat

def datafile_geometry(datafile,chart=False):
    """
    Given arguments, generates relevant geometric information and writes to
    datafile. Includes information needed to convert images to flattened, 
    sorted arrays, and convert flattened, sorted arrays back into images. 
    Includes q_flat, qbin1, qbin2, and q, which are needed for integration. 
    Note that polarization = 1 for Horiz polarization; -1 for Vert polarization. 
    The scale factors are combined into PGFP, which corresponds to pol, geo, 
    filter, and phosphor, whose on-axis amplitude is normalized to unity.
    
    T0_phosphor: phosphor transmittance at normal incidence; angle dependence
        of phosphor is proportional to the absorbance probability, which increases at 
        higher theta angle due to a longer pathlength; normalized to unity
        at normal incidence.
    T0_filter: filter transmittance at normal incidence.

    FS_tilt: tilt of fused silica slab in degrees.
    
    Also generates RPm, whose bits define various masks.
        bit0: edges pixel mask (EPm: pixels with contributions from more than one CCD chip)
        bit1: beamstop pixel mask (BPm: pixels inside the beamstop)
        bit2: defective pixel mask (DPm: permimeter and defective read-out register pixels)
        bit3: outlier pixel mask (OPm)

    'defective_ROR' is a tuple array extracted from datafile that
    specifies which readout registers are defective. The first 
    index in the tuple specifies the CCD id, and the second index 
    specifies the readout register id. CCD_id identifies virtual 
    pixels assigned solely to the corresponding id index. 
    ROR_masks identify all virtual pixels affected by the 
    corresponding ROR index. The algorithm employs binary_dilation to ensure
    all virtual pixels affected by the defective ROR are flagged while
    avoiding false positives from readout registers on adjacent CCD modules. 
    """

    from numpy import zeros_like,ones_like,ones,square,sin,cos,exp,log,argsort,round,pi
    from scipy.ndimage import binary_dilation

    def qbin(Q,RPm,dq):
        """Generates q, qbin1, qbin2, and q_flat given Q, RPm and dq, where
        RPm is used to define the usable range of q in integer multiples of dq,
        and [qbin1[i]:qbin2[i]] specifies the range of q_flat assigned
        to each bin i. """
        from numpy import arange,rint,roll,where
        # Define q_flat, sorted list of q across the entire detector
        q_flat = Q.flatten()[sort_indices]
        # Define qmin, and qmax, limits that exclude beamstop, perimeter, and defective ROR
        reject = ((RPm & 2) == 2) | ((RPm & 1) == 1)
        Qselect = Q[~reject]
        qmin = Qselect[Qselect>0].min()
        qmax = Qselect.max() 
        # Define q, values between qmin and qmax in integer multiples of dq
        q0 = round((qmin + dq/2)/dq)*dq
        q1 = round((qmax - dq/2)/dq)*dq
        q = arange(q0,q1,dq)
        # Define qbin1 and qbin2, the start and stop indices for dq bins along q_flat
        q_boundary = rint(q_flat/dq)
        qbin2 = (where(q_boundary != roll(q_boundary,-1))[0]+1)[int(q0/dq):int(q1/dq)+1][:len(q)]
        qbin1 = roll(qbin2,1)
        qbin1[0] = 0
        qbin2[-1] = len(q_flat)-1
        return q,qbin1,qbin2,q_flat

    def Rayonix_theta_psi_Q(shape,X0,Y0,distance,pixelsize,keV):
        """Given geometric parameters, calculates and returns theta, psi, and Q."""
        from numpy import indices, sqrt,arctan2,arctan,pi,sin
        y_indices,x_indices = indices(shape)
        r = pixelsize*sqrt((y_indices-Y0)**2+(x_indices-X0)**2)
        psi = -arctan2((y_indices-Y0),(x_indices-X0))
        theta = arctan(r/distance)/2
        h = 4.135667696e-15 # Planck's constant [eV-s]
        c = 299792458e10 # speed of light [Angstroms per second]
        wavelength = h*c/(keV*1000) # x-ray wavelength [Angstroms]
        Q = 4*pi*sin(theta)/wavelength
        return theta,psi,Q

    def transmittance_air(theta,BP=760,RH=35,TC=22,t=96.3,keV=11.648):
        """Returns theta-dependent transmittance of x-rays through air along path in
        front of the x-ray detector. Names, thicknesses, and absorption coefficients
        [mm-1] calculated at 12 keV are hard-coded, except for the humidified air
        contribution, which is a function of BP, RH, and TC. The absorbance coefficients 
        are assumed to scale as the inverse cubed power of the x-ray energy."""
        from numpy import exp,cos
        mu = mu_Humidified_Air(BP,RH,TC,keV)
        transmittance = exp(-(mu*t/cos(2*theta)))
        return transmittance

    def transmittance_filters(theta,keV=11.648):
        """Returns theta-dependent transmittance of x-rays through materials 
        encountered between the sample capillary and the phosphor in
        front of the x-ray detector. Names and thicknesses are hardcoded and
        correspond to mu coefficients [mm-1] calculated at specified keV."""
        from numpy import array,exp,cos
        material = ['water', 'fused_silica', 'polyimide', 'parylene_N', 'mylar', 'Be']
        t = array([0,0,0.13,0.02,0.05,0.2])
        mu =  mu_molecular(keV)
        transmittance = exp(-(mu*t).sum()/cos(2*theta))
        return transmittance

    def transmittance_phosphor(theta,keV=11.648,t=0.04,fill_factor=0.58):
        """Returns phosphor transmittance as a function of theta given the
        x-ray energy, phosphor thickness, t [mm], and fill factor. The fill factor
        corresponds to the volume fraction of Gadox in the phosphor film and is 
        estimated to be 0.58.
        """
        from numpy import exp,cos
        mu = mu_Gadox(keV) # inverse 1/e penetration depth [mm-1]
        alpha = mu*t*fill_factor
        return exp(-alpha/cos(2*theta))
    
    def Gadox_responsivity(phi,keV=11.648,t=0.04,fill_factor=0.58):
        """Calculates phi-dependent probability of x-ray absorption in the 
        Gadox phosphor, which is assumed to be proportional to phosphor 
        responsivity."""
        from numpy import exp,log,cos
        mu = mu_Gadox(keV) # inverse 1/e penetration depth [mm-1]
        T0 = exp(-mu*t*fill_factor)
        return (1 - exp(log(T0)/cos(phi)))/(1-T0)

    def mu_Gadox(keV=11.648):
        """Returns inverse 1/e penetration depth [mm-1] of Gadox given x-ray energy
        in keV. The transmission through a 10-um thick slab of Gadox (Gd2O2S) was
        calculated every 100 eV over an energy range spanning 8-16 keV using:
            http://henke.lbl.gov/optical_constants/filter2.html
        This result was then converted to mu and saved as a tab-delimited text
        file. The returned result is calculated using a univariate spline and
        should be valid over the range 8-16 keV."""
        from numpy import loadtxt
        from scipy.interpolate import UnivariateSpline
        E_mu = loadtxt('mu_Gadox.txt',dtype=float,delimiter='\t')
        us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
        return us_mu(1000*keV)

    def mu_molecular(keV=13.613):
        from numpy import loadtxt,log,array
        from scipy.interpolate import UnivariateSpline
        names = array(open('Molecular_Transmission.txt').readlines()[1].strip('\n#\t').split('\t'))
        T = loadtxt('Molecular_Transmission.txt',dtype=float,delimiter='\t')
        mu = []
        for i in range(1,len(T.T)):
            mu.append(-log(UnivariateSpline(T[:,0],T[:,i],s=0)(keV*1000)))
        return array(mu)
    
    def mu_Humidified_Air(BP,RH,TC,keV=11.648):
        """Returns the inverse 1/e penetration depth [mm-1] as a function of
        barometric pressure (BP) in torr, relative humidity (RH) in %, and
        temperature in Celsius. The water vapor pressure VP in torr at TC
        was derived from equations and tabulated data found in:
            https://en.wikipedia.org/wiki/Antoine_equation
        where mu_Water has been converted from units of 1 gm/ml to torr to 
        generate mu_wv. The gas (air) transmission was calculated at 295 K using:
            http://henke.lbl.gov/optical_constants/gastrn2.html
        and used to calculate mu_a in units of mm-1 torr-1. The temperature 
        dependence of the returned absorption coefficient assumes
        ideal gas behavior.
        """
        # Calculate mu for air at 22 Celsius (mm-1)
        mu_a = mu_Air(keV) 
        # Calculate mu for water vapor at STP (mm-1)
        MW = 18
        V_STP = 22.4e3
        density = MW/V_STP
        mu_wv = mu_Water(keV)*density 
        # Calculate water vapor pressure given RH
        A,B,C = 8.07131,1730.63,233.426
        VP = (RH/100)*10**(A - B/(C+TC))
        mu = ((BP-VP)/760*295/(273+TC))*mu_a + (VP/760)*mu_wv
        return mu

    def mu_FusedSilica(keV=11.648):
        """Returns inverse 1/e penetration depth [mm-1] of fused silica given the
        x-ray energy in keV. The transmission through a 0.2-mm thick slab of SiO2
        was calculated every 100 eV over an energy range spanning 8-16 keV using:
            http://henke.lbl.gov/optical_constants/filter2.html
        This result was then converted to mu and saved as a tab-delimited text
        file. The returned result is calculated using a univariate spline and
        should be valid over the range 8-16 keV."""
        from numpy import loadtxt
        from scipy.interpolate import UnivariateSpline
        E_mu = loadtxt('mu_FusedSilica.txt',dtype=float,delimiter='\t')
        us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
        return us_mu(1000*keV)

    def mu_Water(keV=11.648):
        """Returns inverse 1/e penetration depth [mm-1] of water given the
        x-ray energy in keV. The transmission through a 1-mm thick slab of water
        was calculated every 100 eV over an energy range spanning 8-16 keV using:
            http://henke.lbl.gov/optical_constants/filter2.html
        This result was then converted to mu and saved as a tab-delimited text
        file. The returned result is calculated using a univariate spline and
        should be valid over the range 8-16 keV."""
        from numpy import loadtxt
        from scipy.interpolate import UnivariateSpline
        E_mu = loadtxt('mu_Water.txt',dtype=float,delimiter='\t')
        us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
        return us_mu(1000*keV)

    # Load parameters needed from datafile
    defective_ROR = hdf5(datafile,'defective_ROR')
    analysis_dir = hdf5(datafile,'analysis_dir')
    shape = hdf5(datafile,'shape')
    dq = hdf5(datafile,'dq')
    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    BS = hdf5(datafile,'BSm')[0]
    distance = hdf5(datafile,'distance')
    pixelsize = hdf5(datafile,'pixelsize')
    keV = hdf5(datafile,'keV')
    polarization = hdf5(datafile,'polarization')
    FS_tilt = hdf5(datafile,'FS_tilt')
    capillary_thickness = hdf5(datafile,'capillary_thickness')
    capillary_OD = hdf5(datafile,'capillary_OD')

    # Load arrays needed from Rayonix_stats.hdf5
    rayonix_stats = ('/').join(analysis_dir.split('/')[:-3])+'/Rayonix/Rayonix_stats.hdf5'
    CCD_id = hdf5(rayonix_stats,'CCD_id')
    CCD_masks = hdf5(rayonix_stats,'CCD_masks')
    ROR_masks = hdf5(rayonix_stats,'ROR_masks')
    G = hdf5(rayonix_stats,'G')

    # Construct edge pixel mask, EPm, from CCD_masks and write to bit 0 (2^0=1) of RPm
    EPm = CCD_masks.sum(axis=0) > 1
    RPm = EPm.astype('uint8')
    
    # Map saxs beamstop mask onto entire detector; write to bit1 (2^1=2) of RPm (Rayonix Pixel mask)
    offset = int((len(BS)-1)/2)
    RPm[Y0-offset:Y0+offset+1,X0-offset:X0+offset+1] |= 2*BS.astype('uint8')

    # Generate perimeter pixel mask, many of which are defective
    PPm = ones_like(RPm,bool)
    PPm[1:-1,1:-1] = False

    # Identify pixels affected by defective read-out registers
    mask = zeros_like(CCD_id,bool)
    structure = ones((3,3))
    for i,j in defective_ROR:
        mask_j = ROR_masks[j]
        mask_ij = (CCD_id == i) & mask_j
        Ni = mask_ij.sum()
        while True:
            mask_ij = binary_dilation(mask_ij,structure = structure)*mask_j
            N = mask_ij.sum()
            if N == Ni:
                break
            else:
                Ni = N
        mask |= mask_ij

    # Combine PPm and mask to generate defective pixel mask, DPm, and write into bit 2 (2^2 = 4) of RPm 
    DPm = PPm | mask
    RPm |= (4*DPm).astype('uint8')

    # Generate W, weights from G, pixel gain
    W = zeros_like(G)
    W[G>0] = 1/G[G>0]

    # Calculate theta, psi, and Q from geometric parameters
    theta,psi,Q = Rayonix_theta_psi_Q(shape,X0,Y0,distance,pixelsize,keV)

    T0_filter = transmittance_filters(0,keV=keV)*transmittance_air(0,BP=760,RH=35,TC=22,t=96.3,keV=keV)
    #T0_phosphor = transmittance_phosphor(0,keV=keV,t=0.04,fill_factor=0.58)
    #T0_phosphor = transmittance_phosphor(0,keV=keV,t=0.04,fill_factor=0.39)
    phosphor_FF = hdf5(datafile,'phosphor_FF')
    T0_phosphor = transmittance_phosphor(0,keV=keV,t=0.04,fill_factor=phosphor_FF)

    # Calculate and combine relevant scale factors
    pol = 0.5*(1+square(cos(2*theta))-polarization*cos(2*psi)*square(sin(2*theta)))
    geo = cos(2*theta)**3
    filter = exp(log(T0_filter)/cos(2*theta))/T0_filter
    phosphor = (1 - exp(log(T0_phosphor)/cos(2*theta)))/(1-T0_phosphor)
    PGFP = pol*geo*filter*phosphor

    # Calculate PST
    PST = plate_scatterer_transmittance_tilt(theta,psi,FS_tilt*pi/180,0.12,mu_FusedSilica(keV))

    # Calculate CXT and CHT (transmittance of scattering from capillary filled with xenon and helium)
    CXT = capillary_transmittance(theta,psi,capillary_OD/2,capillary_thickness/2,mu_Water(keV),mu_FusedSilica(keV))
    CHT = capillary_transmittance(theta,psi,capillary_OD/2,capillary_thickness/2,0,mu_FusedSilica(keV))

    # Calculate CST (transmittance of scattering from sample in capillary)
    print('Calculating CST... please wait (over 1 minute)')
    CST = sample_transmittance(theta,psi,capillary_OD/2,capillary_thickness/2,mu_Water(keV),mu_FusedSilica(keV))

    # Generate sort_indices and reverse_indices based on 'theta'.
    sort_indices = argsort(theta.flatten())
    reverse_indices = argsort(sort_indices)

    # Determine q, qbin1, qbin2, and q_flat
    q,qbin1,qbin2,q_flat = qbin(Q,RPm,dq)
    
    # Write results in datafile
    hdf5(datafile,'CCD_id',CCD_id)
    hdf5(datafile,'RPm',RPm)
    hdf5(datafile,'W',W.astype('float32'))
    hdf5(datafile,'PGFP',PGFP.astype('float32'))
    hdf5(datafile,'PST',PST.astype('float32'))
    hdf5(datafile,'CXT',CXT.astype('float32'))
    hdf5(datafile,'CHT',CHT.astype('float32'))
    hdf5(datafile,'CST',CST.astype('float32'))
    hdf5(datafile,'Q',Q.astype('float32'))
    hdf5(datafile,'psi',psi)
    hdf5(datafile,'q_flat',q_flat.astype('float32'))
    hdf5(datafile,'q',q)
    hdf5(datafile,'qbin1',qbin1)
    hdf5(datafile,'qbin2',qbin2)
    hdf5(datafile,'sort_indices',sort_indices.astype('uint32'))
    hdf5(datafile,'reverse_indices',reverse_indices.astype('uint32'))
    
    if chart:
        from charting_functions import chart_image,chart_histogram
        from os import path
        beamtime = path.basename(datafile)
        chart_image(RPm,'RPm\n{}'.format(beamtime))
        chart_image(W,'W\n{}'.format(beamtime))
        chart_image(PGFP,'PGFP\n{}'.format(beamtime))
        chart_image(PST,'PST\n{}'.format(beamtime))
        chart_image(CXT,'CXT\n{}'.format(beamtime))
        chart_image(CHT,'CHT\n{}'.format(beamtime))
        chart_image(CST,'CST\n{}'.format(beamtime))
        chart_histogram(W,'W\n{}'.format(beamtime),binsize=0.01,xmax=1.5)

def mu_Air(keV=11.648):
    """Returns inverse 1/e penetration depth [mm-1] of air at 760 torr given the
    x-ray energy in keV. The transmission through a 300-mm thick slab of air
    was calculated every 100 eV over an energy range spanning 5-17 keV using:
        http://henke.lbl.gov/optical_constants/filter2.html
    This result was then converted to mu and saved as a tab-delimited text
    file. The returned result is calculated using a univariate spline and
    should be valid over the range 5-17 keV."""
    from numpy import loadtxt
    from scipy.interpolate import UnivariateSpline
    E_mu = loadtxt('mu_Air.txt',dtype=float,delimiter='\t')
    us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
    return us_mu(1000*keV)

def plate_scatterer_transmittance_tilt(theta,psi,tilt,t,mu=2.30835223):
    """Returns plate scatterer transmittance as a function of theta, psi, thickness (t), 
    vertial tilt (in radians), and inverse 1/e penetration depth (mu). This calculation 
    integrates analytically all x-ray photon trajectories defined by x, the 
    penetration depth into the plate before being diffracted into angle 2*theta. 
    Dimensions are in mm."""
    from numpy import exp,cos,sin,where
    PST = cos(tilt)*(exp(-mu*t/cos(tilt))-exp(-mu*t/(cos(tilt)*cos(2*theta+tilt*sin(psi)))))/(mu*t*(1/cos(2*theta+tilt*sin(psi))-1))
    PST[where(theta==0)] = exp(-mu*t/cos(tilt))
    return PST

def plate_scatterer_transmittance(theta,t,mu=2.30835223):
    """Returns plate scatterer transmittance as a function of theta, 
    thickness (t), and inverse 1/e penetration depth (mu). This 
    calculation integrates analytically all x-ray photon trajectories 
    defined by x, the penetration depth into the plate before being 
    diffracted into angle 2*theta. Dimensions are in mm."""
    from numpy import exp,cos,where
    PST = (exp(-mu*t)-exp(-mu*t/cos(2*theta)))/(mu*t*(1/cos(2*theta)-1))
    PST[where(theta==0)] = exp(-mu*t)
    return PST

def capillary_transmittance(theta,psi,R,t,mu_s=0.2913,mu_c=2.30835223):
    """Returns capillary transmittance assuming its wall thickness, t, is thin
    compared to its radius, R. The parameters mu_s and mu_c correspond to the
    inverse 1/e penetration depth of the capillary contents (zero if empty) and
    fused silica, respectively. Dimensions are in mm."""
    from numpy import exp,sin,cos,sqrt
    a = cos(2*theta)**2+sin(2*theta)**2*sin(psi)**2
    b = -2*R*cos(2*theta)
    ct = -2*R*t-t**2
    root_t = sqrt(b**2-4*a*ct)
    l = -b/a
    dl = (root_t+b)/(2*a)
    PST = plate_scatterer_transmittance(theta,t,mu_c)
    CT = 0.5*(PST*exp(-mu_c*dl-mu_s*l)+PST*exp(-mu_c*t-mu_s*2*R))
    return CT

def sample_transmittance(theta,psi,R,t,mu_s=0.2913,mu_c=2.30835223):
    """Returns sample transmittance as a function of theta and psi given the
    capillary radius (R [mm]) and wall thickness (t [mm]). The x-ray beam is scattered
    and attenuated by sample contained in the capillary, and is further
    attenuated by passing through both entrance and exit walls of the
    capillary. The default inverse 1/e penetration depths, mu_s and mu_c,
    correspond to values for water and fused silica at 12 keV. This
    function averages photon trajectories defined by x, the penetration depth
    into the capillary prior to diffraction at angle 2*theta. Dimensions are in mm."""
    from numpy import exp,sin,cos,sqrt,linspace
    def STx(x):
        a = cos(2*theta)**2+sin(2*theta)**2*sin(psi)**2
        b = -2*x*cos(2*theta)
        c = x**2-R**2
        ct = x**2-(R+t)**2
        root = sqrt(b**2-4*a*c)
        root_t = sqrt(b**2-4*a*ct)
        l = (-b+root)/(2*a)
        dl = (root_t-root)/(2*a)
        return exp(-mu_c*t)*exp(-mu_s*(R-x+l))*exp(-mu_c*dl)
    ST = 0
    N = 50
    xvals = linspace(-R,R,N)
    for x in xvals:
        ST += STx(x)/N
    return ST

def beamtime_nC(datafile):
    """Assigns nC to image files according to timestamps; corresponds to
    integrated I0_PIN signal for each image in units of nano coulombs. 
    Since not all trace files are complete, extrapolates result to the 
    hard-coded, expected number of pulses per measure, NPM. Writes nC, 
    NnC, and nC_td to dataaset. Assumes trace
    files have been processed by beamtime_trc.py"""
    from numpy import diff,where,ones,zeros,nan,roll,rint,argsort
    NPM = 40 # Number of x-ray pulses per measure
    datasets = hdf5(datafile,'datasets')
    for dataset in datasets:
        i,pathname = index_pathname(datafile,dataset)
        try:
            # Load trc information
            trc_timestamps = hdf5(pathname,'trc_timestamps')
            trc_N = hdf5(pathname,'trc_N')
            trc_nC = hdf5(pathname,'trc_nC')
            # Find duplicates and singles, if any, and exclude
            duplicate = trc_timestamps == roll(trc_timestamps,1)
            # Determine Nt, expected number of trace files per image file
            image_period = hdf5(pathname,'period')
            PP_periods = hdf5(datafile,'PP_periods')
            PP_divider = hdf5(datafile,'PP_divider')
            t_error = abs(image_period/PP_periods - 1)
            Nt = PP_divider[where(t_error == t_error.min())][0]
            # Load image timestamps and period
            timestamps = hdf5(pathname,'timestamps')
            # Determine nC extrapolated to NPM (targeted number of x-ray pulses per measure)
            nC = nan*ones(len(timestamps))
            nC_N = zeros(len(timestamps))
            nC_td = nan*ones(len(timestamps))
            for j,timestamp in enumerate(timestamps):
                # Find trc_timestamps that match image timestamps
                select = (trc_timestamps > timestamp - image_period) & (trc_timestamps < timestamp) & ~duplicate
                tdiff = timestamp - trc_timestamps[select]
                if select.sum() > 1:    
                    sort_order = argsort(tdiff)
                    NnCj = trc_N[select][sort_order[:Nt]].sum()
                    nC[j] = Nt*NPM*trc_nC[select][sort_order[:Nt]].sum()/NnCj
                    nC_td[j] = tdiff[sort_order[0]]
                    nC_N[j] = NnCj
                elif select.sum() == 1:
                    nC[j] = Nt*NPM*trc_nC[select]/trc_N[select]
                    nC_td[j] = tdiff
                    nC_N[j] = trc_N[select]
            hdf5(pathname,'nC',nC)
            hdf5(pathname,'nC_N',nC_N)
            hdf5(pathname,'nC_td',nC_td)
        except:
            pass

        image_number = hdf5(pathname,'image_number')
        din = diff(image_number)
        missing = image_number[where(din > 1)] + 1
        if len(missing > 0):
            print('{}: {} missing image numbers {} ({} image files) '.format(i,dataset,missing,len(image_number)))

def beamtime_trc(datafile):
    """Process trace files."""
    from time import time
    datasets = hdf5(datafile,'datasets')
    print('Processing trc files')
    t0 = time()
    for dataset in datasets:
        i,pathname = index_pathname(datafile,dataset)
        dataset_trc(datafile,dataset)
    print('\nProcessed {} datasets in {:0.3f} seconds'.format(len(datasets),time()-t0))

def dataset_trc(datafile,dataset):
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
    i,pathname = index_pathname(datafile,dataset)
    xray_filename = hdf5(pathname,'filenames')[0]

    # prefix = '/Volumes/mirrored_femto-data2'
    # xray_filename = prefix + xray_filename.split('/net/femto-data2.niddk.nih.gov')[-1]
    
    dirname = path.dirname(xray_filename).replace('xray_images','xray_traces')
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
            print('Out-of-order trace files found in {}: {}'.format(i,dataset))

        # Write results
        hdf5(pathname,'trc_filenames',trc_filenames)
        hdf5(pathname,'trc_timestamps',trc_timestamps)
        hdf5(pathname,'trc_nC',trc_nC)
        hdf5(pathname,'trc_N',trc_N)
        print('{:0.3f} seconds to process {} trace files in {}: {}'.format(time()-t0,len(trc_N),i,dataset))
    except:
        print('No trace files found in {}: {}'.format(i,dataset))

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
    
def percus_yevick(D,VF,q,chart=None):
    """Calculates packing structure factor for hard sphere of diameter D and
    volume fraction VF for supplied values of q."""
    from numpy import sin,cos
    from charting_functions import chart_xy_symbol
    l1 = (1+2*VF)**2/(1-VF)**4
    l2 = -(1+VF/2)**2/(1-VF)**4
    NCq1 = l1*(sin(q*D) - q*D*cos(q*D))/(q*D)**3
    NCq2 =  - 6*VF*l2*((q*D)**2*cos(q*D)-2*q*D*sin(q*D) - 2*cos(q*D)+2)/(q*D)**4
    NCq3 = -0.5*VF*l1*((q*D)**4*cos(q*D) - 4*(q*D)**3*sin(q*D)-12*(q*D)**2*cos(q*D)+24*q*D*sin(q*D)+24*cos(q*D)-24)/(q*D)**6
    Sq = 1/(1+24*VF*(NCq1+NCq2+NCq3))
    if chart is not None:
        chart_xy_symbol(q,Sq,'{}\nDiameter = {}; Volume Fraction = {:0.3f}'.format(chart,D,VF),x_label='q')
    return Sq

def atomic_scattering(qs,N,Z,chart=None):
    """Given qs, N, and Z, returns square of the atomic form factor and Compton scattering.
    N and Z can be lists or arrays."""
    from charting_functions import chart_xy_symbol
    from analysis_functions import FF_C
    from numpy import array
    FF2s = 0
    Cs = 0
    if isinstance(N,int):
        N = [N]
        Z = [Z]
    for j,Zi in enumerate(Z):
        FF,C = FF_C(qs,Zi)
        FF2s += N[j]*FF**2
        Cs += N[j]*C
    if chart is not None:
        chart_xy_symbol(qs,array([FF2s+Cs, FF2s,Cs]),'{}\n[FF**2 + C, FF**2, C]'.format(chart),x_label='q')
    return FF2s, Cs

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

def Ss_to_Ssq(Ss,sigSs,qs,qb):
    """Returns Ssq and sigSsq, which represent merged/averaged
    Ss evaluated at qb."""
    from numpy import array
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

def saxs_sphere(q,R):
    """Returns normalized scattering amplitude for a sphere
    of radius R over the range q (scattering amplitude scales as R**3).
        Rg = sqrt(3/5)*R.
    """
    from numpy import sin,cos
    return (3*(sin(q*R)-q*R*cos(q*R))/(q*R)**3)**2

def linefit_weighted(x,y,s):
    """Performs weighted linear least-squares fit of y vs. x 
    with weights calculated from standard deviation s. Returns
    a (intercept), b (slope), and stdev (estimated 
    standard deviation of the mean for xi evaluated within 
    the range of x)."""
    from numpy import dot,outer,sqrt
    w = 1/s**2
    w[s==0] = 0
    sw = w.sum(axis=0)
    swx = dot(x,w)
    swx2 = dot(x**2,w)
    swy = (w*y).sum(axis=0)
    swxy = dot(x,w*y)
    delta = swx2*sw - swx**2
    a =  (1/delta)*(swx2*swy - swx*swxy)
    b =  (1/delta)*(sw*swxy - swx*swy)
    yfit = a + outer(x,b)
    if y.ndim == 2:
        stdev = sqrt((w*(yfit - y)**2).sum(axis=0)/w.sum(axis=0)/(len(x) - 1))
    else:
        stdev = sqrt((w*(yfit - y)**2).sum()/w.sum()/(len(x) - 1))
    return a,b,stdev

def Guinier_fit(q,SB,sigSB,q_min=0,q_max=0.06,chart=False):
    """Performs Guinier analysis of SB and determines I0 and Rg. 
    The upper limit of the fitting range is set at q <= 1/(1.3*Rg),
    which corresponds to the Guinier region. First, performs weighted
    fit of SB over a range up to q=q_max to estimate Rg, then 
    modifies the fitting range according to Rg and refits, iterating
    up to 4 times."""
    from numpy import log,sqrt,exp,stack,ones,zeros
    from numpy.linalg import lstsq
    I0 = zeros(len(SB))
    Rg = zeros(len(SB))
    for i,SBi in enumerate(SB):
        qmax=q_max
        for j in range(4):
            q_Guinier = (q >= q_min) & (q <= qmax)
            q2 = q[q_Guinier]**2
            lSB = log(SBi[q_Guinier])
            siglSB = (sigSB[i]/SB[i])[q_Guinier]
            # a,b,stdev = linefit_weighted(q2,lSB,siglSB)
            w = 1/siglSB
            w[siglSB==0] = 0
            M = stack((ones(len(q2)),q2))
            a,b = lstsq((M*w).T,(lSB*w),rcond=None)[0]
            qmax = 0.5*(1.3/sqrt(-3*b) + qmax)
            if qmax <= q_min+0.003: break
        I0[i] = exp(a)
        Rg[i] = sqrt(-3*b)
        if chart:
            from charting_functions import chart_xy_fit
            chart_xy_fit(q2,lSB,a+b*q2,'log(S)\nI0 = {:0.2f}; Rg = {:0.2f}'.format(exp(a),sqrt(-3*b)),x_label='q^2',ms=3)
    return I0,Rg
     
def Guinier_fit_unweighted(q,SB,q_min=0,q_max=0.06,chart=False):
    """Performs Guinier analysis of SB and determines I0 and Rg. 
    The upper limit of the fitting range is set at q <= 1/(1.3*Rg),
    which corresponds to the Guinier region. First, fits SB over a 
    range up to q=q_max to estimate Rg, then modifies the fitting 
    range according to Rg and refits."""
    from numpy import log,sqrt,exp,stack,ones,zeros
    from numpy.linalg import lstsq
    I0 = zeros(len(SB))
    Rg = zeros(len(SB))
    for i,SBi in enumerate(SB):
        qmax=q_max
        for j in range(4):
            q_Guinier = (q >= q_min) & (q <= qmax)
            q2 = q[q_Guinier]**2
            lSB = log(SBi[q_Guinier])
            M = stack((ones(len(q2)),q2))
            a,b = lstsq(M.T,lSB,rcond=None)[0]
            qmax = 0.5*(1.3/sqrt(-3*b) + qmax)
            if qmax <= q_min+0.003: break
        I0[i] = exp(a)
        Rg[i] = sqrt(-3*b)
        if chart:
            from charting_functions import chart_xy_fit
            chart_xy_fit(q2,lSB,a+b*q2,'log(S)\nI0 = {:0.2f}; Rg = {:0.2f}'.format(exp(a),sqrt(-3*b)),x_label='q^2',ms=3)
    return I0,Rg
    
def datafile_process_saxs(datafile,chart=False):
    """Process saxs datasets: 
            compute background-subtracted, UCp-normalized saxs scattering;
            find zingers;
            determine spurious;
            compute Ss and sigSs;
            write results to pathname."""
    from numpy import sort,array,sqrt,where,around,ones,zeros_like,outer,multiply,zeros,median,roll,clip,unique
    from scipy.ndimage import binary_dilation,binary_erosion
    datasets = hdf5(datafile,'datasets')
    GC_VPC = hdf5(datafile,'GC_VPC')
    GC_UCp = hdf5(datafile,'GC_UCp')
    BSm = hdf5(datafile,'BSm')
    bkg = hdf5(datafile,'bkg')
    dv = hdf5(datafile,'dv')
    Ps = hdf5(datafile,'Ps')
    BS = BSm[0]
    pl,qs = pl_qs(datafile)
    hdf5(datafile,'qs',qs)
    VPC = GC_VPC/GC_UCp**2
    # Generate apodize and annular functions for spurious
    pmin = 14
    pmax = 21
    apodize = clip(1 - (Ps - pmin)/(pmax-pmin),0,1)
    annular = (Ps > pmin) & (Ps < pmax)
    # Construct vpc (variance per count), dvp (dark variance) and ones lists for each value of pl (qs)
    vpc = [] 
    dvp = [] 
    Ip = []
    for p in pl:
        vpc.append(VPC[p[0],p[1]])
        dvp.append(dv[p[0],p[1]])
        Ip.append(ones(len(p[0])))

    for dataset in datasets:
        print('Processing {}'.format(dataset))
        i,pathname = index_pathname(datafile,dataset)
        omit = hdf5(pathname,'omit')
        M0 = hdf5(pathname,'M0')
        saxs = hdf5(pathname,'saxs').astype(float)
        ICs = saxs.sum(axis=(1,2))
        # Determine saxs_bn, background-subtracted and UCp normalized saxs
        saxs_bn = (saxs-bkg)/GC_UCp
        N_omit0 = (omit != 0).sum()
        while True: 
            # Determine ofm, outlier-free mean; eliminate zingers and use omit to flag outlier images
            ofm = sort(saxs_bn[omit==0],axis=0)[2:-2].mean(axis=0)
            # Determine spurious scattering
            spurious = zeros_like(bkg,float)
            for j in range(len(pl)):
                spurious[pl[j]] = ofm[pl[j]] - ofm[pl[j]].min()
            spurious -= (annular*spurious).mean()
            spurious *= apodize
            # Generate corrected images by compensating for spurious scattering
            M0n = M0/M0[omit==0].mean()
            saxs_bnc = saxs_bn - multiply.outer(M0n,spurious)
            
            # Determine relative difference from the median in sigma units for each pixel in every image
            Diff = zeros_like(saxs)
            sigma = zeros_like(saxs)
            for j,p in enumerate(pl):
                cp = saxs_bnc[:,p[0],p[1]] # counts for degenerate set of pixels
                vp = vpc[j]*abs(cp) + dvp[j] # estimated variance for degenerate set of pixels
                cp_median = median(cp,axis=1)
                Dj = cp - outer(cp_median,Ip[j])
                Diff[:,p[0],p[1]] = Dj
                sigma[:,p[0],p[1]] = Dj/sqrt(vp) 

            # Given sigma, define image_outliers, and assign to bit 2 of omit
            image_std = sigma.std(axis=(1,2))
            # Identify stretches of outlier images, which can arise when sample freezes
            image_outliers = binary_dilation(binary_erosion(image_std > 1.1*median(image_std),iterations=2),iterations=2)
            # If Tramp, expand stretch and further extend along the UP ramp to flush volume receiving excess x-ray dose
            if 'Tramp' in dataset:
                image_outliers = binary_dilation(image_outliers,iterations=3)
                kstart = where((image_outliers != roll(image_outliers,1)) & ~image_outliers)[0]
                extend = 10
                for k in kstart:
                    image_outliers[k:k+extend] = True
            omit = omit | 2*image_outliers
            N_omit = (omit != 0).sum()
            # If no new outlier images were found, break, otherwise repeat
            if  N_omit == N_omit0:
                break
            else: N_omit0 = N_omit

        # Given sigma, construct Zt, zinger tuple consisting of [image,pixel] indices and excess counts
        Zb = (sigma.T*(omit==0)).T > abs(sigma[omit==0].min()) + 3
        Z_indices = where(Zb)
        Zt = Z_indices + (around(Diff).astype('uint16')[Z_indices],)

        # Compute weighted mean Ss and standard deviation of the mean, sigSs, omitting zingers
        if (not 'Dark' in dataset) & (not 'PSF' in dataset):
            Ss = zeros((len(saxs),len(pl)))
            sigSs = zeros((len(saxs),len(pl)))
            for j,p in enumerate(pl):
                zp = Zb[:,p[0],p[1]]
                cp = saxs_bnc[:,p[0],p[1]]
                vp = vpc[j]*abs(cp) + dvp[j] 
                Ss[:,j] = (~zp*cp/vp).sum(axis=1)/(~zp*(1/vp)).sum(axis=1)
                sigSs[:,j] = sqrt(1/(~zp/vp).sum(axis=1))

            # Write results in pathname
            hdf5(pathname,'Ss',Ss.astype('float32'))
            hdf5(pathname,'sigSs',sigSs.astype('float32'))
            hdf5(pathname,'spurious',spurious)
        
        # Write results in pathname
        hdf5(pathname,'omit',omit)  
        hdf5(pathname,'Zt',Zt)
        
        if chart:
            from charting_functions import chart_vector,chart_image,chart_xy_symbol,chart_histogram
            chart_vector((sigma*~Zb).std(axis=(1,2)),'(sigma*~Zb).std(axis=(1,2))\n{}: {}'.format(i,dataset))
            chart_vector(ICs,'ICs\n{}: {}'.format(i,dataset))
            if image_outliers.sum() > 0:
                chart_vector(array([ICs,image_outliers*ICs]),'array([ICs,image_outliers*ICs])\n{}: {}'.format(i,dataset))
            chart_vector((saxs*BSm[3]).sum(axis=(1,2))/BSm[3].sum(),'Mbs\n{}: {}'.format(i,dataset))
            chart_histogram(sigma[omit==0],'sigma[omit==0]\n{}: {}'.format(i,dataset),binsize=0.1,xmin=-10,xmax=20)  
            chart_image(sigma[omit==0].max(axis=0),'sigma[omit==0].max(axis=0)\n{}: {}'.format(i,dataset))
            zingers = zeros_like(saxs)
            zingers[Zt[0],Zt[1],Zt[2]] = Zt[3]
            chart_image(zingers.sum(axis=0),'zingers\n{} zinger pixels found in {} images out of {}\n{}: {}'.format(len(Zt[0]),len(unique(Zt[0])),(omit==0).sum(),i,dataset))
            chart_image(spurious,'spurious\n{}: {}'.format(i,dataset))
            chart_xy_symbol(Ps.flatten()/~BS.flatten(),ofm.flatten(),'ofm vs. pixel distance\n{}: {}'.format(i,dataset),x_label='pixel distance')
            chart_xy_symbol(qs,Ss.mean(axis=0),'Ss.mean(axis=0)\n{}: {}'.format(i,dataset),x_label='q')
            chart_xy_symbol(qs,sigSs.mean(axis=0),'sigSs.mean(axis=0)\n{}: {}'.format(i,dataset),x_label='q')

def params_to_datafile(datafile,params):
    """Write parameters to datafile."""
    from numpy import array
    from os import path
    # Ensure analysis_dir exists
    analysis_dir = params['analysis_dir']
    # if not path.isdir(analysis_dir):
    #     makedirs(analysis_dir)
    # Include PP-mode information in datafile
    P0_clk = 271554.454
    HSC_harmonic = 275
    HSC_period = HSC_harmonic/P0_clk
    PP_modes = array(['flythru4','flythru4-i2','flythru4-i4','flythru4-i8','flythru24','flythru48'])
    PP_periods = array([276,2*276,4*276,8*276,1176,2256])*HSC_period
    PP_divider = array([1,2,4,8,1,1])
    hdf5(datafile,'PP_modes',PP_modes)
    hdf5(datafile,'PP_periods',PP_periods)
    hdf5(datafile,'PP_divider',PP_divider)

    # Write parameters in datafile
    for key,value in params.items():
        hdf5(datafile,key,value)

def datafile_datasets(datafile):
    """Find datasets and corresponding xray images in data_dir
    and write to datafile."""
    def pathname_from_image(image,ext='.hdf5'):
        """Parses image and returns pathname."""
        dataset_path = image.split('WAXS/')[-1].split('xray_images')[0]
        dataset = dataset_path.split('/')[-2]
        return dataset_path + dataset + ext 
    data_dir = hdf5(datafile,'data_dir')
    analysis_dir = hdf5(datafile,'analysis_dir')
    exclude = hdf5(datafile,'exclude')
    datasets,xray_images = datasets_filenames(data_dir,'xray_images',exclude)
    # Generate dataset_pathnames from xray_images
    dataset_pathnames = [pathname_from_image(images[0]) for images in xray_images]
    # Write filenames to pathnames
    for i,pathname in enumerate(dataset_pathnames):
        hdf5(analysis_dir+pathname,'filenames',xray_images[i])
    # Write datasets and dataset_pathnames to datafile
    hdf5(datafile,'datasets',datasets)
    hdf5(datafile,'dataset_pathnames',dataset_pathnames)

def index_pathname(datafile,dataset):
    """Returns index and pathname for dataset given datafile."""
    from os import path
    datasets = hdf5(datafile,'datasets')
    i = datasets.index(dataset)
    analysis_dir = path.dirname(datafile) + '/'
    return i,analysis_dir + hdf5(datafile,'dataset_pathnames')[i]

def dataset_chart(datafile,dataset):
    """Generates charts of dataset."""
    from numpy import zeros,log10,unique,minimum
    from charting_functions import chart_image,chart_histogram,chart_vector,chart_xy_scatter,chart_xy_rainbow
    BS = hdf5(datafile,'BSm')[0]
    i,pathname = index_pathname(datafile,dataset)
    spurious = hdf5(pathname,'spurious')
    ofm = hdf5(pathname,'ofm')
    Zt = hdf5(pathname,'Zt')
    NOP_vector = hdf5(pathname,'NOP_vector')
    NOP_image = hdf5(pathname,'NOP_image')
    OPt = hdf5(pathname,'OPt')
    ICs = hdf5(pathname,'ICs')
    Mbs = hdf5(pathname,'Mbs')
    omit = hdf5(pathname,'omit')
    zinger_image = zeros(ofm.shape)
    Y = Zt[1]
    X = Zt[2]
    C = Zt[3] 
    zinger_image[Y,X] += C 
    OP_image = zeros(ofm.shape)
    Y = OPt[1]
    X = OPt[2]
    C = OPt[3] 
    OP_image[Y,X] += C 

    chart_image(log10(zinger_image+1),'{} Zinger pixels found in {} out of {} images\n{}: {}'.format(len(Zt[0]),len(unique(Zt[0])),len(NOP_vector),i,dataset))
    chart_histogram(Zt[3],'zinger amplitude\n{}: {}'.format(i,dataset),binsize=10,xmax=minimum(Zt[3].max(),1000))
    if NOP_vector.sum() > 0:
        chart_image(log10(OP_image+1),'{} Outlier Pixels found in {} out of {} images\n{}: {}'.format(len(OPt[0]),len(unique(OPt[0])),len(NOP_vector),i,dataset))
        chart_vector(NOP_vector,'NOP_vector\n{}: {}'.format(i,dataset),x_label='image number')
        chart_image(NOP_image,'NOP_image\n{}: {}'.format(i,dataset))
    chart_image(ofm/~BS,'ofm\n{}: {}'.format(i,dataset))
    chart_image(spurious,'spurious\n{}: {}'.format(i,dataset))
    
    if 'Tramp' in dataset:
        Up = hdf5(pathname,'Up')
        Repeat = hdf5(pathname,'Repeat')
        RTD_temp = hdf5(pathname,'RTD_temp')
        chart_xy_scatter(RTD_temp,ICs,2*(Repeat-1)+~Up,'ICs\n{}: {}'.format(i,dataset),x_label='RTD_temp')
        chart_xy_scatter(RTD_temp,Mbs,2*(Repeat-1)+~Up,'Mbs\n{}: {}'.format(i,dataset),x_label='RTD_temp')
        chart_xy_scatter(RTD_temp/(omit==0),ICs,2*(Repeat-1)+~Up,'ICs/(omit==0)\n{}: {}'.format(i,dataset),x_label='RTD_temp')

        SsTq = hdf5(pathname,'SsTq')
        Tb = hdf5(pathname,'Tb')
        qb = hdf5(pathname,'qb')
        chart_xy_rainbow(qb,SsTq,'SsTq\n{}: {}'.format(i,dataset),x_label='q')

    # chart_xy_symbol(rn,Temperature/(OPb.sum(axis=(1,2)) == 0),'Temperature (no OP)\n{}: {}'.format(i,dataset),x_label='image number')
    # chart_xy_symbol(rn,Temperature/~(OPb.sum(axis=(1,2)) == 0),'Temperature (OP)\n{}: {}'.format(i,dataset),x_label='image number')
    # chart_xy_symbol(rn,Delay/~(OPb.sum(axis=(1,2)) == 0),'Delay (OP)\n{}: {}'.format(i,dataset),x_label='image number',logy=True)
    # chart_histogram(sigma[OPb],'sigma[OPb]\n{}: {}'.format(i,dataset),binsize=0.1,xmax=20)
    # chart_histogram(sigma[Zb],'sigma[Zb]\n{}: {}'.format(i,dataset),binsize=0.1,xmax=20)
    # chart_image(OPb.sum(axis=0) == 1,'OPb.sum(axis=0) == 1\n{}: {}'.format(i,dataset))
    # chart_image(OPb.sum(axis=0) > 1,'OPb.sum(axis=0) >| 1\n{}: {}'.format(i,dataset))
    # chart_histogram((Diff*GC_UCp)[OPb],'(Diff*GC_UCp)[OPb]\n{}: {}'.format(i,dataset),xmax=(Diff*GC_UCp)[OPb].max())
    # chart_image(ofm/~BS,'ofm/~BS\n{}: {}'.format(i,dataset))
    # chart_xy_symbol(Ps.flatten()/~BS.flatten(),ofm.flatten(),'ofm vs. pixel distance\n{}: {}'.format(i,dataset),x_label='pixel distance')
    # chart_vector((saxs*BSm[3]).sum(axis=(1,2))/BSm[3].sum(),'(saxs*BSm[3]).sum(axis=(1,2))/BSm[3].sum()\n{}: {}'.format(i,dataset))

    # chart_image(spurious,'spurious\n{}: {}'.format(i,dataset))
    # chart_xy_symbol(Ps.flatten()/~BS.flatten(),((ofmc-spurious)/~BS).flatten(),'(ofmc - spurious) vs. pixel distance\n{}: {}'.format(i,dataset),x_label='pixel distance')



    # chart_image(zinger_image,'zinger_image\n{} zinger pixels in {} images\n{}: {}'.format(len(Zt[0]),len(saxs),i,dataset))
    # chart_histogram(sigma,'sigma\n{}: {}'.format(i,dataset),binsize=0.1,xmax=20)
    #     # U,s,VT = image_SVD(saxs)

def inspect_zingers(datafile,dataset):
    """Evaluates zingers found in datasets."""
    from numpy import zeros,log10,array,unique
    from charting_functions import chart_image,chart_histogram
    BS = hdf5(datafile,'BSm')[0]
    i = 101
    dataset = datasets[i]
    datasets = hdf5(datafile,'datasets')
    i,pathname = index_pathname(datafile,dataset)
    zingers = hdf5(pathname,'zingers')
    N_images = len(hdf5(pathname,'filenames'))
    zinger_image = zeros((101,101))
    for zj in zingers:
        Y = zj[2]
        X = zj[3]
        C = zj[4]
        zinger_image[Y,X] += C
    chart_image(log10(zinger_image+1),'{} zingers found in {} images\n{}: {}'.format(len(zingers),N_images,i,datasets[i]))
    spurious = hdf5(pathname,'spurious')
    zfm = hdf5(pathname,'zfm')
    chart_image(spurious,'spurious\n{}: {}'.format(i,datasets[i]))
    chart_image(zfm/~BS,'zfm\n{}: {}'.format(i,datasets[i]))

    saxs = hdf5(pathname,'saxs')
    zinger_indices = unique(zingers[:,1])
    for j in zinger_indices:
        chart_image(saxs[j]/~BS,'saxs\n{}: {}'.format(j,datasets[i]))

    
    

    Zingers = []  
    datasets = hdf5(datafile,'datasets')
    for dataset in datasets:
        i,pathname = index_pathname(datafile,dataset)
        try:
            zingers = hdf5(pathname,'zingers')
            Zingers.extend(zingers)
        except:
            pass 
    Zingers = array(Zingers)   
    chart_histogram(array(Zingers)[:,0],'Number of zingers',xmax=120)

    from numpy import zeros
    zinger_image = zeros((101,101))
    for zj in Zingers:
        Y = zj[2]
        X = zj[3]
        C = zj[4]
        zinger_image[Y,X] += C
    chart_image(log10(zinger_image+1))
     
def datafile_omit0(datafile,reset=False,chart=False):
    """Identify low intensity images due to chopper phase error 
    or beam dump, and encode bit0 of omit."""
    from numpy import roll,maximum,minimum,array,empty,append
    from charting_functions import chart_vector
    print('Executing datafile_omit0')
    datasets = hdf5(datafile,'datasets')
    ICs_a = empty(0,dtype='uint64')
    omit_a = empty(0,dtype='uint8')
    for dataset in datasets:
        i,pathname = index_pathname(datafile,dataset)
        omit = hdf5(pathname,'omit')
        if reset:
            omit[:] = 0
        saxs = hdf5(pathname,'saxs')
        ICs = saxs.sum(axis=(1,2))
        neighbor_max = maximum(roll(ICs,1),roll(ICs,-1))
        neighbor_min = minimum(roll(ICs,1),roll(ICs,-1))
        ICs_max = ICs.max()
        low = (ICs < 0.2*ICs_max) | ((ICs < 0.95*neighbor_max) & (neighbor_min < 0.2*ICs_max))
        omit = omit | low
        ICs_a = append(ICs_a,ICs)
        omit_a = append(omit_a,omit)
        hdf5(pathname,'omit',omit)
        hdf5(pathname,'ICs',ICs)
        if low.sum() > 0:
            print('Flagged {} out of {} images in {} due to low intensity (bit0)'.format((omit&1 == 1).sum(),len(omit),datasets[i]))
    if chart:
        chart_vector(array([ICs_a,ICs_a/(omit_a==0)]),'[ICs,ICs/(omit=0)]',x_label='image index',logy=True)

def pl_qs(datafile):
    """Generate pl, list of pixels for each unique distance from center,
    and qs, the corresponding value of q."""
    from numpy import indices,sqrt,unique,where,arctan,pi,sin,arange
    BS = hdf5(datafile,'BSm')[0]
    height,width = BS.shape
    s = int((width-1)/2)
    y_indices,x_indices = indices((width,height)) - s
    p = sqrt(y_indices**2+x_indices**2)
    pu = unique(p[~BS])
    pl = []
    for k in range(len(pu)):
        pl.append(where((p == pu[k]) & ~BS))
    # Generate qs from pu
    distance = hdf5(datafile,'distance')
    keV = hdf5(datafile,'keV')
    pixelsize = hdf5(datafile,'pixelsize')
    theta = arctan(pu*pixelsize/distance)/2
    h = 4.135667696e-15 # Planck's constant [eV-s]
    c = 299792458e10 # speed of light [Angstroms per second]
    wavelength = h*c/(keV*1000) # x-ray wavelength [Angstroms]
    qs = 4*pi*sin(theta)/wavelength
    dq = hdf5(datafile,'dq')
    qb = arange(0.02,qs.max(),dq)
    hdf5(datafile,'qb',qb)
    hdf5(datafile,'qs',qs)
    return pl,qs

def datafile_M0_M1X_M1Y_Mbs(datafile,chart=False):
    """Processes saxs images to generate M0, M1X, M1Y, and Mbs."""
    from numpy import median,multiply,ones_like,indices,array
    from time import time
    print('Executing datafile_M0_M1X_M1Y_Mbs')
    t0 = time()
    datasets = hdf5(datafile,'datasets')
    GCn = hdf5(datafile,'GCn')
    BSm = hdf5(datafile,'BSm')
    M0 = []
    M1X = []
    M1Y = []
    Mbs = []
    ones_shape = ones_like(GCn)
    h,w = GCn.shape
    s = int((h-1)/2)
    y_mask,x_mask = indices((3,3))-1
    for dataset in datasets:
        i,pathname = index_pathname(datafile,dataset)
        print('\rprocessing {}                               '.format(datasets[i]),end='\r')
        saxs = hdf5(pathname,'saxs')
        Mbss = median(saxs[:,BSm[3]],axis=1)
        saxs_c = saxs - multiply.outer(Mbss,GCn)
        bkg = multiply.outer(saxs_c[:,BSm[2]].mean(axis=1),ones_shape)
        saxs_cb = saxs_c - bkg
        M0s = saxs_cb[:,BSm[1]].sum(axis=1)
        saxs3x3 = saxs_cb[:,s-1:s+2,s-1:s+2]
        M1Ys = -(y_mask*saxs3x3).sum(axis=(1,2))/saxs3x3.sum(axis=(1,2))
        M1Xs = (x_mask*saxs3x3).sum(axis=(1,2))/saxs3x3.sum(axis=(1,2))
        # Write results in pathname
        hdf5(pathname,'M0',M0s)
        hdf5(pathname,'M1X',M1Xs)
        hdf5(pathname,'M1Y',M1Ys)
        hdf5(pathname,'Mbs',Mbss)
        # For charting purposes, concatenate M0,M1X,M1Y,Mbs
        M0.extend(M0s)
        M1X.extend(M1Xs)
        M1Y.extend(M1Ys)
        Mbs.extend(Mbss)
    print('Proccessed {} saxs images in {:0.3f} seconds'.format(len(M0),time()-t0))
    if chart:
        use = M0 > 0.1*median(M0)
        from charting_functions import chart_vector
        chart_vector(array(M0),'M0',x_label='image index')
        chart_vector(array([M1X,M1Y])/use,'M1X, M1Y',ymax=0.1,ymin=-0.1,x_label='image index')

def datafile_glassy_carbon(datafile,chart=False):
    """From glassy carbon zinger-free-mean, which is specified
    by 'dataset', generates BSm, a set of beamstop masks, 
    GCn, normalized glassy carbon scattering, and GC_UCp, a 
    uniformity correction image, all of which are written to datafile."""
    from numpy import sort,indices,sqrt,log,exp,argsort,vstack,ones,array,median,polyfit,polyval
    from scipy.ndimage import binary_erosion,binary_dilation
    from numpy.linalg import lstsq  
    print('Executing datafile_glassy_carbon')
    datasets = hdf5(datafile,'datasets')
    GC_dataset = hdf5(datafile,'GC_dataset')

    # Generate bkg from 'Dark' datasets
    dark_datasets = [dataset for dataset in datasets if 'Dark' in dataset]
    saxs_dark = []
    for dataset in dark_datasets:
        i,pathname = index_pathname(datafile,dataset)
        saxs_dark.extend(hdf5(pathname,'saxs'))
    saxs_dark_sort = sort(array(saxs_dark),axis=0)
    bkg = saxs_dark_sort[:-2].mean(axis=0)
    dv = saxs_dark_sort[:-2].var(axis=0)
    hdf5(datafile,'bkg',bkg)
    hdf5(datafile,'dv',dv)

    # Generate GC_VPC from zinger-free, background subtracted, intensity normalized saxs
    i,pathname = index_pathname(datafile,GC_dataset)
    saxs = hdf5(pathname,'saxs') - bkg
    GC_IC = saxs.sum(axis=(1,2))
    GC_ICn = GC_IC/GC_IC.mean()
    saxsn = (saxs.T/GC_ICn).T
    saxsn_sorted = sort(saxsn,axis=0)
    GC_zfm = saxsn_sorted[:-1].mean(axis=0) 
    GC_zfv = saxsn_sorted[:-1].var(axis=0)  
    GC_VPC = GC_zfv/GC_zfm

    # Generate BSm
    height,width = bkg.shape
    s = int((width-1)/2)
    y_mask,x_mask = indices((width,width))-s
    Ps = sqrt(y_mask**2 + x_mask**2)
    # Perform Guinier fit over hard-coded annular region near the beamstop
    select = (Ps > 12) & (Ps < 16)
    sort_order = argsort(Ps[select])
    ps = Ps[select][sort_order]
    lGCs = log(GC_zfm[select][sort_order])
    M = vstack((ones(select.sum()),ps**2)).T
    intercept,slope = lstsq(M,lGCs,rcond=None)[0]
    fit = exp(slope*Ps**2 + intercept)
    # Switch between Guinier extrapolation and raw data at mid-point of the Guinier fit
    extrapolated = fit*(Ps < 14) + GC_zfm*(Ps >= 14)
    # Define BS according to pixels with greater than -3dB attenuation
    BS = GC_zfm/extrapolated < 1/sqrt(2)
    # From BS, generate spot and annular masks.
    spot = binary_erosion(BS,iterations=3)
    annular_inner = binary_dilation(spot) & ~spot
    annular_outer1 = (binary_dilation(BS) & ~BS)
    annular_outer2 = binary_dilation(annular_outer1) & ~(annular_outer1 | BS)
    annular_outer3 = binary_dilation(annular_outer2) & ~(annular_outer2 | annular_outer1)
    # Combine masks into array of masks
    BSm = array([BS,spot,annular_inner,annular_outer1,annular_outer2,annular_outer3])
    # Generate GC_fit, where the low r range is Guinier extrapolated, and high r range is based on polynomial fit 
    N_poly = 9
    p = polyfit(Ps.flatten(), extrapolated.flatten(), N_poly)
    p_fit = polyval(p, Ps)
    GC_fit = extrapolated*(Ps < 14) + p_fit*(Ps >= 14)
    # Generate GC_UCp, uniformity correction from fit; assign 1 to BS
    GC_UCp = (GC_zfm/GC_fit)
    GC_UCp[BS] = 1.0
    # Generate GCn, normalized GC
    GCn = GC_zfm/median(GC_zfm[annular_outer2])
    # Add results to Rayonix.hdf5 file
    hdf5(datafile,'bkg',bkg)
    hdf5(datafile,'GC_VPC',GC_VPC)
    hdf5(datafile,'BSm',BSm)
    hdf5(datafile,'GCn',GCn)
    hdf5(datafile,'GC_UCp',GC_UCp)
    hdf5(datafile,'Ps',Ps)
    if chart:
        from charting_functions import chart_vector,chart_image,chart_image_mask,chart_xy_fit
        chart_image(bkg,'bkg')
        chart_vector(GC_ICn,'GC_ICn')
        chart_image(GC_VPC,'GC_VPC')
        chart_image(GCn,'GCn')
        chart_image(GC_UCp,'GC_UCp',vmin=0.96,vmax=1.04)
        chart_image_mask(BSm[2],spot,'annular_inner, flagged spot')
        chart_image_mask(BSm[3],BS,'annular_outer1, flagged BS')
        rf = Ps.flatten()
        sort_order = argsort(rf)
        rfs = rf[sort_order]
        GCfs = GC_zfm.flatten()[sort_order]
        GCfs_fit = GC_fit.flatten()[sort_order]
        chart_xy_fit(rfs,GCfs,GCfs_fit,'fit of GC_zfm\nGuinier fit from 12 < r < 16 extrapolated to zero\n{} order polynomial fit'.format(N_poly),x_label='r')

def Timestamps_Values_from_archive(archive,chart=False):
    """Returns sorted date_times, timestamps, and Values from archive; assumes 
    first two columns correspond to date_time and values, respectively."""
    from numpy import loadtxt,array,argsort,diff
    from datetime import datetime
    from charting_functions import chart_xy_symbol
    from time import time
    time_start = time()
    print('Processing {}, please wait...'.format(archive.split('/')[-1]))
    date_time_col = loadtxt(archive,usecols=0,dtype='S',delimiter='\t',skiprows=1)
    value_col = loadtxt(archive,usecols=1,dtype=float,delimiter='\t',skiprows=1)
    date_times = []
    timestamps = []
    values = []
    for i in range(len(date_time_col)):
        date_time = date_time_col[i].decode()
        value = value_col[i]
        try:
            ntp_time = datetime.strptime(date_time,'%Y-%m-%d %H:%M:%S.%f%z').timestamp()
            timestamps.append(ntp_time)
            date_times.append(date_time)
            values.append(value)
        except:
            pass
    date_times = array(date_times)
    timestamps = array(timestamps)
    values = array(values)
    sort_order = argsort(timestamps)
    timestamps = timestamps[sort_order]
    values = values[sort_order]
    date_times = date_times[sort_order]
    print('Processed {} archive entries in {:0.1f} seconds'.format(len(date_time_col),time()-time_start))
    if chart:
        chart_xy_symbol(timestamps[1:],diff(timestamps),'{}\n# bogus date_times: {}'.format(archive.split('/')[-1],len(value_col)-len(values)),x_label='ntp time',y_label='diff(timestamps)',logy=True)
        chart_xy_symbol(timestamps,values,'{}\nstarted: {}'.format(archive.split('/')[-1],date_times[0]),x_label='ntp_time',y_label='value')
    return date_times,timestamps,values

def datafile_RTD_temp(datafile,chart=False):
    """Extracts LightWave temperatures from temperature archive; 
    assigns RTD_temp and corresponding RTD_timestamps nearest to, but 
    less than ntp timestamps found in dataset_pathnames; the temperature 
    archive directory is assumed to be one level up from datafile. 
    If there are no valid entries near the mccd timestamp, RTD_temp 
    is assigned nan. If chart, generates charts that compare RTD_temp 
    with Temperature in datafile, which is parsed from the image filename."""
    from numpy import array,empty,nan,isnan
    from charting_functions import chart_vector,chart_xy_symbol
    print('Executing datafile_RTD_temp')
    # Load relevant information
    temperature_archive = hdf5(datafile,'temperature_archive')
    RTD_archive = ('/').join(datafile.split('/')[:-2]) + '/Archive/' + temperature_archive
    date_times,RTD_timestamps,values = Timestamps_Values_from_archive(RTD_archive,chart=chart)
    # Assign RTD_temp and RTD_timestamps and write to dataset_pathnames 
    datasets = hdf5(datafile,'datasets')
    tT = []
    T = []
    tRTD = []
    RTD = []
    for dataset in datasets:
        i,pathname = index_pathname(datafile,dataset)
        # Find value in archive whose time stamp is nearest but smaller than corresponding mccd_timestamps
        mccd_timestamps = hdf5(pathname,'timestamps')
        select = (RTD_timestamps > mccd_timestamps[0] - 10) & (RTD_timestamps < mccd_timestamps[-1] + 10)
        RTD_ti = RTD_timestamps[select]
        Vi = values[select]
        RTD_temp = empty(len(mccd_timestamps))
        RTD_temp.fill(nan)
        j = 0
        k = 0
        tj = []
        while j < len(mccd_timestamps):
            if RTD_ti[k] < mccd_timestamps[j]:
                while RTD_ti[k] < mccd_timestamps[j]:
                    k += 1
                    if k == len(RTD_ti):
                        break
                RTD_temp[j] = Vi[k-1]
            tj.append(RTD_ti[k-1])
            j += 1
            if k > 0:
                k -= 1
        hdf5(pathname,'RTD_temp',RTD_temp)
        hdf5(pathname,'RTD_timestamps',tj)
        # For charting purposes, assemble lists of timestamps and values spanning all datasets
        tT.extend(mccd_timestamps)
        T.extend(hdf5(pathname,'Temperature'))
        tRTD.extend(tj)
        RTD.extend(RTD_temp)

    if chart:
        tT = array(tT)
        T = array(T)
        tRTD = array(tRTD)
        RTD = array(RTD)
        chart_vector(array([T,RTD]),'[Temperature, RTD_temp]',x_label='image index',y_label='temperature [C]')
        chart_vector(T-RTD,'Temperature - RTD',x_label='image index',y_label='temperature [C]')
        chart_xy_symbol(tT,T-RTD,'Temperature - RTD_temp',x_label='ntp_time',y_label='temperature [C]')
        chart_xy_symbol(tT,T,'Temperature',x_label='ntp_time',y_label='temperature [C]')
        chart_vector((tT - tRTD)/~isnan(RTD),'assigned timestamp differences\n{} nan entries'.format(isnan(RTD_temp).sum()),x_label='image index')

def datafile_timestamps(datafile,chart=False):
    """Generates periodic timestamps from acquire_timestamp and header_timestamp.
    Assumes header_timestamp is slaved to ntp time, but can be delayed relative 
    to the acquire_timestamp if the computer is busy."""
    from numpy import sqrt,zeros,abs,where,diff,median,cumsum,around,array
    from analysis_functions import hdf5
    from charting_functions import chart_xy_symbol
    print('Executing datafile_timestamps')
    # Load needed parameters and arrays from datafile
    PP_periods = hdf5(datafile,'PP_periods')
    PP_divider = hdf5(datafile,'PP_divider')
    TT = []
    TA = []
    TH = []
    datasets = hdf5(datafile,'datasets')
    for dataset in datasets:
        i,pathname = index_pathname(datafile,dataset)
        tsa = hdf5(pathname,'acquire_timestamp')
        tsh = hdf5(pathname,'header_timestamp')
        period_median = median(diff(tsa))
        # Assign PP_period and PP_mode from estimated period
        err = abs(PP_periods - period_median)
        k = where(err == err.min())[0][0]
        PP_period = PP_periods[k]/PP_divider[k]
        dts = diff(tsa)
        Np = zeros(len(tsa))
        Np[1:] = cumsum(around(dts/PP_period))
        tsp = tsh[0] + Np*PP_period
        tsp += (tsh - tsp).min()
        # Validate tsp; if needed, correct for phase shift during data acquisition
        dt = tsa - tsp
        if (dt.max()-dt.min()) > 0.1:
            phase_shift = where(diff(dt)==diff(dt).max())[0][0] + 1
            dthp = tsh - tsp
            tsp[:phase_shift] += dthp[:phase_shift].min()
            tsp[phase_shift:] += dthp[phase_shift:].min()
        TT.extend(tsp)
        TA.extend(tsa)
        TH.extend(tsh)
        hdf5(pathname,'timestamps',tsp)
        period = median(diff(hdf5(pathname,'timestamps')))
        hdf5(pathname,'period',period)

    dta = TA[-1] - TA[0]
    dt = TT[-1]-TT[0]
    slope = 3600*24*(dta-dt)/dt
    if chart:
        TT = array(TT)
        TA = array(TA)
        TH = array(TH)
        chart_xy_symbol(TT,TA-TT,'acquire_timestamps - timestamps\nslope: {:0.3f} seconds per day'.format(slope),x_label='timestamp [s]',y_label='time difference [s]')
        chart_xy_symbol(range(len(TT))[1:],diff(TT),'diff(timestamps)\n{}'.format(datafile.split('/')[-1]),logy=True,x_label='image number',y_label='time difference [s]')
        chart_xy_symbol(TT,TH-TT,'header_timestamps - timestamps',x_label='timestamp [s]',y_label='time difference [s]',logy=True,ymin=0.001)

# def concatenate_datasets(analysis_dir,datasets):
#     """Concatenates image datasets into a single hdf5 file."""
#     from numpy import where,array
#     from os import path
#     term = 'xray_images'
#     dataset_pathnames = term_pathnames(analysis_dir,datasets,term)
#     fragments = analysis_dir.split('/')
#     beamtime = fragments[where(array(fragments) == 'Analysis')[0][0] - 1]
#     datafile = path.join(analysis_dir,beamtime+'.hdf5')
#     acquire_timestamp = []
#     header_timestamp = []
#     filenames = []
#     Temperature = []
#     Delay = []
#     Repeat = []
#     Float = []
#     image_number = []
#     rx_number = []
#     Up = []
#     index = []
#     omit = []
#     for i,dataset in enumerate(datasets):
#         pathname = [name for name in dataset_pathnames if dataset in name][0]
#         f = hdf5(pathname,'filenames')
#         filenames.extend(f)
#         index.extend([i]*len(f))
#         image_number.extend(hdf5(pathname,'image_number'))
#         rx_number.extend(hdf5(pathname,'rx_number'))
#         acquire_timestamp.extend(hdf5(pathname,'acquire_timestamp'))
#         header_timestamp.extend(hdf5(pathname,'header_timestamp'))
#         Repeat.extend(hdf5(pathname,'Repeat'))
#         Up.extend(hdf5(pathname,'Up'))
#         Temperature.extend(hdf5(pathname,'Temperature'))
#         Delay.extend(hdf5(pathname,'Delay'))
#         Float.extend(hdf5(pathname,'Float'))
#         omit.extend(hdf5(pathname,'omit'))
#     # Write datafile
#     hdf5(datafile,'datasets',datasets)
#     hdf5(datafile,'dataset_pathnames',dataset_pathnames)
#     hdf5(datafile,'filenames',array(filenames))
#     hdf5(datafile,'image_number',image_number)
#     hdf5(datafile,'index',index)
#     hdf5(datafile,'rx_number',rx_number)
#     hdf5(datafile,'acquire_timestamp',acquire_timestamp)
#     hdf5(datafile,'header_timestamp',header_timestamp)
#     hdf5(datafile,'Repeat',Repeat)
#     hdf5(datafile,'Up',Up)
#     hdf5(datafile,'Temperature',Temperature)
#     hdf5(datafile,'Delay',Delay)
#     hdf5(datafile,'Float',Float)
#     hdf5(datafile,'omit',omit)
#     print("datafile = '{}'".format(datafile))
#     return datafile

def dataset_attributes(datafile,dataset):
    """From image filenames found in dataset, extracts saxs region 
    defined by [X0,Y0,saxs_boxsize] and writes this 3D array along 
    with image attributes in the corresponding hdf5 file in order
    sorted by acquire_timestamp."""
    from numpy import zeros,argsort,array
    from time import time
    # Extract information needed from datafile
    t0 = time()
    X0 = hdf5(datafile,'X0')
    Y0 = hdf5(datafile,'Y0')
    saxs_boxsize = hdf5(datafile,'saxs_boxsize')
    i,pathname = index_pathname(datafile,dataset)
    filenames = hdf5(pathname,'filenames')
    # Determine relevant attributes
    s = int((saxs_boxsize-1)/2)
    N_images = len(filenames)
    acquire_timestamp = zeros(N_images)
    header_timestamp = zeros(N_images)
    rx_number = zeros(N_images,'uint32')
    saxs = zeros((N_images,saxs_boxsize,saxs_boxsize),'uint16')
    omit = zeros(N_images,'uint8')
    # Extract attributes from path_names
    attributes_dict = parse_pathnames(filenames)
    Temperature = attributes_dict['Temperature']
    Delay = attributes_dict['Delay']
    Float = attributes_dict['Float']
    Repeat = attributes_dict['Repeat']
    Up = attributes_dict['Up']
    image_number = attributes_dict['image_number']
    dataset = pathname.split('/')[-2]
    for i,filename in enumerate(filenames):
        t1 = time()
        rx,tsa,tsh = mccd_header_read(filename)
        saxs[i] = mccd_read(filename,sub=(X0,Y0,s))
        acquire_timestamp[i] = tsa
        header_timestamp[i] = tsh
        rx_number[i] = rx
        print('\r{:0.3f} seconds to process image {} of {}    '.format(time()-t1,i,N_images),end='\r')
    # Sort according to acquire_timestamp and write to hdf5 file
    sort_order = argsort(acquire_timestamp)
    hdf5(pathname,'filenames',array(filenames)[sort_order])
    hdf5(pathname,'image_number',image_number[sort_order])
    hdf5(pathname,'rx_number',rx_number[sort_order])
    hdf5(pathname,'acquire_timestamp',acquire_timestamp[sort_order])
    hdf5(pathname,'header_timestamp',header_timestamp[sort_order])
    hdf5(pathname,'saxs',saxs[sort_order])
    hdf5(pathname,'Repeat',Repeat[sort_order])
    hdf5(pathname,'Up',Up[sort_order])
    hdf5(pathname,'Temperature',Temperature[sort_order])
    hdf5(pathname,'Delay',Delay[sort_order])
    hdf5(pathname,'Float',Float[sort_order])
    hdf5(pathname,'omit',omit[sort_order])
    print('Processed {} images from {} in {:0.3f} seconds                                                '.format(N_images,dataset,time()-t0))

def multiprocess_attributes(datafile,N_proc=8):
    """Multiprocessing dataset_attributes"""
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    print('Multiprocessing with dataset_attributes')
    # Generate argument list
    datasets = hdf5(datafile,'datasets')
    args = [(datafile,dataset) for dataset in datasets]
    with multiprocessing.Pool(N_proc) as pool:
        pool.starmap(dataset_attributes,args)

def datasets_filenames(data_dir,term,exclude=exclude):
    """Finds datasets in data_dir in which 'term' is found; 
    returns lists of 'datasets' and corresponding 'filenames';
    datasets list is sorted according to getmtime for first file 
    found in corresponding filenames, which are sorted alphabetically."""
    from os import path,walk
    from os.path import getmtime
    from numpy import argsort,sort
    filenames = []
    datasets = []
    for root, dirs, files in walk(data_dir):
        excluded = any([e in root for e in exclude])
        if (term in root) & ~excluded:
            datasets.append(root.split('/')[-2])
            # Omit filenames that start with '.'
            files = [f for f in files if f[0] != '.']
            filenames.append([path.join(root,f) for f in sort(files)])
    start_times = [getmtime(f[0]) for f in filenames]
    sort_order = argsort(start_times)
    sorted_datasets = [datasets[i] for i in sort_order]
    sorted_filenames = [filenames[i] for i in sort_order]
    return sorted_datasets,sorted_filenames

def term_pathnames(analysis_dir,datasets,term):
    """Finds pathnames in analysis_dir that contain 'term';
    returns list of pathnames corresponding to datasets."""
    from os import path,walk
    pathnames = []
    for root, dirs, files in walk(analysis_dir):
        filename = [f for f in files if term in f]
        if filename:
            pathnames.append(path.join(root,filename[0]))
    dataset_pathnames = [pathname for dataset in datasets for pathname in pathnames if dataset in pathname]
    return dataset_pathnames

def hdf5_delete(pathname,name):
    """Deletes name from pathname and repacks the file."""
    import h5py
    with h5py.File(pathname,  "a") as f:
        del f[name]
    hdf5_repack(pathname)
    
def hdf5_rename(pathname,name1,name2):
    """Renames name1 in pathname to name2."""
    import h5py
    with h5py.File(pathname,  "a") as f:
        f[name2] = f[name1]
        del f[name1]

def hdf5_repack(pathname):
    """Repacks hdf5 file to eliminate wasted space."""
    from os import rename,system
    tmp = pathname + '.copy'
    rename(pathname,tmp)
    system('h5repack ' + tmp + ' ' + pathname)
    system('rm ' + tmp)

def hdf5(path_name,keyword=None,val=None,index=None, attribute=None) -> ndarray:
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
                    data = convert_bytes_to_string(data)
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

def convert_bytes_to_string(data):
    if hasattr(data, "decode"):
        data = data.decode()
    elif hasattr(data, "__len__") and len(data) > 0 and hasattr(data[0], "decode"):
        data = [val.decode() for val in data]
    return data

def mccd_read(path_name,sub=None):
    """Extracts image or sub image from file specified
    by path_name given hard-coded header_size (4096) and 
    image_size (3840x3840). A sub image is returned if 
    sub is a tuple (X0,Y0,s), where X0 and Y0 correspond 
    to the center of the sub image, whose dimensions are
    (2s+1 x 2s+1)."""
    from numpy import uint16
    from numpy import memmap
    image_size = 3840
    headersize = 4096
    if sub != None:
        X0,Y0,s = sub
        mm = memmap(path_name,uint16,'r',headersize,(image_size,image_size),'C')[Y0-s:Y0+s+1,X0-s:X0+s+1]
        image = mm.copy()
    else:
        mm = memmap(path_name,uint16,'r',headersize,(image_size,image_size),'C')[0:image_size,0:image_size]
        image = mm.copy()
    return image
    
def mccd_header_read(path_name):
    """From header of path_name, extract acquire and header timestamps
    and rx number."""
    def mccd_timestamp_to_seconds(timestamp):
        """Convert mccd timestamp to universal time in seconds. Fails during
        one hour in the Fall when switching from DST to standard time (i.e. 2-3 am)."""
        import pytz
        from datetime import timezone,datetime
        from dateutil.parser import parse
        t0 = parse("1970-01-01 00:00:00+0000")
        #Universal = timezone(timedelta(hours = 0)) 
        months = int(timestamp[:2])
        days = int(timestamp[2:4])
        hours = int(timestamp[4:6])
        mins = int(timestamp[6:8])
        year = int(timestamp[8:12])
        seconds = int(timestamp[13:15])
        microseconds = int(timestamp[16:22])
        # Convert to universal time
        t_image = datetime(year,months,days,hours,mins,seconds,microseconds) 
        local_timezone = pytz.timezone("US/Central")
        local_timezone_offset = local_timezone.utcoffset(t_image, is_dst=True)
        t_image = t_image.replace(tzinfo=timezone(local_timezone_offset))
        return (t_image-t0).total_seconds()
    with open(path_name,"rb") as f:
        header = f.read(4096)
    rxn = header[2304:2304+32]
    tsa = header[2368:2368+32]
    tsh = header[2400:2400+32]
    rx_number = int(str(rxn).split("b'")[-1].split('.rx')[0])
    acquire_timestamp = mccd_timestamp_to_seconds(tsa)
    header_timestamp = mccd_timestamp_to_seconds(tsh)
    return rx_number,acquire_timestamp,header_timestamp

def parse_pathnames(path_names):
    """Parses path_names and assigns attributes."""
    from numpy import array,nan,roll,where,zeros
    from os import path
    # Find common portion of path_names and isolate attribute-specifying segment in path_names
    ext = path.splitext(path_names[0])[-1]
    basename = path_names[0].split('/')[-3]
    names = [path_name.split(basename+'_')[-1].split(ext)[0] for path_name in path_names]
    # Assign attributes according to corresponding name fragments
    fragments = names[0].split('_') 
    # If '0us' is found in names, assign corresponding fragment to Delay
    Delay_match = array(['us' in fragment for fragment in fragments])
    if Delay_match.any():
        delay_list = [name.split('_')[where(Delay_match)[0][0]] for name in names]
        Delay = array([delay_string_to_seconds(delay) for delay in delay_list])
    else:
        Delay = array([nan]*len(path_names))
    # If 'C' and '.' are found in the same name fragment, assign corresponding fragment to Temperature; assign Up 
    Temperature_match = array([('C' in fragment) & ('.' in fragment) for fragment in fragments])
    Temperature = array([nan]*len(names))
    Up = zeros(len(names),bool)
    if Temperature_match.any():
        Temperature = array([float(name.split('_')[where(Temperature_match)[0][0]][:-1]) for name in names])
        if 'Tramp' in path_names[0]:
            Up = ((roll(Temperature,-1) - Temperature) > 0) | (Temperature == Temperature.max())

    # If '.' is found in names, and not 'C', assume it corresponds to Float 
    Float_match = array(['.' in fragment for fragment in fragments]) & ~Temperature_match & ~Delay_match
    if Float_match.any():
        Float = array([float(name.split('_')[where(Float_match)[0][0]]) for name in names])
    else:
        Float = array([nan]*len(path_names))    
    # Assign Repeat (if no explicit Repeat, assign running number to Repeat)
    Repeat_match = ~(Delay_match | Temperature_match | Float_match)
    index = where(Repeat_match)[0][-1]
    Repeat = array([int(name.split('_')[index]) for name in names])
    # Assign running image number (image_number)
    image_number = array([int(name.split('_')[0]) for name in names])
    return {'image_number':image_number,'Temperature':Temperature,'Delay':Delay,'Float':Float,'Repeat':Repeat,'Up':Up}

def delay_string_to_seconds(s):
    """Convert time string to number. e.g. '100ps' -> 1e-10"""
    from numpy import nan
    try: return float(s)
    except: pass
    s = s.replace("s","")
    s = s.replace("p","*1e-12")
    s = s.replace("n","*1e-9")
    s = s.replace("u","*1e-6")
    s = s.replace("m","*1e-3")
    try: return float(eval(s))
    except: return nan 

def z_statistic(N):
    """Returns z, which corresponds to the threshold in units of sigma where
    the probability of measuring a value greater than z*sigma above the mean
    is 1 out of N."""
    from numpy import sqrt
    from scipy.special import erfinv
    return sqrt(2)*erfinv(1-2/N)

def image_SVD(I):
    """Performs SVD on array of images (N,row,column). Returns UT,s,V.
    """
    from scipy.linalg import svd
    from numpy import reshape
    from time import time
    t0 = time()
    N_images,w,h = I.shape
    print("PERFORMING SINGULAR VAULE DECOMPOSITION OF {} IMAGES...".format(N_images))
    I = reshape(I,(N_images,w*h))       # Reshape 3D array to 2D.
    U,s,V = svd(I.T,full_matrices = False)
    UT = reshape(U.T,(len(s),w,h))     # Reshape 2D array to 3D.
    for i in range(len(s)):
        if ((i == 0) & (UT[i][1,1] < 0)) | ((i != 0) & (UT[i][1,1] > 0)):
            UT[i] *= -1
            V[i] *= -1
    t1=time()
    print('{:0.3f} seconds to process {} images.'.format(t1-t0,N_images))
    return UT,s,V

def water_Cv(Tb):
    """Returns Cv for water at Tb using data found in 
    https://www.engineeringtoolbox.com/specific-heat-capacity-water-d_660.html
    where Tb is in Celsius."""
    from numpy import array,polyfit,polyval
    T = array([0.01,10,20,25,30,40,50,60,70,80,90,100,110,120,140,160])
    Cv =  array([4.2174,4.1910,4.1570,4.1379,4.1175,4.0737,4.0264,3.9767,3.9252,3.8729,3.8204,3.7682,3.7167,3.6662,3.5694,3.4788])
    p = polyfit(T,Cv,4)
    #Cv_fit = polyval(p,T)
    #chart_xy_fit(T,Cv,Cv_fit,'Cv and fit of Cv',x_label='Temperature [Celsius]',ms=3)
    return polyval(p,Tb)

def water_isothermal_compressibility(Tb):
    """Returns isothermal compressibility of water evaluated at Tb
    in units of inverse bar."""
    from numpy import loadtxt
    from scipy.interpolate import UnivariateSpline
    Tk,kappa = loadtxt('Water_Isothermal_Compressibility.txt',dtype=float,delimiter='\t').T
    us_kappa = UnivariateSpline(Tk,kappa,s=0)
    return us_kappa(Tb)*1e-6

def water_density(T):
    """Returns density of water given temperature T in Celsius. The 6th order
    polynomial reported in https://aip.scitation.org/doi/10.1063/1.453710
    was derived from data spanning  -33.41 < T < - 5.23·C from that work
    (using a a 300 um ID capillary), and higher accuracy data from Kell 
    in the range - 4 < T < 10 C. Data above 10 C come from a different
    function found in https://pubs.acs.org/doi/pdf/10.1021/je60064a005."""
    from numpy import where,array
    ai = [0.99986,6.690e-5,-8.486e-6,1.518e-7,-6.9484e-9,-3.6449e-10,-7.497e-12]
    rho_a = ai[0] + ai[1]*T + ai[2]*T**2 + ai[3]*T**3 + ai[4]*T**4 + ai[5]*T**5 + ai[6]*T**6

    bi = [999.83952,16.945176,-7.9870401e-3,-46.170461e-6,105.56302e-9,-280.54253e-12,16.879850e-3]
    rho_b = (bi[0] + bi[1]*T + bi[2]*T**2 + bi[3]*T**3 + bi[4]*T**4 + bi[5]*T**5)/(1 + bi[6]*T)/1000
    
    return array(where(T>=10,rho_b,rho_a))    

def electron_density_normalization(T,red=1.37,tec=115e-6,chart=False):
    """Returns temperature-dependent normalization of SAXS intensity 
    given 'red', the relative electron density of the particle at 4
    Celsius (compared to water/buffer), and 'tec', the linear thermal expansion 
    coefficient for the particle.
        'red' for RNA is 1.57 
        'red' for protein is 1.37 """
    sd = water_density(T)
    pd = red/(1+3*tec*(T-4))
    norm = sd*(pd - sd)**2/(red - 1)**2
    if chart:
        from charting_functions import chart_xy_symbol
        chart_xy_symbol(T,norm,'electron density normalization\nred = {:0.3f}, tec = {:0.3e}'.format(red,tec),x_label='Temperature [Celsius]',ms=3)
    return norm

# tec=115e-6
# from numpy import array,arange
# T = arange(-16,121)
# red = arange(1.2,1.5,0.1)
# ed_norm = []
# for redi in red:
#     ed_norm.append(electron_density_normalization(T,red=redi,tec=115e-6))
# ed_norm = array(ed_norm)
# chart_xy_symbol(T,ed_norm,'electron density normalization\nred = {}; tec = {:0.3e}'.format(red,tec),x_label='Temperature [Celsius]')

def Compton_scattering(q,Sq,formula,q_match=5):
    """Returns Compton scattering from elements specified 
    in formula and scaled to Sq assuming scattering at
    q ~ q_match is dominated by respective atomic form factors."""
    from numpy import where
    FF2,C = FF2_C_from_formula(formula,q)
    i = where(q > q_match)[0][0]
    Sq_scale = Sq[i-10:i+11].mean()
    FF2C_scale = (FF2+C)[i-10:i+11].mean()
    C_scaled = C*Sq_scale/FF2C_scale
    return C_scaled

def FF_C(q,Z):
    """Reads form factor and Compton scattering files and uses
    UnivariateSpline to return FF(q) and C(q)."""
    from numpy import loadtxt,pi,where
    from scipy.interpolate import UnivariateSpline
    FF = loadtxt('FF(1-60 and 71-85).txt',dtype=float,delimiter='\t')
    q_data = 4*pi*FF[0,1:]#*qdict['wavelength']
    us_FF = UnivariateSpline(q_data,FF[where(FF[:,0]==Z)[0][0],1:],s=0)
    C = loadtxt('S(1-60 and 71-85).txt',dtype=float,delimiter='\t')
    us_C = UnivariateSpline(q_data,C[where(C[:,0]==Z)[0][0],1:],s=0)
    return us_FF(q),us_C(q)

def F_FF_C_from_formula(formula,q):
    """Given chemical formula, returns stoichometrically-weighted
    averages of F, FF, and C, all evaluated at q. Formula consists
    of element designation followed by stoichiometric integer, e.g., H2O1."""
    from numpy import arange,zeros,array
    N,Z,NE,MW = chemical_formula_to_N_Z_NE_MW(formula)
    FF = zeros(q.shape)
    F = zeros(q.shape)
    C = zeros(q.shape)
    N = array(N,float)
    W = N/N.sum()
    # Generate stoichiometrically weighted sum
    for i,Zi in enumerate(Z):
        Fi,Ci = FF_C(q,Zi)
        F += W[i]*Fi
        FF += W[i]*Fi**2
        C += W[i]*Ci
    return F,FF,C

def rms(V):
    """Computes rms of vector V; excludes slow variations in V."""
    from numpy import sqrt
    return sqrt(0.5*((V[1:] - V[:-1])**2).sum()/(len(V) - 1))

def chemical_formula_to_N_Z_NE_MW(formula):
    """Given a chemical formula (string with each element followed by number),
    returns N, the number of atoms for each element, Z, the atomic number for 
    each element, NE, the total number of electrons, and MW, the molecular weight. """
    from numpy import array
    PT_dict = PeriodicTable_dict()
    N_str = ''.join((ch if ch in '0123456789' else ' ') for ch in formula)
    N = [int(i) for i in N_str.split()]
    symbol_str = ''.join((ch if ch not in '0123456789' else ' ') for ch in formula)
    symbol = [sym for sym in symbol_str.split()]
    Z = [int(PT_dict['AtomicNumber'][PT_dict['Symbol'].index(sym)] )for sym in symbol]
    M = [float(PT_dict['AtomicMass'][PT_dict['Symbol'].index(sym)]) for sym in symbol]
    NE = (array(N)*Z).sum()
    MW = (array(M)*N).sum()
    return N,Z,NE,MW

def PeriodicTable_dict():
    """Converts Periodic Table in csv file into a dictionary."""
    import csv
    with open('Periodic Table of Elements.csv', 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        csv_dict = {elem: [] for elem in csv_reader.fieldnames}
        for row in csv_reader:
            for key in csv_dict.keys():
                csv_dict[key].append(row[key])
    return csv_dict

def S_AbsoluteScale_ComptonCorrected(q,S,qe,FF2,C,q_switch=5,chart=False):
    """Given S vs. q and FF2,C vs. qe, Returns SA,
    which is S vs. qe on an absolute scale after correction 
    for Compton scattering. S is extrapolated to q=0 using an
    even order polynomial and the switch from molecular to 
    atomic scattering is made at q = q_switch."""
    from numpy.linalg import lstsq 
    from numpy import vstack,ones,where,array,sqrt
    # Extrapolate S to q = 0 using results from weighted even order polynomial fit
    qs = q < 0.04
    w = sqrt(q[qs])
    M = vstack((ones(len(q[qs])),q[qs]**2,q[qs]**4))
    coeffs = lstsq((M*w).T,(S[qs]*w).T,rcond=None)[0]
    fit = coeffs.T@M    
    residual = S[qs] - fit
    # Put S on qe scale including extrapolation to zero q.
    dq = qe[1]
    Se = coeffs.T@vstack((ones(len(qe)),qe**2,qe**4))
    select = (qe > q.min()-dq) & (qe < q.max()+dq)
    Se[select] = S
    # Rescale Se according to FF2 and subtract Compton scattering
    iswitch = where(qe>q_switch)[0][0]  
    S_scale = Se[iswitch-3:iswitch+4].mean()
    FF2C_scale = (FF2+C)[iswitch-3:iswitch+4].mean()
    SA = Se*FF2C_scale/S_scale
    SA[iswitch:] = (FF2+C)[iswitch:]
    SAC = SA - C
    SAC[iswitch:] = FF2[iswitch:]
    SACn = SAC/FF2[0]
    if chart:
        from charting_functions import chart_xy_rainbow,chart_xy_symbol
        chart_xy_symbol(q[qs],S[qs],'S',x_label='q',y_label='Counts',logx=True)  
        chart_xy_symbol(q[qs],residual,'residual',x_label='q',y_label='Counts',logx=True)  
        chart_xy_symbol(qe,array([SAC,SA,FF2+C,FF2,C]),'[SAC,SA,FF2+C,FF2,C]',x_label='q',logx=True,xmin=0.01)
        chart_xy_symbol(qe,SACn,'SACn',x_label='q',logx=True,xmin=0.01,logy=True)
    return SACn

def S_AbsoluteScale_ComptonCorrected_old(q,S,qe,FF2,C,q_switch=5,chart=False):
    """Given S vs. q and FF2,C vs. qe, Returns SA,
    which is S vs. qe on an absolute scale after correction 
    for Compton scattering. S is extrapolated to q=0 using an
    even order polynomial and the switch from molecular to 
    atomic scattering is made at q = q_switch."""
    from numpy.linalg import lstsq 
    from numpy import vstack,ones,where,array,sqrt
    # Extrapolate S to q = 0 using results from weighted even order polynomial fit
    qs = q < 0.2
    w = sqrt(q[qs])
    M = vstack((ones(len(q[qs])),q[qs]**2,q[qs]**4))
    coeffs = lstsq((M*w).T,(S[:,qs]*w).T,rcond=None)[0]
    fit = coeffs.T@M    
    residual = S[:,qs] - fit
    # Put S on qe scale including extrapolation to zero q.
    dq = qe[1]
    Se = coeffs.T@vstack((ones(len(qe)),qe**2,qe**4))
    select = (qe > q.min()-dq) & (qe < q.max()+dq)
    Se[:,select] = S
    # Rescale Se according to FF2 and subtract Compton scattering
    iswitch = where(qe>q_switch)[0][0]  
    S_scale = Se[:,iswitch-3:iswitch+4].mean(axis=1)
    FF2C_scale = (FF2+C)[iswitch-3:iswitch+4].mean()
    SA = ((Se*FF2C_scale).T/S_scale).T
    SA[:,iswitch:] = (FF2+C)[iswitch:]
    SAC = SA - C
    SAC[:,iswitch:] = FF2[iswitch:]
    SACn = SAC/FF2[0]
    if chart:
        from charting_functions import chart_xy_rainbow,chart_xy_symbol
        chart_xy_rainbow(q[qs],S[:,qs],'S',x_label='q',y_label='Counts',logx=True)  
        chart_xy_rainbow(q[qs],residual,'residual',x_label='q',y_label='Counts',logx=True)  
        chart_xy_symbol(qe,array([SAC.mean(axis=0),SA.mean(axis=0),FF2+C,FF2,C]),'[SAC_mean,SA_mean,FF2+C,FF2,C]',x_label='q',logx=True,xmin=0.01)
        chart_xy_rainbow(qe,SACn,'SACn',x_label='q',logx=True,xmin=0.01)
    return SACn

def Pair_Distribution_Function(qe,SAC,title=None,dr=0.025,rmax=100,sig=10,chart=None):
    """Converts I(q) to pair distribution function, PDF according to:
        p(r) = dq/(2*pi**2)*integral(rq*sin(rq)*I(q))
    """
    from scipy.integrate import simpson
    from numpy import pi,arange,sin,multiply,exp,zeros
    # Calculate PDF after apodization with gaussian of specified sig
    r = arange(0,rmax+dr,dr)
    rq = multiply.outer(r,qe)
    PDF = zeros((len(SAC),len(r)))
    dq = qe[1]
    scale = dq/(2.0*pi*pi)
    apodize = exp(-0.5*(qe/sig)**2)
    for i,SACi in enumerate(SAC):
        PDF[i] = scale*simpson(rq*sin(rq)*SACi*apodize)
    if chart:
        from charting_functions import chart_xy_rainbow
        chart_xy_rainbow(r,PDF,'PDF\n{}'.format(title),x_label= 'r [Å]')
    return r, PDF

def mu_Water(keV=11.648):
    """Returns inverse 1/e penetration depth [mm-1] of water given the
    x-ray energy in keV. The transmission through a 1-mm thick slab of water
    was calculated every 100 eV over an energy range spanning 8-16 keV using:
        http://henke.lbl.gov/optical_constants/filter2.html
    This result was then converted to mu and saved as a tab-delimited text
    file. The returned result is calculated using a univariate spline and
    should be valid over the range 8-16 keV."""
    from numpy import loadtxt
    from scipy.interpolate import UnivariateSpline
    E_mu = loadtxt('mu_Water.txt',dtype=float,delimiter='\t')
    us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
    return us_mu(1000*keV)
    
def mu_FusedSilica(keV=11.648):
    """Returns inverse 1/e penetration depth [mm-1] of fused silica given the
    x-ray energy in keV. The transmission through a 0.2-mm thick slab of SiO2
    was calculated every 100 eV over an energy range spanning 8-16 keV using:
        http://henke.lbl.gov/optical_constants/filter2.html
    This result was then converted to mu and saved as a tab-delimited text
    file. The returned result is calculated using a univariate spline and
    should be valid over the range 8-16 keV."""
    from numpy import loadtxt
    from scipy.interpolate import UnivariateSpline
    E_mu = loadtxt('mu_FusedSilica.txt',dtype=float,delimiter='\t')
    us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
    return us_mu(1000*keV)

def mu_Kapton(keV=11.648):
    """Returns inverse 1/e penetration depth [mm-1] of Kapton given the
    x-ray energy in keV. The attemuation length of Kapton was calculated
    every 100 eV over an energy range spanning 8-16 keV using:
        https://henke.lbl.gov/optical_constants/atten2.html
    Choose from a list of common materials: polyimide
    Chemical Formula: C22H10N2O5
    Density: 1.43 g/cm^3
    This result was then converted to mu and saved as a tab-delimited text
    file. The returned result is calculated using a univariate spline and
    should be valid over the range 8-16 keV."""
    from numpy import loadtxt
    from scipy.interpolate import UnivariateSpline
    E_mu = loadtxt('mu_Kapton.txt',dtype=float,delimiter='\t')
    us_mu = UnivariateSpline(E_mu[:,0],E_mu[:,1],s=0)
    return us_mu(1000*keV)

def mu_for_Z(Z,keV=11.648):
    """Returns inverse 1/e penetration depth [mm-1] for atomic
    number Z at specified x-ray energy and density of 1 g/cm3. 
    The transmission through a 1-mm thick slab was calculated for each
    Z every 100 eV over an energy range spanning 5-17 keV using
        http://henke.lbl.gov/optical_constants/filter2.html
    and saved as a tab-delimited text file. The transmission curve
    for specified Z from this file is fitted with a univariate spline 
    and converted to mu."""
    from numpy import loadtxt,log
    from scipy.interpolate import UnivariateSpline
    T = loadtxt('Atomic_Transmission(gm_cm3_mm).txt',dtype=float,delimiter='\t')
    us_T = UnivariateSpline(T[:,0],T[:,Z],s=0)
    return -log(us_T(1000*keV))

def sample_analysis(datafile,sample):
    """Processes datasets identified by 'sample', a list that 
    includes corresponding air and capillary scattering datasets, 
    generates ST, the average scattering for each dataset on
    a common (binned) temperature scale, Tb, and writes results in
    an hdf5 file using the first entry in the list as the file name."""
    from numpy import arange,zeros,zeros_like,array,nan,isnan,ones
    from os import path
    # Load datasets and q
    datasets = hdf5(datafile,'datasets')
    q = hdf5(datafile,'q')

    # Assign sample_filename
    analysis_dir = path.dirname(datafile) + '/'
    sample_filename = analysis_dir+sample[0] + '.hdf5'
    
    # Identify 'sample' datasets 
    S_datasets = array([dataset for dataset in datasets for s in sample if s in dataset])
    
    # Load information needed from S_datasets
    IC = []
    S = []
    sigS = []
    omit = []
    index = []
    temp = []
    nC = []
    Delay = []
    Repeat = []
    Temperature = []
    RTD_temp = []
    Up = []
    timestamps = []
    M0 = []
    iS = zeros(len(S_datasets),'uint16')
    for j,dataset in enumerate(S_datasets):
        iS[j],pathname = index_pathname(datafile,dataset)
        ICj = hdf5(pathname,'IC')
        IC.extend(ICj)
        index.extend(len(ICj)*[j])
        S.extend(hdf5(pathname,'S'))
        sigS.extend(hdf5(pathname,'sigS'))
        try:
            temp.extend(hdf5(pathname,'temp'))
        except:
            temp.extend(hdf5(pathname,'RTD_temp'))
        omit.extend(hdf5(pathname,'omit'))
        Delay.extend(hdf5(pathname,'Delay'))
        Repeat.extend(hdf5(pathname,'Repeat'))
        Temperature.extend(hdf5(pathname,'Temperature'))
        Up.extend(hdf5(pathname,'Up'))
        timestamps.extend(hdf5(pathname,'timestamps'))
        M0.extend(hdf5(pathname,'M0'))
        RTD_temp.extend(hdf5(pathname,'RTD_temp'))
        try:
            nC.extend(hdf5(pathname,'nC'))
        except:
            nC.extend([nan]*len(ICj))
    IC = array(IC)
    S = array(S)
    sigS = array(sigS)
    temp = array(temp)
    omit = array(omit)
    index = array(index)
    nC = array(nC)
    Delay = array(Delay)
    Repeat = array(Repeat)
    Temperature = array(Temperature)
    Up = array(Up)
    timestamps = array(timestamps)
    M0 = array(M0)
    RTD_temp = array(RTD_temp)

    # Determine Tb
    Tb = arange(Temperature.min(),Temperature.max()+1)
    
    # Generate Ss, sigSs, which correspond to S and sigS normalized by IC for each dataset
    # Generate ICm, M0m, which correspond to mean IC and M0 values for each dataset (omitting outliers)
    Ss = zeros_like(S)
    sigSs = zeros_like(S)
    ICm = zeros(len(S_datasets))
    M0m = zeros(len(S_datasets))
    for j,dataset in enumerate(S_datasets):
        ji = index == j
        jio = ji & (omit==0)
        ICm[j] = IC[jio].mean()
        M0m[j] = M0[jio].mean()
        Ss[ji] = ICm[j]*(S[ji].T/IC[ji]).T
        sigSs[ji] = ICm[j]*(sigS[ji].T/IC[ji]).T
    
    # Generate ST, sigST, and dSs for each dataset
    # Generate (M0Tn,ICTn) normalized, nC-scaled, Tb-binned M0 and IC
    ST = zeros((len(S_datasets),len(Tb),len(q)))
    sigST = zeros((len(S_datasets),len(Tb),len(q)))
    M0Tn = zeros((len(S_datasets),len(Tb)))
    ICTn = zeros((len(S_datasets),len(Tb)))
    sigM0Tn = zeros((len(S_datasets),len(Tb)))
    sigICTn = zeros((len(S_datasets),len(Tb)))
    dSs = zeros_like(S)
    for j,dataset in enumerate(S_datasets):
        ji = index == j
        jio = ji & (omit==0)
        ST[j],sigST[j] = S_to_ST(Ss[jio],sigSs[jio],temp[jio],Tb)
        dSs[ji] = Ss[ji] - S_to_ST(ST[j],sigST[j],Tb,temp[ji])[0]
        jion = jio & ~isnan(nC)
        M0nj = M0[jion]/M0[jion].mean()
        ICnj = IC[jion]/IC[jion].mean()
        nCnj = nC[jion]/nC[jion].mean()
        Tj = temp[jion]
        M0Tn[j],sigM0Tn[j] = S_to_ST(M0nj/nCnj,ones(len(Tj)),Tj,Tb)
        ICTn[j],sigICTn[j] = S_to_ST(ICnj/nCnj,ones(len(Tj)),Tj,Tb)            

    # Write results to hdf5 file
    hdf5(sample_filename,'S_datasets',S_datasets)
    hdf5(sample_filename,'timestamps',timestamps)
    hdf5(sample_filename,'index',index)
    hdf5(sample_filename,'temp',temp)
    hdf5(sample_filename,'omit',omit)
    hdf5(sample_filename,'Ss',Ss)
    hdf5(sample_filename,'sigSs',sigSs)
    hdf5(sample_filename,'Tb',Tb)
    hdf5(sample_filename,'ST',ST)
    hdf5(sample_filename,'sigST',sigST)
    hdf5(sample_filename,'M0m',M0m)
    hdf5(sample_filename,'ICm',ICm)
    hdf5(sample_filename,'M0Tn',M0Tn)
    hdf5(sample_filename,'ICTn',ICTn)
    hdf5(sample_filename,'sigM0Tn',sigM0Tn)
    hdf5(sample_filename,'sigICTn',sigICTn)
    hdf5(sample_filename,'Repeat',Repeat)
    hdf5(sample_filename,'Up',Up)

def sample_CBP(datafile,sample_name,VF=None,chart=False):
    """Extracts data from hdf5 file specified by sample_name, 
    generates C and B, which correspond to capillary and buffer   
    scattering curves corrected for sample absorbance, and uses VF
    to generate P, which corresponds to solute scattering; 
    writes results to same hdf5 file. """
    from numpy import array,where,exp,log,zeros_like,sqrt,outer
    from os import path
    from charting_functions import chart_xy_fit, chart_xy_rainbow, chart_xy_symbol

    # Determine sample_filename and load information 
    analysis_dir = path.dirname(datafile) + '/'
    sample_filename = analysis_dir + sample_name + '.hdf5'
    q = hdf5(datafile,'q')

    # Extract needed data
    S_datasets = array(hdf5(sample_filename,'S_datasets'))
    index = hdf5(sample_filename,'index')
    Tb = hdf5(sample_filename,'Tb')
    ST = hdf5(sample_filename,'ST')
    sigST = hdf5(sample_filename,'sigST')
    M0m = hdf5(sample_filename,'M0m')
    ICm = hdf5(sample_filename,'ICm')
    M0Tn = hdf5(sample_filename,'M0Tn')
    ICTn = hdf5(sample_filename,'ICTn')
    sigM0Tn = hdf5(sample_filename,'sigM0Tn')
    sigICTn = hdf5(sample_filename,'sigICTn')
    Repeat = hdf5(sample_filename,'Repeat')

    # Determine number of repeats for each dataset
    Rj = array([Repeat[index==i].max() for i in range(index.max()+1)])

    # Identify indices for air, buffer, capillary, and 'protein' datasets
    aj = where(['Tramp_A' in dataset for dataset in S_datasets])[0][0]
    bj = where(['Tramp_B' in dataset for dataset in S_datasets])[0]
    cj = where(['Tramp_C' in dataset for dataset in S_datasets])[0][0]
    pj = where(['Tramp_PC' in dataset for dataset in S_datasets])[0]

    # Assign appropriately indexed ST and sigST to AT, CT, BT, and PT
    AT = ST[aj]
    CT = ST[cj]
    BT = ST[bj]
    PT = ST[pj]
    sigAT = sigST[aj]
    sigCT = sigST[cj]
    sigBT = sigST[bj]
    sigPT = sigST[pj]

    # Extract scaling parameters for air
    M0m_A = M0m[aj]
    ICm_A = ICm[aj]
    ICTn_A = ICTn[aj]
    sigICTn_A = sigICTn[aj]

    # Extract scaling parameters for capillary
    M0m_C = M0m[cj]
    ICm_C = ICm[cj]
    ICTn_C = ICTn[cj]
    sigICTn_C = sigICTn[cj]

    # Extract scaling parameters for buffer; average over B datasets
    M0m_B = M0m[bj]
    M0m_Bm = (M0m_B*Rj[bj]).sum()/Rj[bj].sum()
    ICm_B = ICm[bj]
    ICTn_B = ICTn[bj]
    ICTn_Bm = (ICTn_B.T*Rj[bj]).sum(axis=1)/Rj[bj].sum()
    sigICTn_B = sigICTn[bj]
    sigICTn_Bm = (sigICTn_B.T*Rj[bj]).sum(axis=1)/Rj[bj].sum()
    ICTn_Bfit = us_fit(ICTn_Bm,s_scale=1,sig=sigICTn_Bm,chart=False)

    # Extract scaling parameters for 'protein'
    M0m_P = M0m[pj]
    ICm_P = ICm[pj]
    ICTn_P = ICTn[pj]
    sigICTn_P = sigICTn[pj]

    # Determine rci from PCi series, assuming 1/3 dilution factor
    rci = array([(1/3)**int(name.split('_PC')[-1].split('-')[0]) for name in S_datasets[pj]])
    
    if VF is None:
        # Determine VF for salts assuming hexacoordinated ions, with PC0 having 1 M chloride
        if ('NaCl' in sample_name):
            VF = 6*2/55.5    
            formula = 'Na1Cl1H24O12'
        elif 'KCl' in sample_name:
            VF = 6*2/55.5
            formula = 'K1Cl1H24O12'
        elif ('MgCl2' in sample_name):
            VF = 6*(1 + 1/2)/55.5
            formula = 'Mg1Cl2H36O18'
        elif ('CaCl2' in sample_name):
            VF = 6*(1 + 1/2)/55.5
            formula = 'Ca1Cl2H36O18'
        elif 'AlCl3' in sample_name:
            VF = 6*(1 + 1/3)/55.5
            formula = 'Al1Cl3H48O24'
        # Rescale VF according to volume deficit at 1 M chloride
        VF *= 0.96 

    # Determine abs_C, abs_B and abs_P, absorbance of buffer and protein from corresponding M0m
    abs_C = -log(M0m[cj]/M0m[aj])
    abs_B = -log(M0m_Bm/M0m[cj])
    abs_P = -log(M0m[pj[where(rci==1)]]/M0m[cj]) - (1-VF)*abs_B

    # Determine wd, TRB, TRP (temperature-dependent density, and transmittances through buffer and protein)
    wd = water_density(Tb)
    TRC = exp(-abs_C)
    TRB = exp(-abs_B*(1-outer(rci,VF))*wd)    
    TRP = exp(-abs_P*outer(rci,wd))

    # Generate C 
    C = CT/TRC
    sigC = sigCT/TRC

    # Generate B for all buffer datasets
    B = array([(BT[i].T*ICTn_Bfit/(TRC*exp(-abs_B)*wd) - C.T/wd).T for i in range(len(BT))])
    sigB = array([sqrt((sigBT[i].T*ICTn_Bfit/(TRC*exp(-abs_B)*wd))**2 + (sigCT.T/(TRC*wd))**2).T for i in range(len(BT))]) 

    # Generate P for all 'protein' datasets; use B[1] for PC0 dataset, and B[0] for all others
    P = zeros_like(PT)
    sigP = zeros_like(PT)
    for i in range(len(PT)):
        ICTn_fit = us_fit(ICTn_P[i],s_scale=1,sig=sigICTn_P[i],chart=False)
        j = where(rci[i] == 1,1,0)
        P[i] = (PT[i].T*ICTn_fit/(TRC*TRB[i]*TRP[i]*wd)).T - ((1-rci[i]*VF)*BT[j].T*ICTn_Bfit/(TRC*TRB[i]*wd)).T - (rci[i]*VF*CT.T/(TRC*wd)).T
        sigP[i] = sqrt(((sigPT[i].T*ICTn_fit/(TRC*TRB[i]*TRP[i]*wd)).T)**2 + (((1-rci[i]*VF)*sigBT[j].T*ICTn_Bfit/(TRC*TRB[i]*wd)).T)**2 + ((rci[i]*VF*sigCT.T/(TRC*wd)).T)**2)

    hdf5(sample_filename,'B',B)
    hdf5(sample_filename,'sigB',sigB)
    hdf5(sample_filename,'P',P)
    hdf5(sample_filename,'sigP',sigP)

    print('Processed {} datasets'.format(sample_name))

    if chart:
        chart_xy_fit(Tb,ICTn_B,ICTn_Bfit,'ICTn_B\n{}'.format(sample_name),x_label='Temperature [Celsius]')
        chart_xy_rainbow(q,AT,'AT\n{}'.format(S_datasets[aj]),x_label='q',y_label='Counts',logx=True,ymin=0)
        chart_xy_rainbow(q,C,'C\n{}'.format(S_datasets[cj]),x_label='q',y_label='Counts',logx=True,ymin=0)
    
        chart_xy_symbol(Tb,TRB,'TRB\n{}'.format(S_datasets[bj]),x_label='Temperature [Celsius]',y_label='Counts')
        chart_xy_symbol(Tb,TRP,'TRP\n{}'.format(S_datasets[pj]),x_label='Temperature [Celsius]',y_label='Counts')
        chart_xy_symbol(Tb,TRB*TRP,'TRB*TRP\n{}'.format(S_datasets[pj]),x_label='Temperature [Celsius]',y_label='Counts')
        
        for i in range(len(BT)):
            chart_xy_rainbow(q,BT[i],'BT[{}]\n{}'.format(i,S_datasets[bj[i]]),x_label='q',y_label='Counts',logx=True,ymin=0)
            chart_xy_rainbow(q,B[i],'B[{}]\n{}'.format(i,S_datasets[bj[i]]),x_label='q',y_label='Counts',logx=True,ymin=0)

        for i in range(len(P)):
            chart_xy_rainbow(q,PT[i],'PT\n{}'.format(S_datasets[pj[i]]),x_label='q',y_label='Counts',logx=True,ymin=0)
            chart_xy_rainbow(q,P[i],'Pi\n{}'.format(S_datasets[pj[i]]),x_label='q',y_label='Counts',logx=True,ymin=0)

def sample_analysis_old(datafile,sample):
    """Processes datasets identified by 'sample', which is a list of 
    keywords, and writes results into an hdf5 file using the first entry 
    in the list as the file name."""
    from numpy import arange,zeros,zeros_like,array,where,isnan,ones,nan
    from os import path
    # Load datasets and q
    datasets = hdf5(datafile,'datasets')
    q = hdf5(datafile,'q')

    # Assign sample_filename
    analysis_dir = path.dirname(datafile) + '/'
    sample_filename = analysis_dir+sample[0] + '.hdf5'
    
    # Identify 'sample' datasets 
    S_datasets = array([dataset for dataset in datasets for s in sample if s in dataset])
    
    # Load information needed from S_datasets
    IC = []
    S = []
    sigS = []
    omit = []
    index = []
    temp = []
    nC = []
    Delay = []
    Repeat = []
    Temperature = []
    RTD_temp = []
    Up = []
    timestamps = []
    M0 = []
    iS = zeros(len(S_datasets),'uint16')
    for j,dataset in enumerate(S_datasets):
        iS[j],pathname = index_pathname(datafile,dataset)
        ICj = hdf5(pathname,'IC')
        IC.extend(ICj)
        index.extend(len(ICj)*[j])
        S.extend(hdf5(pathname,'S'))
        sigS.extend(hdf5(pathname,'sigS'))
        try:
            temp.extend(hdf5(pathname,'temp'))
        except:
            temp.extend(hdf5(pathname,'RTD_temp'))
        omit.extend(hdf5(pathname,'omit'))
        Delay.extend(hdf5(pathname,'Delay'))
        Repeat.extend(hdf5(pathname,'Repeat'))
        Temperature.extend(hdf5(pathname,'Temperature'))
        Up.extend(hdf5(pathname,'Up'))
        timestamps.extend(hdf5(pathname,'timestamps'))
        M0.extend(hdf5(pathname,'M0'))
        RTD_temp.extend(hdf5(pathname,'RTD_temp'))
        try:
            nC.extend(hdf5(pathname,'nC'))
        except:
            nC.extend([nan]*len(ICj))
    IC = array(IC)
    S = array(S)
    sigS = array(sigS)
    temp = array(temp)
    omit = array(omit)
    index = array(index)
    nC = array(nC)
    Delay = array(Delay)
    Repeat = array(Repeat)
    Temperature = array(Temperature)
    Up = array(Up)
    timestamps = array(timestamps)
    M0 = array(M0)
    RTD_temp = array(RTD_temp)

    # Determine Tb
    Tb = arange(Temperature.min(),Temperature.max()+1)
    
    # Generate Ss, sigSs, which correspond to S and sigS divided by IC for each dataset
    # Generate ICm, M0m, which correspond to mean IC and M0 values for each dataset (omitting outliers)
    Ss = zeros_like(S)
    sigSs = zeros_like(S)
    ICm = zeros(len(S_datasets))
    M0m = zeros(len(S_datasets))
    for j,dataset in enumerate(S_datasets):
        ji = index == j
        Ss[ji] = (S[ji].T/IC[ji]).T
        sigSs[ji] = (sigS[ji].T/IC[ji]).T
        jio = ji & (omit==0)
        ICm[j] = IC[jio].mean()
        M0m[j] = M0[jio].mean()

    # Determine ICg, M0g, global scaled IC and M0 from buffer datasets
    bj = where(['Tramp_B' in dataset for dataset in S_datasets])[0]
    ICg = ICm[bj].mean()
    M0g = M0m[bj].mean()

    # Generate Sg,sigSg, which correspond to Ss and sigSs on a global scale
    #   Tramp_A and Tramp_C are scaled according to M0g
    #   Tramp_B and Tramp_PCi are scaled according to ICg  
    Sg = zeros_like(S)
    sigSg = zeros_like(S)
    for j,dataset in enumerate(S_datasets):
        ji = index == j
        if ('Tramp_A' in S_datasets[j]) or ('Tramp_C' in S_datasets[j]):
            Sg[ji] = Ss[ji]*ICm[j]*M0g/M0m[j]
            sigSg[ji] = sigSs[ji]*ICm[j]*M0g/M0m[j]
        else:
            Sg[ji] = Ss[ji]*ICg
            sigSg[ji] = sigSs[ji]*ICg
    
    # Generate ST, sigST, and dSg for each dataset
    # Generate (M0Tn,ICTn) normalized, nC-scaled, Tb-binned M0 and IC
    ST = zeros((len(S_datasets),len(Tb),len(q)))
    sigST = zeros((len(S_datasets),len(Tb),len(q)))
    M0Tn = zeros((len(S_datasets),len(Tb)))
    ICTn = zeros((len(S_datasets),len(Tb)))
    sigM0Tn = zeros((len(S_datasets),len(Tb)))
    sigICTn = zeros((len(S_datasets),len(Tb)))
    dSg = zeros_like(S)
    for j,dataset in enumerate(S_datasets):
        ji = index == j
        jio = ji & (omit==0)
        ST[j],sigST[j] = S_to_ST(Sg[jio],sigSg[jio],temp[jio],Tb)
        dSg[ji] = Sg[ji] - S_to_ST(ST[j],sigST[j],Tb,temp[ji])[0]
        jion = jio & ~isnan(nC)
        M0nj = M0[jion]/M0[jion].mean()
        ICnj = IC[jion]/IC[jion].mean()
        nCnj = nC[jion]/nC[jion].mean()
        Tj = temp[jion]
        M0Tn[j],sigM0Tn[j] = S_to_ST(M0nj/nCnj,ones(len(Tj)),Tj,Tb)
        ICTn[j],sigICTn[j] = S_to_ST(ICnj/nCnj,ones(len(Tj)),Tj,Tb)            

    # Write results to hdf5 file
    hdf5(sample_filename,'S_datasets',S_datasets)
    hdf5(sample_filename,'timestamps',timestamps)
    hdf5(sample_filename,'index',index)
    hdf5(sample_filename,'temp',temp)
    hdf5(sample_filename,'omit',omit)
    hdf5(sample_filename,'Sg',Sg)
    hdf5(sample_filename,'sigSg',sigSg)
    hdf5(sample_filename,'Tb',Tb)
    hdf5(sample_filename,'ST',ST)
    hdf5(sample_filename,'sigST',sigST)
    hdf5(sample_filename,'M0g',M0g)
    hdf5(sample_filename,'ICg',ICg)
    hdf5(sample_filename,'M0m',M0m)
    hdf5(sample_filename,'ICm',ICm)
    hdf5(sample_filename,'M0Tn',M0Tn)
    hdf5(sample_filename,'ICTn',ICTn)
    hdf5(sample_filename,'sigM0Tn',sigM0Tn)
    hdf5(sample_filename,'sigICTn',sigICTn)
    hdf5(sample_filename,'Repeat',Repeat)
    hdf5(sample_filename,'Up',Up)

def image_psi(ePix_filename, image,mask,name,vmin=0.6,vmax=1.4,chart=False,font_size=18):
    """Charts image normalized to median in each q bin.
    Excludes and zeros masked pixels."""
    from numpy import zeros, median, arange
    from charting_functions import chart_image, chart_xy_symbol

    q = hdf5(ePix_filename,'q')
    qbin1 = hdf5(ePix_filename,'qbin1')
    qbin2 = hdf5(ePix_filename,'qbin2')
    sort_indices = hdf5(ePix_filename,'sort_indices')
    reverse_indices = hdf5(ePix_filename,'reverse_indices')
    I_flat = image.flatten()[sort_indices]
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
    In = In_flat[reverse_indices].reshape(image.shape)
    if chart:
        chart_image(In,'I/I_median for each q-bin\n{}'.format(name),vmin=vmin,vmax=vmax,font_size=font_size)
        chart_xy_symbol(q,Iq,'Median vs. q\n{}'.format(name),x_label='q',font_size=font_size)
    return In,Iq
