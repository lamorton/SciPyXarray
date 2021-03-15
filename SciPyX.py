import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.mlab import specgram as mplspec
from scipy.integrate import cumtrapz as trapezoidalIntegral

pi=np.pi
ma=np.ma

def deriv(signal,dimension=None,coordinate=None,**kwargs):
    """
    Take a derivative with respect to a coordinate along a dimension. Allows 
    for general partial derivatives.
    """
    if dimension is None and signal.ndim>1:
        raise Exception("Warning: you must supply a dimension name if data has more than one to choose from.")
    elif dimension is None:
        axisNumber = 0
        dimension = signal.dims[0] #only one dimension 
    else:
        axisNumber=signal.get_axis_num(dimension)
    dy = np.gradient(signal.data,axis=axisNumber)
    if coordinate is None:
        coordinate = dimension
    x=getattr(signal,coordinate)
    axisNumber = x.get_axis_num(dimension)    
    dx = np.gradient(x,axis=axisNumber)
    return xr.DataArray(dy/dx,coords=signal.coords)

from scipy.integrate import cumtrapz as trapezoidalIntegral
def integral(signal,dimension=None,**kwargs):
    """
    Integrate an xarray variable over a dimension
    """
    if dimension is None and signal.ndim>1:
        raise Exception("Warning: you must supply a dimension name if data has more than one to choose from.")
    elif dimension is None:
        axisNumberSignal = 0
    else:
        axisNumberSignal=signal.get_axis_num(dimension)
    
    x=getattr(signal,dimension)   
    result=trapezoidalIntegral(signal,x,axis=axisNumberSignal,initial=0)
    return xr.DataArray(result,coords=signal.dims, dims=dimension)

def FFT(signal,NFFT=None,timeName='time',norm=None,freqName='frequency',real=True):
    """
    Automatically take the real or complex FFT of an xarray.DataArray You must
    supply the name of the frequency axis. Assumes that time is in seconds,
    and returns frequency in kHz. 
    """
    timeAxisNum=signal.get_axis_num(timeName)
    timeAxisData=getattr(signal,timeName).data
    dt=(timeAxisData[-1]-timeAxisData[0])/(len(timeAxisData)-1)
    if NFFT is None:
        NFFT=len(timeAxisData)
    if real:
        freq=rfftfreq(NFFT,dt)*1e-3
        signalFFT=rfft(signal.data,n=NFFT,axis=timeAxisNum,norm=norm)
    else:
        freq=fftshift(fftfreq(NFFT,dt))*1e-3
        signalFFT=fftshift(fft(signal.data,n=NFFT,axis=timeAxisNum,norm=norm),axes=timeAxisNum)
    #pack things back into a dataarray
    name=signal.name
    coords=dict(signal.coords.items()) 
    coords.pop(timeName)
    coords[freqName]=freq
    olddims=signal.dims
    newdims=olddims[0:timeAxisNum]+(freqName,)+olddims[timeAxisNum+1:]
    return xr.DataArray(signalFFT,coords=coords,name=name,dims=newdims)

#TODO: make the kwargs symmetric between FFT and FFTinverse! 
#TODO: resolve issue with numerical precision of time coordinate during round-trip!
def FFTinverse(signal,NFFT=None,norm=None,freqName='frequency',timeName='time',real=True,t0=0):
    """
    Undo the FFT of an xarray.DataArray object.
    Assumes frequency in kHz and time in seconds.
    """
    freqAxisNum = signal.get_axis_num(freqName)
    freqAxisData = getattr(signal,freqName).data
    df = (freqAxisData[-1]-freqAxisData[0])/(len(freqAxisData)-1)

    if real:
        if NFFT is None:
            raise Exception("If real=True, you must supply NFFT.")
        time = rfftfreq((NFFT-1)*2,0.5*df*1e3)+t0 #same function forward or backward
        signalFFT = irfft(signal.data,n=NFFT,axis=freqAxisNum,norm=norm)
    else:
        if NFFT is None:
            NFFT=len(freqAxisData)
        time = t0 + np.linspace(0,1,NFFT)/(df*1e3)
        signalFFT = ifft(ifftshift(signal.data,axes=freqAxisNum),n=NFFT,axis=freqAxisNum,norm=norm)
    #pack variables back into a dataarray
    name = signal.name
    coords = dict(signal.coords.items()) 
    coords.pop(freqName)
    coords[timeName] = time
    olddims = signal.dims
    newdims = olddims[0:freqAxisNum]+(timeName,)+olddims[freqAxisNum+1:]
    return xr.DataArray(signalFFT,coords=coords,name=name,dims=newdims)

from numpy.fft import rfft,irfft,rfftfreq,fft,ifft,fftfreq,fftshift,ifftshift
def bpfilt(signal,f0=None,f1=None,timeName='time',real=True):
    """
    Bandpass-filter the signals using a step-function frequency response.
    Arguments:
        signal: xarray.DataArray 
    Keyword arguments:
        f0: the low-frequency edge of the passband. Defaults to lowest possible
            frequency, which is zero for if real=True, and negative Nyquist 
            otherwise. 
        f1: the upper-frequency edge of passband. Default: highest frequency.
        timeName: name of the dimension to transform over. Defaults to 'time'.
        real: if True, use numpy.fft.rfft, so that only non-negative frequencies
              may be selected.  If signal is complex, set real=False, otherwise
              the imaginary component of signal will be silently discarded.
    Returns:
        An xarray.DataArray containing bandpassed version of the signal, with
        same dimensions.
    """    
    NFFT=len(signal[timeName].data)
    signalFFT = FFT(signal,timeName=timeName,NFFT=NFFT,real=real,freqName='frequency')
    freq = signalFFT.frequency.data
    if f0 is None:
        f0 = freq.min()
    if f1 is None:
        f1 = freq.max()

    #identify frequencies within range f0 to f1
    mask = (freq >= f0) & (freq <= f1) #using Bool True equiv to float 1.
    maskArray = xr.DataArray(mask,(freq,),('frequency'))
    filteredFFT = maskArray * signalFFT
    return FFTinverse(filteredFFT,real=real,t0=signal[timeName].data[0],NFFT=NFFT,timeName=timeName,freqName='frequency')

from matplotlib.mlab import psd
def powerSpectrum(signal,timeName='time',frequencyName='frequency',show=True,**kwargs):
    """
    Wrapper for power spectrum to use xarray.DataArrays. Allows only 1-D arrays
    (ie, not vectorized).
    Assumes that time axis is uniformly spaced, without checking. Be warned!
    """
    time = getattr(signal,timeName).data
    dt = (time[-1]-time[0])/(len(time)-1)
    spec,freq = psd(signal.data,Fs=1e-3/dt,**kwargs)
    xrspec = xr.DataArray(spec,(freq,),('frequency',),name='power')
    if show: 
        xrspec.plot()
        plt.yscale('log')
    return xrspec

def spec(data,NFFT=1024,timeName='time',frequencyName='frequency'):   
    """
    Wrapper for spectrogram creation (but not plotting). Can handle non-zero start time, unlike usual spectrogram which autostarts from zero.
    Puts time in seconds and frequency in kHz. The window overlap is 1/2 the window
    length, and a Hanning window is used.
    Inputs:
        data: an xr.DataArray object with one dimensional data
	      the dimension is assumed to be uniformly sampled, so the
	      sample frequency is determined by the difference between
	      the first two coordinate points.
	NFFT: the length of window used for short-time FFT window.
    Outputs:
        
	
    """
    assert xr.core.dataarray.DataArray == type(data), "Datatype=%s must be equal to %s"%(type(data),type(xr.core.dataarray.DataArray))
    assert len(data.dims)==1, "Data must be one-dimensional."
    t=data.coords[timeName].data
    hn=data.data
    deltaT=(t[1]-t[0]) 
    Fs=1/deltaT  #everything comes out in terms of Hz
    noverlap=NFFT/2
    s=mplspec(hn,Fs=Fs,NFFT=NFFT,noverlap=noverlap)
    variable=s[0]*1e3
    coords={frequencyName:s[1]*1e-3,timeName:s[2]+t[0]}
    dims=(frequencyName,timeName)
    return xr.DataArray(variable,coords=coords,dims=dims,name=data.name)

def tupleReplace(tpl,value,index):
    #utils.tupleReplace(field.data.shape,nfreqs,timeAxisNum)
    lst = list(tpl)
    lst[index] = value
    return tuple(lst)
    
from matplotlib.pylab import hanning    
#TODO: figure out how to do the power spectral density normalization  
#TODO: figure out how to do the transform: real or complex?  
def shortTimeFourierTransform(field,timeName,frequencyName,NFFT=2048):
    """
    Get the spectrogram of an arbitrary-dimensional data structure, along the
    time axis as defined by 'timeName'. Uses half-overlap with hanning window.
    Arguments:
       field: DataArray object that must have coordinate named by 'timeName'
              for time in seconds.
       timeName: name of the time-like dimension in the incoming array.
       frequencyName: name of the frequency-like dimension in the result.
    Keyword arguments:
       NFFT: number of time samples from series to take FFT
    Returns an xr.DataArray containing:
        data: Fourier Transformed signal. No normalization, this is not a PSD!
        <time>: number of time steps is floor(len(t)/tstep).
        <frequency>: nfreqs=NFFT/2+1.  In kHz, if time is in seconds!
        Note that names of <time>, <frequency> are set by user.
    """
    timeAxisNum=field.get_axis_num(timeName)
    t=getattr(field,timeName).data
    tstep=int(NFFT/2)
    ntimes=int((len(t)-NFFT)/tstep)+1
    nfreqs=int(NFFT/2)+1
    #need to know shape of array with time dimension removed for setup of fft 
    #output-catching array.  
    transformedDimensions=tupleReplace(field.data.shape,nfreqs,timeAxisNum)
    overallDimensions=(ntimes,)+transformedDimensions#put new time basis first,
    #leaving frequency where the time dimension was in the original array
    ft=np.empty(overallDimensions,dtype=complex)
    window=hanning(NFFT)
    broadcaster=(1,)*len(transformedDimensions) 
    #put window length where you want window to line up with data array
    broadcaster=tupleReplace(broadcaster,len(window),timeAxisNum)
    window=window.reshape(broadcaster)#line up window so it will broadcast right
    #do the short-time Fourier transforms to get the spectrogram
    for i in range(ntimes):
        selectTimes=slice(tstep*i,(tstep*i+NFFT))
        darr = field.isel({timeName:selectTimes})
        ft[i]=np.fft.rfft(darr*window,axis=timeAxisNum)
    #convert from Hz to kHz: 
    timebins=np.linspace(t[0],t[-1],ntimes,endpoint=False)
    fbins=np.fft.rfftfreq(NFFT,t[1]-t[0])          
    #set up new coordinates from old ones
    coords=dict(field.coords.items())
    coords.pop(timeName)
    coords[timeName]=timebins
    coords[frequencyName]=fbins*1e-3        
    #set up dims explicitly (dimensional inference is being deprecated)
    olddims=field.dims
    dims=(timeName,)+tupleReplace(olddims,frequencyName,timeAxisNum)     
    name=field.name 
    return xr.DataArray(ft,coords=coords,dims=dims,name=name)


                      
#TODO: make the transform incorporate the coil sensitivities --- that way, a bad coil
#can have a bad SVD mode, and you can leave it out.
def nuFFT(signal,NFFT,angleName='phi',wavenumberName='torModeNum'):
    """
    Fourier transform of non-uniformly-sampled toroidal array data.
    Transform is done in the angular direction, in parallel over all 
    time points (no transform in time). Uses angular harmonics up to NFFT.
    """
    #get ready
    npoints=len(signal[angleName])
    #assert npoints==2*NFFT+1
    #do the work                   
    D=np.empty((2*NFFT+1,npoints),dtype=complex)
    eikonal=np.exp(1j*signal[angleName].data)
    for i in range(NFFT+1):
        D[NFFT-i]=0.5*(eikonal**i).conj()#positive / negative rotation directions
        D[NFFT+i]=0.5*(eikonal**i)
        if i == 0:
            D[NFFT-i] = 1.0
    Dm=np.matrix(D.T)
    Dinv=np.linalg.pinv(Dm)

    #Inverse of Dm is sensitivity matrix.  Multiply data by this to get transform
    #Found out the hard way that leaving the angular coordinate unspecified 
    #results in the alignment between the two arrays failing. Must supply coord.
    sensitivity=xr.DataArray(Dinv,dims=(wavenumberName,angleName),coords={wavenumberName:range(-NFFT,NFFT+1),angleName:signal[angleName].data})  
    btrf=sensitivity*signal #use dataArray functionality to align toroidal angle
    btrf=btrf.sum(angleName)#with the matrix, and then sum over the axis!  nifty, eh?
    btrf.name=signal.name
    
    return btrf

   
def svd(signal,sampleDimension,sensorDimension):
    """
    Wrapper for singular value decomposition using xarray.DataArray. Warning:
    this method transposes the data, which is non-lazy.
    Assumes that sensor dim length <= sample dim length!
    """
    dimensions = list(signal.dims)
    dimensions.remove(sampleDimension)
    dimensions.remove(sensorDimension)
    dimensionsShort = dimensions[:] #save copy for later
    #svd expects the sensor dimension second to last, sample dimension last
    dimensions+=[sensorDimension,sampleDimension]
    #print dimensions
    #print signal.dims
    transposed = signal.transpose(*dimensions)
    U,S,V = np.linalg.svd(transposed.data,full_matrices=0)
    UI = np.linalg.inv(U) #Assumes that sensor dim length <= sample dim length!
    #this is the annoying part that I'm saving you from having to puzzle out
    dimensionsU = dimensionsShort+[sensorDimension,'singularIndex']
    dimensionsUI = dimensionsShort + ['singularIndex',sensorDimension]
    dimensionsV = dimensionsShort+['singularIndex',sampleDimension]
    coordinates = dict(signal.coords.items())
    data_vars={'sensorMode':(tuple(dimensionsU),U),
               'sensorInverse':(tuple(dimensionsUI),UI),
               'sampleMode':(tuple(dimensionsV),V),
               'singularValue':(('singularIndex',),np.array(S))
            }
    return xr.Dataset(data_vars,coordinates)
    
    
###  Newer versions

def xderiv(y,x=None):
    """
    Take discrete derivative of y (with respect to x) for an xarray.dataArray.
    Arguments:
        y: dataArray to take derivative of
        x: -if y is 1-D and x is None, use the sole dimension of y.
           -if y is multidimensional (or if you wish to), you must supply x as: 
               (A) a string, specifying  the name of a 1-D coordinate of y
               (B) a 1-D dataArray, whose sole dimension matches one of the
                   dimensions of y. 
    Returns:
       applies numpy.gradient to y along specified axis and divides by gradient
       of specified x-variable along same axis. The dataArray will have the same shape as y.
    """
    if y.ndim>1:#try to infer intended dimension from x, if supplied
        if x is None:
            raise Exception("Error: must supply x if y.ndim>1.")
        if type(x)==str:#assume x refers to variable in y
            if x in y.dims:
                axisNumber = y.get_axis_num(x)
                xcoord=getattr(y,x)
            elif x in y.coords:
                xcoord=getattr(y,x)
                if xcoord.ndim>1:
                    raise Exception("Error: x=y.%s has %d dimensions. Supplied x-variable must be one-dimensional."%(x,xcoord.ndim))
                dimension = xcoord.dims[0]#only one dimension by this point
                axisNumber = y.get_axis_num(dimension)
        elif type(x)==xr.core.dataarray.DataArray:#try to infer from x
            if not x.ndim == 1:
                raise Exception("Error: x must be one-dimensional if it is a dataArray")
            else:#can infer dimension from x
                xcoord=x
                axisNumber=y.get_axis_num(xcoord.dims[0])#this still might fail...
        else:
            raise Exception("Unable to infer dimension.")
    elif y.ndim==1:#infer x if not supplied, or if supplied, check that it is one-dimensional
        axisNumber=0
        if x is None:
            xcoord = y.coords[y.dims[axisNumber]] #y only has one dimension at this point, pick out corresponding coordinate
        elif type(x)==str:
            xcoord = getattr(y,x)#it can't be higher-dimensional than y, I'm fairly sure
        elif type(x)==xr.core.dataarray.DataArray:
            assert x.dims[0]==y.dims[0], "Error: x and y don't have the same dimension name."
            xcoord=x 
        else: #TODO: Could maybe extend this to ndarrays? would require change to
              #supply the correct coordinates right at tne end
            raise Exception("Error: type of x must be either string or dataArray.")
        assert len(xcoord.shape)==1, "Error: x must be 1-D."
    else:
        raise Exception("y must have at least one dimension.")
    #need to do this because of the auto-broadcasting ability in the case that
    #y is multidimensional but x is only one-dimensional.  However, it can allow
    #x and y to be outer-producted by mistake if 
    dy = xr.DataArray(np.gradient(y,axis=axisNumber),coords=y.coords)
    dx = xr.DataArray(np.gradient(xcoord,axis=0),coords=xcoord.coords)
    return dy/dx

def xinterp(function,argument,coordinate=None,**kwargs):
    """
    xarray wrapper for np.interp.  Exact same functionality & optional argument
    names.
    Arguments:
        function: needs to be a 1-D xarray.DataArray. 
        argument: either a DataArray or an ndarray.
        coordinate: string specifying name of coordinate to use as basis to 
                    interpolate from (ie, argument will need to be in same units)
                    If unspecified, use coordinate corresponding to sole dimension
                    of 'function'
    Returns:
        DataArray or ndarray with same shape as argument
    """
    assert function.ndim==1,"function must be 1-D."
    if coordinate is None:
        coordinateName=function.dims[0]
        coordinateValue=function.coords[coordinateName].data
    elif type(coordinate)==str:
        coordinateValue=getattr(function,coordinate).data
        coordinateName=coordinate
    else:
        raise Exception("You must supply a string.")
    if type(argument)==xr.core.dataarray.DataArray:
        assert argument.dims[0]==function.dims[0], "If argument is a dataArray, it must have same dimension name as function."
        argument=argument.data
    else:
        argument=np.asanyarray(argument)
    if len(argument.shape)==0:#handle case where a scalar gets turned into an array: you probably just want a scalar back
        return np.interp(argument[()],coordinateValue,function.data,**kwargs)
    newFunction=np.interp(argument,coordinateValue,function.data,**kwargs)
    if type(argument)== xr.core.dataarray.DataArray:
        return xr.DataArray(newFunction,coords=argument.coords,name=function.name)
    else:
        return newFunction

def xinterpInverse(function,argument,coordinate=None,**kwargs):
    """
    xarray wrapper for np.interp.  Exact same functionality & optional argument
    names.
    Arguments:
        function: needs to be a 1-D xarray.DataArray. 
        argument: either a DataArray or an ndarray, 1-D.
        coordinate: string specifying name of coordinate to use as basis to 
                    interpolate from (ie, argument will need to be in same units)
                    If unspecified, use coordinate corresponding to sole dimension
                    of 'function'
    Returns:
        DataArray with 
    """
    assert function.ndim==1,"function must be 1-D."
    if coordinate is None:
        coordinateName=function.dims[0]
        coordinateValue=function.coords[coordinateName].data
    elif type(coordinate)==str:
        coordinateValue=getattr(function,coordinate).data
        coordinateName=coordinate
    else:
        raise Exception("You must supply a string.")
    argumentArray=np.asanyarray(argument)#downcast (or upcast!) to use np.interp       
    if len(argumentArray.shape)==0:#handle case where a scalar gets turned into an array: you probably just want a scalar back
        return np.interp(argumentArray[()],function.data,coordinateValue,**kwargs)
    newFunction=np.interp(argumentArray,function.data,coordinateValue,**kwargs)
    if type(argument)== xr.core.dataarray.DataArray:
        return xr.DataArray(newFunction,coords=argument.coords,name=coordinateName)
    else:
        return newFunction
def xintegral(y,x=None,initial=0):
    """
    xarray wrapper for scipy.cumtrapz
    Arguments:
        y: dataArray to take derivative of
        x: -if y is 1-D and x is None, use the sole dimension of y.
           -if y is multidimensional (or if you wish to), you must supply x as: 
               (A) a string, specifying  the name of a 1-D coordinate of y
               (B) a 1-D dataArray, whose sole dimension matches one of the
                   dimensions of y. 
    Returns:
       applies scipy.cumtrapz to y along specified axis with x as the
       independent variable.  The dataArray will have the same shape as y.
    """
    assert not initial is None, "Do not set initial to 'None' because this will violate shape constancy of output."
    if y.ndim>1:#try to infer intended dimension from x, if supplied
        if x is None:
            raise Exception("Error: must supply x if y.ndim>1.")
        if type(x)==str:#assume x refers to variable in y
            if x in y.dims:
                axisNumber = y.get_axis_num(x)
                xcoord=getattr(y,x)
            elif x in y.coords:
                xcoord=getattr(y,x)
                if xcoord.ndim>1:
                    raise Exception("Error: x=y.%s has %d dimensions. Supplied x-variable must be one-dimensional."%(x,xcoord.ndim))
                dimension = xcoord.dims[0]#only one dimension by this point
                axisNumber = y.get_axis_num(dimension)
        elif type(x)==xr.core.dataarray.DataArray:#try to infer from x
            if not x.ndim == 1:
                raise Exception("Error: x must be one-dimensional if it is a dataArray")
            else:#can infer dimension from x
                xcoord=x
                axisNumber=y.get_axis_num(xcoord.dims[0])#this still might fail...
        else:
            raise Exception("Unable to infer dimension.")
    elif y.ndim==1:#infer x if not supplied, or if supplied, check that it is one-dimensional
        axisNumber=0
        if x is None:
            xcoord = y.coords[y.dims[axisNumber]] #y only has one dimension at this point, pick out corresponding coordinate
        elif type(x)==str:
            xcoord = getattr(y,x)#it can't be higher-dimensional than y, I'm fairly sure
        elif type(x)==xr.core.dataarray.DataArray:
            assert x.dims[0]==y.dims[0], "Error: x and y don't have the same dimension name."
            xcoord=x 
        else: #TODO: Could maybe extend this to ndarrays? would require change to
              #supply the correct coordinates right at tne end
            raise Exception("Error: type of x must be either string or dataArray.")
        assert len(xcoord.shape)==1, "Error: x must be 1-D."
    else:
        raise Exception("y must have at least one dimension.")
    return xr.DataArray(trapezoidalIntegral(y,xcoord,axisNumber,initial=initial),coords=y.coords)

def recarray2xarray(recarray,dim=None,coords=[]):
    """
    Convert a structured numpy array to an xarray dataset.
    If you know which variable you want to use as your dimension, you can provide
    the name as the optional argument 'dim', otherwise it will just be 'dim_0'
    This is useful for the outputs of numpy.genfromtxt and scipy.io.readsav
    which are in recarrays.
    """
    #turn the record array into a dictionary first
    keylist=recarray.dtype.fields.keys()
    coord_dict={}
    dvars={}
    
    if dim in keylist: #name the dimension
        for key in keylist:
            if key in coords:
                coord_dict[key]=((dim),recarray[key])
            else:
                dvars[key]=((dim),recarray[key])
                
    else:#don't name the dimension
        for key in keylist:
            if key in coords:
                coord_dict[key]=recarray[key]
            else:
                dvars[key]=recarray[key]
    return xr.Dataset(data_vars=dvars,coords=coord_dict)