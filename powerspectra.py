import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import yt
yt.mylog.setLevel("ERROR")
import warnings
warnings.filterwarnings('ignore')
from yt.units import kpc
from scipy.stats import binned_statistics
from scipy import fft
from astropy.convolution import convolve_fft, Gaussian2DKernel
from scipy.optimize import minimize
import emcee
import pandas as pd

def projection(ds, width, res):
    emis, l, pemis=yt.add_xray_emissivity_field(ds, 0.5, 3.5,
                                        table_type='apec', metallicity=.5)
    proj=ds.proj(emis, axis='y')
    # This performs the projection by summing over the entire domain.
    frb=proj.to_frb((width,'kpc'), [res,res], center=[0,0,0])
    # This applies a Fixed Resolution Buffer, which specifies a resolution
    # and changes the simulation's AMR grid to fit that resolution.
    arr=np.array(frb[emis])
    return arr

def projection_no_grid(ds, level=0, direction=1):
    # The SCG has to be perfectly aligned with the AMR grid, so we can no
    # longer specify any resolution we want. We have to specify the maximum
    # level of AMR cell to include. In these simulations, level 4 gives
    # a 512x512 image for the entire domain.
    ds.periodicity=(True,True,True)
    #This must be set, otherwise the SCG overlapping with the domain edge
    #is going to cause an error.
    l_edge=ds.domain_left_edge
    #This is the most negative point in the simulation in three 
    #dimensions. In this case it is [-250, -250, -500] kpc
    emis, l, pemis= yt.add_xray_emissivity_field(ds, 0.5, 3.5,
                                        table_type='apec', metallicity=.5)
    dims=ds.domain_dimensions
    scg=ds.smoothed_covering_grid(level, l_edge, dims*2**(level))
    # This is the smoothed covering grid object, which is currently in 3D.
    # To project over it, we'll sum over the specified axis. The direction
    # argument specifies the axis, with 0:x, 1:y, 2:z
    proj=np.array(np.sum(scg[emis], axis=direction))
    return proj

def concatenating_projection(ds, level=0, direction=2):
    # First we make an empty list of lists to store each octant.
    # The list is 2x2x2, just as our divided domain is.
    projections=[[['#' for n in range(2)] for n in range(2)] for n in range(2)]
    ds.periodicity=(True, True, True)
    # This sets the boundary conditions for the simulation.
    # This must be included otherwise the SCG will be outside the simulation domain.
    emis, lumis, pemis = yt.add_xray_emissivity_field(ds, 0.5, 3.5,
                                        table_type='apec', metallicity=0.5)
    l_edge=ds.domain_left_edge
    dimensions=ds.domain_dimensions
    # The below for loop does a lot of the heavy lifting here.
    # We change each coordinate of the left edge to select the appropriate octant.
    for i in range(2):
        for j in range(2):
            for k in range(2):
                scg=ds.smoothed_covering_grid(level,
                                [l_edge[0]*i, l_edge[1]*j, l_edge[2]*k], 
                                dimensions*2**(level-1))
                proj=np.array(np.sum(scg[emis], axis=direction))
                projections[i][j][k]=proj
    # Now we add together octants along the sight line and concatenate the
    # adjacent octants. This has to be done differently for each projection
    # axis, which has been worked out and programmed below.
    if direction==2:
        top=np.concatenate((projections[1][1][1]+projections[1][1][0],
                            projections[1][0][1]+projections[1][0][0]), axis=1)
        bot=np.concatenate((projections[0][1][1]+projections[0][1][0], 
                            projections[0][0][1]+projections[0][0][0]), axis=1)
        proj=np.concatenate((top,bot), axis=0)
        return proj
    if direction==1:
        top=np.concatenate((projections[1][1][1]+projections[1][0][1], 
                            projections[1][1][0]+projections[1][0][0]), axis=1)
        bot=np.concatenate((projections[0][1][1]+projections[0][0][1], 
                            projections[0][1][0]+projections[0][0][0]), axis=1)
        proj=np.concatenate((top,bot), axis=0)
        return proj 
    if direction==0:
        top=np.concatenate((projections[1][1][1]+projections[0][1][1], 
                            projections[1][1][0]+projections[0][1][0]), axis=1)
        bot=np.concatenate((projections[1][0][1]+projections[0][0][1], 
                            projections[1][0][0]+projections[0][0][0]), axis=1)
        proj=np.concatenate((top,bot), axis=0)
        return proj
    
def get_power_spectrum_fft(img):
    img_k = np.fft.fft2(img)
    kx, ky = np.fft.fftfreq(img.shape[0]), np.fft.fftfreq(img.shape[1])
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2+ky**2)
    stat = binned_statistic(k.flatten(), np.abs(img_k.flatten())**2, bins=32, statistic='mean')
    bin_center = (stat.bin_edges[:-1]+stat.bin_edges[1:])/2
    return bin_center, stat.statistic

def mask_annulus(img, inrad, outrad, width=None):
    # I include the width argument such that we can input the inner radius 
    # and outer radius of the image in kpc when we apply the mask to our 
    # simulation images later. The below if statement will handle the 
    # conversion to pixels.
    if width!=None:
        scale=img.shape[0]/width
        inrad=inrad*scale
        outrad=outrad*scale
    X, Y=np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    center=[img.shape[0]/2, img.shape[1]/2]
    # We can use the numpy logic functions to create the mask 
    mask = np.logical_and(((X-center[1])**2+(Y-center[0])**2)>inrad**2,
                          ((X-center[1])**2+(Y-center[0])**2)<outrad**2)
    # Now we apply the mask to the image.
    masked_img=img.copy()
    masked_img[~mask]=0
    return masked_img, mask

def mexican_hat(img, k):
    # In practice, we use an epsilon of 10^-3. Smaller is more accurate,
    # however there are diminishing returns after a point.
    epsilon=.001
    sigma=0.225/k
    
    G1 = Gaussian2DKernel(sigma/np.sqrt(1+epsilon))
    G2 = Gaussian2DKernel(sigma*np.sqrt(1+epsilon))
    F=G1-G2
    #We take our filter and convolve it with the power-law generated image
    convolved_img= convolve_fft(img, F,  normalize_kernel=False)
    return convolved_img

def get_power_spectrum_variance(img, ks, mask=None):
    #ks will be an array of our desired wavefactors in units of 1/pixel
    P=np.zeros(len(ks))
    epsilon= 0.001
    if mask is None or mask.shape != img.shape:
        mask=np.ones(img.shape)
    #We will use a for loop to iterate through ks and calculate the power
    for i, k in enumerate(ks):
        sigma= 0.225/k
        
        G1=Gaussian2DKernel(sigma/np.sqrt(1+epsilon))
        #Here we have G1*I and G1*M
        convolved_img1 = convolve_fft(img, G1)
        convolved_mask1 = convolve_fft(mask, G1)
        
        G2 = Gaussian2DKernel(sigma*np.sqrt(1+epsilon))
        #and here is G2*I and G2*M
        convolved_img2 = convolve_fft(img, G2)
        convolved_mask2 = convolve_fft(mask, G2)
        S=np.nan_to_num(convolved_img1/convolved_mask1-convolved_img2/convolved_mask2)*mask
        # Vk has to also be normalized by the area of the original
        # image/the area of the mask.
        Vk=np.sum(np.ones(img.shape))/np.sum(mask)*np.sum(S**2)
        P[i]=Vk/(epsilon**2*k**2*np.pi)
    return P

def beta_normalize(proj, width=500, nsteps=25000):
    # First we can retrieve the radial data and dimensions, as above.
    radial_prof=radial_data(proj)
    r=radial_prof.r
    s_mean=radial_prof.mean
    s_std=radial_prof.std
    dim=proj.shape
    # We can define a function that will compare our radial data to our
    # model, and return the likelihood that this model is correct.
    def log_likelihood(theta, r, s, serr):
        s0, beta, Rc, log_f=theta
        if Rc<0 or 3*beta<0.5:
            return -np.inf
        model=s0*(1+(r/Rc)**2)**(-3*beta+0.5)
        sigma2=serr**2+model**2*np.exp(2*log_f)
        return -0.5*np.sum((s-model)**2/sigma2 + np.log(sigma2))
    # The variable theta in this function will contain our parameters
    # We can now use this to find some starting values for the MCMC.
    nll= lambda *args: -log_likelihood(*args)
    initial=np.array([s_mean.max(), 0.53, dim[0]/width*26, np.log(.1)])
    soln=minimize(nll, initial, args=(r,s_mean,s_std)) 
    # The minimize function will step through each argument to find values
    # for our parameters in the soln variable. We also give an initial
    # guess, which is based on the simulation's initial beta model 
    # We now make a function to include priors, which are any hard limits
    # on the parameters. This isn't especially important in this 
    # application, however we will certainly use this later on.
    # For now we want to make sure our core radius is positive, and our
    # exponent remains negative.
    def log_prior(theta):
        s0, beta, Rc, log_f=theta
        if Rc>0 and 3*beta>0.5:
            return 0
        else:
            return -np.inf
    def log_probability(theta, r, s, serr):
        lp=log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp+log_likelihood(theta,r,s, serr)
    # This function essentially makes it so that if either prior is
    # violated, the likelihood of that combination of parameters becomes 
    # negative infinity, which the MCMC algorithm will throw out.
    pos = soln.x+ 1e-10* np.random.randn(32, 4)
    nwalkers,ndim=pos.shape
    sampler=emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                  args=(r,s_mean,s_std))
    sampler.run_mcmc(pos, nsteps, progress=True)
    flat_samples = sampler.get_chain(discard=500,thin=200,flat=True)
    mcs0, mcbeta, mcRc, mclogf= tuple([np.percentile(flat_samples[:, i], 50) for i in range(ndim)])
    # Now we have run the MCMC algorithm and it has found the most likely
    # values for each parameter. Let's pack them up in a dict.
    beta_params={'s0':mcs0,'beta':mcbeta, 'Rc': mcRc, 'logf':mclogf}
    # We can now create an image that follows the model, and divide
    # the original image by the beta model image.
    y=np.linspace(-dim[0]/2, dim[0]/2, dim[0])
    x=np.linspace(-dim[1]/2, dim[1]/2, dim[1])
    X, Y= np.meshgrid(x,y)
    r2=X**2+Y**2
    beta=mcs0*(1+r2/(mcRc**2))**(-3*mcbeta+0.5)
    normalized_proj=proj/beta
    
    return normalized_proj, beta_params, soln

def model(r, params):
    s0=params['s0']
    beta=params['beta']
    Rc=params['Rc']
    return s0*(1+(r/Rc)**2)**(-3*beta+0.5)

def get_amplitude_spectrum(img, kkpc, width=500, Rc=26, mask=None):
    # This function is essentially identical to the variance method, 
    # with the final conversion described above. It also has parameters for
    # using physical units (kpc) instead of pixels. Our array of k-values is 
    # now input in units of 1/kpc, while the width and core radius are in kpc.
    A = np.zeros(len(kkpc))
    scale=img.shape[0]/width
    ks=kkpc/scale
    epsilon = 0.001
    for i, k in enumerate(ks):
        sigma = 0.225/k
        if mask is None or mask.shape != img.shape:
            mask = np.ones(img.shape)

        G1 = Gaussian2DKernel(sigma/np.sqrt(1+epsilon))
        convolved_img1 = convolve_fft(img, G1)
        convolved_mask1 = convolve_fft(mask, G1)
        G2 = Gaussian2DKernel(sigma*np.sqrt(1+epsilon))
        convolved_img2 = convolve_fft(img, G2)
        convolved_mask2 = convolve_fft(mask, G2)
        S = np.nan_to_num((convolved_img1/convolved_mask1-convolved_img2/convolved_mask2)*mask)
        Vk = np.sum(np.ones(img.shape))/np.sum(mask)*np.sum(S**2)
        
        P = Vk/epsilon**2/k**2/np.pi
        A[i]= np.sqrt(P*k**3/(Rc*scale))
    return A

def concatenate_data(key, max_ann=2):
    data=[]
    for a in range(max_ann):
        annulus_data=pd.read_csv('ann'+str(a+1)+'.csv')
        spectrum=annulus_data[key].to_numpy()
        data.append(spectrum)
    return np.concatenate(tuple(data))

def radial_data(data,annulus_width=1,working_mask=None,x=None,y=None,rmax=None):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)
    
    A function to reduce an image to a radial cross-section.
    
    :INPUT:
      data   - whatever data you are radially averaging.  Data is
              binned into a series of annuli of width 'annulus_width'
              pixels.

      annulus_width - width of each annulus.  Default is 1.

      working_mask - array of same size as 'data', with zeros at
                        whichever 'data' points you don't want included
                        in the radial data computations.

      x,y - coordinate system in which the data exists (used to set
               the center of the data).  By default, these are set to
               integer meshgrids

      rmax -- maximum radial value over which to compute statistics
    
    :OUTPUT:
        r - a data structure containing the following
                   statistics, computed across each annulus:

          .r      - the radial coordinate used (outer edge of annulus)

          .mean   - mean of the data in the annulus

          .sum    - the sum of all enclosed values at the given radius

          .std    - standard deviation of the data in the annulus

          .median - median value in the annulus

          .max    - maximum value in the annulus

          .min    - minimum value in the annulus

          .numel  - number of elements in the annulus

    :EXAMPLE:        
      ::
        
        import numpy as np
        import pylab as py
        import radial_data as rad

        # Create coordinate grid
        npix = 50.
        x = np.arange(npix) - npix/2.
        xx, yy = np.meshgrid(x, x)
        r = np.sqrt(xx**2 + yy**2)
        fake_psf = np.exp(-(r/5.)**2)
        noise = 0.1 * np.random.normal(0, 1, r.size).reshape(r.shape)
        simulation = fake_psf + noise

        rad_stats = rad.radial_data(simulation, x=xx, y=yy)

        py.figure()
        py.plot(rad_stats.r, rad_stats.mean / rad_stats.std)
        py.xlabel('Radial coordinate')
        py.ylabel('Signal to Noise')
    """
    
# 2012-02-25 20:40 IJMC: Empty bins now have numel=0, not nan.
# 2012-02-04 17:41 IJMC: Added "SUM" flag
# 2010-11-19 16:36 IJC: Updated documentation for Sphinx
# 2010-03-10 19:22 IJC: Ported to python from Matlab
# 2005/12/19 Added 'working_region' option (IJC)
# 2005/12/15 Switched order of outputs (IJC)
# 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
# 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory
 
    import numpy as ny

    class radialDat:
        """Empty object container.
        """
        def __init__(self): 
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None

    #---------------------
    # Set up input parameters
    #---------------------
    data = ny.array(data)
    
    if working_mask==None:
        working_mask = ny.ones(data.shape,bool)
    
    npix, npiy = data.shape
    if x==None or y==None:
        x1 = ny.arange(-npix/2.,npix/2.)
        y1 = ny.arange(-npiy/2.,npiy/2.)
        x,y = ny.meshgrid(y1,x1)

    r = abs(x+1j*y)

    if rmax==None:
        rmax = r[working_mask].max()

    #---------------------
    # Prepare the data container
    #---------------------
    dr = ny.abs([x[0,0] - x[0,1]]) * annulus_width
    radial = ny.arange(rmax/dr)*dr + dr/2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = ny.zeros(nrad)
    radialdata.sum = ny.zeros(nrad)
    radialdata.std = ny.zeros(nrad)
    radialdata.median = ny.zeros(nrad)
    radialdata.numel = ny.zeros(nrad, dtype=int)
    radialdata.max = ny.zeros(nrad)
    radialdata.min = ny.zeros(nrad)
    radialdata.r = radial
    
    #---------------------
    # Loop through the bins
    #---------------------
    for irad in range(nrad): #= 1:numel(radial)
      minrad = irad*dr
      maxrad = minrad + dr
      thisindex = (r>=minrad) * (r<maxrad) * working_mask
      #import pylab as py
      #pdb.set_trace()
      if not thisindex.ravel().any():
        radialdata.mean[irad] = ny.nan
        radialdata.sum[irad] = ny.nan
        radialdata.std[irad]  = ny.nan
        radialdata.median[irad] = ny.nan
        radialdata.numel[irad] = 0
        radialdata.max[irad] = ny.nan
        radialdata.min[irad] = ny.nan
      else:
        radialdata.mean[irad] = data[thisindex].mean()
        radialdata.sum[irad] = data[r<maxrad].sum()
        radialdata.std[irad]  = data[thisindex].std()
        radialdata.median[irad] = ny.median(data[thisindex])
        radialdata.numel[irad] = data[thisindex].size
        radialdata.max[irad] = data[thisindex].max()
        radialdata.min[irad] = data[thisindex].min()
    
    #---------------------
    # Return with data
    #---------------------
    
    return radialdata