
# coding: utf-8

# # HD202628

# In[1]:


import os
import numpy as np
import emcee
import scipy.optimize
import matplotlib.pyplot as plt
import pymultinest as pmn
import corner
import galario.double as gd
from galario import arcsec

import alma.image

#get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


# this may be needed to avoid emcee hanging when using multiple threads
gd.threads(num=1)


# ### Extract the visibilities
# ```python
# statwt(vis='hd202682.ms', datacolumn='DATA')
# import uvplot
# uvplot.io.export_uvtable('uv.txt', tb, vis='hd202628.ms',
#                          datacolumn='DATA',
#                          channel='all')
# ```

# In[3]:


# import the data, this assumes we're getting the output from uvplot
uv_file = 'uv.txt'
u, v, Re, Im, w = np.require( np.loadtxt(uv_file, unpack=True),requirements=["C_CONTIGUOUS"])

# meaning we can get the mean wavelength like so
with open(uv_file) as f:
    _ = f.readline()
    tmp = f.readline()

wavelength = float(tmp.strip().split('=')[1])
#print('wavelength is {} mm'.format(wavelength*1e3))
    
u /= wavelength
v /= wavelength

# estimate re-weighting factor (so that chi^2 for null model would be 1, and d.o.f = 2*len(w))
# weights would need to be multiplied by this number
#reweight_factor = 2*len(w) / np.sum( (Re**2.0 + Im**2.0) * w )
#print('reweighting factor is {}'.format(reweight_factor))


# In[4]:


# set image properties, can alter f_max for higher or lower resolution
nxy, dxy = gd.get_image_size(u, v, verbose=True)
dxy_arcsec = dxy / arcsec


# In[5]:


# decide what density model we want to use
model_name = 'peri_glow'


# In[6]:


# make the image object. by default this is asisymmetric
# (i.e. model='los_image_axisym', and has no anomaly parameter)
ii = alma.image.Image(arcsec_pix=dxy_arcsec, image_size=(nxy, nxy), model='los_image',
                      dens_model=model_name, z_fact=1, wavelength=wavelength, star=True)


# In[7]:


# drop-in function for ii.image, with fixed image parameters
# star_fwhm is small since images are centered on 0,0
def image_func(p):
    image = alma.image.eccentric_ring_image(p[:-3], nxy, dxy_arcsec, n=10000000, star_fwhm=0.1)
    
    # add pt source
    x = np.arange(ii.nx)
    x, y = np.meshgrid(x - (ii.nx-1)/2 - p[-3] / ii.arcsec_pix,
                       x - (ii.nx-1)/2 - p[-2] / ii.arcsec_pix)
    rxy2 = x**2 + y**2
    sigma = 4. / 2.35482
    image += p[-1] * np.exp(-0.5*rxy2/sigma**2) / (2*np.pi*sigma**2)

    return image

# set in Image object
ii.image = lambda p: image_func(p)


# In[8]:


# add point source
ii.params = ii.params + ['$x_p$','$y_p$','$F_p$']
ii.p_ranges = ii.p_ranges + [[-5,-2],[1,3],[1e-5,1e-3]]
ii.n_params += 3


# In[9]:


# add weight factor to parameter list
ii.params += ['$f_{w}$']
ii.p_ranges += [[0,10]]
ii.n_params += 1


# In[10]:


# need finite ranges for multinest
ii.p_ranges[0] = [-1,1]
ii.p_ranges[1] = [-1,1]
ii.p_ranges[2] = [100,160]
ii.p_ranges[3] = [100,200]
ii.p_ranges[4] = [50,70]
ii.p_ranges[5] = [0.0005,0.002]
ii.p_ranges[6] = [2,10]
ii.p_ranges[7] = [0,1]
ii.p_ranges[13] = [0,0.0001]


# In[11]:


# pericentre glow model
p0 = [0.022917028151767577,0.05347781855098957,
      130.53003111507562,149.32877003696768,57.79454189974377,
      0.0012643363554985956,6.653440815847018,0.14113567134196148,
      0.09337039584457309,0.1623220940696366,0.03056157420808084,
      0.05870136614497326,0.14630245760350327,2.5667229030720946e-05,
      -2.875182934899027,1.7896256528679597,0.00021363256681495132, 1.7]

p0 = np.array(p0)

#print('parameters and ranges for {}'.format(model_name))
#for i in range(ii.n_params):
#    print('{}\t{}\t{}\t{}'.format(i,p0[i],ii.p_ranges[i],ii.params[i]))


# In[12]:


# set size of cutout used to generate images, which is based on the
# initial parameters. The tolerance in compute_rmax might be
# varied if the crop size turns out too large. We set 'zero_node'
# to True because we'll generate unrotated images, and let galario
# do the rest
# ii.compute_rmax(np.append([0, 0], p0[14:]), tol=1e-2, expand=10, zero_node=False)

# this gives an idea of how long an mcmc might take
# %timeit ii.image(p0)

# show an image and the primary beam
#im = ii.image(p0[:-1])
#fig,ax = plt.subplots(1,3, figsize=(9.5,5), sharey=True, sharex=True)
#ax[0].imshow(im, origin='bottom', vmax=np.percentile(im, 99.99))
#ax[1].imshow(ii.pb, origin='bottom')
#ax[2].imshow(im*ii.pb, origin='bottom', vmax=np.percentile(im, 99.99))
#fig.tight_layout()


# In[13]:


def lnpostfn(p):
    """ Log of posterior probability function """

    for x,r in zip(p,ii.p_ranges):
        if x < r[0] or x > r[1]:
            return -np.inf

    # galario
    chi2 = gd.chi2Image(ii.image(p[:-1]) * ii.pb, dxy, u, v, Re, Im, w, origin='lower')
    return -0.5 * ( chi2*p[-1] + np.sum(2*np.log(2*np.pi/(w*p[-1]))) )

nlnpostfn = lambda p: -lnpostfn(p)


# In[14]:


# check it works
#lnpostfn(p0)


# ### multinest

# In[15]:


# multinest
pmn_out = 'multinest/'
model_name = pmn_out[:-1]

def mn_prior(cube, ndim, nparam):
    pars = ii.p_ranges
    for i in range(ndim):
        cube[i] = pars[i][0] + cube[i] * (pars[i][1]-pars[i][0])

def mn_lnlike(cube, ndim, nparam):  
    param = np.array([])
    for i in range(ndim):
        param = np.append(param,cube[i])
    return lnpostfn(param)


# In[ ]:


# run it (call python script of this notebook with >nice -11 mpiexec -n 40 python3 vis_model_mn.py)
pmn.run(mn_lnlike, mn_prior, ii.n_params, n_live_points=75, verbose=True,
        outputfiles_basename=pmn_out, multimodal=True)


# In[17]:


# output, start here if multinest was run outside this notebook
#a = pmn.Analyzer(outputfiles_basename=pmn_out, n_params=ii.n_params)
## print(a.get_stats())
#
#p = [a.get_stats()['marginals'][i]['median'] for i in range(ii.n_params)]
## print(p)
#
#for i in range(ii.n_params):
#    print(ii.params[i], '\t', 
#          a.get_stats()['marginals'][i]['median'], '\t',
#          p[i]-a.get_stats()['marginals'][i]['1sigma'][0], '\t',
#          a.get_stats()['marginals'][i]['1sigma'][1]-p[i], '\t',
#          a.get_stats()['marginals'][i]['3sigma'][1])
#
#
## In[14]:
#
#
## corner plot
#d = a.get_data()
#mask = d[:,0] > 1e-8
#fig = corner.corner(d[mask,2:], weights=d[mask,0], labels=ii.params, show_titles=True)
#fig.savefig('{}corner.pdf'.format(pmn_out))
#
#
## ### emcee
#
## In[15]:
#
#
## set up and run mcmc fitting
#ndim = ii.n_params        # number of dimensions
#nwalkers = 36             # number of walkers
#nsteps = 1000              # total number of MCMC steps
#nthreads = 8              # CPU threads that emcee should use
#
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpostfn, threads=nthreads)
#
## initialize the walkers with an ndim-dimensional Gaussian ball
#pos = [p0 + p0*0.01*np.random.randn(ndim) for i in range(nwalkers)]
#
## execute the MCMC
#pos, prob, state = sampler.run_mcmc(pos, nsteps)
#
#
## In[16]:
#
#
#print(sampler.acceptance_fraction)
#
#
## In[18]:
#
#
## save the chains to file
#model_name = 'emcee'
#np.savez_compressed(model_name+'/chains-'+model_name+'.npz', sampler.chain, sampler.lnprobability)
#
#
## In[19]:
#
#
## load chains, start here if emcee was run outside this notebook
#with np.load(model_name+'/chains-'+model_name+'.npz') as data:
#    chain = data['arr_0']
#    lnprobability = data['arr_1']
#
#
## In[20]:
#
#
#nwalkers, nsteps, ndim = chain.shape
#print(chain.shape)
#
#
## In[21]:
#
#
## see what the chains look like, skip a burn in period if desired
#burn = 900
#fig,ax = plt.subplots(ndim+1,2,figsize=(9.5,9),sharex='col',sharey=False)
#
#for j in range(nwalkers):
#    ax[-1,0].plot(lnprobability[j,:burn])
#    for i in range(ndim):
#        ax[i,0].plot(chain[j,:burn,i])
#        ax[i,0].set_ylabel(ii.params[i])
#
#for j in range(nwalkers):
#    ax[-1,1].plot(lnprobability[j,burn:])
#    for i in range(ndim):
#        ax[i,1].plot(chain[j,burn:,i])
#        ax[i,1].set_ylabel(ii.params[i])
#
#ax[-1,0].set_xlabel('burn in')
#ax[-1,1].set_xlabel('sampling')
#fig.savefig(model_name+'/chains-'+model_name+'.pdf')
#
#
## In[22]:
#
#
## make the corner plot
#fig = corner.corner(sampler.chain[:,burn:,:].reshape((-1,ndim)), labels=ii.params,
#                    show_titles=True)
#
#fig.savefig(model_name+'/corner-'+model_name+'.png')
#
#
## In[23]:
#
#
## get the median parameters
#p = np.median(sampler.chain[:,burn:,:].reshape((-1,ndim)),axis=0)
#s = np.std(sampler.chain[:,burn:,:].reshape((-1,ndim)),axis=0)
#print(','.join(p.astype(str)))
#print(s)
#
#
## ### post-fitting stuff (regardless of fitting method)
#
## In[24]:
#
#
## see what it looks like
#im = ii.image(p[:-1])
#fig,ax = plt.subplots()
#ax.imshow(im, origin='bottom', vmax=np.percentile(im, 99.9))
#fig.tight_layout()
#fig.savefig(model_name+'/best-'+model_name+'.pdf', dpi=500)
#
#
## In[25]:
#
#
## save the visibilities for subtraction from the data
#vis_mod = gd.sampleImage(ii.pb * ii.image(p), dxy, u, v, origin='lower')
#np.save(model_name+'/vis-'+model_name+'.npy', vis_mod)
#
#
## In[26]:
#
#
## just point source
#_ = p.copy()
#_[5]=0
#vis_mod = gd.sampleImage(ii.pb * ii.image(_), dxy, u, v, origin='lower')
#np.save(model_name+'/vis-ptsrc.npy', vis_mod)
#
#
## ## Creating a map of the residuals
## This must be done within CASA. First the script 'residual' is run to subtract the model visibilities created above from the ms we made at the top.
## ```python
## import alma.casa
## alma.casa.residual('../../../../data/alma/hd202628/hd202628.ms',
##                    'vis-multinest.npy', tb, datacolumn='DATA')
## tclean(vis='residual.ms', imagename='residual', imsize=[512,512],
##        cell='0.05arcsec', interactive=False, niter=0)
## exportfits(imagename='residual.image',fitsimage='residual.fits')
## alma.casa.residual('../../../../data/alma/hd202628/hd202628.ms',
##                    'vis-ptsrc.npy', tb, datacolumn='DATA',ms_new='residual-ptsrc.ms')
## tclean(vis='residual-ptsrc.ms', imagename='residual-ptsrc',
##        imsize=[512,512], cell='0.05arcsec', interactive=False, niter=0)
## exportfits(imagename='residual-ptsrc.image',fitsimage='residual-ptsrc.fits')
## ```
