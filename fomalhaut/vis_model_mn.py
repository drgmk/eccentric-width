
# coding: utf-8

# # Fomalhaut A's vertical structure
# Multiple pointings... See splits.py for splits, statwt, and uv table creation

# In[1]:


import os
import numpy as np
import emcee
import scipy.optimize
import scipy.signal
import matplotlib.pyplot as plt
import corner
import pymultinest as pmn
import galario.double as gd
from galario import arcsec

import alma.image

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


# this may be needed to avoid emcee hanging when using multiple threads
gd.threads(num=1)


# In[3]:


# import the data, this assumes we're getting the output from uvplot
uvdata = []
all_weights = np.array([])
wavelength = []
fw = []
nfield = 7
nspw = 4
for i in range(nfield):
    for j in range(nspw):
    
        f = 'uv-field{}-spw{}.txt'.format(i,j)

        u, v, Re, Im, w = np.require( np.loadtxt(f, unpack=True),requirements=["C_CONTIGUOUS"])

        # meaning we can get the mean wavelength like so
        with open(f) as tmp:
            _ = tmp.readline()
            tmp = tmp.readline()

        wavelength_tmp = float(tmp.strip().split('=')[1])    
        u /= wavelength_tmp
        v /= wavelength_tmp
        wavelength.append(wavelength_tmp)

        # estimate re-weighting factor (so that chi^2 for null model would be 1, and d.o.f = 2*len(w))
        # weights would need to be multiplied by this number
        fw.append( 2*len(w) / np.sum( (Re**2.0 + Im**2.0) * w ) )

#        print('{}, {} rows, \twave {:9.7f}mm,'
#              '\treweight factor {:g}'.format(os.path.basename(f),len(w),
#                                              wavelength_tmp*1e3,
#                                              fw[-1]))

        uvdata.append( (u, v, Re, Im, w) )
        all_weights = np.append(all_weights, w)


# In[4]:


# set image properties, take greatest resolution needed
nxy = 0
dxy = 1
for vis in uvdata:
    u, v, _, _, _ = vis
    nxy_tmp, dxy_tmp = gd.get_image_size(u, v, verbose=False)
    if nxy_tmp > nxy and dxy_tmp < dxy:
        nxy = nxy_tmp
        dxy = dxy_tmp
        
dxy_arcsec = dxy / arcsec
#print('Final nxy:{}, dxy:{}, dxy arcsec:{}'.format(nxy, dxy, dxy_arcsec))


# In[5]:


# decide what density model we want to use
model_name = 'peri_glow'


# In[6]:


# make the image object, one will be used for all fields
# use an empirical pb from CASA, since it may matter here
ii = alma.image.Image(arcsec_pix=dxy_arcsec, image_size=(nxy, nxy), model='los_image',
                      dens_model=model_name, z_fact=1, wavelength=wavelength[0],
                      star=True, pb_fits='tmp.pb.fits')


# In[7]:


# drop-in function for ii.image, with fixed image parameters
# star_fwhm is small since images are centered on 0,0
ii.image = lambda p: alma.image.eccentric_ring_image(p, nxy, dxy_arcsec, n=10000000, star_fwhm=0.1, da_gauss=False)


# In[8]:


# create offset primary beam
def get_pb(ii, x0, y0):
    return ii.primary_beam_image(x0=x0, y0=y0)


# In[9]:


# add weight factor to parameter list
ii.params += ['$f_{w}$']
ii.p_ranges += [[0,10]]
ii.n_params += 1


# In[10]:


# need finite ranges for multinest, modify for problem at hand
ii.p_ranges[0] = [-0.1,0.1]
ii.p_ranges[1] = [-0.1,0.1]
ii.p_ranges[2] = [140,170]
ii.p_ranges[3] = [-20,60]
ii.p_ranges[4] = [50,80]
ii.p_ranges[5] = [0.01,0.1]
ii.p_ranges[6] = [10,25]
ii.p_ranges[7] = [0,3]
#ii.p_ranges[9] = [0,0.005]   #
#ii.p_ranges[11] = [0,0.005]  # for flat MacGregor model
#ii.p_ranges[12] = [0,0.005]  #
ii.p_ranges[13] = [0,0.0015]


# In[11]:


# offsets in x, y (i.e. -ra, dec), delta RA are (a-b)*15*cos(dec)
# we assume the pointing is good and that these are fixed
# (but can all move relative to some center together)
off = [0.0, 0.0,                # field 0
       -8.898409, -19.98620,    # field 1 relative to 0
       +8.898396, +19.98620,    # 2 relative etc.
       +10.22100,  +7.42670,    # 3
        -1.32100, +12.56580,    # 4
        +1.32100, -10.93879,    # 5
       -10.22100,  -7.42670]    # 6

# pericentre glow model
p0 = [-0.06542785156837794,0.048660868203270105,156.37496611018472,40.74382055684831,66.64319590251502,
      0.026595648373342343,18.173138876150325,1.5,0.1238431709670846,0.0025,
      0.06,0.0025,0.0025,0.000673553452336449, 1.1]

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
#im = ii.image(p0)
#fig,ax = plt.subplots(1,3, figsize=(9.5,5), sharey=True, sharex=True)
#ax[0].imshow(im, origin='bottom', vmax=np.percentile(im, 99.9))
#ax[1].imshow(get_pb(ii,0,0), origin='bottom')
#ax[2].imshow(im*get_pb(ii,0,0), origin='bottom', vmax=np.percentile(im, 99.9))
#fig.tight_layout()


# In[13]:


# sanity check on field offsets
#fig,ax = plt.subplots(2,4, figsize=(9.5,5), sharey=True, sharex=True)
#for i in range(nfield):
#    tmp = np.append([(p0[0]-off[2*i]), (p0[1]-off[2*i+1])],p0[2:])
#    im = ii.image(tmp)
#    ax[np.unravel_index(i,ax.shape)].imshow(im * get_pb(ii,0,0), origin='bottom', vmax=np.percentile(im,99))
#    ax[np.unravel_index(i,ax.shape)].contour(im, origin='lower', alpha=0.5)
#    ax[np.unravel_index(i,ax.shape)].set_title('field {}'.format(i))
#    ax[np.unravel_index(i,ax.shape)].plot(nxy/2, nxy/2, '+')
#
#fig.tight_layout()


# In[14]:


# sanity check on pb offsets
#fig,ax = plt.subplots(2,4, figsize=(9.5,5), sharey=True, sharex=True)
#tmp = np.append([0,0],p0[2:])
#image = ii.image(tmp)
#for i in range(nfield):
#    x0, y0 = p0[0]-off[2*i], p0[1]-off[2*i+1]
#    pb = get_pb(ii, -x0, -y0)
#    ax[np.unravel_index(i,ax.shape)].imshow(image * pb, origin='bottom', vmax=np.percentile(image,99))
#    ax[np.unravel_index(i,ax.shape)].set_title('field {}:{:5.2f},{:5.2f}'.format(i,x0,y0))
#    ax[np.unravel_index(i,ax.shape)].plot(nxy/2-x0/dxy_arcsec, nxy/2-y0/dxy_arcsec, '+w')
#
#fig.tight_layout()


# In[21]:


def lnpostfn(p):
    """ Log of posterior probability function """

    for x,r in zip(p,ii.p_ranges):
        if x < r[0] or x > r[1]:
            return -np.inf

    # images, star-centered with offset primary beam (i.e. second e.g. above), shifted by galario
    chi2 = 0.0
    tmp = np.append([0,0],p[2:-1])
    image = ii.image(tmp)
    for i in range(nfield):
        x0, y0 = p[0]-off[2*i], p[1]-off[2*i+1]
        pb = get_pb(ii, -x0, -y0)
        for j in range(nspw):
            u, v, Re, Im, w = uvdata[i*nspw + j]
            chi2_tmp = gd.chi2Image(image*pb , dxy, u, v, Re, Im, w, origin='lower',
                                    dRA = -x0*arcsec, dDec = y0*arcsec)
            chi2 += chi2_tmp
        
    # we include a weight factor to force reasonable uncertainties
    return -0.5 * ( chi2*p[-1] + np.sum(2*np.log(2*np.pi/(all_weights*p[-1]))) )

nlnpostfn = lambda p: -lnpostfn(p)


# In[22]:


# check it works
#lnpostfn(p0)


# ### multinest fitting

# In[15]:


# where results go
pmn_out = 'multinest-da-full/'
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


# run it (call python script of this notebook with >nice -5 mpiexec -n 40 python3 vis_model.py)
pmn.run(mn_lnlike, mn_prior, ii.n_params, n_live_points=75, verbose=True,
        outputfiles_basename=pmn_out, multimodal=True)


# In[16]:


# output, start here if multinest was run outside this notebook
#a = pmn.Analyzer(outputfiles_basename=pmn_out, n_params=ii.n_params)
## print(a.get_stats())
#
#p = [a.get_stats()['marginals'][i]['median'] for i in range(ii.n_params)]
#print(p)
#
#for i in range(ii.n_params):
#    print(ii.params[i], '\t', 
#          a.get_stats()['marginals'][i]['median'], '\t',
#          p[i]-a.get_stats()['marginals'][i]['1sigma'][0], '\t',
#          a.get_stats()['marginals'][i]['1sigma'][1]-p[i], '\t',
#          a.get_stats()['marginals'][i]['3sigma'][1])
#
#
## In[17]:
#
#
## corner plot
#d = a.get_data()
#mask = d[:,0] > 1e-8
#fig = corner.corner(d[mask,2:], weights=d[mask,0], labels=ii.params, show_titles=True)
#fig.savefig('{}corner.pdf'.format(pmn_out))
#
#
## ### emcee fitting
#
## In[25]:
#
#
## set up and run mcmc fitting
#ndim = ii.n_params        # number of dimensions
#nwalkers = 28             # number of walkers
#nsteps = 10              # total number of MCMC steps
#nthreads = 4              # CPU threads that emcee should use
#
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpostfn, threads=nthreads)
#
## initialize the walkers with an ndim-dimensional Gaussian ball
#pos = [p0 + p0*0.05*np.random.randn(ndim) for i in range(nwalkers)]
#
## execute the MCMC
#pos, prob, state = sampler.run_mcmc(pos, nsteps)
#
#
## In[27]:
#
#
#print(sampler.acceptance_fraction)
#
#
## In[29]:
#
#
## save the chains to file
#model_name = 'emcee_full'
#np.savez_compressed(model_name+'/chains-'+model_name+'.npz', sampler.chain, sampler.lnprobability)
#
#
## In[16]:
#
#
## load chains, start here if emcee was run outside this notebook
#with np.load(model_name+'/chains-'+model_name+'.npz') as data:
#    chain = data['arr_0']
#    lnprobability = data['arr_1']
#
#
## In[17]:
#
#
#nwalkers, nsteps, ndim = chain.shape
#print(chain.shape)
#
#
## In[18]:
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
## In[19]:
#
#
## make the corner plot
#fig = corner.corner(chain[:,burn:,:].reshape((-1,ndim)), labels=ii.params,
#                    show_titles=True)
#
#fig.savefig(model_name+'/corner-'+model_name+'.pdf')
#
#
## In[19]:
#
#
## get the median parameters
#p = np.median(chain[:,burn:,:].reshape((-1,ndim)),axis=0)
#s = np.std(chain[:,burn:,:].reshape((-1,ndim)),axis=0)
#print(','.join(p.astype(str)))
#print(s)
#
#
## ### post-fitting stuff (regardless of fitting method)
#
## In[20]:
#
#
## see what it looks like
#im = ii.image(p)
#fig,ax = plt.subplots()
#ax.imshow(im, origin='bottom', vmax=np.percentile(im, 99.9))
#fig.tight_layout()
#fig.savefig(model_name+'/best-'+model_name+'.pdf', dpi=500)
#
#
## In[21]:
#
#
## save the visibilities for subtraction from the data
#tmp = np.append([0,0],p[2:])
#image = ii.image(tmp)
#for i in range(nfield):
#    x0, y0 = p[0]-off[2*i], p[1]-off[2*i+1]
#    pb = get_pb(ii, -x0, -y0)
#    for j in range(nspw):
#        u, v, Re, Im, w = uvdata[i*nspw + j]
#        vis_mod = gd.sampleImage(image * pb, dxy, u, v, origin='lower', dRA = -x0*arcsec, dDec = y0*arcsec)
#        np.save(model_name+'/vis-{}-field{}-spw{}.npy'.format(model_name, i, j), vis_mod)
#
#
## ## Creating a map of the residuals
## See splits.py
