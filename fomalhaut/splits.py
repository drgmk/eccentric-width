# we start with a continuum ms from the pipeline
# called calibrated_final.ms

# a couple of typos in the scriptForImagingPrep.py:
# comment es.generateReducScript on line 38
# fix line 198 to include typo in ms filename (fluxscale -> fluxcale)

# CASA 5.4

# time and spectrally average this down to something more manageable
# there are seven observations
# end result has 4 channels per spw
split(vis='calibrated_final.ms',
      width=[32,32,32,960,32,32,32,960,32,32,32,960,
             32,32,32,960,32,32,32,960,32,32,32,960,32,32,32,960],
      spw='',
      keepflags=False,timebin='30s',field='Fomalhaut',
      datacolumn='data',
      outputvis='calibrated_final_cont_w32_t30.ms')

# do statwt on the continuum ms
statwt('calibrated_final_cont_w32_t30.ms', datacolumn='DATA')

# split out a table per spw per field for modelling, further averaging
# each spw down to a single channel
import uvplot
for i in range(7): # over fields
     for j in range(4): # over spw
         uvplot.io.export_uvtable(
         'uv-field{}-spw{}.txt'.format(i,j), tb,
         vis='calibrated_final_cont_w32_t30.ms',
         split=split, keep_tmp_ms=True, datacolumn='DATA',channel='all',
         split_args={'vis':'calibrated_final_cont_w32_t30.ms',
                     'outputvis':'calibrated_final_t30_field{}_spw{}.ms'.format(i,j),
                     'datacolumn':'DATA', 'field':'{},{}'.format(i,i+7),
                     'spw':'{},{},{},{},{},{},{}'.format(0+j,4+j,8+j,12+j,16+j,20+j,24+j),
                     'width':4, 'keepflags':False
                     }
                                 )

# a quick image for each field, no masking
# there are two field ids for each sky location observed, combine these
for i in range(7):
    tclean(vis='calibrated_final_cont_w32_t30.ms/',imagename='fomalhaut-field{}'.format(i),
          field='{},{}'.format(i,i+7), imsize=[512,512],cell='0.1arcsec',interactive=False)
    exportfits(imagename='fomalhaut-field{}.image'.format(i),fitsimage='fomalhaut-field{}.fits'.format(i))

# proper clean for final image, must use gridder='mosaic'
tclean(vis='calibrated_final_cont_w32_t30.ms',imagename='fomalhaut',imsize=[512,512], cell='0.1arcsec', interactive=True, niter=5000, mask='fom.mask', gridder='mosaic', pbcor=True)
exportfits(imagename='fomalhaut.image',fitsimage='fomalhaut.fits')
exportfits(imagename='fomalhaut.image.pbcor',fitsimage='fomalhaut.pbcor.fits')

# once we've done the modelling, make a residual image
import alma.casa
model = 'multinest_g_full'
for i in range(7):
    for j in range(4):
        alma.casa.residual('../calibrated_final_t30_field{}_spw{}.ms'.format(i,j),
                           'vis-{}-field{}-spw{}.npy'.format(model,i,j),
                           tb, ms_new='residual-field{}-spw{}.ms'.format(i,j),
                           datacolumn='DATA')
#        tclean(vis='residual-field{}-spw{}.ms'.format(i,j),
#               imagename='residual-field{}-spw{}'.format(i,j),
#               imsize=[512,512],cell='0.1arcsec',
#               interactive=False,pbcor=False)
#        exportfits(imagename='residual-field{}-spw{}.image'.format(i,j),
#                   fitsimage='residual-field{}-spw{}.fits'.format(i,j), overwrite=True)
#        alma.casa.model_ms('../calibrated_final_t30_field{}_spw{}.ms'.format(i,j),
#                           'vis-{}-field{}-spw{}.npy'.format(model,i,j),
#                           tb, datacolumn='DATA',
#                           ms_new='model-field{}-spw{}.ms'.format(i,j), remove_new=True)
#        tclean(vis='model-field{}-spw{}.ms'.format(i,j),
#               imagename='model-field{}-spw{}'.format(i,j),
#               imsize=[512,512],cell='0.1arcsec',interactive=False,pbcor=False)
#        exportfits(imagename='model-field{}-spw{}.image'.format(i,j),
#                   fitsimage='model-field{}-spw{}.fits'.format(i,j), overwrite=True)

concat(vis=['residual-field{}-spw{}.ms'.format(i,j) for i in range(7) for j in range(4)],concatvis='residuals.ms', copypointing=False)
tclean(vis='residuals.ms', imagename='residuals', imsize=[512,512], cell='0.1arcsec', interactive=False, niter=0, gridder='mosaic')
exportfits(imagename='residuals.image',fitsimage='residuals.fits')
