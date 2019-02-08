
import numpy as np
import fsps
import sedpy
from time import time
import prospect

from sedpy.observate import load_filters
from prospect.models.templates import TemplateLibrary

def load_sps(zcontinuous=1, **extras):
    """Instantiate and return the Stellar Population Synthesis object.

    :param zcontinuous: (default: 1)
        python-fsps parameter controlling how metallicity interpolation of the
        SSPs is acheived.  A value of `1` is recommended.
        * 0: use discrete indices (controlled by parameter "zmet")
        * 1: linearly interpolate in log Z/Z_\sun to the target
             metallicity (the parameter "logzsol".)
        * 2: convolve with a metallicity distribution function at each age.
             The MDF is controlled by the parameter "pmetals"
    """
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps

def load_model(add_dust=False, add_neb=False, add_zred=True, add_burst=False, **extras):

    from prospect.models import priors, SedModel
    from prospect.models.templates import TemplateLibrary

    #TemplateLibrary.show_contents()
    #TemplateLibrary.describe("alpha")
    model_params = TemplateLibrary["parametric_sfh"]

    if add_zred:
        model_params['zred']['isfree'] = True
        model_params["zred"]["prior"] = priors.TopHat(mini=0.01, maxi=2.0)

    if add_burst:
        model_params.update(TemplateLibrary["burst_sfh"])
        model_params['fage_burst']['isfree'] = True

    if add_dust:
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_umin']['isfree'] = True
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_gamma']['isfree'] = True

    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logz']['isfree'] = True
        model_params['gas_logu']['isfree'] = True

    return SedModel(model_params)

class Sedmodel:

    def __init__(self, add_dust=False, add_neb=False, add_zred=True, add_burst=False):

        run_params = {}
        run_params['add_dust'] = add_dust
        run_params['add_neb'] = add_neb
        run_params['add_zred'] = add_zred
        run_params['add_burst'] = add_burst
        run_params["fixed_metallicity"] = None
        run_params["zcontinuous"] = 1
        run_params['filterset'] = ['sdss_{0}0'.format(b) for b in ['u','g','r','i','z']]

        self.model = load_model(**run_params)
        self.sps = load_sps(**run_params)

        self.num_dim = self.model.ndim
        self.num_filters = len(run_params['filterset'])
        print("Number of dimensions:", self.num_dim)

        from prospect.utils.obsutils import fix_obs
        self.obs = {}
        self.obs['filters'] = load_filters(run_params['filterset'])
        self.obs["phot_wave"] = [f.wave_effective for f in self.obs["filters"]]
        self.obs['wavelength'] = None
        self.obs["spectrum"] = None
        self.obs = fix_obs(self.obs)

        print("\nFilters:\n  {}".format(self.obs["filters"]))
        print("\nFree Parameters:\n  {}".format(self.model.free_params))
        print("\nFixed Parameters:\n  {}".format(self.model.fixed_params))

    def compute_photsed_grid(self, u_grid, plot_sed=True):

        num_samples = u_grid.shape[0]
        theta_grid = np.zeros_like(u_grid)
        phot_grid = np.zeros((num_samples, len(self.obs['filters'])))
        times = []

        if plot_sed:
            import matplotlib.pyplot as plt
            wspec = self.sps.wavelengths

        for i in range(num_samples):
            u = u_grid[i, :]
            theta = self.model.prior_transform(u)
            theta_grid[i, :] = theta

            t1 = time()
            spec, phot, mfrac = self.model.mean_model(theta, self.obs, sps=self.sps)
            phot_grid[i, :] = phot
            t2 = time()
            times.append(t2-t1)

            title_text = ', '.join([p+"=%.2g" % self.model.params[p][0] for p in self.model.free_params])\
                                    + ' in %.2g' % (t2-t1) + ' sec'
            if plot_sed:
                plt.loglog(wspec, spec, label=title_text,  lw=0.7, color='navy', alpha=0.7)
            #plt.plot(model["phot_wave"], phot, marker='s',markersize=10, alpha=0.8, ls='', lw=3,
            #     markerfacecolor='none', markeredgecolor='blue', markeredgewidth=3)

        if plot_sed:
            plt.xlabel('Wavelength [A]')
            plt.ylabel('Flux Density [maggies]')
            plt.tight_layout()

        return theta_grid, phot_grid, times
