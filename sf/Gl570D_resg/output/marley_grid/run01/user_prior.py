def user_defined_lnprior(p):
    '''
    Takes a vector of stellar parameters and returns the ln prior.
    '''
    if not ((p[5] > 0) and 
            (p[7] > 500) and (p[7] < 5000) and
    		(p[6] > -19) and (p[6] < -17)):
    	return -np.inf

    lnp_sigamp = -(p[5] - 1.0)**2/(2.0*0.05**2)

    ln_prior_out = lnp_sigamp

    return ln_prior_out
