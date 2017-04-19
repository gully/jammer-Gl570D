def user_defined_lnprior(p):
    '''
    Takes a vector of stellar parameters and returns the ln prior.
    '''
    if not ((p[10] > 100) and (p[10] < 2000) and 
            (p[9] > -8) and (p[9] < -2) and
            (p[8] > 0) and
    		(p[2] > -1000) and (p[2] < 1000) and
			(p[3] > 2.0) and (p[3] < 1000)):
    	return -np.inf

    lnp_vz = -(p[2] - 0.0)**2/(2.0*100.0**2)
    lnp_vsini = -(p[3] - 30.0)**2/(2.0*200.0**2)
    lnp_c1 = -(p[5] - 0.0)**2/(2.0*0.000001**2)
    lnp_c2 = -(p[6] - 0.0)**2/(2.0*0.000001**2)
    lnp_c3 = -(p[7] - 0.0)**2/(2.0*0.000001**2)
    lnp_lamp = -(p[9] + 4.0)**2/(2.0*0.01**2)
    lnp_sigamp = -(p[8] - 1.0)**2/(2.0*0.001**2)

    ln_prior_out = lnp_vz + lnp_vsini + lnp_c1 + lnp_c2 + lnp_c3 + lnp_lamp + lnp_sigamp

    return ln_prior_out

