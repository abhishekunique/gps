
// pneumatic thinport cylinder dynamics
// inputs:   voltage u, length x, velocity v, pressure p, timestep dt
//			parameters c[] = {p_atm, p_comp, bore area}
// final outputs: p_next(say q) dqdz[0.3]=[qu, qx, qv, qp]
// dm = u_c*c_f(p) - u_a*f-a(p), where
//		- u_c and u_a are the compressor and atm port area resp.
//		- f_c and f_a are influx and outflux
// dp = n*(Rs*T*dm - p*dv)/v


#include "pr_Dyn.h"
#include <stdio.h>
#include <math.h>

const mjtNum pATM = 101325.0;

typedef enum area_
{
	none,
	sigmoid,
	rectify2
} areaFcn;


double pressure(mjtNum length, mjtNum velocity, 
                             int type, const mjtNum* prm, 
                             mjtNum act, mjtNum ctrl)
{ 
	double par, tp;
    double prmLocal_par[] = {0.093971070967613, 0.513477320952826, 0.035877578312754,
            1.235977874629122, 8.404091015001519, 0.485030592118657, 6.042436037292539,
            1.021871164868839, 0.175296791929956, 0.121642029266126};
	par = pr_pneumatic(ctrl, length, velocity, act, prmLocal_par, 0, NULL, 0);

	double prmLocal_tp[] = {
			092296.8746522295,
			521485.7661869445,
			0.000063617251235,
			-0.000000100000000,
			-0.279542527698693,
			0.000000394757380,
			0.664815462980302,
			-0.491965547790831,
			-0.000000000286402,
			0.446588486605370,
			0.000000365128968,
			0.213516990988350,
			0.999999999999928
	};
	tp = pr_thinPort(ctrl, length, velocity, act, prmLocal_tp, 0, 5, NULL);
	printf("actL l:%2.2f, v:2.4%f, a:%f, par:%f, tp:%f \n", length, velocity, act, par, tp);
	return tp;
}


// Thin Port =========================================================

// Rectify from both direction
// params    = [Yoff, Xoff, spread, kink, tilt] & kink > 0
//           NOTE: {-1< tilt <1}  && {scale>0} && {Yoff>0} ensures positivity of rectify2(x)
//
// y         = Yoff + spread*[ tilt*(x-Xoff) + sqrt((x-Xoff)^2+kink^2)]/2
// dydx      = spread * [ tilt + (x-Xoff)/sqrt(kink^2 + (x-Xoff)^2)]/2

mjtNum area_rectify2(mjtNum u, const mjtNum* params, mjtNum* dareadu)
{
	mjtNum temp = params[3]*params[3] + (u-params[1])*(u-params[1]); 
	mjtNum ar = params[0] + params[2]*( params[4]*(u-params[1]) + sqrt(temp) )/2.0;

	if(dareadu)
		dareadu[0] = params[2]*( params[4] + (u-params[1])/sqrt(temp) )/2.0;
	if (ar<0)
	{	  printf("Negative area. Returning 0\n");
		return 0.0;
	}
	return ar;
}



// params = {Yoffset, Xoffset, scale, r}
// ar = Yoffset + abs(scale) + scale*(x-Xoffset)/ sqrt(r^2 + (x-Xoffset)^2)
// dar = scale* r^2/ (r^2 + (x-Xoffset)^2)^(3/2)
mjtNum area_sigmoid(mjtNum u, const mjtNum* params, mjtNum* dareadu)
{
	mjtNum temp = params[3]*params[3] + (u-params[1])*(u-params[1]); 
	mjtNum ar = params[0] + fabs(params[2]) + params[2]*(u-params[1])/sqrt(temp);
	if(dareadu)
		dareadu[0] = params[2]*params[3]*params[3]/pow(temp, 1.5);
	if (ar<0)
	{	printf("Negative area. Returning 0\n");
		return 0.0;
	}
	return ar;
}


mjtNum area_a(mjtNum u, mjtNum* params, mjtNum* dareadu)
{
	if (u < -1.70)
		u = -1.70;
	if (u > 2.0)
		u = 2.0;
	mjtNum ar = params[0]*pow(u,7) + 
		params[1]*pow(u,6) + 
		params[2]*pow(u,5) + 
		params[3]*pow(u,4) + 
		params[4]*pow(u,3) + 
		params[5]*pow(u,2) + 
		params[6]*u + 
		params[7];

	// get derivatives
	if(dareadu)
	{	dareadu[0] =	params[0]*7*pow(u,6) +
						params[1]*6*pow(u,5) + 
						params[2]*5*pow(u,4) + 
						params[3]*4*pow(u,3) + 
						params[4]*3*pow(u,2) + 
						params[5]*2.0*u + 
						params[6];
	}
	return ar;
}

mjtNum area_c(mjtNum u, mjtNum* params, mjtNum* params_ratio, mjtNum* dareadu)
{
	// u_c and u_a
	auto ratio = [](mjtNum u, mjtNum* par_ratio, mjtNum* dratiodu)
	{
		mjtNum ratio = 0, p, e;

		// ratio = par_ratio[0] * exp(-util_pow((u - par_ratio[1]) / par_ratio[2], 2)) 
		// 		+ par_ratio[3] * exp(-util_pow((u - par_ratio[4]) / par_ratio[5], 2)) 
		// 		+ par_ratio[6] * exp(-util_pow((u - par_ratio[7]) / par_ratio[8], 2)) 
		// 		+ par_ratio[9] * exp(-util_pow((u - par_ratio[10]) / par_ratio[11], 2));

		p = (u - par_ratio[1]) / par_ratio[2];
		e = par_ratio[0] * exp(-pow(p, 2));
		ratio = e;
		if(dratiodu)
			dratiodu[0] = -2.0*p*e/par_ratio[2];

		p = (u - par_ratio[4]) / par_ratio[5];
		e = par_ratio[3] * exp(-pow(p, 2));
		ratio += e;
		if(dratiodu)
			dratiodu[0] += -2.0*p*e/par_ratio[5];

		p = (u - par_ratio[7]) / par_ratio[8];
		e = par_ratio[6] * exp(-pow(p, 2));
		ratio += e;
		if(dratiodu)
			dratiodu[0] += -2.0*p*e/par_ratio[8];

		p = (u - par_ratio[10]) / par_ratio[11];
		e = par_ratio[9] * exp(-pow(p, 2));
		ratio += e;
		if(dratiodu)
			dratiodu[0] += -2.0*p*e/par_ratio[11];
		
		return ratio;
	};

	if(dareadu)
	{
		mjtNum dar_adu = 0.0;
		mjtNum drdu = 0.0;
		mjtNum ar_a = area_a(u, params, &dar_adu);
		mjtNum r = ratio(u, params_ratio, &drdu);

		dareadu[0] = r*dar_adu + drdu*ar_a; 
		return r*ar_a;
	}
	else
	{
		return ratio(u, params_ratio, NULL)*area_a(u, params, NULL);
	}
}

// c[11] = {p_a, p_c, boreArea, {areaPrms}_a, {areaPrms}_c}
//mjtNum pr_thinPort(mjtNum u, mjtNum x, mjtNum v, mjtNum p, const mjtNum* c, 
//       mjtNum dt, mjtNum* dpdz)
mjtNum pr_thinPort_subdt(mjtNum u, mjtNum x, mjtNum v, mjtNum p, const mjtNum* c, 
	mjtNum dt, mjtNum* dpdz)
{
	mjtNum dm = 0.0, dp = 0.0;
	const mjtNum p_a = c[0];			// atm pressure
	const mjtNum p_c = c[1];			// compressor pressure
	const mjtNum boreArea = c[2];		// cylinder bore area
	const mjtNum* params_a = NULL;		// {Yoffset, scale, Xoffset, r}_a
	const mjtNum* params_c = NULL;		// {Yoffset, scale, Xoffset, r}_c
	const mjtNum n = 1.4;				// air's specific heat
	const mjtNum M = .0289645;			// kg/mol, dry air molecular mass
	const mjtNum R = 8.314472;			// jule/mole/K, universal gas law constant
	const mjtNum Rs = R/M;				// 287.0574 jule/kg/K, Specific gas law constant for air.
	const mjtNum T = 293;				// temperature 
	
	// boreArea = 6.3617251235e-5;
	// p_a = p_a*1e6 + pATM;
	// p_c = p_c*1e6 + pATM;
	// p = p*1e6 + pATM; //convert to pascal
	// parameters
	mjtNum Oparams_a[8] = {-2.127e-09, 3.219e-09, 1.601e-08, -2.937e-08, -3.387e-08, 1.004e-07,-3.243e-08, 9.391e-09};	// atm port area sysId params
	mjtNum Oparams_ratio[12] = {3.987,0.9852,0.7052,0.5404,0.6171,0.1939,0.4721,0.4596,0.1529,0.216,0.0003629,0.1308};	// compressor port area sysId params
	

	// flux :: f_c, f_a
	auto flux = [](mjtNum p_u, mjtNum p_d, mjtNum* dfluxdp_u, mjtNum* dfluxdp_d)
	{
		auto h = [](mjtNum p_u, mjtNum p_d, mjtNum* dhdp_u, mjtNum* dhdp_d)
		{
			static bool firstFlag = true;
			static mjtNum alpha = 0.0;
			static mjtNum theta = 0.0;
			static mjtNum beta  = 0.0;
			static mjtNum k;

			if(firstFlag)
			{
				mjtNum f = 5.0;			// for air, dimentionless
				mjtNum c = 0.72;		// discharge coeff
				mjtNum Z = 0.99;		// gas compressibility factor
				mjtNum M = .0289645;	// kg/mol, dry air molecular mass
				mjtNum R = 8.314472;	// jule/mole/K, universal gas law constant
				mjtNum Rs = R/M;		// 287.0574 jule/kg/K, Specific gas law constant for air.
				mjtNum T = 293;			// temperature 
				k = 1.0 + 2.0/f;		// specific heat ratio 
				alpha = c*sqrt(2.0*k/(Z*Rs*T*(k-1.0)));
				theta = pow((k+1.0)/2.0, k/(k-1.0));
				beta = c*sqrt( (pow(2.0/(k+1), (k+1.0)/(k-1.0))) * k/(Z*Rs*T) );
				firstFlag = false;
			}

			if(p_d>p_u)
				printf("Error in h(p_u:%f, p_d:%f): p_d>p_u\n", p_u*1e-6, p_d*1e-6);

			if(p_u/p_d<=theta)
			{
				mjtNum w =  pow(p_d/p_u, 2.0/k) - pow(p_d/p_u, (k+1.0)/k);
				if(w<0)
				{	printf("Error in h(p_u:%f, p_d:%f): Negative value\n", p_u*1e-6, p_d*1e-6);
					if(dhdp_u)
						dhdp_u[0] = 0.0;
					if(dhdp_d)
						dhdp_d[0] = 0.0;
					return 0.0;
				}


				mjtNum g = p_u*sqrt(w);
				// derivatives
				if(dhdp_u)
					dhdp_u[0] = alpha*( (2.0-2.0/k)*pow(p_u, 1.0-2.0/k)*pow(p_d, 2.0/k) 
										- (1.0-1.0/k)*pow(p_u, -1.0/k)*pow(p_d,1.0+1.0/k) )/(2.0*g);
				if(dhdp_d)
					dhdp_d[0] = alpha*( (2.0/k)*pow(p_u, 2.0-2.0/k)*pow(p_d, 2.0/k-1.0) 
										- (1.0+1.0/k)*pow(p_u, 1.0-1.0/k)*pow(p_d,1.0/k) )/(2.0*g);
				return alpha*g;
			}
			else
			{
				if(dhdp_u)
					dhdp_u[0] = beta;
				if(dhdp_d)
					dhdp_d[0] = 0.0;
				return beta*p_u;
			}
		};
	
		mjtNum f = 0.0;
		mjtNum sign = 0.0;

		if(p_u>=p_d)
		{	sign = 1.0;
			if( (dfluxdp_u)||(dfluxdp_d) )
			{	// derivatives
				mjtNum dhdp_u, dhdp_d;
				f = sign*h(p_u, p_d, &dhdp_u, &dhdp_d);
				if(dfluxdp_u)
					dfluxdp_u[0] = sign*dhdp_u;
				if(dfluxdp_d)
					dfluxdp_d[0] = sign*dhdp_d;
			}
			else
				f = sign*h(p_u, p_d, NULL, NULL);
		}
		else
		{	sign = -1.0;
			if( (dfluxdp_u)||(dfluxdp_d) )
			{	// derivatives
				mjtNum dhdp_u, dhdp_d;
				f = sign*h(p_d, p_u, &dhdp_d, &dhdp_u);
				if(dfluxdp_u)
					dfluxdp_u[0] = sign*dhdp_u;
				if(dfluxdp_d)
					dfluxdp_d[0] = sign*dhdp_d;
			}
			else
				f = sign*h(p_d, p_u, NULL, NULL);
		}
		return f;
	};

	mjtNum h = 1e-6;
	// pressure dynaimcs
	mjtNum *dflux_cdp = NULL;
	mjtNum *dflux_adp = NULL;
	mjtNum *darea_cdu = NULL;
	mjtNum *darea_adu = NULL;
	mjtNum f_c, f_a, ar_c, ar_a, ddmdp, ddmdu;

	//if ( prs_isbad( &u, 1) )	 // bad voltage, default to constant dynamics
	//	return p;
	//else
	{
		// mark derivatives
		if(dpdz)
		{
			mjtNum dflux_cdp1, dflux_adp1, darea_cdu1, darea_adu1;
			dflux_cdp = &dflux_cdp1;
			dflux_adp = &dflux_adp1;
			darea_cdu = &darea_cdu1;
			darea_adu = &darea_adu1;
		}

		// dm = u_c*f_c(p) - u_a*f-a(p)
		f_c = flux(p_c, p, NULL, dflux_cdp);
		f_a = flux(p, p_a, dflux_adp, NULL);

		areaFcn areaFcn_inUse = rectify2;
 		switch(areaFcn_inUse)
		{
			case none: // polynomial
				ar_c = area_c(u, Oparams_a, Oparams_ratio, darea_cdu);
				ar_a = area_a(u, Oparams_a, darea_adu);
				break;
			case sigmoid:
				params_a = c+3;
				params_c = c+7;
				ar_c = area_sigmoid(u, params_c, darea_cdu);
				ar_a = area_sigmoid(u, params_a, darea_adu);
				break;
			case rectify2:
				params_a = c+3;
				params_c = c+8;
				ar_c = area_rectify2(u, params_c, darea_cdu);
				ar_a = area_rectify2(u, params_a, darea_adu);
				break;
			default: 
				printf("Unrecognized area function\n");
		}
		
		dm = ar_c*f_c - ar_a*f_a;

		// dp = n*(Rs*T*dm - p*dv)/v
		dp = n*(Rs*T*dm - p*v*boreArea)/(x*boreArea);

		// get derivatives:: dqdz[0.3]=[qu, qx, qv, qp]
		if(dpdz)
		{
			// dq/du
			ddmdu = darea_cdu[0]*f_c - darea_adu[0]*f_a;
			dpdz[0] = n*Rs*T*ddmdu/(x*boreArea);
			//printf("\n%e", dt*n*Rs*T/(x*boreArea)); //(1.301115e+07)

			// dq/dx
			dpdz[1] = (-1.0)*dp/x;

			// dq/dv
			dpdz[2] = (-1.0)*n*p/x;
				
			// dq/dp
			ddmdp = ar_c*dflux_cdp[0] - ar_a*dflux_adp[0];
			dpdz[3] = n*(Rs*T*ddmdp - v*boreArea)/(x*boreArea);
		}

		//printf("DP == f_c:%e, \tf_a:%e, ar_c:%e, ar_a:%e, dm:%e, \tdp:%e, p:%e\n", f_c, f_a, ar_c, ar_a, dm, dp, p+dp*dt);
		return dp;
	}
}


mjtNum pr_thinPort(mjtNum u, mjtNum x, mjtNum v, mjtNum p, const mjtNum* c, 
	mjtNum dt, mjtNum ndt, mjtNum* dpdz)
{
	mjtNum subdt = dt/ndt;
	mjtNum t0 = 0;
	mjtNum dpdz_l[4]={0.0, 0.0, 0.0, 0.0};
	mjtNum *dpdz_p = NULL;

	// mark derivatives
	if(dpdz)
	{	dpdz_p = dpdz_l;
		dpdz[0] = 0.0;
		dpdz[1] = 0.0;
		dpdz[2] = 0.0;
		dpdz[3] = 0.0;
	}

	if(dt==0.0) // retun contineous time dynamics
		return pr_thinPort_subdt(u, x, v, p, c, 0.0, dpdz); //derivs
	else // integrate forward for pressure
	{
		if(dpdz)
			dpdz[3] = 1.0;

		for(int i=0; i<ndt; i++) // integrate for n steps
		{
			// integrate pressure forward (pnext = p0+ subdt*dp0 + subdt*dp1 + subdt*dp2 ....)
			p = p + subdt*pr_thinPort_subdt(u, x, v, p, c, subdt, dpdz_p);
			
			// accummulate derivatives
			if(dpdz_p)
			{
				dpdz[0] += subdt*dpdz_p[0];
				dpdz[1] += subdt*dpdz_p[1];
				dpdz[2] += subdt*dpdz_p[2];
				dpdz[3] += subdt*dpdz_p[3];
			}
		}
		return p;
		/*
		while(t0+2.0*subdt<=dt)
		{	p = pr_thinPort_subdt(u, x, v, p, c, subdt, NULL, 0); //no derivs and always discreteTimeDynamics
			t0 += subdt;
			//printf("%f, ", t0);
		}
		//printf("\n");
		return pr_thinPort_subdt(u, x, v, p, c, subdt, dpdz, isContineousTimeDynamics); //derivs
		*/
	}
}
