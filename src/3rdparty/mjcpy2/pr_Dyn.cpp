#include "pr_Dyn.h"
#include <math.h>

//------------------------------Local Utils ------------------------------
#define prMAXVAL		1E+15		// maximum value allowed (for autofix)

// max function, avoid re-evaluation
mjtNum pr_max(mjtNum a, mjtNum b)
{
	if( a >= b )
		return a;
	else 
		return b;
}

bool pr_isbad(mjtNum* x, int n)
{
	bool bad=0;
	int i;
	for( i=0; i<n && !bad; i++ )
		bad |= ( x[i]!=x[i] || x[i]>prMAXVAL || x[i]<-prMAXVAL );
	return bad;
}

// long-tailed sigmoid, y = x/sqrt(x^2+beta^2)
mjtNum pr_sigmoid(mjtNum x, mjtNum beta, mjtNum *yx,  mjtNum *yxx)
{
	mjtNum s, y;
	if( beta==0.0 )       // step function
	{
		if (yx) *yx = 0.0;      
		return (x > 0.0 ? 1.0 : -1.0);
	}
	else                // sigmoid
	{
		s	= 1.0/sqrt(x*x + beta*beta);
		y	= s*x;
		if (yx) 
			*yx  = s * (1.0-y*y);
		if (yxx) 
			*yxx  = -3.0 * s * y * yx[0];
		return y;
	}
}

// smooth abs, output y = sqrt(x^2+beta^2)-beta, yx= dy/dx
mjtNum pr_softAbs(mjtNum x, mjtNum beta, mjtNum *yx, mjtNum *yxx)
{
	mjtNum tmp;
	if( beta==0 ) // hard abs
	{
		if ( yx )
			*yx = (x >= 0 ? 1.0 : -1.0);      
		return (x >= 0.0 ? x : -x);
	}
	else          // smooth abs
	{
		tmp   = sqrt(x*x + beta*beta);
		if (yxx)
			yx[0] = pr_sigmoid(x, beta, yxx, nullptr);
		else if ( yx )
			*yx   = x / tmp;
		return tmp - beta;
	}
}


//------------------------------ pressure models ------------------------------

// next_pressure q, voltage u, position x, velocity v, pressure p
// integrate linear ODE :: dp/dt = (a(z) - p) * r(z)
// Compute q = p(tNext)
// Additionally given the derivatives [da/du da/dx da/dv dr/du dr/dx dr/dv]
// compute [dq/du dq/dx dq/dv dq/dp]'
mjtNum tOpt_linear_ODE(mjtNum p, mjtNum a, mjtNum r, mjtNum t, int n, mjtNum *derivs)
{
	mjtNum d, q, g, gr, qp;
	int i;

	d  = a - p;
	g = exp(-t*r);
	q = a - d*g;

	//overwrite y derivatives into a,r derivatives
	if ( derivs )
	{		
		qp = g;
		gr = -t*g;
		for ( i=0; i<n; i++ )   // chain rule q_z = a_z*(1-g) - d*g_r*r_z
			derivs[i] = derivs[i]*(1-g) - d*gr*derivs[i+n];  
		derivs[n] = qp;
	}
	return q;
}



// inputs:  10 parameters c[0..9], voltage u, position x, velocity v
// outputs: asymptotic pressure A, flow rate R, derivs[0..5]=[au, ax, av, ru, rx, rv]
static void pr_pressure(mjtNum* A, mjtNum* R, const mjtNum* c, mjtNum u, mjtNum x, mjtNum v, mjtNum *derivs)
{
	mjtNum bias, scale, w, s, sw, tau, orifice, ou, b, r0, h;
	mjtNum wu, su, au, ax, av, ru, rx, rv, a, r;
	mjtNum *psw, *pou;

	if ( derivs )
	{
		psw = &sw;
		pou = &ou;
	}
	else
		psw = pou = 0;

	// get bias and scale of pressure sigmoid from min and max
	bias	= (c[1] + c[0]) / 2.0;
	scale   = (c[1] - c[0]) / 2.0;

	// origin shift for voltage    3 < c[2] < 7
	u		= u - c[2];

	// 'kink' around u==0          0 < c[3],c[4] < 20
	w		= c[3]*u + c[4]*u*u*u;

	// asymptotic pressure sigmoid
	s		= pr_sigmoid(w, 1, psw, nullptr);

	// asymptotic pressure with no velocity
	b		= bias + scale*s;

	// timescale, proportional to 1/(air volume)
	tau		= 1 + c[6]*x;

	// area of the valve orifice with linear correction  -1 < c[8] < 1
	orifice = pr_softAbs(u, 0.1, pou, nullptr) + c[8]*u;

	// flow rate with velocity correction
	r0		= (c[7]*orifice + c[9]) / tau; // c[7] > 0, c[9] > eps > 0
	h		= c[5] * v / tau;
	r		= r0 + h;

	//asymptotic pressure
	a		= b * r0 / r;

	// derivatives -- all two-letter variables denote derivatives
	// e.g. au := d_a/d_u
	if (derivs)
	{
		wu = c[3] + c[4]*3*u*u;
		su = sw*wu;
		ax = 0;

		ru =  (ou + c[8]) * c[7] / tau;
		rx = -(r / tau) * c[6];
		rv = c[5] / tau;

		au = (scale*su*(r-h) + b*ru*h/r) / r;
		av = -b*rv*(1-h/r)/ r;

		derivs[0] = au;
		derivs[1] = ax;
		derivs[2] = av;
		derivs[3] = ru;
		derivs[4] = rx;
		derivs[5] = rv;
	}
	*R = r;
	*A = a;
}

// pneumatic cylinder dynamics
// inputs:   voltage u, length x, velocity v, pressure p, 10 parameters c[0..9], timestep dt
// intermediate outputs: asymptotic pressure a, flowRate r, derivs[0..5]=[au ax av ru rx rv]
// final outputs(say q): dp for dt==0, p_next otherwise; dqdz[0.3]=[qu, qx, qv, qp]
mjtNum pr_pneumatic(mjtNum u, mjtNum x, mjtNum v, mjtNum p, const mjtNum* c, mjtNum dt, mjtNum* dpdz, bool isContineousTimeDynamics)
{
	
	mjtNum a,r,derivs[6],p_next; 
	int i;

	if ( pr_isbad( &u, 1) )			// bad voltage, default to constant dynamics
		return p;
	else
	{	
		// u = util_max(3.0, util_min(7.0, u));
		pr_pressure(&a, &r, c, u, x, v, dpdz ? derivs : 0); // intermediate output

		/*if(isContineousTimeDynamics)	// integrate forward
		{
			p_next = p + dt*r*(a-p);
			if (dpdz)
			{
				for (i=0; i<3; i++)
					dpdz[i] = dt* (r*derivs[i] + derivs[3+i]*(a-p) );
				dpdz[3] = 1-dt*r;
			}
		}*/

		if(dt==0) // return continuous time dynamics p_dot = r * (a-p)
		{
			if (dpdz)
			{
				for (i=0; i<3; i++)
					dpdz[i] = (r*derivs[i] + derivs[3+i]*(a-p));
				dpdz[3] = -r;
			}
			return r*(a-p);
		}
		else	// discrete time integration within dt period
		{
			// final outputs
			p_next = pr_max(tOpt_linear_ODE(p, a, r, dt, 3, dpdz ? derivs : 0), 0.0); //pressure is nonegative
			if (dpdz)
				for (i = 0; i < 4; i++)
					dpdz[i] = p_next ? derivs[i] : 0;
			return p_next;
		}
	}
}