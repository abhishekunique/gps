#ifndef _PR_DYN_H_
#define _PR_DYN_H_

#ifdef __cplusplus
extern "C"
{
#endif

//#include "tOpt_util.h"

#ifdef DLL_EXPORT
    #define DLL_API __declspec(dllexport)
#elif defined(DLL_IMPORT)
    #define DLL_API __declspec(dllimport)
#else
	#define DLL_API
#endif

typedef double mjtNum;
// ------------------- pneumatic cylinder dynamics -----------------
// inputs:   voltage u, length x, velocity v, pressure p, 10 parameters c[0..9], timestep dt
// intermediate outputs: asymptotic pressure a, flowRate r, derivs[0..5]=[au ax av ru rx rv]
// final outputs: p_next(say q) dqdz[0.3]=[qu, qx, qv, qp]
DLL_API mjtNum pr_pneumatic(mjtNum u, mjtNum len, mjtNum vel, mjtNum p, const mjtNum* c, 
	mjtNum dt, mjtNum* dpdz, bool isContineousTimeDynamics);


DLL_API mjtNum pr_thinPort(mjtNum u, mjtNum x, mjtNum v, mjtNum p, const mjtNum* c, 
	mjtNum dt, mjtNum ndt, mjtNum* dpdz);

/*
DLL_API mjtNum pressure(mjtNum length, mjtNum velocity, 
                             int type, const mjtNum* prm, 
                             mjtNum act, mjtNum ctrl);
							 */

#ifdef __cplusplus
}
#endif

#endif 