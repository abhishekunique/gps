#include "pr_Dyn.h"
#include "pr_Dyn.cpp"
#include "pr_thinPort.cpp"

// actuator_user = [pr_model, len_gain, len_bias, pr_model_prm....]
double pressureLocal(const mjModel* m, const mjData* d, int id)
{
	// resolve user space 
	const int pr_model = (int)m->actuator_user[id*m->nuser_actuator + 0];
	const mjtNum len_gain = m->actuator_user[id*m->nuser_actuator + 1];
	const mjtNum len_bias = m->actuator_user[id*m->nuser_actuator + 2];
	const mjtNum* pr_model_prm = m->actuator_user + id*m->nuser_actuator + 3;

	// resolve variables
	mjtNum length = len_gain*(d->actuator_length[id] - m->actuator_length0[id]) + len_bias;
	mjtNum velocity = len_gain*d->actuator_velocity[id];
	mjtNum act = pr_model_prm[0] + d->act[id]; // pressure = c[0] + d->act
	mjtNum ctrl = d->ctrl[id];
	mjtNum dp;

	// Ensure minimum positive volume
	if (length<0.005)
	{
		printf("Cylinder length (%f) out of bounds. Adjusting to %f\n", length, 0.005);
		length = 0.005;
	}

	// Enforce pressure bounds
	if (act<pr_model_prm[0])
	{
		printf("Pressure (%0.2f) under bounds (by %10.2f). Adjusting to %0.2f\n", act, act-pr_model_prm[0], pr_model_prm[0]);
		act = pr_model_prm[0];
	}
	else if (act>pr_model_prm[1])
	{
		printf("Pressure (%0.2f) over bounds (by %10.2f). Adjusting to %0.2f\n", act, act-pr_model_prm[1], pr_model_prm[1]);
		act = pr_model_prm[1];
	}
	
	// pneumatic dynamics
	if (pr_model == 0) // parametric
	{
		if (m->nuser_actuator< 3 + 10)
			mju_error("User buffer is too small to carry all the dynamics parameters");
		else
			dp = pr_pneumatic(ctrl, length, velocity, act, pr_model_prm, 0, NULL, 1);
	}
	else if (pr_model == 1) // thinport
	{
		if (m->nuser_actuator< 3 + 15)
			mju_error("User buffer is too small to carry all the dynamics parameters");
		else
			dp = pr_thinPort(ctrl, length, velocity, act, pr_model_prm, 0, 5, NULL);
	}

	//printf("id: %2d, u:%0.4f, l:%0.4f, v:%7.4f, a:%0.2f, \tdp:%0.2f\n", id,  ctrl, length, velocity, act, dp);
	return dp; // constant dynamics
}
