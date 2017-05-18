
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include "lib\glew\glew.h"
#include "lib\freeglut\freeglut.h"
#include <time.h>
#include <ctime>
#include "config.h"
#include "Parameters.h"

using namespace std;



void simulation();

int main() {
	glutInit(&__argc, __argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(920, 00);
	glutInitWindowSize(1000, 1000);
	glutCreateWindow("Smoothed Particle Hydrodynamics");
	glutDisplayFunc(simulation);
	glutMainLoop();
	getchar();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////// FUNCTIONS
int render_scene(int ntotal, double *x1, double *x2, double* mass, int* fluid_code, int* wall, int* state, int* test_array, int* ghost, double** temp, vector<int> &marker_vector, double part_spacing);
int render_scene_mass(int ntotal, double *x1, double *x2, double* mass, int *wall, int *state, int *ghost, double** temp, double part_spacing);
void dump_data(int ntotal, double *x1, double *x2, double *v1, double *v2, double *rho, double *ref_rho, double *pressure, double *mass, double *eta, double *hsml, double *temp, double *energy, double *therm_cond, double *sp_heat, int *fluid_code, int *state, int *ghost, int* ghost_state, int* wall, int* boundry, int* Interface, double *v1_old, double *v2_old, double *energy_old, double *rho_old, double *x1_h, double *x2_h, double *v1_h, double *v2_h, double *rho_h, double *ref_rho_h, double *pressure_h, double *mass_h, double *eta_h, double *hsml_h, double *temp_h, double *energy_h, double *therm_cond_h, double *sp_heat_h, int *fluid_code_h, int *state_h, int *ghost_h, int* ghost_state_h, int* wall_h, int* boundry_h, int* Interface_h, double *v1_old_h, double *v2_old_h, double *energy_old_h, double *rho_old_h, int arraySize, double length, double part_spacing, int index, pair <int, double> solid_line);
double initial_total_mass_util(int ntotal, double *mass_h, int *wall_h);
void write_array(int *arr, int len, string name);
void write_array(double *arr, int len, string name);
void force_comparison_util(double *dv1dt, double *dv2dt, double *dS1dt, double *dS2dt, double *exdv1dt, double *exdv2dt, int len);

// Function to find the maximum smoothing length required in the calculation of timestep.
__global__ void max_hsml(int ntotal, double *hsml, double *hsml_max, int *fluidcode)
{
	hsml_max[0] = hsml[1];
	for (int i = 1; i <= ntotal; i++) if (hsml[i] > *hsml_max && fluidcode[i] != -1)hsml_max[0] = hsml[i];
}

// A voxel search is implemented to find the neighbors of particles. Each particle is assigned to a square of side = smoothing length based on its position
__global__ void index_particles(int ntotal, int *bucket_index, double *x1, double *x2, double length, double kern_constant, double *hsml_max, int *nvoxel_length)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > 0 && idx <= ntotal)
	{

		nvoxel_length[0] = (int)(length / (kern_constant*hsml_max[0]) + 1);
		bucket_index[idx] = (x1[idx]) / (kern_constant*hsml_max[0]) + nvoxel_length[0] * (int)((x2[idx]) / (kern_constant*hsml_max[0]));

	}
}

__global__ void equation_of_state(int ntotal, double *pressure, double *rho, double *ref_rho, int* fluidcode, int* Interface, double p0, double gamma, double background_pressure, double c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > 0 && idx <= ntotal)
	{
		if (fluidcode[idx] == 1)pressure[idx] = p0*(pow(rho[idx] / ref_rho[idx], gamma) - 1) + background_pressure;  //Simulating Free Surface flows Monaghan, Applied Mathematics
																													 //if (fluidcode[idx] == 2)pressure[idx] = c*c*(pow(rho[idx] / ref_rho[idx], 1.4) - 1) + 0.003;
																													 // pressure[idx]= 0.2*c*c*rho[idx];


		if (fluidcode[idx] > 3)pressure[idx] = 0;
	}
}

__global__ void compute_kernel(int ntotal, int *bucket_index, int *nvoxel_length, double *x1, double *x2, double *hsml, int *neighbors, int *neighbors_count, double *w, double *dwdx, double *dwdy, int *ghost, int *ghost_state, int *wall, int *fluid_code, int *Interface, double kern_constant, double part_spacing, int max_neighbors)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	double dx, dy, r;
	int index;
	neighbors_count[idx] = 0;
	Interface[idx] = 0;
	for (int i = 1; i <= ntotal; i++)
	{
		if ((bucket_index[i] == bucket_index[idx] || bucket_index[idx] == bucket_index[i] + 1 || bucket_index[idx] == bucket_index[i] - 1 || bucket_index[idx] == bucket_index[i] + nvoxel_length[0] || bucket_index[idx] == bucket_index[i] + nvoxel_length[0] - 1 || bucket_index[idx] == bucket_index[i] + nvoxel_length[0] + 1 || bucket_index[idx] == bucket_index[i] - nvoxel_length[0] - 1 || bucket_index[idx] == bucket_index[i] - nvoxel_length[0] + 1 || bucket_index[idx] == bucket_index[i] - nvoxel_length[0]) && i != idx && !wall[idx] && fluid_code[idx] != -1 && fluid_code[i] != -1 && idx>0)
		{
			dx = x1[idx] - x1[i];
			dy = x2[idx] - x2[i];
			r = dx*dx + dy*dy;
			if (r <= kern_constant*kern_constant*(hsml[i] + hsml[idx])*(hsml[i] + hsml[idx]) / 4)
			{
				if (neighbors_count[idx] == max_neighbors)
				{
					printf("\n\nFATAL!! too many neighbhors for the particle #%d\n\n", idx);
					return;
				}
				neighbors_count[idx]++;
				index = idx*max_neighbors + neighbors_count[idx];

				neighbors[index] = i;


				//Tag all non-ghost particles close to interface
				if (ghost[i] && !ghost[idx] && fluid_code[idx] == 1 && !wall[idx])Interface[idx] = 1;

				/////////////////////////////////////////////////////////////////////////////////////////////
				//////////////// KERNEL COMPUTATION
				double hsml_avg = (hsml[i] + hsml[idx]) / 2;
				r = sqrt(r) / hsml_avg;
				double factor = 15.0 / (7.0 * hsml_avg * hsml_avg * 3.14159265358979323846);
				if (r >= 0 && r <= 1.0)
				{
					w[index] = factor * (2.0 / 3.0 - (r * r) + (r * r * r) / 2.0);
					dwdx[index] = factor * (-2.0 + 1.5 * r) / (hsml_avg*hsml_avg) * dx;    // CHECK THIS FUNCTION!!!
					dwdy[index] = factor * (-2.0 + 1.5 * r) / (hsml_avg*hsml_avg) * dy;
				}
				else if (r > 1.0 && r <= 2.0)
				{
					w[index] = factor * (1.0 / 6.0) * (2.0 - r) * (2.0 - r) * (2.0 - r);
					dwdx[index] = -factor * 0.5* (2.0 - r) * (2.0 - r) / hsml_avg * (dx / (r*hsml_avg));
					dwdy[index] = -factor * 0.5* (2.0 - r) * (2.0 - r) / hsml_avg * (dy / (r*hsml_avg));
				}
				else
				{
					w[index] = 0;
					dwdx[index] = 0;
					dwdy[index] = 0;
				}
				/*double factor = 7 / (478 * hsml_avg*hsml_avg*3.14159265358979323846);       //MOrris JOURNAL OF COMPUTATIONAL PHYSICS 136,214–226(1997) Reason for using quintic spline given
				if (r >= 0 && r < 1)
				{
					w[index] = factor*(pow(3 - r, 5) - 6 * pow(2 - r, 5) + 15 * pow(1 - r, 5));
					dwdx[index] = factor * 5 * (-pow(3 - r, 4) + 6 * pow(2 - r, 4) - 15 * pow(1 - r, 4))*dx / (r*hsml_avg*hsml_avg);
					dwdy[index] = factor * 5 * (-pow(3 - r, 4) + 6 * pow(2 - r, 4) - 15 * pow(1 - r, 4))*dy / (r*hsml_avg*hsml_avg);
				}
				else if (r >= 1 && r < 2)
				{
					w[index] = factor*(pow(3 - r, 5) - 6 * pow(2 - r, 5));
					dwdx[index] = factor * 5 * (-pow(3 - r, 4) + 6 * pow(2 - r, 4))*dx / (r*hsml_avg*hsml_avg);
					dwdy[index] = factor * 5 * (-pow(3 - r, 4) + 6 * pow(2 - r, 4))*dy / (r*hsml_avg*hsml_avg);
				}
				else if (r >= 2 && r < 3)
				{
					w[index] = factor*(pow(3 - r, 5));
					dwdx[index] = factor * 5 * (-pow(3 - r, 4))*dx / (r*hsml_avg*hsml_avg);
					dwdy[index] = factor * 5 * (-pow(3 - r, 4))*dy / (r*hsml_avg*hsml_avg);
				}
				else
				{
					w[index] = 0;
					dwdx[index] = 0;
					dwdy[index] = 0;
				}*/
				/////////////////////////////////////////////////////////////////////////////////////////////
			}
		}
	}
}

__global__ void compute_derivatives(int ntotal, double *x1, double *x2, double *v1, double *v2, double *rho, double *mass, double *pressure, double *hsml, int *neighbors, int *neighbors_count, double *w, double *dwdx, double *dwdy, double *eta, double *therm_cond, double *temp, double *dv1dt, double *dv2dt, double *dS1dt, double *dS2dt, double *exdv1dt, double *exdv2dt, double *av1, double *av2, double *drhodt, double *dhdt, double *dmgdt, double * dhsmldt, int *wall, int *fluid_code, int *Interface, int *state, int *ghost_state, int *ghost, double length, double epsilon, double gamma, double c, double back_press, int max_neighbors, int kern_constant, double constant_gravity, double S_ll, double S_ls, double part_spacing)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;


	if (idx > 0 && (idx - 1) / ntotal == 0)
	{
		double dvx, dvy;
		int neighbor_index;
		int index = max_neighbors*idx;
		int idy = idx;
		drhodt[idy] = 0;
		double w_norm = 0;
		//if (state[idy] == LIQUID)
		{

			for (int i = 1; i <= neighbors_count[idy]; i++)
			{
				neighbor_index = neighbors[index + i];
				//////////////////////////////////////////////////////////////////////////////////////////////////
				/////////////// SUMMATION DENSITY
				/*if (state[neighbor_index] == SOLID && !ghost[neighbor_index])
				{
				drhodt[idy] += 1000 * part_spacing*part_spacing*w[index + i];
				w_norm += part_spacing*part_spacing*w[index + i];
				}
				else
				{
				drhodt[idy] += mass[neighbor_index] * w[index + i];
				w_norm += mass[neighbor_index]*w[index + i] / rho[neighbor_index];
				}*/
				//////////////////////////////////////////////////////////////////////////////////////////////////
				/////////////// CONTINUITY DENSITY MULTIPHASE
				dvx = (v1[idy] - v1[neighbor_index]);
				dvy = (v2[idy] - v2[neighbor_index]);
				if (fluid_code[idy] == fluid_code[neighbor_index]) drhodt[idy] += rho[idy] * (dvx*dwdx[index + i] + dvy*dwdy[index + i])*mass[neighbor_index] / rho[neighbor_index];
				//if (fluid_code[idy] == 1 && fluid_code[neighbor_index] == 1) drhodt[idy] += dmgdt[neighbor_index] * w[index + i];

			}
			//w_norm += mass[idy] * 297471206 / rho[idy];
			//drhodt[idy] += 297471206 * mass[idy];
			//if(drhodt[idy]<=995)drhodt[idy] /= w_norm;
		}
	}

	if ((idx - 1) / ntotal == 1)
	{
		double dvx, dvy, dx, dy;
		int neighbor_index;
		int index = max_neighbors*(idx - ntotal);
		int idy = idx - ntotal;
		dv1dt[idy] = 0;
		dv2dt[idy] = 0;

		for (int i = 1; i <= neighbors_count[idy]; i++)
		{
			neighbor_index = neighbors[index + i];
			dvx = (v1[idy] - v1[neighbor_index]);
			dvy = (v2[idy] - v2[neighbor_index]);
			dx = (x1[idx] - x1[neighbor_index]);
			dy = (x2[idx] - x2[neighbor_index]);

			//if (idy == 854)printf("gradients: %.12f neigh: %d\n", (pressure[neighbor_index]),neighbor_index);
			//////////////////////////////////////////////////////////////////////////////////////////////////
			/////////////// MOMENTUM (Morris et. al. low Re flows)
			if (fluid_code[idy])
			{
				dv1dt[idy] -= (pressure[idy] / pow(rho[idy], 2) + pressure[neighbor_index] / pow(rho[neighbor_index], 2))*dwdx[index + i] * mass[neighbor_index];
				dv2dt[idy] -= (pressure[idy] / pow(rho[idy], 2) + pressure[neighbor_index] / pow(rho[neighbor_index], 2))*dwdy[index + i] * mass[neighbor_index];
			}

			//////////////////////////////////////////////////////////////////////////////////////////////////
			////////////// Low Re term from Morris et.al.
			if (fluid_code[idy])
			{
				dv1dt[idy] += (eta[idy] + eta[neighbor_index])*mass[neighbor_index] * dvx*(dx*dwdx[index + i] + dy*dwdy[index + i]) / (rho[idy] * rho[neighbor_index] * (dx*dx + dy*dy));
				dv2dt[idy] += (eta[idy] + eta[neighbor_index])*mass[neighbor_index] * dvy*(dx*dwdx[index + i] + dy*dwdy[index + i]) / (rho[idy] * rho[neighbor_index] * (dx*dx + dy*dy));
			}

		}
		dv2dt[idy] -= constant_gravity;
	}

	if ((idx - 1) / ntotal == 2)
	{
		double dvx, dvy;
		int neighbor_index;
		int index = max_neighbors*(idx - 2 * ntotal);
		int idy = idx - 2 * ntotal;
		av1[idy] = 0;
		av2[idy] = 0;

		for (int i = 1; i <= neighbors_count[idy]; i++)
		{
			neighbor_index = neighbors[index + i];
			dvx = (v1[idy] - v1[neighbor_index]);
			dvy = (v2[idy] - v2[neighbor_index]);
			//////////////////////////////////////////////////////////////////////////////////////////////////
			////////////// XSPH
			if (fluid_code[idy] == fluid_code[neighbor_index])
			{
				av1[idy] -= 2 * epsilon*mass[neighbor_index] * dvx / (rho[idy] + rho[neighbor_index])*w[index + i];
				av2[idy] -= 2 * epsilon*mass[neighbor_index] * dvy / (rho[idy] + rho[neighbor_index])*w[index + i];
			}
		}
	}


	if ((idx - 1) / ntotal == 3)
	{

		int neighbor_index;
		double dx, dy, hsml_avg, r;
		int index = max_neighbors*(idx - 3 * ntotal);
		int idy = idx - 3 * ntotal;
		dhdt[idy] = 0;

		for (int i = 1; i <= neighbors_count[idy]; i++)
		{
			neighbor_index = neighbors[index + i];
			dx = (x1[idy] - x1[neighbor_index]);
			dy = (x2[idy] - x2[neighbor_index]);
			hsml_avg = (hsml[idy] + hsml[neighbor_index]) / 2;
			r = dx*dx + dy*dy;
			////////////////////////////////////////////////////////////////////////////////////////////////////
			///////////// ENERGY EQUATION

			if (fluid_code[idy] == 1 && fluid_code[neighbor_index] == 1)
			{
				double factor = 4 * therm_cond[idy] * therm_cond[neighbor_index] / (therm_cond[idy] + therm_cond[neighbor_index]);
				dhdt[idy] += factor*(temp[idy] - temp[neighbor_index])*mass[neighbor_index] / (rho[neighbor_index]/**rho[idy]*/) * (dx*dwdx[index + i] + dy*dwdy[index + i]) / (r + 0.01*hsml_avg*hsml_avg);

			}
		}
	}
	if ((idx - 1) / ntotal == 4)
	{
		double dvx, dvy;
		int neighbor_index;
		int index = max_neighbors*(idx - 4 * ntotal);
		int idy = idx - 4 * ntotal;
		dhsmldt[idy] = 0;

		for (int i = 1; i <= neighbors_count[idy]; i++)
		{
			neighbor_index = neighbors[index + i];
			dvx = (v1[idy] - v1[neighbor_index]);
			dvy = (v2[idy] - v2[neighbor_index]);
			////////////////////////////////////////////////////////////////////////////////////////////////////
			////////////// SMOOTHING LENGTH EVOLUTION
			//dhsmldt[idy] -= hsml[idy] * mass[neighbor_index] * (dvx *dwdx[index + i] + dvy *dwdy[index + i]) / (rho[neighbor_index] * 2);
		}
	}
	/*if ((idx - 1) / ntotal == 5)
	{
		int neighbor_index;
		double dx, dy, hsml_avg, r;
		int index = max_neighbors*(idx - 5 * ntotal);
		int idy = idx - 5 * ntotal;
		dS1dt[idy] = 0;
		dS2dt[idy] = 0;
		for (int i = 1; i <= neighbors_count[idy]; i++)
		{
			neighbor_index = neighbors[index + i];
			if (fluid_code[idy] == 1 && state[idy] == LIQUID && !ghost[neighbor_index] && !wall[neighbor_index])
			{
				dx = (x1[idy] - x1[neighbor_index]);
				dy = (x2[idy] - x2[neighbor_index]);
				hsml_avg = (hsml[idy] + hsml[neighbor_index]) / 2;
				r = sqrt(dx*dx + dy*dy);
				double factor = S_ll;
				if (state[neighbor_index] != LIQUID)factor = S_ls;
				//Surface Tension
				dS1dt[idy] += factor*cos(r*3.14159265358979323846 / (2 * kern_constant*hsml_avg) + 3.14159265358979323846 / 6)*dx / (r*mass[idy]);
				dS2dt[idy] += factor*cos(r*3.14159265358979323846 / (2 * kern_constant*hsml_avg) + 3.14159265358979323846 / 6)*dy / (r*mass[idy]);
			}
		}
	}*/
	/*else if ((idx - 1) / ntotal == 6)
	{
	double dx, dy, r;
	int neighbor_index;
	int index = max_neighbors*(idx - 6 * ntotal);
	int idy = idx - 6 * ntotal;
	exdv1dt[idy] = 0;
	int min_index = 0;
	double dx_min = 1e30;

	for (int i = 1; i < neighbors_count[idy]; i++)
	{
	neighbor_index = neighbors[index + i];
	if (ghost[neighbor_index] && !ghost[idy] && state[idy] == LIQUID)
	{

	dx = (x1[idy] - x1[neighbor_index]);
	if (abs(dx) < abs(dx_min) && abs(x2[idy] - x2[neighbor_index]) < 0.7*part_spacing)
	{
	dx_min = dx;
	min_index = neighbor_index;
	}
	}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////
	///////////// EXTERNAL FORCE
	if (min_index)
	{
	double rr0 = 0.5*part_spacing;
	double dd = 0.01;//was 0.01
	r = 0.5*part_spacing - abs(dx_min);
	double f = (pow(rr0 / r, 12) - pow(rr0 / r, 4)) / pow(r, 2);
	//force scaled using mass of the ghost particle

	if(dx_min>0)exdv1dt[idy] -= dd*r*f;
	else exdv1dt[idy] += dd*r*f;
	//exdv2dt[idy] += mass[neighbor_index] / (917 * part_spacing*part_spacing)*dd*dy*f;

	}

	}
	if ((idx - 1) / ntotal == 7)
	{
	double dy, r;
	int neighbor_index;
	int index = max_neighbors*(idx - 7 * ntotal);
	int idy = idx - 7 * ntotal;
	exdv2dt[idy] = 0;
	int min_index = 0;
	double dy_min = 1e30;

	for (int i = 1; i < neighbors_count[idy]; i++)
	{
	neighbor_index = neighbors[index + i];
	if (ghost[neighbor_index] && !ghost[idy] && state[idy] == LIQUID)
	{

	dy = (x2[idy] - x2[neighbor_index]);
	if (abs(dy) < abs(dy_min) && abs(x1[idy] - x1[neighbor_index]) < 0.7*part_spacing)
	{
	dy_min = dy;
	min_index = neighbor_index;
	}
	}
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////
	///////////// EXTERNAL FORCE
	if (min_index)
	{
	double rr0 = 0.35*part_spacing;
	double dd = 0.01;//was 0.01
	r = abs(dy_min);
	double f = (pow(rr0 / r, 12) - pow(rr0 / r, 4)) / pow(r, 2);
	//force scaled using mass of the ghost particle

	if (dy_min > 0)exdv2dt[idy] += dd*r*f;
	else exdv2dt[idy] -= dd*r*f;
	//exdv2dt[idy] += mass[neighbor_index] / (917 * part_spacing*part_spacing)*dd*dy*f;

	}

	}*/
}

__global__ void leapfrog_part_1(int ntotal, double *v1, double *v2, double *mass, double *ref_rho, double *energy, double *dhdt, double *v1_old, double *v2_old, double *rho, double *rho_old, double *energy_old, int *wall, int *fluidcode, int *ghost, double *drhodt, double *dv1dt, double *dv2dt, double *dt)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > 0 && idx <= ntotal)
	{
		if (!wall[idx] && !ghost[idx] && fluidcode[idx] != -1)
		{
			if (ref_rho[idx] != 917)rho_old[idx] = rho[idx];
			if (ref_rho[idx] != 917)rho[idx] += *dt / 2 * drhodt[idx];
			//if (ref_rho[idx] != 917)rho[idx] = drhodt[idx];
			if (ref_rho[idx] != 917)v1_old[idx] = v1[idx];
			if (ref_rho[idx] != 917)v2_old[idx] = v2[idx];
			if (ref_rho[idx] != 917)v1[idx] += *dt / 2 * dv1dt[idx];
			if (ref_rho[idx] != 917)v2[idx] += *dt / 2 * dv2dt[idx];
			energy_old[idx] = energy[idx];
			energy[idx] += *dt / 2 * mass[idx] * dhdt[idx] / rho[idx];
		}
		if (ghost[idx])
		{
			rho_old[idx] = rho[idx];
			rho[idx] += *dt / 2 * drhodt[idx];
			energy_old[idx] = energy[idx];
			energy[idx] += *dt / 2 * mass[idx] * dhdt[idx] / rho[idx];
		}
	}
}

__global__ void leapfrog_part_2(int ntotal, double *v1, double *v2, double *mass, double *ref_rho, double *temp, double *spheat, double *energy, double *dhdt, double *v1_old, double *v2_old, double *av1, double *av2, double *x1, double *x2, double *rho, double *rho_old, double *energy_old, double *hsml, int *wall, int *fluidcode, int* ghost, int* state, double *drhodt, double *dv1dt, double *dv2dt, double *dhsmldt, double *dt, int *high_compression_flag)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx == 0)*high_compression_flag = 0;
	if (idx > 0 && idx <= ntotal)
	{
		if (!wall[idx] && !ghost[idx] && fluidcode[idx] != -1)
		{
			if (ref_rho[idx] != 917)
			{
				rho[idx] = rho_old[idx] + *dt*drhodt[idx];
				//if (ref_rho[idx] != 917)rho[idx] = drhodt[idx];
				if ((rho[idx] - ref_rho[idx]) > 0.03*ref_rho[idx])*high_compression_flag = 1;
			}
			if (ref_rho[idx] != 917)v1[idx] = v1_old[idx] + *dt*dv1dt[idx] + av1[idx];
			if (ref_rho[idx] != 917)v2[idx] = v2_old[idx] + *dt*dv2dt[idx] + av2[idx];
			if (ref_rho[idx] != 917)x1[idx] = x1[idx] + *dt*v1[idx];
			if (ref_rho[idx] != 917)x2[idx] = x2[idx] + *dt*v2[idx];
			hsml[idx] = hsml[idx] + *dt*dhsmldt[idx];
			energy[idx] = energy_old[idx] + *dt* mass[idx] * dhdt[idx] / rho[idx];
		}
		if (ghost[idx])
		{
			rho[idx] = rho_old[idx] + *dt*drhodt[idx];
			energy[idx] = energy_old[idx] + *dt* mass[idx] * dhdt[idx] / rho[idx];
		}

	}
}

__global__ void find_extremes(int ntotal, double *hsml, double *eta, double *rho, double *v1, double *v2, double *dv1dt, double *dv2dt, double *hsml_min, double *eta_max, double *dvdt_max, double *rho_min, double *v2_max, int *fluidcode)
{

	for (int i = 1; i <= ntotal; i++)
	{
		if (i == 1)
		{
			*hsml_min = hsml[1];
			*eta_max = eta[1];
			*dvdt_max = (dv1dt[1] * dv1dt[1] + dv2dt[1] * dv2dt[1]);
			*rho_min = rho[1];
			*v2_max = (v1[1] * v1[1] + v2[1] * v2[1]);
		}
		else if (fluidcode[i] != -1)
		{
			*hsml_min = min(*hsml_min, hsml[i]);
			*eta_max = max(*eta_max, eta[i]);
			*dvdt_max = max(*dvdt_max, (dv1dt[i] * dv1dt[i] + dv2dt[i] * dv2dt[i]));
			*rho_min = min(*rho_min, rho[i]);
			*v2_max = max(*v2_max, (v1[i] * v1[i] + v2[i] * v2[i]));
		}
	}
	*dvdt_max = sqrt(*dvdt_max);

}

__global__ void timestep_calc(double c, double *hsml_min, double *eta_max, double *rho_min, double *dvdt_max, double *dt, int *reduce_timestep_flag)
{
	*dt = 100;
	*dt = min(*dt, 0.25**hsml_min / c);
	*dt = min(*dt, 0.125**hsml_min**hsml_min**rho_min / *eta_max);
	*dt = min(*dt, 0.25*sqrt(*hsml_min / *dvdt_max));
	//if (*reduce_timestep_flag)*dt /= 10;
	//printf("%.12f\n", *dt);
}

__global__ void particle_state(int ntotal, double *temp, double *energy, double *sp_heat, double *therm_cond, int *fluidcode, int* wall, int* state, int* Interface, int* ghost, double *mass, double latent_heat, double *dmgdt, int *redis_flag, double *dt)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx == 0)*redis_flag = 0;
	if (idx > 0 && idx <= ntotal)
	{

		dmgdt[idx] = 0;
		if (fluidcode[idx] == 1 && !wall[idx])
		{
			if (state[idx] == SOLID)
			{
				if (mass[idx] != 0)temp[idx] = energy[idx] / (mass[idx] * sp_heat[idx]);
				else temp[idx] = 0;
				Interface[idx] = 0;
			}
			if (state[idx] == LIQUID)
			{
				if (energy[idx] - mass[idx] * latent_heat <= 0)
				{
					dmgdt[idx] = (mass[idx] * latent_heat - energy[idx]) / latent_heat;
					temp[idx] = 0;
					mass[idx] -= dmgdt[idx];
					dmgdt[idx] = dmgdt[idx] / (*dt);
					//if (mass[idx] < 0.2*mass[1]) *redis_flag = 1;
					energy[idx] = mass[idx] * latent_heat;

				}
				else
				{
					//if (idx == 20)printf("yup");
					temp[idx] = (energy[idx] - latent_heat*mass[idx]) / (mass[idx] * sp_heat[idx]);
					//Interface[idx] = 0;
				}
			}
		}
	}
}

__global__ void derivative_sum_util(int ntotal, double *dv1dt, double *dv2dt, double *dS1dt, double *dS2dt, double *exdv1dt, double *exdv2dt)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > 0 && idx <= ntotal)
	{
		dv1dt[idx] += dS1dt[idx] + exdv1dt[idx];
		dv2dt[idx] += dS2dt[idx] + exdv2dt[idx];
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////// PRIMARY VARIABLES ASSOCIATED WITH PARTICLES
////// x1
////// x2
////// v1
////// v2
////// rho
////// rho_ref
////// pressure
////// mass
////// eta
////// hsml - smoothing length
////// temp
////// therm_cond
////// sp_heat
////// energy
////// fluid_code
////// state
////// ghost
////// ghost_state
////// wall
////// boundry
////// Interface
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


void simulation()
{
	int ntotal;
	double length;
	double part_spacing;
	int timestep = 1;
	int thermal_properties_input_flag = 0;
	ifstream gridfile;
	gridfile.open("grid.txt");
	gridfile >> timestep >> length >> part_spacing >> ntotal;
	int arraySize = 5 * ntotal;
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	//////// PARTICLE PROPERTIES

	double *mass_h = new double[arraySize + 1], *mass;
	double *rho_h = new double[arraySize + 1], *rho;
	double *ref_rho_h = new double[arraySize + 1], *ref_rho;
	double *eta_h = new double[arraySize + 1], *eta;
	double *pressure_h = new double[arraySize + 1], *pressure;
	double *v1_h = new double[arraySize + 1], *v1;
	double *v2_h = new double[arraySize + 1], *v2;
	double *x1_h = new double[arraySize + 1], *x1;
	double *x2_h = new double[arraySize + 1], *x2;
	double *hsml_h = new double[arraySize + 1], *hsml;
	double *temp_h = new double[arraySize + 1], *temp;
	double *therm_cond_h = new double[arraySize + 1], *therm_cond;
	double *spheat_h = new double[arraySize + 1], *spheat;
	double *energy_h = new double[arraySize + 1], *energy;
	double *dmgdt_h = new double[arraySize + 1], *dmgdt;
	double *v1_old_h = new double[arraySize + 1], *v1_old;
	double *v2_old_h = new double[arraySize + 1], *v2_old;
	double *rho_old_h = new double[arraySize + 1], *rho_old;
	double *energy_old_h = new double[arraySize + 1], *energy_old;

	//COLOR CODES
	int *master_tag_h = new int[arraySize + 1], *master_tag;
	int *fluid_code_h = new int[arraySize + 1], *fluid_code;
	int *boundry_h = new int[arraySize + 1], *boundry;
	int *wall_h = new int[arraySize + 1], *wall;
	int *Interface_h = new int[arraySize + 1], *Interface;
	int *state_h = new int[arraySize + 1], *state;
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////// VARIABLE DEFAULTS
	for (int i = 0; i <= arraySize; i++)
	{
		master_tag_h[i] = 1;
		mass_h[i] = 0;
		rho_h[i] = 0;
		ref_rho_h[i] = 0;
		eta_h[i] = 0;
		pressure_h[i] = 0;
		v1_h[i] = 0;
		v2_h[i] = 0;
		x1_h[i] = 0;
		x2_h[i] = 0;
		hsml_h[i] = 0;
		temp_h[i] = 0;
		therm_cond_h[i] = 0;
		spheat_h[i] = 0;
		energy_h[i] = 0;
		dmgdt_h[i] = 0;
		fluid_code_h[i] = 0;
		boundry_h[i] = 0;
		wall_h[i] = 0;
		Interface_h[i] = 0;
		state_h[i] = 0;
		v1_old_h[i] = 0;
		v2_old_h[i] = 0;
		rho_old_h[i] = 0;
		energy_old_h[i] = 0;

	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////// FILE READ
	for (int i = 1; i <= ntotal; i++) {
		gridfile >> x1_h[i] >> x2_h[i] >> v1_h[i] >> v2_h[i] >> rho_h[i] >> ref_rho_h[i] >> pressure_h[i] >> mass_h[i] >> eta_h[i] >> hsml_h[i];
		gridfile >> fluid_code_h[i] >> boundry_h[i] >> wall_h[i];
	}


	// CHECK IF THERMAL PROPERTIS ARE INCLUDED IN THE FILE
	gridfile >> thermal_properties_input_flag;
	// READ THERMAL PROPERTIES
	//if (thermal_properties_input_flag)thermal_properties_input_flag = 0;
	if (thermal_properties_input_flag)
	{
		for (int i = 1; i <= ntotal; i++) {
			gridfile >> temp_h[i] >> energy_h[i] >> therm_cond_h[i] >> spheat_h[i] >> state_h[i];
		}
	}
	//thermal_properties_input_flag = 0;
	// CHECK if 'old' variable values are included in the file;
	int old_variables_flag;
	gridfile >> old_variables_flag;
	if (old_variables_flag)
	{
		string s;
		getline(gridfile, s);
		for (int i = 1; i <= ntotal; i++) {
			gridfile >> v1_old_h[i] >> v2_old_h[i] >> energy_old_h[i] >> rho_old_h[i];
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////



	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////// DEVICE-SIDE MEMORY ALLOCATION
	size_t size = (arraySize + 1) * sizeof(double);
	cudaMalloc((void**)&x1, size);
	cudaMalloc((void**)&x2, size);
	cudaMalloc((void**)&v1, size);
	cudaMalloc((void**)&v2, size);
	cudaMalloc((void**)&rho, size);
	cudaMalloc((void**)&mass, size);
	cudaMalloc((void**)&pressure, size);
	cudaMalloc((void**)&ref_rho, size);
	cudaMalloc((void**)&eta, size);
	cudaMalloc((void**)&hsml, size);
	cudaMalloc((void**)&temp, size);
	cudaMalloc((void**)&energy, size);
	cudaMalloc((void**)&spheat, size);
	cudaMalloc((void**)&therm_cond, size);
	cudaMalloc((void**)&dmgdt, size);
	cudaMalloc((void**)&v1_old, size);
	cudaMalloc((void**)&v2_old, size);
	cudaMalloc((void**)&energy_old, size);
	cudaMalloc((void**)&rho_old, size);

	size = (arraySize + 1) * sizeof(int);
	cudaMalloc((void**)&master_tag, size);
	cudaMalloc((void**)&Interface, size);
	cudaMalloc((void**)&fluid_code, size);
	cudaMalloc((void**)&boundry, size);
	cudaMalloc((void**)&wall, size);
	cudaMalloc((void**)&state, size);

	size = (arraySize + 1) * sizeof(double);
	cudaMemcpy(x1, x1_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(x2, x2_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v1, v1_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v2, v2_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(rho, rho_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(mass, mass_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pressure, pressure_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(ref_rho, ref_rho_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(eta, eta_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(hsml, hsml_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(temp, temp_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(energy, energy_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(therm_cond, therm_cond_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(spheat, spheat_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dmgdt, dmgdt_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v1_old, v1_old_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(v2_old, v2_old_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(energy_old, energy_old_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(rho_old, rho_old_h, size, cudaMemcpyHostToDevice);

	size = (arraySize + 1) * sizeof(int);
	cudaMemcpy(master_tag, master_tag_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Interface, Interface_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(fluid_code, fluid_code_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(boundry, boundry_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(wall, wall_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(state, state_h, size, cudaMemcpyHostToDevice);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////// AUXILLIARY ARRAYS
	int *bucket_index;
	size = (arraySize + 1) * sizeof(int);
	cudaMalloc((void**)&bucket_index, size);

	double *dv1dt, *dv2dt, *dS1dt, *dS2dt, *exdv1dt, *exdv2dt, *drhodt, *av1, *av2, *dhdt, *dhsmldt;
	size = (arraySize + 1) * sizeof(double);
	cudaMalloc((void**)&dv1dt, size);
	cudaMalloc((void**)&dv2dt, size);
	cudaMalloc((void**)&dS1dt, size);
	cudaMalloc((void**)&dS2dt, size);
	cudaMalloc((void**)&exdv1dt, size);
	cudaMalloc((void**)&exdv2dt, size);
	cudaMalloc((void**)&drhodt, size);
	cudaMalloc((void**)&av1, size);
	cudaMalloc((void**)&av2, size);
	cudaMalloc((void**)&dhdt, size);
	cudaMalloc((void**)&dhsmldt, size);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////// GHOST PARTICLE VARIABLES
	int *ghost_state, *ghost_state_h = new int[arraySize + 1];
	int *ghost, *ghost_h = new int[arraySize + 1];
	double *dist = new double[arraySize + 1];
	size = (arraySize + 1) * sizeof(int);
	cudaMalloc((void**)&ghost_state, size);
	cudaMalloc((void**)&ghost, size);
	for (int i = 0; i < arraySize; i++)
	{
		ghost_h[i] = 0;
		ghost_state_h[i] = 0;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////// BOUNDARY VARIABLES
	///// used to store the center coordinates and radius of the boundry circle, eliminating the need to tag boundry particles.
	pair <int, double> solid_line(GHOST_START_INDEX, 0);
	int boundary_variables_input_flag = 0;
	gridfile >> boundary_variables_input_flag;
	if (boundary_variables_input_flag)
	{
		gridfile >> solid_line.first;
		gridfile >> solid_line.second;
	}
	//set ghost state
	for (int i = 1; i <= arraySize; i++)
	{
		if (i >= GHOST_START_INDEX && i <= GHOST_END_INDEX)
		{
			if (i <= solid_line.first || i >= GHOST_END_INDEX - (solid_line.first - GHOST_START_INDEX))ghost_state_h[i] = 0;
			else ghost_state_h[i] = 1;
		}
		else ghost_state_h[i] = 0;
	}
	cudaMemcpy(ghost_state, ghost_state_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(ghost, ghost_h, size, cudaMemcpyHostToDevice);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////// KERNEL RELATED ARRAYS
	int *neighbors, *neighbors_count, *neighbors_h = new int[(int)(max_neighbors*0.4*arraySize) + 1], *neighbors_count_h = new int[(int)(max_neighbors*0.4*arraySize) + 1];
	double *w, *dwdx, *dwdy;
	size = (max_neighbors*0.4*arraySize + 1) * sizeof(int);
	cudaMalloc((void**)&neighbors, size);
	cudaMalloc((void**)&neighbors_count, size);
	size = (max_neighbors*0.4*arraySize + 1) * sizeof(double);
	cudaMalloc((void**)&w, size);
	cudaMalloc((void**)&dwdx, size);
	cudaMalloc((void**)&dwdy, size);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	double *hsml_max, *tmp = new double[arraySize + 1]; cudaMalloc((void**)&hsml_max, sizeof(double));

	int *nvoxel_length; cudaMalloc((void**)&nvoxel_length, sizeof(int));


	double epsilon = 0.3;
	double alpha = 0.03;
	int block_size = 320;
	int n_blocks = ntotal / block_size + 1;
	bool density_norm = false;
	int *compression_flag, compression_flag_h[1]; cudaMalloc((void**)&compression_flag, sizeof(int));
	int *evolution_flag, flag[1]; cudaMalloc((void**)&evolution_flag, sizeof(int));
	int *redis_flag, redis_flag_h[1]; cudaMalloc((void**)&redis_flag, sizeof(int));
	int *reduce_timestep_flag, reduce_timestep_flag_h[1]; cudaMalloc((void**)&reduce_timestep_flag, sizeof(int));
	reduce_timestep_flag_h[0] = 0; cudaMemcpy(reduce_timestep_flag, reduce_timestep_flag_h, sizeof(int), cudaMemcpyHostToDevice);
	int count_timestep = 0;
	int *new_ntotal, new_ntotal_h[1]; cudaMalloc((void**)&new_ntotal, sizeof(int));
	int old_ntotal = ntotal;
	int latent_heat = 80 * 4180 / 7;
	double total_mass = 0, initial_total_mass;
	double liquid_mass_old = 0;

	double *hsml_min, *eta_max, *dvdt_max, *rho_min, *dt, *v2_max;
	double *dt_h = new double[1];
	double *v2max_h = new double[1];
	cudaMalloc((void**)&hsml_min, sizeof(double));
	cudaMalloc((void**)&eta_max, sizeof(double));
	cudaMalloc((void**)&rho_min, sizeof(double));
	cudaMalloc((void**)&dvdt_max, sizeof(double));
	cudaMalloc((void**)&dt, sizeof(double));
	cudaMalloc((void**)&v2_max, sizeof(double));

	//////////////////////////////////////////////////////////////////////////////////////
	/////// RENDER RELATED ARRAYS
	double **tmp_render = new double*[3]; for (int i = 0; i < 3; i++) tmp_render[i] = new double[arraySize + 1];
	vector <int> marker_vector;
	//////////////////////////////////////////////////////////////////////////////////////

	double st, en, totTime = 0;

	int max_timestep = 130000;
	double time_elapsed = 0;
	if (timestep > max_timestep)max_timestep = timestep;

	//Get total mass of liquid+solid+ghost particles;
	initial_total_mass = initial_total_mass_util(ntotal, mass_h, wall_h);
	bool recompute_kernel = true;
	cout << "total mass: " << 1000 * initial_total_mass << endl;



	//*************************************************************************************
	// The primary simulation loop
	//*************************************************************************************
	while (timestep <= max_timestep)
	{


		st = clock();
		*flag = 0;
		n_blocks = ntotal / block_size + 1;

		if (timestep != 1) leapfrog_part_1 << <n_blocks, block_size >> >(ntotal, v1, v2, mass, ref_rho, energy, dhdt, v1_old, v2_old, rho, rho_old, energy_old, wall, fluid_code, ghost, drhodt, dv1dt, dv2dt, dt);
		max_hsml << <1, 1 >> >(ntotal, hsml, hsml_max, fluid_code);

		if (recompute_kernel)
		{
			index_particles << <n_blocks, block_size >> >(ntotal, bucket_index, x1, x2, length, kern_constant, hsml_max, nvoxel_length);
			compute_kernel << <n_blocks, block_size >> >(ntotal, bucket_index, nvoxel_length, x1, x2, hsml, neighbors, neighbors_count, w, dwdx, dwdy, ghost, ghost_state, wall, fluid_code, Interface, kern_constant, part_spacing, max_neighbors);
		}

		equation_of_state << <n_blocks, block_size >> >(ntotal, pressure, rho, ref_rho, fluid_code, Interface, p0, gamma, back_press, c);



		n_blocks = 8 * ntotal / block_size + 1;
		compute_derivatives << <n_blocks, block_size >> >(ntotal, x1, x2, v1, v2, rho, mass, pressure, hsml, neighbors, neighbors_count, w, dwdx, dwdy, eta, therm_cond, temp, dv1dt, dv2dt, dS1dt, dS2dt, exdv1dt, exdv2dt, av1, av2, drhodt, dhdt, dmgdt, dhsmldt, wall, fluid_code, Interface, state, ghost_state, ghost, length, epsilon, gamma, c, back_press, max_neighbors, kern_constant, constant_gravity, S_ll, S_ls, part_spacing);
		n_blocks = ntotal / block_size + 1;

		//force_comparison_util(dv1dt, dv2dt,dS1dt, dS2dt, exdv1dt, exdv2dt, ntotal);

		derivative_sum_util << <n_blocks, block_size >> >(ntotal, dv1dt, dv2dt, dS1dt, dS2dt, exdv1dt, exdv2dt);

		find_extremes << <1, 1 >> >(ntotal, hsml, eta, rho, v1, v2, dv1dt, dv2dt, hsml_min, eta_max, dvdt_max, rho_min, v2_max, fluid_code);

		timestep_calc << <1, 1 >> >(c, hsml_min, eta_max, rho_min, dvdt_max, dt, reduce_timestep_flag);
		cudaMemcpy(dt_h, dt, sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(v2max_h, v2_max, sizeof(double), cudaMemcpyDeviceToHost);
		if (ref_velocity*ref_velocity < *v2max_h)
		{
			cout << "Particle velocity exceeded reference velocity: " << sqrt(*v2max_h) << endl;
			getchar();
		}
		time_elapsed += dt_h[0];
		if (timestep == 1 && !old_variables_flag) leapfrog_part_1 << <n_blocks, block_size >> >(ntotal, v1, v2, mass, ref_rho, energy, dhdt, v1_old, v2_old, rho, rho_old, energy_old, wall, fluid_code, ghost, drhodt, dv1dt, dv2dt, dt);
		else leapfrog_part_2 << <n_blocks, block_size >> >(ntotal, v1, v2, mass, ref_rho, temp, spheat, energy, dhdt, v1_old, v2_old, av1, av2, x1, x2, rho, rho_old, energy_old, hsml, wall, fluid_code, ghost, state, drhodt, dv1dt, dv2dt, dhsmldt, dt, compression_flag);
		cudaMemcpy(compression_flag_h, compression_flag, sizeof(int), cudaMemcpyDeviceToHost);

		if (*compression_flag_h)
		{
			cout << "particles density deviation beyond 3%" << endl;
			//getchar();
		}

		//if(timestep >300 && timestep<1000) temp_move_ghost_util << <n_blocks, block_size >> >(ntotal, x2, v2,mass, part_spacing, dt);

		cudaDeviceSynchronize();





	

		cudaMemcpy(x1_h, x1, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(x2_h, x2, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);

		en = clock();
		render_scene(ntotal, x1_h, x2_h, mass_h, fluid_code_h, wall_h, state_h, ghost_state_h, ghost_h, tmp_render, marker_vector, part_spacing);
		//render_scene_mass(ntotal, x1_h, x2_h, mass_h, wall_h, state_h, ghost_h, tmp_render, part_spacing);
		time_elapsed += *dt_h;

		cout << "timestep no: " << timestep << " timestep length: " << *dt_h << " total time elapsed: " << time_elapsed << endl;
		//cout << "Time elapsed: " << time_elapsed << endl;
		//render_scene_mass(ntotal, x1_h, x2_h, mass_h, wall_h, state_h, ghost_h, tmp_render, length, 40, part_spacing);
		cout << "timestep no: " << timestep << "      Computation time: " << en - st << endl;
		//cout << "ntotal: " << ntotal << "old ntotal: " << old_ntotal <<  endl;
		//cout << "mass ratio at 800: : " << mass_h[865]/(917*part_spacing*part_spacing) << endl;
		//cout << "mass balance (should be 1): " << total_mass / initial_total_mass << endl;
		//cout << "liquid mass/old mass: ";


		//Dump data every hundred steps
		if (timestep % 1000 == 0 || timestep == 50998)
		{

			cout << "DUMPING DATA....." << endl;
			dump_data(ntotal, x1, x2, v1, v2, rho, ref_rho, pressure, mass, eta, hsml, temp, energy, therm_cond, spheat, fluid_code, state, ghost, ghost_state, wall, boundry, Interface, v1_old, v2_old, energy_old, rho_old, x1_h, x2_h, v1_h, v2_h, rho_h, ref_rho_h, pressure_h, mass_h, eta_h, hsml_h, temp_h, energy_h, therm_cond_h, spheat_h, fluid_code_h, state_h, ghost_h, ghost_state_h, wall_h, boundry_h, Interface_h, v1_old_h, v2_old_h, energy_old_h, rho_old_h, arraySize, length, part_spacing, timestep, solid_line);

		}
		timestep++;
		//getchar();
		marker_vector.clear();
		if (timestep > max_timestep)
		{
			cout << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl << endl;
			cout << "Max timestep reached. Enter the number of extra iterations to perform:" << endl;
			int n; cin >> n;
			max_timestep += n;
		}
	}



}

////////////////////////////////////////////////////////////////////////////////////////////
//////  RENDER SCENE
int render_scene(int ntotal, double *x1, double *x2, double *mass, int *fluid_code, int *wall, int *state, int *test_array, int *ghost, double** temp, vector <int> &marker_vector, double part_spacing)
{
	int i;
	for (i = 1; i <= ntotal; i++)
	{
		//temp[1][i] = (x1[i] + 5 * length / nPartSide) / (length + 10 * length / nPartSide);
		//temp[2][i] = (x2[i] + 5 * length / nPartSide) / (0.7*length + 3 * length / nPartSide);
		temp[1][i] = x1[i] / RENDER_X_SCALE;
		temp[2][i] = x2[i] / RENDER_Y_SCALE;
		temp[1][i] = 2 * (temp[1][i] + RENDER_X_OFFSET);
		temp[2][i] = 2 * (temp[2][i] + RENDER_Y_OFFSET);
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //this was taken from the render function

	glBegin(GL_TRIANGLES);


	for (i = 1; i <= ntotal; i++)
	{
		if (true)glColor3f(1, 1, 1);
		if (fluid_code[i] == 1)glColor3f(0, 1, 0);

		if (state[i] == 0)glColor3f(0, 0, 1);
		if (wall[i])glColor3f(1, 0, 0);
		if (ghost[i])glColor3f(1, 1, 1);
		//if (i > ntotal/5) glColor3f(1, 0, 0);
		//if (test_array[i])glColor3f(1, 0.647059, 0);
		//if (i == 357)glColor3f(1, 0, 0);
		if (fluid_code[i] == -1)glColor3f(0, 0, 0);

		if (i)
		{

			glVertex3f(temp[1][i], temp[2][i] + 0.004, 0.0);
			glVertex3f(temp[1][i] - 0.004, temp[2][i] - 0.004, 0.0);
			glVertex3f(temp[1][i] + 0.004, temp[2][i] - 0.004, 0.0);
		}



	}
	for (int j = 0; j < marker_vector.size(); j++)
	{
		i = marker_vector[j];
		glColor3f(1, 0, 0);
		glVertex3f(temp[1][i], temp[2][i] + 0.004, 0.0);
		glVertex3f(temp[1][i] - 0.004, temp[2][i] - 0.004, 0.0);
		glVertex3f(temp[1][i] + 0.004, temp[2][i] - 0.004, 0.0);
	}
	glEnd();

	glutSwapBuffers();

	return 0;
}
////////////////////////////////////////////////////////////////////////////////////////////
int render_scene_mass(int ntotal, double *x1, double *x2, double* mass, int *wall, int *state, int *ghost, double** temp, double part_spacing)
{
	int i;
	for (i = 1; i <= ntotal; i++)
	{
		temp[1][i] = x1[i] / RENDER_X_SCALE;
		temp[2][i] = x2[i] / RENDER_Y_SCALE;
		temp[1][i] = 2 * (temp[1][i] + RENDER_X_OFFSET);
		temp[2][i] = 2 * (temp[2][i] + RENDER_Y_OFFSET);
	}
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLineWidth(2.5);
	glColor3f(1, 1, 1);
	glBegin(GL_LINES);
	for (i = 1; i <= ntotal; i++)
	{
		if (ghost[i])
		{

			glVertex3f(temp[1][i], 0.0, 0.0);
			glVertex3f(temp[1][i], mass[i] / (917 * part_spacing*part_spacing), 0.0);
		}
	}
	glEnd();

	glutSwapBuffers();
	return 0;
}
////////////////////////////////////////////////////////////////////////////////////////////
///// DUMP DATA
void dump_data(int ntotal, double *x1, double *x2, double *v1, double *v2, double *rho, double *ref_rho, double *pressure, double *mass, double *eta, double *hsml, double *temp, double *energy, double *therm_cond, double *sp_heat, int *fluid_code, int *state, int *ghost, int* ghost_state, int* wall, int* boundry, int* Interface, double *v1_old, double *v2_old, double *energy_old, double *rho_old, double *x1_h, double *x2_h, double *v1_h, double *v2_h, double *rho_h, double *ref_rho_h, double *pressure_h, double *mass_h, double *eta_h, double *hsml_h, double *temp_h, double *energy_h, double *therm_cond_h, double *sp_heat_h, int *fluid_code_h, int *state_h, int *ghost_h, int* ghost_state_h, int* wall_h, int* boundry_h, int* Interface_h, double *v1_old_h, double *v2_old_h, double *energy_old_h, double *rho_old_h, int arraySize, double length, double part_spacing, int index, pair <int, double> solid_line)
{
	cudaMemcpy(x1_h, x1, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(x2_h, x2, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(v1_h, v1, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(v2_h, v2, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(rho_h, rho, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(ref_rho_h, ref_rho, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(pressure_h, pressure, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(mass_h, mass, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(eta_h, eta, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hsml_h, hsml, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(temp_h, temp, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(energy_h, energy, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(therm_cond_h, therm_cond, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(sp_heat_h, sp_heat, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);

	cudaMemcpy(fluid_code_h, fluid_code, ((arraySize + 1) * sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy(state_h, state, ((arraySize + 1) * sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy(ghost_h, ghost, ((arraySize + 1) * sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy(ghost_state_h, ghost_state, ((arraySize + 1) * sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy(wall_h, wall, ((arraySize + 1) * sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy(boundry_h, boundry, ((arraySize + 1) * sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy(Interface_h, Interface, ((arraySize + 1) * sizeof(int)), cudaMemcpyDeviceToHost);

	cudaMemcpy(v1_old_h, v1_old, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(v2_old_h, v2_old, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(energy_old_h, energy_old, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(rho_old_h, rho_old, (arraySize + 1) * sizeof(double), cudaMemcpyDeviceToHost);

	ofstream pos;
	pos.open("dump_" + to_string(index) + ".txt", 'w');
	pos << index + 1 << endl << length << endl << part_spacing << endl << ntotal << endl;

	for (int i = 1; i <= ntotal; i++)
	{
		pos << x1_h[i] << ' ' << x2_h[i] << ' ' << v1_h[i] << ' ' << v2_h[i] << ' ' << rho_h[i] << ' ' << ref_rho_h[i] << ' ' << pressure_h[i] << ' ' << mass_h[i] << ' ' << eta_h[i] << ' ' << hsml_h[i];
		pos << ' ' << fluid_code_h[i] << ' ' << boundry_h[i] << ' ' << wall_h[i] << endl;
	}
	pos << '1' << endl;
	for (int i = 1; i <= ntotal; i++)
	{
		pos << temp_h[i] << ' ' << energy_h[i] << ' ' << therm_cond_h[i] << ' ' << sp_heat_h[i] << ' ' << state_h[i] << endl;
	}
	pos << '1' << endl;
	for (int i = 1; i <= ntotal; i++)
	{
		pos << v1_old_h[i] << ' ' << v2_old_h[i] << ' ' << energy_old_h[i] << ' ' << rho_old_h[i] << endl;
	}
	pos << '1' << endl;
	pos << solid_line.first << ' ' << solid_line.second << endl;
	//Uncomment below to take screenshots. Needs FreeImage Library.
	/*BYTE* pixels = new BYTE[3 * 1000 * 1000];

	glReadPixels(0, 0, 1000, 1000, GL_RGB, GL_UNSIGNED_BYTE, pixels);

	// Convert to FreeImage format & save to file
	FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, 1000, 1000, 3 * 1000, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
	string name = to_string(index) + ".bmp";
	char *a = new char[name.size() + 1];
	a[name.size()] = 0;
	memcpy(a, name.c_str(), name.size());

	FreeImage_Save(FIF_BMP, image, a, 0);

	// Free resources
	FreeImage_Unload(image);
	delete[] pixels;*/
	pos.close();
}
///////////////////////////////////////////////////////////////////////////////////////////
////// TOTAL MASS CALCULATOR
double initial_total_mass_util(int ntotal, double *mass_h, int *wall_h)
{
	double tot = 0;
	for (int i = 1; i <= ntotal; i++)
	{
		if (!wall_h[i])tot += mass_h[i];
	}
	return tot;
}
///////////////////////////////////////////////////////////////////////////////////////////
//////ARRAY DUMP FUNCTION
void write_array(int *arr, int len, string name)
{
	int size = len * sizeof(int);
	int *host_array = new int[len];
	cudaMemcpy(host_array, arr, size, cudaMemcpyDeviceToHost);
	ofstream pos;
	pos.open(name + ".txt");
	for (int i = 0; i < len; i++)
	{
		pos << host_array[i] << endl;
	}
}
void write_array(double *arr, int len, string name)
{
	int size = len * sizeof(double);
	double *host_array = new double[len];
	cudaMemcpy(host_array, arr, size, cudaMemcpyDeviceToHost);
	ofstream pos;
	pos.open(name + ".txt");
	for (int i = 0; i < len; i++)
	{
		pos << host_array[i] / 7.20269e-7 << endl;
	}

	delete[] host_array;
}
///////////////////////////////////////////////////////////////////////////////////////////
////// PARTICLE FORCE COMPARISON UTIL
void force_comparison_util(double *dv1dt, double *dv2dt, double *dS1dt, double *dS2dt, double *exdv1dt, double *exdv2dt, int len)
{
	int size = len * sizeof(double);
	double *dv1dt_h = new double[len + 1];
	double *dv2dt_h = new double[len + 1];
	double *dS1dt_h = new double[len + 1];
	double *dS2dt_h = new double[len + 1];
	double *exdv1dt_h = new double[len + 1];
	double *exdv2dt_h = new double[len + 1];
	cudaMemcpy(dv1dt_h, dv1dt, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dv2dt_h, dv2dt, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dS1dt_h, dS1dt, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dS2dt_h, dS2dt, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(exdv2dt_h, exdv2dt, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(exdv1dt_h, exdv1dt, size, cudaMemcpyDeviceToHost);

	ofstream pos;
	pos.open("force_comparison.txt");
	pos << "pressure - surface tension - external force - gravity" << endl;
	for (int i = 1; i <= len; i++)
	{
		pos << sqrt(pow(dv1dt_h[i], 2) + pow(dv2dt_h[i] + constant_gravity, 2)) / constant_gravity << "  " << sqrt(pow(dS1dt_h[i], 2) + pow(dS2dt_h[i], 2)) / constant_gravity << ' ' << sqrt(pow(exdv1dt_h[i], 2) + pow(exdv2dt_h[i], 2)) / constant_gravity << "  1" << endl;
	}
	pos.close();
	delete[] dv1dt_h, dv2dt_h, dS1dt_h, dS2dt_h, exdv1dt_h, exdv2dt_h;
}
