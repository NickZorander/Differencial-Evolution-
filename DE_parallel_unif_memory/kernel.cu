#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <curand_kernel.h>
#include <curand.h>

#include <time.h>
#include <random>

#include <stdio.h>
#include <iomanip>
#include <csignal>

typedef unsigned long ul;
typedef unsigned int ui;
using namespace std;

#define CHECK_CUDA_RESULT { cudaError_t result = cudaGetLastError(); if (result != 0) { printf("CUDA call on line %d returned error %d: %s\n", __LINE__, result, cudaGetErrorString(result)); exit(1);}}


#define MY_PI 3.14159265358979323846

__device__
double RastriginFunction(float* X, ul GENOME_SIZE)
{
	double sum = 0;
	for (ul i = 0; i < GENOME_SIZE; i++)
	{
		sum += X[i] * X[i] - 10 * cos(2 * MY_PI * X[i]) + 10;
	}
	return sum;
}


__device__
double PlayFunction(float* x, ul gen_size)
{
	return 1/(1+RastriginFunction(x, gen_size));
}


struct Participant
{
	double* fitness;
	float* genome;
};



__global__
void init_state_array_kernel(time_t *seed, curandState_t *state, ul populationSize)
{
	ul tid = threadIdx.x + blockDim.x*blockIdx.x;
	
	if (tid < populationSize)
		curand_init(seed[tid], tid, 0, &state[tid]);
}


__global__
void init_first_population_kernel(
	Participant* population_1, 
	Participant* population_2,
	ul populationSize, ul genomeSize, 
	curandState_t *state_arr, 
	float a, float b)
{
	ul tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid < populationSize)
	{
		curandState_t state = state_arr[tid];

		for (ul i = 0; i < genomeSize; ++i)
		{
			population_1[tid].genome[i] = a + curand_uniform(&state)*(b - a);
			population_2[tid].genome[i] = 1.0;
		}
		state_arr[tid] = state;

		population_1[tid].fitness[0] = PlayFunction(population_1[tid].genome, genomeSize);
		population_2[tid].fitness[0] = 0;
	}
}

__global__ 
void differencial_evolution_kernel_maximize(
	Participant *i_population,										//входная (текущая) популяция
	Participant *o_population,										//выходная ппопуляция
	curandState_t *state,											//массив состояний для генерации случайных величин 
	float **y_genomes_memory,
	const ul populationSize,										//мощность популяции
	const ul genomeSize)
{
	ul tid = threadIdx.x + (blockIdx.x * blockDim.x);

	if (tid < populationSize)
	{
		//// Parametrs //////////////////
		const float f_parameter = 0.7;
		const float cr_parameter = 0.5;
		////////////////////////////////
		//printf("tid: %d\n", tid);

		curandState_t local_state = state[tid];
		

		ul a, b, c;
		do
		{
			a = (ul)curand(&local_state) % populationSize;
		} while (a == tid);

		do
		{
			b = (ul)curand(&local_state) % populationSize;
		} while (b == tid || b == a);

		do
		{
			c = (ul)curand(&local_state) % populationSize;
		} while (c == tid || c == b || c == a); //выбрали три неравных индекса для создания мутантного вектора

		
		float* y_genome = nullptr;  
		y_genome = y_genomes_memory[tid];
		if (y_genome == nullptr)
			printf("tid = %d, y_genome init fail! \n", tid);

		
		float rand_val = 0;
		for (int j = 0; j < genomeSize; ++j)
		{
			rand_val = curand_uniform(&local_state);
			if (rand_val < cr_parameter)
			{				
				y_genome[j] = i_population[a].genome[j] + f_parameter * (i_population[b].genome[j] - i_population[c].genome[j]);
			}
			else
			{
				y_genome[j] = i_population[tid].genome[j];
			}
		} //собрали мутантный вектор
		
		state[tid] = local_state; //вернули обновленное зерно генератора в общий массив
		
		double y_fitness;
		y_fitness = PlayFunction(y_genome, genomeSize);

		double x_fitness;
		x_fitness = i_population[tid].fitness[0];
		//printf("tid=%d \n    y is identical to x in %d genome parts\n    y fitness= %f  | x fitness = %f\n", tid, count, y_fitness, x_fitness);

		float *winner_genome;
		double winner_fitness;

		if (y_fitness > x_fitness)
		{
			winner_genome = y_genome;
			winner_fitness = y_fitness;
		}
		else
		{
			winner_genome = i_population[tid].genome;
			winner_fitness = x_fitness;
		} //записали в результат более приспособленного


		for (ul i = 0; i < genomeSize; ++i)
		{
			o_population[tid].genome[i] = winner_genome[i];
		}
		o_population[tid].fitness[0] = winner_fitness;

		//free(y_genome);
	}
}

struct Enviroment
{
	const ul genomeSize;
	const ul populationSize;

	Participant *population_1;
	Participant *population_2;

	curandState_t *state_arr;

	float **y_genomes;

	Enviroment(ul gs, ul ps, float a, float b)
		:genomeSize(gs), populationSize(ps)
	{
		///////Init randoms//////////////////////////////////
		time_t *seed_arr;
		cudaMallocManaged(&seed_arr, populationSize * sizeof(time_t)); CHECK_CUDA_RESULT
		srand(time(0));
		for (ul i = 0; i < populationSize; ++i)
		{
			seed_arr[i] = (time_t)rand();
		} //инициализировали на хосте массив сидов для дальнейшей инициализации генераторов

		cudaMallocManaged(&state_arr, populationSize * sizeof(curandState_t)); CHECK_CUDA_RESULT
		
		ui blockSize = 256;
		ui numBlocks = (populationSize / blockSize) + 1;

		init_state_array_kernel<<<numBlocks, blockSize>>>(seed_arr, state_arr, populationSize); CHECK_CUDA_RESULT
		//////////////////////////////////////////////////////

		////////////// Allocate memory  & Initiate start population//////////////////////
		cudaMallocManaged(&population_1, populationSize * sizeof(Participant)); CHECK_CUDA_RESULT
		cudaMallocManaged(&population_2, populationSize * sizeof(Participant)); CHECK_CUDA_RESULT
		
		float *temp_genome;
		double *temp_fitness;

		for (ul i = 0; i < populationSize; ++i)
		{
			cudaMallocManaged(&temp_genome, genomeSize* sizeof(float)); CHECK_CUDA_RESULT
			cudaMallocManaged(&temp_fitness, sizeof(double)); CHECK_CUDA_RESULT
			population_1[i].genome = temp_genome;
			population_1[i].fitness = temp_fitness;
			
			cudaMallocManaged(&temp_genome, genomeSize * sizeof(float)); CHECK_CUDA_RESULT
			cudaMallocManaged(&temp_fitness, sizeof(double)); CHECK_CUDA_RESULT
			population_2[i].genome = temp_genome;
			population_2[i].fitness = temp_fitness;
		}

		init_first_population_kernel << <numBlocks, blockSize >> > (population_1, population_2, populationSize, genomeSize, state_arr, a, b); CHECK_CUDA_RESULT
		/////////////////////////////////////////////////////
		
		////Allocate memory for temp genome/////
		cudaMallocManaged(&y_genomes, populationSize*sizeof(float*)); CHECK_CUDA_RESULT
		for (ul i = 0; i < populationSize; ++i)
		{
			cudaMallocManaged(&y_genomes[i], genomeSize * sizeof(float)); CHECK_CUDA_RESULT

			for (ul j=0; j<genomeSize; ++j)
			{
				y_genomes[i][j] = 1;
			}
		}

		bool all_ok = true;
		for (ul i = 0; i < populationSize; ++i) //check results
		{
			bool check_y_gen = false;
			ul count_err = 0;
			for (ul j=0; j<genomeSize; ++j)
			{
				if (y_genomes[i][j] != 1)
				{
					check_y_gen = true;
					all_ok = false;
					count_err++;
				}					
			}
			if (check_y_gen)
				cout << "y_genome[" << i << "] init fail! Error count =" << count_err << endl << endl; 
		}
		if (all_ok)
			cout << "Mutant vector genomes initiated SUCCCES" << endl << endl;

		//////////////////////////////////////

		//		
		cudaDeviceSynchronize();
		cout << "check_populations result:" << endl;
		check_populations(a, b);
		
		/*cout << endl << "initiated population 1: " << endl;
		print_population(population_1);

		cout << endl << "initiated population 2: " << endl;
		print_population(population_2);*/
	}

	void print_population(Participant* population)
	{
		for (int i = 0; i < populationSize; ++i)
		{
			cout << "participant №" << i << " | fitness: " << population[i].fitness[0] << " genome: ( ";
			for (int j = 0; j < genomeSize; ++j)
			{
				cout << population[i].genome[j] << " , ";
			}
			cout << ")" << endl;
		}
	}

	ul find_best_fitness_index(Participant* population)
	{
		ul max_fit_ind = 0;

		for (ul i = 0; i < populationSize; ++i)
		{
			if (population[i].fitness[0] > population[max_fit_ind].fitness[0])
				max_fit_ind = i;
		}

		return max_fit_ind;
	}

	void check_populations( float a, float b)
	{
		bool all_ok = true;
		
		for (ul i = 0; i < populationSize; ++i)
		{
			bool check_pop_1 = false;
			ul err_count_1 = 0;

			bool check_pop_2 = false;
			ul err_count_2 = 0;

			for (ul j = 0; j < genomeSize; ++j)
			{
				if (population_1[i].genome[j] < a || population_1[i].genome[j] > b)
				{
					err_count_1++;
					check_pop_1 = true;
					all_ok = false;
				}
				
				if (population_2[i].genome[j] != 1)
				{
					err_count_2++;
					check_pop_2 = true;
					all_ok = false;
				}		
			}

			if (check_pop_1)
			{
				cout << "population_1[" << i << "] init failure, number err_genes:"<< err_count_1 <<" ,genome: (";

				for (ul k = 0; k < genomeSize; ++k)
					cout << population_1[i].genome[k] << " , ";
				cout << ")" << endl << endl;
			}
				
			if (check_pop_2)
			{
				cout << "population_2[" << i << "] init failure, number err_genes:" << err_count_2 << " ,genome: (";

				for (ul k = 0; k < genomeSize; ++k)
					cout << population_2[i].genome[k] << " , ";
				cout << ")" << endl << endl;
			}				
		}
		
		if (all_ok)
			cout << "Popultions init SUCCES" << endl << endl;
	}

	void DifEv(float* result, ui number_of_generations)
	{
		//Participant* current_population = population_1;
		//Participant* new_population = population_2;
		
		ui blockSize = 256;
		ui numBlocks = (populationSize  / blockSize) +1;

		size_t heapsize = sizeof(float) * genomeSize * populationSize * 3;
		cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapsize);

		for (ui i = 1; i <= number_of_generations; ++i)
		{
			if (i % 2) //1, 3, 5...
			{
				differencial_evolution_kernel_maximize <<<numBlocks, blockSize >>> (population_1, population_2, state_arr, y_genomes, populationSize, genomeSize); CHECK_CUDA_RESULT

				cudaDeviceSynchronize();
				ul best_ind = find_best_fitness_index(population_2);
				cout << "Generation: " << i << " | Best index: " << best_ind << " | Fitnes:  " << population_2[best_ind].fitness[0] << endl;
			}
			else //2, 4, 6 ...
			{
				differencial_evolution_kernel_maximize << <numBlocks, blockSize >> > (population_2, population_1, state_arr, y_genomes ,populationSize, genomeSize); CHECK_CUDA_RESULT

				cudaDeviceSynchronize();
				ul best_ind = find_best_fitness_index(population_1);
				cout << "Generation: " << i << " | Best index: " << best_ind << " | Fitnes:  " << population_1[best_ind].fitness[0] << endl;
			}
		}

		ul best_ind = find_best_fitness_index(population_1); //Дописать для числа поколений!!!!
		for (ul i = 0; i < genomeSize; ++i)
		{
			result[i] = population_1[best_ind].genome[i];
		}
	}
};


int main()
{
	ul genomeSize = 10000;
	ul populationSize = genomeSize*7;
	ui numberOfGenerations = 5000000;
	
	Enviroment e(genomeSize, populationSize,  -10, 10);

	float* res = new float[genomeSize];

	e.DifEv(res, numberOfGenerations);

	return 0; 
}