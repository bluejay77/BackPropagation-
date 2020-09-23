/*

A backpropagation algorithm example, from Prof Daniel Graupe's book,
2nd edition.

The backpropagation in the C++ language.  Example:
the character recognition in the C++ and the
backpropagation algorithm,

Dr Antti Juhani Ylikoski 2020-09-10

The Daniel Graupe example from the pp. 69 --> 76.

This file is bp-cpp.cpp

Usage:

g++ bp-cpp-cpp -o bp
./bp

This one will need some debugging and maybe some 
adptation and porting work.


*/

#include<math.h>
#include<iostream>
#include<fstream>

using namespace std;

#define N_DATASETS 9
#define N_INPUTS 36
#define N_OUTPUTS 2
#define N_LAYERS 3

// {# inputs, # of neurons in L1, # of neurons in L2, # of neurons in L3}
short conf[4] = {N_INPUTS, 2, 2, N_OUTPUTS};

float **w[3], *z[3], *y[3], *Fi[3], eta;
// According to the number of layers

// the w[i,j,k] is the ith layer, the jth neuron, her kth weight
//
// the z[i,j] are the bias
//
// the y[i,j] are the outputs of the activation function
//
// the Fi[i,j] are aux variables



ofstream ErrorFile("error.txt", ios::out);



// 3 training sets


bool dataset[N_DATASETS][N_INPUTS] = {
{ 0, 0, 1, 1, 0, 0,    // ‘A’
  0, 1, 0, 0, 1, 0,
  1, 0, 0, 0, 0, 1,
  1, 1, 1, 1, 1, 1,
  1, 0, 0, 0, 0, 1,
  1, 0, 0, 0, 0, 1},

{ 1, 1, 1, 1, 1, 0,    // ‘B’
  1, 0, 0, 0, 0, 1,
  1, 1, 1, 1, 1, 0,
  1, 0, 0, 0, 0, 1,
  1, 0, 0, 0, 0, 1,
  1, 1, 1, 1, 1, 0},

{ 0, 1, 1, 1, 1, 1,    // 'C'
  1, 0, 0, 0, 0, 0,
  1, 0, 0, 0, 0, 0,
  1, 0, 0, 0, 0, 0,
  0, 1, 1, 1, 1, 1},


{ 1, 1, 1, 1, 1, 0,    // 'D'
  1, 0, 0, 0, 0, 1,
  1, 0, 0, 0, 0, 1,
  1, 0, 0, 0, 0, 1,
  1, 1, 1, 1, 1, 0},

{ 1, 1, 1, 1, 1, 1,    // 'E'
  1, 0, 0, 0, 0, 0,
  1, 1, 1, 1, 1, 1,
  1, 0, 0, 0, 0, 0,
  1, 1, 1, 1, 1, 1},

{ 1, 1, 1, 1, 1, 1,    // 'F'
  1, 0, 0, 0, 0, 0,
  1, 1, 1, 1, 1, 1,
  1, 0, 0, 0, 0, 0,
  1, 0, 0, 0, 0, 0},

{ 0, 1, 1, 1, 1, 1,    // 'G' 
  1, 0, 0, 0, 0, 0,
  1, 0, 0, 0, 0, 0,
  1, 0, 1, 1, 1, 1,
  1, 0, 0, 0, 0, 1,
  0, 1, 1, 1, 1, 1},

{ 1, 0, 0, 0, 0, 1,    // 'H'
  1, 0, 0, 0, 0, 1,
  1, 1, 1, 1, 1, 1,
  1, 0, 0, 0, 0, 1,
  1, 0, 0, 0, 0, 1},

{ 0, 0, 1, 1, 1, 0,   // 'I'
  0, 0, 0, 1, 0, 0,
  0, 0, 0, 1, 0, 0,
  0, 0, 0, 1, 0, 0,
  0, 0, 1, 1, 1, 0},


// Below are the datasets for checking "the rest of the world".
// They are not the ones the NN was trained on.


/* { 1, 0, 0, 0, 0, 1,    // ‘X’
0, 1, 0, 0, 1, 0,
0, 0, 1, 1, 0, 0,
0, 0, 1, 1, 0, 0,
0, 1, 0, 0, 1, 0,
1, 0, 0, 0, 0, 1},


{ 0, 1, 0, 0, 0, 1,        // ‘Y’
0, 0, 1, 0, 1, 0,
0, 0, 0, 1, 0, 0,
0, 0, 0, 1, 0, 0,
0, 0, 0, 1, 0, 0,
0, 0, 0, 1, 0, 0},


{ 1, 1, 1, 1, 1, 1,        // ‘Z’
0, 0, 0, 0, 1, 0,
0, 0, 0, 1, 0, 0,
0, 0, 1, 0, 0, 0,
0, 1, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1}   */
},


datatrue[N_DATASETS][N_OUTPUTS] = {{0,1}, {1,0}, {1,1},
				   {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}};



// Memory allocation and initialization functions


void MemAllocAndInit(char S)
{
  if(S == 'A')
    for(int i = 0; i < N_LAYERS; i++)
      {
	w[i] = new float*[conf[i + 1]];
	z[i] = new float[conf[i + 1]];
	y[i] = new float[conf[i + 1]]; Fi[i] = new float[conf[i + 1]];
	for(int j = 0; j < conf[i + 1]; j++)
	  {
	    w[i][j] = new float[conf[i] + 1];
	    // Initializing in the range (-0.5;0.5) (including bias weight)
	    for(int k = 0; k <= conf[i]; k++)
	      w[i][j][k] = rand()/(float)RAND_MAX - 0.5;
	  }
      }
  if(S == 'D')
    {
      for(int i = 0; i < N_LAYERS; i++)
	{
	  for(int j = 0; j < conf[i + 1]; j++)
	    {
	      delete[] w[i][j];
	      delete[] w[i], z[i], y[i], Fi[i];
	    }
	}
    }
}



// Activation function

float FNL(float z)
{
  float y;
  y = 1. / (1. + exp(-z));
  return y;
}



// Applying input
void ApplyInput(short sn)
{
float input;
 for(short i = 0; i < N_LAYERS; i++) // Counting layers
   for(short j = 0; j < conf[i + 1]; j++) // Counting neurons in each layer
     {
       z[i][j] = 0.;
       // Counting input to each layer (= # of neurons in the previous layer)
       for(short k = 0; k < conf[i]; k++)
	 {
	   if(i) // If the layer is not the first one input = y[i - 1][k];
	     {
	       input = dataset[sn][k];
	       z[i][j] += w[i][j][k] * input;
	     }
	 }
       z[i][j] += w[i][j][conf[i]]; // Bias term
       y[i][j] = FNL(z[i][j]);
     }
}


// Training function, tr - # of runs

void Train(int tr)
{
  short i, j, k, m, sn;
  float eta, prev_output, multiple3, SqErr, eta0;
  eta0 = 1.5; // Starting learning rate eta = eta0;
  for(m = 0; m < tr; m++) // Going through all tr training runs
    {
      SqErr = 0.;
      // Each training run consists of runs through
      // each training set. id est every DATASET
      for(sn = 0; sn < N_DATASETS; sn++)
	{
	  ApplyInput(sn);  // sn = set number
	  // Counting the layers down
	  for(i = N_LAYERS - 1; i >= 0; i--)
	    // Counting neurons in the layer i
	    for(j = 0; j < conf[i + 1]; j++)
	      {
		if(i == 2) // If it is the output layer pro input?
		  multiple3 = datatrue[sn][j] - y[i][j];
		else
		  multiple3 = 0.;
		// Counting neurons in the following layer
		for(k = 0; k < conf[i + 2]; k++)
		  multiple3 += Fi[i + 1][k] * w[i + 1][k][j];
		Fi[i][j] = y[i][j] * (1 - y[i][j]) * multiple3;
		// Counting weights in the neuron
		// (neurons in the previous layer)
		for(k = 0; k < conf[i]; k++)
		  {
		    if(i) // If it is not a first layer
		      prev_output = y[i - 1][k];
		    else
		      prev_output = dataset[sn][k];
		    w[i][j][k] += eta * Fi[i][j] * prev_output;
		  }
		// Bias weight correction
		w[i][j][conf[i]] += eta * Fi[i][j];
	      }
	  SqErr += pow((y[N_LAYERS - 1][0] - datatrue[sn][0]), 2) +
	    pow((y[N_LAYERS - 1][1] - datatrue[sn][1]), 2);
	}
    }
  // ErrorFile << 0.5 * SqErr << endl;
// Decrease learning rate every 100th iteration
  if(!(m % 100))
  eta /= 2.;
// Go back to original learning rate every 400th iteration
  if(!(m % 400))
    eta = eta0;
}



// Prints complete information about the network
void PrintInfo(void)
{
  for(short i = 0; i < N_LAYERS; i++) // Counting layers
    {
      cout << "LAYER " << i << endl;
      // Counting neurons in each layer
      for(short j = 0; j < conf[i + 1]; j++)
	{
	  cout << "NEURON " << j << endl;
	  // Counting input to each layer (= # of neurons in the previous layer)
	  for(short k = 0; k < conf[i]; k++)
	    cout << "w[" << i << "][" << j << "][" << k << "]=" << w[i][j][k]
		 << ' ';
	  cout << "w[" << i << "][" << j << "][BIAS]=" << w[i][j][conf[i]]
	       << ' ' << endl;
	  cout << "z[" << i << "][" << j << "]=" << z[i][j] << endl;
	  cout << "y[" << i << "][" << j << "]=" << y[i][j] << endl;
	}
    }
}




      
// Prints the output of the network
void PrintOutput(void)
 {
   // Counting number of datasets
     for(short sn = 0; sn < N_DATASETS; sn++)
       {
	 ApplyInput(sn);
	 cout << "TRAINING SET " << sn << ": [ ";
	 // Counting neurons in the output layer
	 for(short j = 0; j < conf[3]; j++)
	   cout << y[N_LAYERS - 1][j] << ' ';
	 cout << "] ";
	 if (y[N_LAYERS - 1][0] > (datatrue[sn][0] - 0.1)
	     && y[N_LAYERS - 1][0] < (datatrue[sn][0] + 0.1)
	     && y[N_LAYERS - 1][1] > (datatrue[sn][1] - 0.1)
	     && y[N_LAYERS - 1][1] < (datatrue[sn][1] + 0.1))
	   cout << "--- RECOGNIZED ---";
	 else
	   cout << "--- NOT RECOGNIZED ---";
	 cout << endl;
       }
 }




	 

// Loads weights from a file
void LoadWeights(void)
{
  float in;
  ifstream file("weights.txt", ios::in);
  // Counting layers
  for(short i = 0; i < N_LAYERS; i++)
    // Counting neurons in each layer
    for(short j = 0; j < conf[i + 1]; j++)
      // Counting input to each layer (= # of neurons in the previous layer)
      for(short k = 0; k <= conf[i]; k++)
	{
	  file >> in;
	  w[i][j][k] = in;
	}
  file.close();
}




// Saves weights to a file
void SaveWeights(void)
{
  ofstream file("weights.txt", ios::out);
  // Counting layers
  for(short i = 0; i < N_LAYERS; i++)
    // Counting neurons in each layer
    for(short j = 0; j < conf[i + 1]; j++)
      // Counting input to each layer (= # of neurons in the previous layer)
      for(short k = 0; k <= conf[i]; k++)
	file << w[i][j][k] << endl;
  file.close();
}



// Gathers recognition statistics for 1 and 2 false bit cases void
void GatherStatistics(void)
{
  short sn, j, k, TotalCases;
  int cou;
  cout << "WITH 1 FALSE BIT PER CHARACTER:" << endl; TotalCases = conf[0];
  // Looking at each dataset
  for(sn = 0; sn < N_DATASETS; sn++)
    {
      cou = 0;
      // Looking at each bit in a dataset
      for(j = 0; j < conf[0]; j++)
	{
	  if(dataset[sn][j])
	    dataset[sn][j] = 0;
	  else
	    dataset[sn][j] = 1; ApplyInput(sn);
	  if (y[N_LAYERS - 1][0] > (datatrue[sn][0] - 0.1)
	      && y[N_LAYERS - 1][0] < (datatrue[sn][0] + 0.1)
	      && y[N_LAYERS - 1][1] > (datatrue[sn][1] - 0.1)
	      && y[N_LAYERS - 1][1] < (datatrue[sn][1] + 0.1))
	    cou++;
	  if(dataset[sn][j]) // Switching back
	    dataset[sn][j] = 0;
	  else
	    dataset[sn][j] = 1;
	}
      cout << "TRAINING SET " << sn << ": " << cou << '/' << TotalCases
	   << " recognitions (" << (float)cou / TotalCases * 100. << "%)"
	   << endl;
      cout << "WITH 2 FALSE BITS PER CHARACTER:" << endl;
      TotalCases = conf[0] * (conf[0] - 1.);
      // Looking at each dataset
      for(sn = 0; sn < N_DATASETS; sn++)
	{
	  cou = 0;
	  // Looking at each bit in a dataset for(j = 0; j < conf[0]; j++)
	  for(k = 0; k < conf[0]; k++)
	    {
	      if(j == k)
		continue;
	      if (dataset[sn][j])
		dataset[sn][j] = 0;
	      else
		dataset[sn][j] = 1;
	      if (dataset[sn][k])
		dataset[sn][k] = 0;
	      else
		dataset[sn][k] = 1; ApplyInput(sn);
	      if (y[N_LAYERS - 1][0] > (datatrue[sn][0] - 0.1)
		  && y[N_LAYERS - 1][0] < (datatrue[sn][0] + 0.1)
		  && y[N_LAYERS - 1][1] > (datatrue[sn][1] - 0.1)
		  && y[N_LAYERS - 1][1] < (datatrue[sn][1] + 0.1))
		cou++;
	      if(dataset[sn][j]) // Switching back
		dataset[sn][j] = 0;
	      else
		dataset[sn][j] = 1;
	      if (dataset[sn][k])
		dataset[sn][k] = 0;
	      else
		dataset[sn][k] = 1;
	    }
	}
      cout << "TRAINING SET " << sn << ": " << cou << '/' << TotalCases
	   << " recognitions (" << (float)cou / TotalCases * 100. << "%)"
	   << endl;
    }
}



// Entry point: main menu

void hidden_main(void)
{
  short ch;
  int x;
  MemAllocAndInit('A');
  do
    {
      system("clear");
      cout << "MENU" << endl;
      cout << "1. Apply input and print parameters" << endl;
      cout << "2. Apply input (all training sets) and print output" << endl;
      cout << "3. Train network" << endl; cout << "4. Load weights" << endl;
      cout << "5. Save weights" << endl;
      cout << "6. Gather recognition statistics" << endl;
      cout << "0. Exit" << endl;
      cout << "Your choice: "; cin >> ch;
      cout << endl;
      switch(ch)
	{
	case 1: cout << "Enter set number: ";
	  cin >> x;
	  ApplyInput(x);
	  PrintInfo(); break;
	case 2: PrintOutput();
	  break;
	case 3: cout << "How many training runs?: ";
	  cin >> x; Train(x); break;
	case 4: LoadWeights();
	  break;
	case 5: SaveWeights();
	  break;
	case 6: GatherStatistics();
	  break;
	case 0: MemAllocAndInit('D');
	  return;
	}
      cout << endl;
      cin.get();
      cout << "Press ENTER to continue..." << endl;
      cin.get();
    }
  while(ch);
}



int main()
{
  printf("Hello, world!\n");
}


