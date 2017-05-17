#define LIQUID 1
#define SOLID 0

// KERNEL INFLUENCE REGION CONSTANT
double kern_constant = 2;
int max_neighbors = 50;


// EQUATION OF STATE
// -> Refernce velocity will be used to determine c and B in the equation of state. Refer to J. Comput. Phys., 110 (1994), pp. 399–406.
// -> Reference velocity should indicate the probable upper bound on the bulk velocity of the fluid medium.
double ref_velocity = 0.5;
// ->calculation of other variables based on ref_velocity
double c = 10 * ref_velocity;
double gamma = 7, back_press = 0;
double p0 = (1e7) / (1000 * gamma);//    3.04e8; was 3


// BODY FORCE
double constant_gravity = 9.8;

// SURFACE TENSION 
double S_ll = 2e-4;
double S_ls = 2e-4;

//NOTE: There are numerous other factors/parameters present in the main source to tweak/fine tune. I'll expose them to the user through this file in subsequent updates.