/****************************************************************************
  This program is the first in a series that looks at adding a penny to a
  bank account balance. It simply sets up a bank account balance as an
  integer and calls a function to add a penny to the account. This is done 
  in a convoluted way (instead of just b++). This is for consistency with 
  later versions that will demonstrate intermittent problems in the increment
  operation that can be accentuated when a delay is introduced.
  
  Compile with:

    cc -o penny01 penny01.c 

  Dr. Kevan Buckley, University of Wolverhampton, 2018
*****************************************************************************/

#include <stdio.h>
#include <unistd.h>

void add_penny(int *balance) {
  int b = *balance;
  usleep(1000000);
  b = b + 1;
  *balance = b;
}

int main(){
  int account = 0;
  add_penny(&account);
  printf("accumulated %dp\n", account);
  return 0;
} 
