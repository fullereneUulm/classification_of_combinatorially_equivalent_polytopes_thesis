function [ result ] = r_n_betta( betta )

global data;

sum=0;
 for i=1:length(data)
     sum=sum+rankMinimum(i)^betta;
 end
 
 result=sum;   


end

