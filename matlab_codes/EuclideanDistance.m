function [ result ] = EuclideanDistance( x1,x2 )

sum=0;
for i=1:length(x1)
    sum=sum+(x1(i)-x2(i))^2;
end
result=sqrt(sum);

end

